from algorithms.DMRG_anyH import dmrg_finite_size
from library import Parameters as Pm
import numpy as np
from os import path
from library.BasicFunctions import load_pr, save_pr, plot, output_txt, get_size
from algorithms.DeepMPSfinite import parameters_dmps, deep_mps_qubit, \
    fidelities_to_original_state, act_umpo_on_mps
from library.TensorBasicModule import open_mps_product_state_spin_up


"""
NOTE: for Linux, add the fold "T-Nalg" in your system path
"""
para_dmrg = Pm.generate_parameters_dmrg('chain')  # set default parameters of DMRG
para_dmrg['spin'] = 'half'  # spin-half model
para_dmrg['bound_cond'] = 'open'  # open boundary condition
para_dmrg['chi'] = 48  # dimension cut-off of DMRG
para_dmrg['l'] = 48  # system size
para_dmrg['jxy'] = 0  # coupling constant - jx and jy in the Heisenberg model
para_dmrg['jz'] = 1  # coupling constant - jz in the Heisenberg model
para_dmrg['hx'] = 0.3  # magnetic field in the x direction
para_dmrg['hz'] = 0  # magnetic field in the z direction
para_dmrg = Pm.make_consistent_parameter_dmrg(para_dmrg)  # check consistency of the parameters

para_dmps = parameters_dmps()  # set default parameters of MPS encoding
para_dmps['num_layers'] = 9  # number of the MPU layers in the circuit
para_dmps['chi_overlap'] = 256  # dimension cut-off of the disentangled MPS


# calculate the ground state by DMRG
pre_fix = path.basename(__file__)[:-3] + '_'
if path.isfile(path.join(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr')):
    # check if data exist in para_dmrg['data_path']; load if exist.
    # Note: you may need to modify para_dmrg['data_path'] to a local folder when
    #       running this code with different computers or OS
    print('Load existing MPS data ...')
    a, ob = load_pr(path.join(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr'), ['a', 'ob'])
else:
    # run DMRG if no data is found in para_dmrg['data_path']
    ob, a, info, para_dmrg = dmrg_finite_size(para_dmrg)
    save_pr(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr', (ob, a, info, para_dmrg),
            ('ob', 'a', 'info', 'para'))
print('Energy per site = ' + str(ob['e_per_site']))

# calculate entanglement properties and fidelity
a.calculate_entanglement_spectrum()
a.calculate_entanglement_entropy()
ent0_mid = a.ent[round(a.length/2)]
fid0 = a.fidelity_log_to_product_state()
print('Mid entanglement entropy and fid0 = %.12g, %.12g' % (ent0_mid, fid0))

# calculate the MPUs
save_path = path.join(para_dmrg['project_path'], 'data_dMPS/')
save_exp = 'UMPO_layer' + str(para_dmps['num_layers']) + para_dmrg['data_exp'] + '.pr'
if path.isfile(path.join(save_path, save_exp)):
    print('Load existing MPO data ...')
    mpo, fid = load_pr(path.join(save_path, save_exp), ['mpo', 'fid'])
else:
    fid, _, _, mpo, _ = deep_mps_qubit(a, para_dmps, para_dmrg)
    save_pr(save_path, save_exp, [mpo, fid], ['mpo', 'fid'])

fid = fid[1:]  # |<0|(|U_dag\psi>)|
output_txt(fid.reshape(-1, ), pre_fix + 'fid')  # save results as txt file

fid_prod0 = np.zeros((para_dmps['num_layers'], ))  # # |(<0|U)|GS>)|
for n in range(para_dmps['num_layers']):
    mps_ini = open_mps_product_state_spin_up(a.length, a.mps[0].shape[1])[0]
    fid_prod0[n] = fidelities_to_original_state(a, mps_ini, mpo[:n+1], para_dmps['chi_overlap'])[-1]
# plot(fid1)
output_txt(fid_prod0.reshape(-1, ), pre_fix + 'fid_prod0')  # save results as txt file

