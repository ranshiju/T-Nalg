from algorithms.DMRG_anyH import dmrg_finite_size
from library import Parameters as Pm
import numpy as np
from os import path
from library.BasicFunctions import load_pr, save_pr, plot, output_txt, get_size
from algorithms.DeepMPSfinite import parameters_dmps, deep_mps_qubit, \
    fidelities_to_original_state, act_umpo_on_mps
from library.TensorBasicModule import open_mps_product_state_spin_up
import copy, sys


num_theta = 200
theta = np.arange(0, 1, 1/num_theta)
fid = np.zeros((theta.size, ))
fid_prod0 = np.zeros((theta.size, ))
fid_recover = np.zeros((theta.size, ))


lattice = 'chain'
para_dmrg = Pm.generate_parameters_dmrg(lattice)
para_dmrg['spin'] = 'half'
para_dmrg['bound_cond'] = 'open'
para_dmrg['chi'] = 32
para_dmrg['l'] = 24
para_dmrg['jxy'] = 0
para_dmrg['jz'] = 1
para_dmrg['hx'] = 0.5
para_dmrg['hz'] = 0
para_dmrg = Pm.make_consistent_parameter_dmrg(para_dmrg)

para_dmps = parameters_dmps()
para_dmps['num_layers'] = 1
para_dmps['chi_overlap'] = 256
para_dmps['theta'] = 0
para_dmps['num_theta'] = 100


if path.isfile(path.join(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr')):
    print('Load existing MPS data ...')
    a, ob = load_pr(path.join(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr'), ['a', 'ob'])
else:
    ob, a, info, para_dmrg = dmrg_finite_size(para_dmrg)
    save_pr(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr', (ob, a, info, para_dmrg),
            ('ob', 'a', 'info', 'para'))
print('Energy per site = ' + str(ob['e_per_site']))

a.calculate_entanglement_spectrum()
a.calculate_entanglement_entropy()
ent0_mid = a.ent[round(a.length/2)]
fid0 = a.fidelity_log_by_spins_up()
print('Mid entanglement entropy and fid0 = %.12g, %.12g' % (ent0_mid, fid0))

for tt in range(theta.size):
    para_dmps['theta'] = theta[tt]
    print('theta = ' + str(para_dmps['theta']))
    fid_tmp, _, _, mpo, _ = deep_mps_qubit(a, para_dmps)
    print('Fidelity = ' + str(fid_tmp))
    fid[tt] = fid_tmp[-1]
    # mps_ini = open_mps_product_state_spin_up(a.length, a.mps[0].shape[1])[0]
    # fid_prod0[tt] = fidelities_to_original_state(a, mps_ini, mpo,
    #                                              para_dmps['chi_overlap'])[-1]
    # mps = copy.deepcopy(a)
    # for n in range(para_dmps['num_layers']):
    #     mps_data = act_umpo_on_mps(mps.mps, mpo[n])
    #     mps.input_mps(mps_data, if_deepcopy=False)
    #     mps.orthogonalize_mps(mps.length - 1, 0, normalize=True, is_trun=False)
    #     mps.center = 0
    #     mps.orthogonalize_mps(0, mps.length - 1, normalize=True, is_trun=True,
    #                           chi=para_dmps['chi_overlap'])
    #     mps.center = mps.length - 1
    # fid_recover[tt] = fidelities_to_original_state(a, mps.mps, mpo, para_dmps['chi_overlap'])[-1]


output_txt(fid.reshape(-1, ), 'fid')
# output_txt(fid_prod0.reshape(-1, ), 'fid_prod0')
# output_txt(fid_recover.reshape(-1, ), 'fid_recover')

# exp = 'plot(np.arange(ent[0].size),'
# for n in range(0, ent.__len__(), 2):
#     exp += 'ent[' + str(n) + '],'
# exp = exp[:-1] + ')'
# exec(exp)

# ent = np.hstack(ent)
# output_txt(ent, 'ent')
