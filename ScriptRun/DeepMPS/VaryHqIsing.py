from algorithms.DMRG_anyH import dmrg_finite_size
from library import Parameters as Pm
import numpy as np
from os import path
from library.BasicFunctions import load_pr, save_pr, plot, output_txt, get_size
from algorithms.DeepMPSfinite import parameters_dmps, deep_mps_qubit, \
    fidelities_to_original_state, act_umpo_on_mps
from library.TensorBasicModule import open_mps_product_state_spin_up
import copy, sys


hx = np.arange(0.1, 0.9, 0.8/40)

pre_fix = path.basename(__file__)[:-3] + '_'
num_hx = hx.size
fid = np.zeros((num_hx, ))
fid_prod0 = np.zeros((num_hx, ))
ent0_mid = np.zeros((num_hx, ))
fid_ini = np.zeros((num_hx,))

lattice = 'chain'
para_dmrg = Pm.generate_parameters_dmrg(lattice)
para_dmrg['spin'] = 'half'
para_dmrg['bound_cond'] = 'open'
para_dmrg['chi'] = 64
para_dmrg['l'] = 48
para_dmrg['jxy'] = 0
para_dmrg['jz'] = 1
para_dmrg['hx'] = 0.5
para_dmrg['hz'] = 0
para_dmrg = Pm.make_consistent_parameter_dmrg(para_dmrg)

para_dmps = parameters_dmps()
para_dmps['num_layers'] = 1
para_dmps['chi_overlap'] = 256
para_dmps['theta'] = 0
para_dmps['num_theta'] = 1

for n in range(num_hx):
    para_dmrg['hx'] = hx[n]
    para_dmrg = Pm.make_consistent_parameter_dmrg(para_dmrg)

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
    ent0_mid[n] = a.ent[round(a.length/2)]
    fid_ini[n] = a.fidelity_log_to_product_state()
    # print('Mid entanglement entropy and fid0 = %.12g, %.12g' % (ent0_mid, fid_ini))

    save_path = path.join(para_dmrg['project_path'], 'data_dMPS\\')
    save_exp = 'UMPO_layer' + str(para_dmps['num_layers']) + para_dmrg['data_exp'] + '.pr'
    if path.isfile(path.join(save_path, save_exp)):
        print('Load existing MPO data ...')
        mpo, fid_tmp = load_pr(path.join(save_path, save_exp), ['mpo', 'fid'])
    else:
        fid_tmp, _, _, mpo, _ = deep_mps_qubit(a, para_dmps)
        save_pr(save_path, save_exp, [mpo, fid_tmp], ['mpo', 'fid'])
    fid[n] = fid_tmp[-1]

    mps_ini = open_mps_product_state_spin_up(a.length, a.mps[0].shape[1])[0]
    fid_prod0[n] = fidelities_to_original_state(a, mps_ini, mpo, para_dmps['chi_overlap'])[-1]

output_txt(fid.reshape(-1, ), pre_fix + 'fid')
output_txt(fid_prod0.reshape(-1, ), pre_fix + 'fid_prod0')
output_txt(ent0_mid.reshape(-1, ), pre_fix + 'ent0_mid')
output_txt(fid_ini.reshape(-1, ), pre_fix + 'fid_ini')

