from library.MPSClass import MpsOpenBoundaryClass
import numpy as np
from os import path
from library.BasicFunctions import load_pr, save_pr, plot, output_txt, get_size
from algorithms.DeepMPSfinite import parameters_dmps, deep_mps_qubit, \
    fidelities_to_original_state, act_umpo_on_mps
from library.TensorBasicModule import open_mps_product_state_spin_up
import copy, sys


length = np.arange(4, 104, 4, dtype=int)
d = 2
chi = 3

para_dmps = parameters_dmps()
para_dmps['num_layers'] = 9
para_dmps['chi_overlap'] = 256
para_dmps['theta'] = 0
para_dmps['num_theta'] = 1

pre_fix = path.basename(__file__)[:-3] + '_'
num_len = length.size
fid = np.zeros((num_len, ))
fid_prod0 = np.zeros((num_len, ))
ent0_mid = np.zeros((num_len, ))
fid_ini = np.zeros((num_len,))

for n in range(num_len):
    length_now = int(length[n])
    a = MpsOpenBoundaryClass(length_now, d, chi, spin='half', way='svd', ini_way='r')
    a.central_orthogonalization(0, normalize=True)

    a.calculate_entanglement_spectrum()
    a.calculate_entanglement_entropy()
    ent0_mid[n] = a.ent[round(a.length/2)]
    fid_ini[n] = a.fidelity_log_by_spins_up()
    # print('Mid entanglement entropy and fid0 = %.12g, %.12g' % (ent0_mid, fid_ini))

    fid_tmp, _, _, mpo, _ = deep_mps_qubit(a, para_dmps)
    fid[n] = fid_tmp[-1]

    mps_ini = open_mps_product_state_spin_up(a.length, a.mps[0].shape[1])[0]
    fid_prod0[n] = fidelities_to_original_state(a, mps_ini, mpo, para_dmps['chi_overlap'])[-1]

output_txt(fid.reshape(-1, ), pre_fix + 'fid')
output_txt(fid_prod0.reshape(-1, ), pre_fix + 'fid_prod0')
output_txt(ent0_mid.reshape(-1, ), pre_fix + 'ent0_mid')
output_txt(fid_ini.reshape(-1, ), pre_fix + 'fid_ini')

