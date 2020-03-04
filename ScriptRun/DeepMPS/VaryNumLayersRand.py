import numpy as np
from os import path
from library.BasicFunctions import load_pr, save_pr, plot, output_txt, get_size, project_path
from algorithms.DeepMPSfinite import parameters_dmps, deep_mps_qubit, \
    fidelities_to_original_state, act_umpo_on_mps
from library.TensorBasicModule import open_mps_product_state_spin_up
import copy, sys
from library.MPSClass import MpsOpenBoundaryClass


length = 24
d = 2
chi = 3
pre_fix = path.basename(__file__)[:-3] + '_'

para_dmps = parameters_dmps()
para_dmps['num_layers'] = 20
para_dmps['chi_overlap'] = 256
para_dmps['theta'] = 0
para_dmps['num_theta'] = 1

a = MpsOpenBoundaryClass(length, d, chi, spin='half', way='svd', ini_way='r')
a.central_orthogonalization(0, normalize=True)

a.calculate_entanglement_spectrum()
a.calculate_entanglement_entropy()
ent0_mid = a.ent[round(a.length/2)]
fid0 = a.fidelity_log_to_product_state()
print('Mid entanglement entropy and fid0 = %.12g, %.12g' % (ent0_mid, fid0))

fid, _, _, mpo, _ = deep_mps_qubit(a, para_dmps)

fid = fid[1:]
# plot(fid)
output_txt(fid.reshape(-1, ), pre_fix + 'fid')

fid_prod0 = np.zeros((para_dmps['num_layers'], ))
for n in range(para_dmps['num_layers']):
    mps_ini = open_mps_product_state_spin_up(a.length, a.mps[0].shape[1])[0]
    fid_prod0[n] = fidelities_to_original_state(a, mps_ini, mpo[:n+1], para_dmps['chi_overlap'])[-1]
# plot(fid1)
output_txt(fid_prod0.reshape(-1, ), pre_fix + 'fid_prod0')

# fid_recover = np.zeros((para_dmps['num_layers'], ))
# for n in range(1, para_dmps['num_layers']+1):
#     mps = copy.deepcopy(a)
#     for nn in range(n):
#         mps_data = act_umpo_on_mps(mps.mps, mpo[nn])
#         mps.input_mps(mps_data, if_deepcopy=False)
#         mps.orthogonalize_mps(mps.length - 1, 0, normalize=True, is_trun=False)
#         mps.center = 0
#         mps.orthogonalize_mps(0, mps.length - 1, normalize=True, is_trun=True,
#                               chi=para_dmps['chi_overlap'])
#         mps.center = mps.length - 1
#     fid_recover[n-1] = fidelities_to_original_state(a, mps.mps, mpo[:n], para_dmps['chi_overlap'])[-1]
# # plot(fid2)
# output_txt(fid_recover.reshape(-1, ), pre_fix + 'fid_recover')


