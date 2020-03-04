from algorithms.QESalgorithms import qes_1d_ed
from library.Parameters import parameters_ed_ground_state, make_para_consistent_ed
from library.BasicFunctions import output_txt
import numpy as np


var = []
var_name = 'l'

para = parameters_ed_ground_state('chain')
para['jxy'] = 0
para['jz'] = 1
para['hx'] = 0.5
para['hz'] = 0
para['l'] = 18  # number of physical sites in the bulk
para['tau'] = 1e-5
para['bound_cond'] = 'open'

n_var = var.__len__()
e_error = np.zeros((n_var, 1))
corr_xx = np.zeros((n_var, para['l_phys'] - 1))
corr_zz = np.zeros((n_var, para['l_phys'] - 1))

for n in range(n_var):
    exec('para[\'' + var_name + '\'] = ' + str(var[n]))
    para = make_para_consistent_ed(para)
    para['pos4corr'] = np.hstack((np.ones((para['l'] - 1, 1), dtype=int), np.arange(
        2, para['l']-1, dtype=int).reshape(-1, 1)))
    bath, solver, ob0, ob = qes_1d_ed(para)
    e_error[n] = abs(-0.318309886183529 - ob['e_site'])
    if n == 0:
        corr_xx = np.array(ob['corr_xx']).reshape(-1, 1)
        corr_zz = np.array(ob['corr_zz']).reshape(-1, 1)
    else:
        corr_xx = np.hstack((corr_xx, np.array(ob['corr_xx']).reshape(-1, 1)))
        corr_zz = np.hstack((corr_zz, np.array(ob['corr_zz']).reshape(-1, 1)))

output_txt(e_error0, 'e_error0')
output_txt(e_error, 'e_error')
output_txt(abs(corr_xx), 'corr_xx')
output_txt(abs(corr_zz), 'corr_zz')
