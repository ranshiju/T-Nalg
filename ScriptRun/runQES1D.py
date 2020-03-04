from algorithms.QESalgorithms import qes_gs_1d_ed
from library.Parameters import parameter_qes_gs_by_ed, make_para_consistent_qes_gs
from library.BasicFunctions import output_txt
import numpy as np


var = list(range(2, 17))
var_name = 'chi'

para = parameter_qes_gs_by_ed()
para['jxy'] = 0
para['jz'] = 1
para['hx'] = 0.5
para['hz'] = 0
para['l_phys'] = 16  # number of physical sites in the bulk
para['chi'] = 8  # Virtual bond dimension cut-off
para['tau'] = 1e-5
para['dmrg_type'] = 'white'
para['if_load_bath'] = True

n_var = var.__len__()
e_error0 = np.zeros((n_var, 1))
e_error = np.zeros((n_var, 1))
corr_xx = np.zeros((n_var, para['l_phys'] - 1))
corr_zz = np.zeros((n_var, para['l_phys'] - 1))

for n in range(n_var):
    exec('para[\'' + var_name + '\'] = ' + str(var[n]))
    para = make_para_consistent_qes_gs(para)
    bath, solver, ob0, ob = qes_gs_1d_ed(para)
    if para['dmrg_type'] is 'mpo':
        e_error0[n] = abs(-0.318309886183529 - sum(ob0['eb'])/3)
    else:
        e_error0[n] = abs(-0.318309886183529 - ob0['eb'])
    e_error[n] = abs(-0.318309886183529 - ob['e_site'])
    print('Error of E/site by iDMRG = ' + str(e_error0[n]))
    print('Error of E/site by QES = ' + str(e_error[n]))
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
