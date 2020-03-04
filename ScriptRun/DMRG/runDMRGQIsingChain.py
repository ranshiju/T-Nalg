from library import Parameters as pm
from algorithms import DMRG_anyH as dmrg
from library.BasicFunctions import save_pr as save
from library.BasicFunctions import mkdir
import numpy as np


var1 = [1]
var2 = np.arange(4, 40, 4, dtype=int)

lattice = 'chain'
para = pm.generate_parameters_dmrg(lattice)
para['spin'] = 'half'
para['bound_cond'] = 'periodic'
para['chi'] = 128
para['l'] = 32
para['jxy'] = 0
para['jz'] = 1
para['hx'] = 0.5
para['hz'] = 0
para['data_path'] = '..\\data_dmrg\\QIsing\\'
mkdir(para['data_path'])

nvar1 = len(var1)
nvar2 = len(var2)
for n1 in range(0, nvar1):
    for n2 in range(0, nvar2):
        para['jz'] = var1[n1]
        para['l'] = var2[n2]
        # para['is_save_op'] = False
        para = pm.make_consistent_parameter_dmrg(para)
        # Run DMRG
        ob, A, info, para = dmrg.dmrg_finite_size(para)
        save(para['data_path'], para['data_exp'] + '.pr', (ob, A, info, para),
             ('ob', 'A', 'info', 'para'))

# Calculate the two-body RDM of the p1 and p2 sites
# p1 = 2
# p2 = 3
# rho = A.reduced_density_matrix_two_body(p1, p2)

# from library.MPSClass import ln_fidelity_per_site as fid
# z = fid(A.mps, A.mps)
