from library import Parameters as pm
from algorithms import DMRG_anyH as dmrg
from library.BasicFunctions import save_pr as save
from library.BasicFunctions import mkdir
import numpy as np


j = [1]
h = np.arange(0, 1, 0.05)

lattice = 'chain'
para = pm.generate_parameters_dmrg(lattice)
para['spin'] = 'half'
para['bound_cond'] = 'open'
para['chi'] = 32
para['l'] = 32
para['jxy'] = 0
para['hz'] = 0
para['data_path'] = '..\\data_dmrg\\MPS4ML\\'
mkdir(para['data_path'])

nj = len(j)
nh = len(h)
for n1 in range(0, nj):
    for n2 in range(0, nh):
        para['jz'] = j[n1]
        para['hx'] = h[n2]
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
