from algorithms import DMRG_anyH as dmrg
from library import HamiltonianModule as hm, Parameters as pm
from library.BasicFunctions import save_pr as save
from library.BasicFunctions import mkdir
import numpy as np


j = [1]
h = np.arange(0.05, 0.25, 0.05)

lattice = 'arbitrary'
para = pm.generate_parameters_dmrg(lattice)
para['spin'] = 'half'
para['chi'] = 30
para['l'] = 16
para['index1'] = np.arange(0, para['l']).reshape(1, -1)
para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
para['coeff1'] = np.ones((para['l'], 1))

para['positions_h2'] = np.zeros((24, 2), dtype=int)
para['positions_h2'] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [0, 3],
    [3, 4],
    [2, 5],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [7, 8],
    [5, 10],
    [2, 13],
    [1, 14],
    [8, 10],
    [10, 13],
    [13, 14],
    [8, 9],
    [10, 11],
    [12, 13],
    [14, 15],
    [9, 11],
    [11, 12],
    [12, 15]
]
para['positions_h2'] = np.array(para['positions_h2'], dtype=int)
para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])

para['data_path'] = '..\\data_dmrg\\MPS4ML\\'
mkdir(para['data_path'])

nj = len(j)
nh = len(h)
for n1 in range(0, nj):
    for n2 in range(0, nh):
        para['jz'] = j[n1]
        para['hx'] = h[n2]
        para['data_exp'] = 'MPS_square4x4_jz' + str(para['jz']) + '_hx'\
                           + str(para['hx']) + '_chi' + str(para['chi'])
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0]*3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            para['coeff2'][n*3+2] = para['jz']
        para = pm.make_consistent_parameter_dmrg(para)
        # Run DMRG
        ob, A, info, para = dmrg.dmrg_finite_size(para)
        save(para['data_path'], para['data_exp'] + '.pr', (ob, A, info, para),
             ('ob', 'A', 'info', 'para'))

# Calculate the two-body RDM of the p1 and p2 sites
# p1 = 2
# p2 = 3
# rho = A.reduced_density_matrix_two_body(p1, p2)
