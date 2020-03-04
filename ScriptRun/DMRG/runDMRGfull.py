from library import Parameters as pm
from algorithms import DMRG_anyH as dmrg
from library.BasicFunctions import save_pr as save
from library.BasicFunctions import mkdir
import numpy as np


theta = np.arange(0.5, 2.6, 0.1)
alpha = np.arange(0, 0.25, 0.05)
print(theta)
print(alpha)

lattice = 'longRange'
para = pm.generate_parameters_dmrg(lattice)
para['l'] = 10  # number of sites
para['jxy'] = 0  # Jxy coupling constant
para['jz'] = 1  # Jz coupling constant
para['hx'] = 0.5  # magnetic field in x direction
para['hz'] = 0  # magnetic field in z direction
para['alpha'] = 0  # decaying parameter of the interaction strength
para['chi'] = 80  # dimension cut-off of DMRG
para['bound_cond'] = 'open'  # boundary condition: open or periodic
para['is_pauli'] = True
para['data_path'] = '.\\data_dmrg\\fullyConnected\\'
mkdir(para['data_path'])

nt = len(theta)
na = len(alpha)
for n1 in range(0, nt):
    for n2 in range(0, na):
        para['theta'] = theta[n1]
        para['alpha'] = alpha[n2]
        para['hx'] = np.cos(theta[n1])
        para['jz'] = np.sin(theta[n1])
        para = pm.make_consistent_parameter_dmrg(para)
        # Run DMRG
        ob, A, info, para = dmrg.dmrg_finite_size(para)
        save(para['data_path'], para['data_exp'] + '.pr', (ob, A, info, para),
             ('ob', 'A', 'info', 'para'))

# Calculate the two-body RDM of the p1 and p2 sites
# p1 = 2
# p2 = 3
# rho = A.reduced_density_matrix_two_body(p1, p2)
