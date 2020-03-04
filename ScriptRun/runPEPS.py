import numpy as np
from algorithms.PEPSalgorithms import super_orthogonalization_honeycomb as so_peps
from library.Parameters import generate_parameters_so_honeycomb

para = generate_parameters_so_honeycomb()
para['beta'] = (0.02, 0.05) + tuple(np.arange(0.1, 2.1, 0.1)) + tuple(np.arange(2.2, 4.2, 0.2)) + \
               (4.5, 5) + tuple(range(6, 11)) + tuple(range(12, 32, 2))
para['beta'] = np.array(para['beta'])
para['so_time'] = 10

name = ['chi', 'tau']
value = [[12, 24, 32], [0.01]]

num = [len(value[0]), len(value[1])]
for n0 in range(0, num[0]):
    for n1 in range(0, num[1]):
        print('para[\'' + name[0] + '\'] = ' + str(value[0][n0]))
        exec('para[\'' + name[0] + '\'] = ' + str(value[0][n0]))
        print('para[\'' + name[1] + '\'] = ' + str(value[1][n1]))
        exec('para[\'' + name[1] + '\'] = ' + str(value[1][n1]))
        so_peps(para)
