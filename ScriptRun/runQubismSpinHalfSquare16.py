import numpy as np
from algorithms import DMRG_anyH as dmrg
from library import Parameters as pm, Qubism
from library.BasicFunctions import mkdir, save_pr, load_pr
import library.HamiltonianModule as hm
import os.path as path

is_load = False

h = np.array(range(0, 10)) * 0.1

# delta = 0.5
# num_samples = 100
# h1 = np.random.rand(num_samples, 1) * delta
# h2 = np.random.rand(num_samples, 1) * delta + (1 - delta)
# h = np.vstack((h1, h2))

j = [1]

lattice = 'arbitrary'
boundary_cond = 'Periodic'
para = pm.generate_parameters_dmrg(lattice)
para['spin'] = 'half'
para['chi'] = 128
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
# Boundary interactions
if boundary_cond is 'Periodic':
    para['positions_h2_edge'] = np.zeros((8, 2), dtype=int)
    para['positions_h2_edge'] = [
        [0, 6],
        [1, 7],
        [8, 14],
        [9, 15],
        [0, 15],
        [3, 12],
        [4, 11],
        [6, 9],
    ]
    para['positions_h2'] = np.vstack((para['positions_h2'], para['positions_h2_edge']))

para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
model = 'Spin_' + para['spin'] + '_Square' + boundary_cond
# para['data_path'] = '..\\dataQubism\\states_' + model + '\\'
para['data_path'] = 'C:\\Users\\ransh\\Desktop\\tmpdata\\states_' + model + '\\'
para['image_path'] = '..\\dataQubism\\images_' + model + '\\'
mkdir(para['data_path'])
mkdir(para['image_path'])

nj = len(j)
nh = len(h)
mx = np.zeros((nh, ))
for n1 in range(0, nj):
    for n2 in range(0, nh):
        para['jz'] = j[n1]
        para['hx'] = h[n2]
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            para['coeff2'][n * 3 + 2] = para['jz']
        para['data_exp'] = 'jz' + str(para['jz']) + '_hx' \
                           + str(para['hx']) + '_chi' + str(para['chi'])
        para = pm.make_consistent_parameter_dmrg(para)
        # Run DMRG
        if path.isfile(path.join(para['data_path'], para['data_exp'] + '.pr')) and is_load:
            print('Load existing data ...')
            a = load_pr(path.join(para['data_path'], para['data_exp'] + '.pr'), 'a')
        else:
            print('Start DMRG calculation ...')
            ob, a, info, para = dmrg.dmrg_finite_size(para)
            save_pr(para['data_path'], para['data_exp'] + '.pr', (ob, a, info, para),
                    ('ob', 'a', 'info', 'para'))
        mx[n2] = sum(ob['mx']) / para['l']
        if mx[n2] < 1e-8:
            exp_image = 'Phase1_' + para['data_exp']
        else:
            exp_image = 'Phase2_' + para['data_exp']
        state = a.full_coefficients_mps()
        image = Qubism.state2image(state * 256, para['d'], is_rescale=True)
        # image = Image.fromarray(image.astype(np.uint8))
        image = Qubism.image2rgb(image, if_rescale_1=False)
        image.save(path.join(para['image_path'], exp_image + '.jpg'))
print(mx)
