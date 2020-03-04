import numpy as np
from algorithms import DMRG_anyH as dmrg
from library import Parameters as pm, Qubism
from library.BasicFunctions import mkdir, save_pr, load_pr
import os.path as path

is_save_state = False
is_load_state = False

h = np.arange(0.9, 1, 0.002)

# delta = 0.45
# num_samples = 100
# h1 = np.random.rand(num_samples, 1) * delta
# h2 = np.random.rand(num_samples, 1) * delta + (1 - delta)
# h = np.vstack((h1, h2))

j = [1]
lattice = 'chain'
para = pm.generate_parameters_dmrg(lattice)
para['spin'] = 'half'
para['bound_cond'] = 'periodic'
para['chi'] = 128
para['l'] = 16
para['jxy'] = 0
para['hz'] = 0
model = 'Spin_' + para['spin'] + '_' + lattice
para['data_path'] = '..\\dataQubism\\states_' + model + '\\'
# para['data_path'] = 'C:\\Users\\ransh\\Desktop\\tmpdata\\states_' + model + '\\'
para['image_path'] = '..\\dataQubism\\images_test_' + str(h[0]) + '-' + str(h[-1]) + '\\'
mkdir(para['data_path'])
mkdir(para['image_path'])

nj = len(j)
nh = len(h)
for n1 in range(0, nj):
    for n2 in range(0, nh):
        para['jz'] = j[n1]
        para['hx'] = h[n2]
        para = pm.make_consistent_parameter_dmrg(para)
        # Run DMRG
        if path.isfile(path.join(para['data_path'], para['data_exp'] + '.pr')) and is_load_state:
            print('Load existing data ...')
            a = load_pr(path.join(para['data_path'], para['data_exp'] + '.pr'), 'a')
        else:
            print('Start DMRG calculation ...')
            ob, a, info, para = dmrg.dmrg_finite_size(para)
            if is_save_state:
                save_pr(para['data_path'], para['data_exp'] + '.pr', (ob, a, info, para),
                        ('ob', 'a', 'info', 'para'))
        if h[n2] < 0.5:
            exp_image = 'Phase1_' + para['data_exp']
        else:
            exp_image = 'Phase2_' + para['data_exp']
        state = a.full_coefficients_mps()
        image = Qubism.state2image(state * 256, para['d'], is_rescale=True)
        # image = Image.fromarray(image.astype(np.uint8))
        image = Qubism.image2rgb(image, if_rescale_1=False)
        image.save(path.join(para['image_path'], exp_image + '.jpg'))

