import numpy as np
from algorithms import DMRG_anyH as dmrg
from library import Parameters as pm, Qubism
from library.BasicFunctions import mkdir, save_pr, load_pr
import os.path as path
from library.BasicFunctions import output_txt


# h = np.arange(0, 0.82, 0.82/100)
# print(h)

# delta = 0.4  # Critical field: 0.41
# num_samples = 100
# h1 = np.random.rand(num_samples, 1) * delta
# h2 = np.random.rand(num_samples, 1) * delta + (0.82 - delta)
# h = np.vstack((h1, h2))

gap = [0.5]
for delta in gap:
    num_samples = 100
    h1 = np.random.rand(num_samples, 1) * delta
    h2 = np.random.rand(num_samples, 1) * delta + (1 - delta)
    h = np.vstack((h1, h2))

    j = [1]
    tol = 1e-4  # to judge if the state has two-fold degeneracy
    lattice = 'chain'
    para = pm.generate_parameters_dmrg(lattice)
    para['spin'] = 'one'
    para['bound_cond'] = 'periodic'
    para['chi'] = 128
    para['l'] = 12
    para['jxy'] = 1
    para['hx'] = 0
    model = 'Spin_' + para['spin'] + '_' + lattice
    # para['data_path'] = '..\\dataQubism\\states_' + model + '\\'
    para['data_path'] = 'E:\\tmpData\\states_' + model + '\\'
    para['image_path'] = '..\\dataQubism\\images_' + model + '\\train 0-' + str(delta)
    mkdir(para['data_path'])
    mkdir(para['image_path'])

    n_mid = int(para['l']/2)
    nj = len(j)
    nh = len(h)
    for n1 in range(0, nj):
        for n2 in range(0, nh):
            para['jz'] = j[n1]
            para['hz'] = h[n2]
            para = pm.make_consistent_parameter_dmrg(para)
            # Run DMRG
            if path.isfile(path.join(para['data_path'], para['data_exp'] + '.pr')):
                print('Load existing data ...')
                a = load_pr(path.join(para['data_path'], para['data_exp'] + '.pr'), 'a')
            else:
                print('Start DMRG calculation ...')
                ob, a, info, para = dmrg.dmrg_finite_size(para)
                save_pr(para['data_path'], para['data_exp'] + '.pr', (ob, a, info, para),
                        ('ob', 'a', 'info', 'para'))
            print('The entanglement gap is ' +
                  str((a.lm[n_mid][0] - a.lm[n_mid][1]) / a.lm[n_mid][0]))
            if (a.lm[n_mid][0] - a.lm[n_mid][1]) / a.lm[n_mid][0] < tol:
                exp_image = 'Phase1_' + para['data_exp']
            else:
                exp_image = 'Phase2_' + para['data_exp']
            state = a.full_coefficients_mps()
            image = Qubism.state2image(state * 256, para['d'], is_rescale=True)
            # image = Image.fromarray(image.astype(np.uint8))
            image = Qubism.image2rgb(image, if_rescale_1=False)
            image.save(path.join(para['image_path'], exp_image + '.jpg'))

