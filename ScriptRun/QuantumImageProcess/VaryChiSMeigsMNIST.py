from algorithms.TNmachineLearningAlgo import gtn_one_class
from library.Parameters import parameters_gtn_one_class
from library import TNmachineLearning
import os
import numpy as np
import copy
from library.BasicFunctions import load_pr, save_pr, mean_squared_error, output_txt
import skimage


var_name = 'chi'
var_values = np.arange(44, 52, 4)
# var_values = np.hstack((np.arange(2, 20, 2), np.arange(20, 44, 4)))
# var_values = np.arange(0, 104, 4)
# var_values = np.hstack((np.arange(0, 100, 2), np.arange(100, 700, 10),
#                        np.arange(700, 780, 5), np.arange(780, 784, 2)))
select_way = 'SequencedMeasure'  # 'SequencedMeasure', 'RandomMeasure', 'MaxSEE', 'Variance'
generate_way = 'eigs'  # 'eigs': grey images, 'rand': black-and-white (Lei's way)

# Initialize parameters
dataset = 'mnist'
samples = 'all'
num_features = 100
which_class = 3
chi = 40

# =============================================================
para = parameters_gtn_one_class()
para['dct'] = False
para['d'] = 2
para['step'] = 0.2  # initial gradient step
para['if_save'] = True
para['if_load'] = True
para['dataset'] = dataset
para['class'] = which_class
para['chi'] = chi

psnr_av = np.zeros(var_values.shape)
mse_av = np.zeros(var_values.shape)
ssim = np.zeros(var_values.shape)
b = TNmachineLearning.MachineLearningFeatureMap(para['d'], para['dataset'])
if var_name is not 'which_class':
    b.load_data(data_path=os.path.join(b.project_path, '..\\..\\MNIST\\' + para['dataset'] + '\\'),
                file_sample='t10k-images.idx3-ubyte',
                file_label='t10k-labels.idx1-ubyte', is_normalize=True)
    b.select_samples([para['class']])

is_order_calculated = False
for t in range(var_values.size):
    # Modify parameter
    print('For ' + var_name + ' = ' + str(var_values[t]))
    exec(var_name + ' = ' + str(var_values[t]))
    if var_name in para:
        exec('para[\'' + var_name + '\'] = ' + str(var_values[t]))

    print('Train the generative TN')
    a, para_gtn = gtn_one_class(para)
    if var_name is 'which_class':
        b.load_data(data_path='..\\..\\..\\MNIST\\' + para['dataset'] + '\\',
                    file_sample='t10k-images.idx3-ubyte',
                    file_label='t10k-labels.idx1-ubyte', is_normalize=True)
        b.select_samples([para['class']])

    if select_way is 'SequencedMeasure':
        print('Calculate the sequence of the measurements')
        order_file = os.path.join(para['data_path'], 'Order_'+para['save_exp'])
        if (not is_order_calculated) and os.path.isfile(order_file):
            order = load_pr(order_file, 'order')
        else:
            order = a.mps.markov_measurement(if_restore=True)[0]
            save_pr(para['data_path'], 'Order_'+para['save_exp'], [order], ['order'])
        order_now = copy.copy(order.reshape(-1, )[:num_features])
        is_order_calculated = True
    elif select_way is 'MaxSEE':
        if not is_order_calculated:
            ent = a.mps.calculate_onsite_reduced_density_matrix()[0]
            order = np.argsort(ent)[::-1]
            is_order_calculated = True
        order_now = copy.copy(order.reshape(-1, )[:num_features])
    elif select_way is 'Variance':
        if not is_order_calculated:
            variance = b.variance_pixels()
            order = np.argsort(variance)[::-1]
            is_order_calculated = True
        order_now = copy.copy(order.reshape(-1, )[:num_features])
    else:
        print('Generate random sequence of the measurements')

    print('Generate samples and calculate PSNR for the dataset')
    if var_name is 'which_class':
        b.select_samples([para['class']])
    if samples is 'all':
        samples = list(range(b.images.shape[1]))
    for n in range(samples.__len__()):
        if select_way is 'RandomMeasure':
            order_now = np.random.permutation(a.length)[:num_features]
        img = b.images[order_now, n]
        img_new = a.generate_features(img, pos=order_now, f_max=1, f_min=0,
                                      is_display=False, way=generate_way)
        psnr_av[t] += (skimage.measure.compare_psnr(b.images[:, n], img_new) / samples.__len__())
        mse_av[t] += (mean_squared_error(b.images[:, n], img_new) / samples.__len__())
        ssim[t] += (skimage.measure.compare_ssim(
            b.images[:, n], img_new, data_range=1) / samples.__len__())
    print('Average PSNR = ' + str(psnr_av[t]))
file_name_fixed = os.path.basename(__file__)[:-3] + para['dataset']
output_txt(psnr_av, filename='PSNR'+file_name_fixed)
output_txt(mse_av, filename='MSE'+file_name_fixed)
output_txt(ssim, filename='SSIM'+file_name_fixed)
print('All simulations completed')


