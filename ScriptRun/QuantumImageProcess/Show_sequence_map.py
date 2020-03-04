from library.Parameters import parameters_gtn_one_class
from algorithms.TNmachineLearningAlgo import gtn_one_class
from library import TNmachineLearning
from library.BasicFunctions import load_pr, save_pr, show_multiple_images_v1, \
    join_imgs_in_one_row, save_one_image
import os, copy, numpy as np


dataset = 'mnist'
which_class = 3
nums_features = [10, 20, 40, 80, 120, 400]
select_way = 'Variance'  # 'SequencedMeasure', 'RandomMeasure', 'MaxSEE', 'Variance'


para = parameters_gtn_one_class()
para['dataset'] = dataset
para['class'] = which_class
para['chi'] = 16
para['dct'] = False
para['d'] = 2
para['step'] = 0.2  # initial gradient step
para['if_save'] = True
para['if_load'] = True


def get_sequence(gtn, num_f, param, order_way):
    if order_way is 'SequencedMeasure':
        order_file = os.path.join(param['data_path'], 'Order_' + param['save_exp'])
        if os.path.isfile(order_file):
            order = load_pr(order_file, 'order')
        else:
            order = gtn.mps.markov_measurement(if_restore=True)[0]
            save_pr(param['data_path'], 'Order_' + param['save_exp'], [order], ['order'])
        order_now = copy.copy(order[:num_f])
    elif order_way is 'MaxSEE':
        ent = gtn.mps.calculate_onsite_reduced_density_matrix()[0]
        order = np.argsort(ent)[::-1]
        order_now = copy.copy(order[:num_f])
    elif order_way is 'Variance':
        tmp = TNmachineLearning.MachineLearningFeatureMap(param['d'], param['dataset'])
        tmp.load_data()
        tmp.select_samples([param['class']])
        variance = tmp.variance_pixels()
        order = np.argsort(variance)[::-1]
        order_now = copy.copy(order[:num_f])
    elif order_way is 'RandomMeasure':
        order = np.random.permutation(gtn.length)
        order_now = copy.copy(order[:num_f])
    else:
        order = None
        order_now = None
    return order, order_now


a, para = gtn_one_class(para)
imgs = list()
for n in range(nums_features.__len__()):
    seq, seq_now = get_sequence(a, nums_features[n], para, select_way)
    imgs.append(a.show_sequence(seq_now, if_show=False))
save_one_image(join_imgs_in_one_row(imgs), 'Sequence_maps', if_show=True)
