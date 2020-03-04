from library.Parameters import parameters_gtn_one_class
from algorithms.TNmachineLearningAlgo import gtn_one_class
from library import TNmachineLearning
from library.BasicFunctions import load_pr, save_pr, show_multiple_images_v1, \
    join_imgs_in_one_row, save_one_image
import os, copy, numpy as np
from skimage.measure import compare_ssim


'''
Good examples:
MNIST: [3, 9], [[8], [21]]
Fashion_MNIST: [3], [[13]]
'''

dataset = 'mnist'
which_class = 3
num = 100


para = parameters_gtn_one_class()
para['dataset'] = dataset
para['class'] = which_class

b = TNmachineLearning.MachineLearningFeatureMap(2, para['dataset'])
# file_sample='t10k-images.idx3-ubyte', file_label='t10k-labels.idx1-ubyte'
b.load_data(data_path=b.data_path)
b.select_samples([para['class']])
b.show_image(list(np.random.permutation(b.numVecSample)[:num]), title_way=None)


