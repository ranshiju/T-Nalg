from library.Parameters import parameters_gtn_one_class
from algorithms.TNmachineLearningAlgo import gtn_one_class
from library import TNmachineLearning
from library.BasicFunctions import load_pr, save_pr, show_multiple_images_v1, \
    join_imgs_in_one_row, save_one_image
import os, copy, numpy as np
from skimage.measure import compare_ssim


'''
Good examples:
MNIST: [3, 9], [[8, 196], [21]]
Fashion_MNIST: [3], [[13]]
'''

dataset = 'mnist'
classes = [3]
samples = [[196]]
nums_features = [80]
# nums_features = [0, 5, 10] + list(range(20, 200, 30))
select_way = 'SequencedMeasure'  # 'SequencedMeasure', 'RandomMeasure', 'MaxSEE', 'Variance'
generate_way = 'eigs'  # 'eigs': grey images, 'rand': black-and-white (Lei's way)
if_mark = False


para = parameters_gtn_one_class()
para['dataset'] = dataset
para['class'] = 3
para['chi'] = 40
para['dct'] = False
para['d'] = 2
para['step'] = 0.2  # initial gradient step
para['if_save'] = True
para['if_load'] = True


def get_marked_imgs(gtn, imgs_test, num_f, sample, param, order_way):
    if order_way is 'SequencedMeasure':
        order_file = os.path.join(param['data_path'], 'Order_' + param['save_exp'])
        if os.path.isfile(order_file):
            order = load_pr(order_file, 'order')
        else:
            order = gtn.mps.markov_measurement(if_restore=True)[0]
            save_pr(para['data_path'], 'Order_' + para['save_exp'], [order], ['order'])
        order_now = copy.copy(order.reshape(-1,)[:num_f])
    elif order_way is 'MaxSEE':
        ent = gtn.mps.calculate_onsite_reduced_density_matrix()[0]
        order = np.argsort(ent)[::-1]
        order_now = copy.copy(order.reshape(-1,)[:num_f])
    elif order_way is 'Variance':
        tmp = TNmachineLearning.MachineLearningFeatureMap(param['d'], param['dataset'])
        tmp.load_data()
        tmp.select_samples([param['class']])
        variance = tmp.variance_pixels()
        order = np.argsort(variance)[::-1]
        order_now = copy.copy(order.reshape(-1,)[:num_f])
    elif order_way is 'RandomMeasure':
        order = np.random.permutation(gtn.length)
        order_now = copy.copy(order.reshape(-1,)[:num_f])
    else:
        order = None
        order_now = None
    img_part = imgs_test.images.copy()[order_now, sample]
    img_new = gtn.generate_features(img_part, pos=order_now, f_max=1, f_min=0,
                                    is_display=False, way=generate_way)
    img_new = img_new.reshape(imgs_test.img_size)
    marked_img = imgs_test.mark_pixels_on_full_image(img_new, order_now)
    return imgs_test.images[:, sample].reshape(imgs_test.img_size), img_new, marked_img, order


num_c = list(classes).__len__()
num_f = list(nums_features).__len__()
imgs = list()
titles = list()
num_s_tot = 0
for nc in range(num_c):
    num_s = list(samples[nc]).__len__()
    num_s_tot += num_s
    para['class'] = classes[nc]
    a, para = gtn_one_class(para)
    b = TNmachineLearning.MachineLearningFeatureMap(para['d'], para['dataset'])
    b.load_data(data_path=b.data_path, file_sample='t10k-images.idx3-ubyte',
                file_label='t10k-labels.idx1-ubyte')
    b.select_samples([para['class']])
    for ns in range(num_s):
        imgs_now = list()
        print('Processing the image ' + str(samples[nc][ns]) + ' in class ' + str(para['class']))
        for n in range(num_f):
            img0, img1, marked_img, order = get_marked_imgs(
                a, b, nums_features[n], samples[nc][ns], para, select_way)
            if n == 0:
                imgs_now.append(img0.copy())
                titles.append('Original img')
            if if_mark:
                imgs_now.append(marked_img.copy())
            else:
                imgs_now.append(img1.copy())
            # tmp = img1.reshape(-1, )
            # tmp[order[:nums_features[n]]] = 0
            # show_multiple_images_v1([tmp.reshape(img1.shape)])
            ssim = compare_ssim(img0, img1)
            title = 'N=' + str(nums_features[n]) + ', SSIM=' + str(ssim)
            titles.append(title)
        save_one_image(join_imgs_in_one_row(imgs_now), '0'+select_way)
        imgs = imgs + imgs_now.copy()

save_exp = str(classes) + '_' + dataset
show_multiple_images_v1(imgs, lxy=(num_s_tot, num_f + 1),
                        titles=titles, save_name=save_exp)


