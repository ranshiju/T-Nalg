import numpy as np
from library import BasicFunctions as bf, TensorBasicModule as tm
from library.MPSClass import MpsOpenBoundaryClass as MPS
import os.path as path
import cv2
import copy
from scipy.sparse.linalg import eigsh
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt

is_debug = False
is_GPU = False


class MachineLearningBasic:

    def __init__(self, dataset='mnist'):
        self.dataset = dataset
        self.dataInfo = dict()
        self.is_normalize_data = True
        self.is_there_labels = True  # For generative tasks, no labels are needed
        self.data_samples = ''
        self.data_labels = ''
        self.images = np.zeros(0)
        self.labels = np.zeros(0, dtype=int)
        self.length = -1
        self.img_size = [0, 0]
        self.is_dct = False
        self.dct_info = dict()
        self.is_label_added = False
        self.project_path = bf.project_path()
        self.data_path = None

        self.identify_dataset()

        self.tmp = None
        if is_debug:
            self.check_consistency()

    def clear_tmp_data(self):
        self.tmp = None

    def identify_dataset(self):
        if self.dataset is 'mnist':
            self.img_size = [28, 28]
            self.data_path = path.join(self.project_path, '..\\..\\MNIST\\')
            file_sample = 'train-images.idx3-ubyte'
            file_label = 'train-labels.idx1-ubyte'
            self.data_samples = path.join(self.data_path, file_sample)
            self.data_labels = path.join(self.data_path, file_label)
            self.is_normalize_data = True
            self.is_there_labels = True
        elif self.dataset is 'fashion_mnist':
            self.img_size = [28, 28]
            self.data_path = path.join(self.project_path, '..\\..\\MNIST\\fashion_mnist\\')
            file_sample = 'train-images.idx3-ubyte'
            file_label = 'train-labels.idx1-ubyte'
            self.data_samples = path.join(self.data_path, file_sample)
            self.data_labels = path.join(self.data_path, file_label)
            self.is_normalize_data = True
            self.is_there_labels = True
        elif self.dataset is 'custom':
            self.is_there_labels = False

    def load_data(self, data_path=None, file_sample=None, file_label=None,
                  is_normalize=None):
        # MNIST files for training: 'train-images.idx3-ubyte', 'train-labels.idx1-ubyte'
        # MNIST files for testing:  't10k-images.idx3-ubyte',  't10k-labels.idx1-ubyte'
        if data_path is None:
            data_path = self.data_path
        if file_label is not None:
            self.data_samples = path.join(data_path, file_sample)
            if self.is_there_labels:
                self.data_labels = path.join(data_path, file_label)
        if not (is_normalize is None):
            self.is_normalize_data = is_normalize
        self.images = bf.decode_idx3_ubyte(self.data_samples)
        if self.is_there_labels:
            self.labels = bf.decode_idx1_ubyte(self.data_labels)
        self.length = self.images.shape[0]
        if self.is_normalize_data:
            self.images /= (254 * (self.images.max() > 2) + 1)

    def input_data(self, images, labels):
        self.data_samples = 'from input'
        self.data_labels = 'from input'
        self.is_normalize_data = True  # the input date are required to be normalized
        self.images = images
        self.labels = labels
        self.length = self.images.shape[0]

    def dct(self, shift=None, factor=None):
        images_dct = np.zeros(self.images.shape)
        if self.is_label_added is False:
            for n in range(self.images.shape[1]):
                images_dct[:, n] = cv2.dct(self.images[:, n].reshape(self.img_size)).reshape(-1, )
        else:
            images_dct[0, :] = self.images[0, :]
            for n in range(self.images.shape[1]):
                images_dct[1:, n] = cv2.dct(self.images[1:, n].reshape(self.img_size)).reshape(-1, )
        self.images = images_dct
        if shift is None:
            self.dct_info['shift'] = self.images.min()
        else:
            self.dct_info['shift'] = shift
        self.images -= self.dct_info['shift']
        if factor is None:
            self.dct_info['factor'] = self.images.max() + 1
        else:
            self.dct_info['factor'] = factor
        self.images /= self.dct_info['factor']
        self.is_dct = True
        # print(self.dct_info['shift'])
        # print(self.dct_info['factor'])

    def analyse_dataset(self):
        # Total number of samples
        self.dataInfo['NumTotalTrain'] = self.labels.__len__()
        # Order the samples and labels
        order = np.argsort(self.labels)
        self.images = bf.sort_vecs(self.images, order, axis=1)
        self.labels = np.array(sorted(self.labels), dtype=int)
        # Total number of classes
        self.dataInfo['NumClass'] = int(self.labels[-1] + 1)
        # Detailed information
        self.dataInfo['nStart'] = np.zeros((self.dataInfo['NumClass'], ), dtype=int)
        self.dataInfo['nClassNum'] = np.zeros((self.dataInfo['NumClass'],), dtype=int)
        self.dataInfo['nStart'][0] = 0
        for n in range(1, self.dataInfo['NumClass']):
            x = tm.arg_find_array(self.labels[self.dataInfo['nStart'][n-1] + 1:] == n,
                                  1, 'first')
            self.dataInfo['nClassNum'][n-1] = x + 1
            self.dataInfo['nStart'][n] = x + self.dataInfo['nStart'][n-1] + 1
        self.dataInfo['nClassNum'][-1] = \
            self.dataInfo['NumTotalTrain'] - self.dataInfo['nStart'][-1]

    def show_average_image(self):
        return np.average(self.images, axis=1).reshape(self.img_size)

    def show_image(self, n, title_way='which_class'):
        # n can be a number or an image
        if type(n) is int:
            l_side = int(np.sqrt(self.length))
            bf.show_multiple_images_v1([self.images[:, n].reshape(l_side, l_side)], titles=[n])
        elif type(n) is np.ndarray:
            bf.show_multiple_images_v1([n])
        else:
            l_side = int(np.sqrt(self.length))
            num = n.__len__()
            n_now = 0
            while num > 0.5:
                dn = min(num, 100)
                if title_way is 'which_sample':
                    titles = [x for x in n[n_now:n_now + dn]]
                elif title_way is 'which_class':
                    titles = [int(self.labels[x]) for x in n[n_now:n_now + dn]]
                else:
                    titles = None
                # bf.show_multiple_images_v0(
                #     [self.images[:, x].reshape(l_side, l_side) for x in n[n_now:n_now + dn]],
                #     gap=0)
                bf.show_multiple_images_v1(
                    [self.images[:, x].reshape(l_side, l_side) for x in n[n_now:n_now+dn]],
                    titles=titles)
                n_now += dn
                num -= dn

    def show_incomplete_image(self, pixels, pos, way=1, if_show=True):
        if way == 0:
            image = np.zeros((self.img_size[0] * self.img_size[1]))
            image[pos] = pixels
            image = image.reshape(self.img_size)
        else:
            image = np.zeros((self.img_size[0] * self.img_size[1], 3))
            image[pos, 0] = pixels
            image = image.reshape(self.img_size + [3])
        if if_show:
            viewer = ImageViewer(image)
            viewer.show()
        return image

    def mark_pixels_on_full_image(self, image0, pos, if_plot=False):
        image = np.zeros((self.img_size[0] * self.img_size[1], 3))
        for n in range(3):
            image[:, n] = image0.reshape(-1, ).copy()
        # image[pos, 0] = (image[pos, 0] + np.ones((len(pos), ))) / 2
        image[pos, :] /= 1.5
        image[pos, 0] = np.ones((len(pos),))
        # image[pos, 1] = np.ones((len(pos),))
        # image[pos, 2] = np.ones((len(pos),))
        image = image.reshape(self.img_size + [3])
        if if_plot:
            viewer = ImageViewer(image)
            viewer.show()
        return image

    def show_sequence(self, pos, save_name=None, if_show=False, back_color=0):
        x = 10
        pos1 = np.arange(pos.size+x, x, -1) / (pos.size+x)
        img = np.ones((self.img_size[0] * self.img_size[1], 3)) * back_color
        img[pos, 0] = pos1
        img = img.reshape(self.img_size + [3])
        if (type(save_name) is str) or if_show:
            plt.figure(1)
            plt.imshow(img)
            if type(save_name) is str:
                plt.savefig('.\\' + save_name)
            if if_show:
                plt.show(1)
        return img

    def variance_pixels(self, pos='all'):
        if pos is 'all':
            pos = range(self.length)
        var = np.zeros(len(pos), )
        for n in range(len(pos)):
            var[n] = np.var(self.images[pos[n], :])
        return var

    def report_dataset_info(self):
        print('There are ' + str(self.dataInfo['NumClass']) + ' classes in the dataset')
        print('Total number of samples: ' + str(self.dataInfo['NumTotalTrain']))
        for n in range(0, self.dataInfo['NumClass']):
            print('\t There are ' + str(self.dataInfo['nClassNum'][n]) +
                  ' samples in the ' + str(n) + '-th class')

    def check_consistency(self):
        if self.dataInfo['NumTotalTrain'] != sum(self.dataInfo['nClassNum']):
            bf.print_error('The total number in the dataset is NOT consistent '
                           'with the sum of the samples of all classes')
        for n in range(0, self.dataInfo['NumClass']):
            start = self.dataInfo['nStart'][n]
            end = start + self.dataInfo['nClassNum'][n]
            tmp = self.labels[start:end]
            if not np.prod(tmp == tmp[0]):
                bf.print_error('In the ' + str(tmp[0]) + '-th labels, not all labels are '
                               + str(tmp[0]))
                print(bf.arg_find_array(tmp != tmp[0]))


class MachineLearningFeatureMap(MachineLearningBasic):

    def __init__(self, d, dataset):
        MachineLearningBasic.__init__(self, dataset)
        self.d = d
        self.vec_classes = list()
        self.vecsImages = np.zeros(0)
        self.vecsLabels = np.zeros(0)
        self.numVecSample = 0
        self.mps = None

    def input_mps(self, mps):
        self.mps = copy.deepcopy(mps)

    def select_samples(self, classes, numbers=None, how='random'):
        self.vec_classes = classes
        num_class = classes.__len__()
        if numbers is None:
            numbers = ['all'] * num_class
        ind_tot = list()
        for n in range(num_class):
            ind = list(np.where(self.labels == classes[n])[0])
            if not (numbers[n] is 'all'):
                numbers[n] = min(numbers[n], ind.__len__())
                if how is 'random':
                    order = np.random.permutation(ind.__len__())[:numbers[n]]
                elif how is 'first':
                    order = list(range(numbers[n]))
                else:
                    order = list(range(ind.__len__() - numbers[n], ind.__len__()))
                ind = [ind[x] for x in list(order)]
            ind_tot += ind
        self.images = self.images[:, ind_tot]
        if self.is_there_labels:
            self.labels = self.labels[ind_tot]
        # self.images = self.images[:, :10]
        # self.labels = self.labels[:10]
        self.numVecSample = ind_tot.__len__()

    def to_black_and_white(self):
        self.images[self.images > 0.5] = 1
        self.images[self.images <= 0.5] = 0

    def true_label2label_order(self):
        label = np.zeros(self.labels.shape)
        for n in range(self.vec_classes.__len__()):
            label += (self.labels == self.vec_classes[n]) * n
        return label

    def add_labels_to_images(self):
        if self.is_label_added is False:
            label = np.array(self.true_label2label_order(), dtype=float)
            label /= (self.vec_classes.__len__() - 1)
            self.images = np.vstack((label.reshape(1, -1), self.images))
            self.length += 1
        else:
            print('Warning: it seems that the labels have already been added.')
        self.is_label_added = True

    def images2vecs(self, theta_max=np.pi/2):
        # The pixels should have been normalized to [0, 1)
        s = self.images.shape
        self.numVecSample = s[1]
        self.images *= theta_max
        self.vecsImages = np.zeros((self.d, ) + s)
        for nd in range(1, self.d+1):
            self.vecsImages[nd-1, :, :] = (np.sqrt(tm.combination(self.d-1, nd-1)) * (
                    np.cos(self.images)**(self.d-nd)) * (np.sin(self.images)**(nd-1)))

    def vecs2images(self, theta_max=np.pi/2, data=None):
        # only apply for d=2
        # img = np.zeros(self.vecsImages.shape[1:])
        if data is None:
            return np.arccos(self.vecsImages[0, :, :]) / theta_max
        else:
            return np.arccos(data[0, :, :]) / theta_max

    def map_to_vectors(self, x, d=None, theta_max=np.pi/2):
        if d is None:
            d = self.d
        x *= theta_max
        y = np.zeros((self.d, ) + x.shape)
        for nd in range(1, self.d + 1):
            y[nd - 1, :, :] = (np.sqrt(tm.combination(d - 1, nd - 1)) * (
                    np.cos(x) ** (d - nd)) * (np.sin(x) ** (nd - 1)))
        return y

    def label2vectors(self, num_channels=None):
        # num_channels is the dimension of the label bonds; it doesn't has to be the number of classes
        num_class = self.vec_classes.__len__()
        if num_channels is None:
            num_channels = num_class
        dn1 = int(num_channels/num_class)
        dn0 = int((num_channels - dn1) / (num_class - 1))
        self.vecsLabels = np.zeros((num_channels, self.numVecSample))
        for n in range(0, self.numVecSample):
            which_c = self.vec_classes.index(self.labels[n])
            n_start = dn0 * which_c
            self.vecsLabels[n_start:n_start + dn1, n] = np.ones((dn1,)) / np.sqrt(dn1)

    def fidelity_mps_image(self, mps_ref, ni):
        # Calculate the fidelity between an MPS and one image
        fid = 0
        length = mps_ref.__len__()
        v0 = np.ones((1, ))
        image = self.vecsImages[:, :, ni]
        for n in range(0, self.length):
            v0 = tm.absorb_vectors2tensors(mps_ref[n], (v0, image[:, n]), (0, 1))
            norm = np.linalg.norm(v0)
            v0 /= norm
            fid += np.log(norm)
        return fid / length

    def compute_fidelities(self, mps):
        fid = np.zeros((self.numVecSample, 1))
        vecs = np.ones((1, self.numVecSample))
        for nt in range(self.length):
            s = mps[nt].shape
            vecs = np.tensordot(mps[nt], tm.khatri(
                vecs, self.vecsImages[:, nt, :]), ([0, 1], [0, 1]))
            norm = np.linalg.norm(vecs, axis=0)
            vecs /= norm.repeat(s[2]).reshape(self.numVecSample, s[2]).T
            fid -= np.log(norm.reshape(self.numVecSample, 1))
        return fid / self.length

    def compute_bond_vectors(self, nb=0):
        # Contract the samples to the MPS but left the nb-th physical bond empty
        # The bond vectors are normalized
        vecsL = np.ones((1, self.numVecSample))
        for nt in range(0, nb):
            s = self.mps.mps[nt].shape
            tmp = tm.khatri(vecsL, self.vecsImages[:, nt, :].squeeze())
            vecsL = np.tensordot(self.mps.mps[nt], tmp, ([0, 1], [0, 1]))
            norm = np.linalg.norm(vecsL, axis=0)
            vecsL /= norm.repeat(s[2]).reshape(self.numVecSample, s[2]).T
        vecsR = np.ones((1, self.numVecSample))
        for nt in range(self.length-1, nb, -1):
            s = self.mps.mps[nt].shape
            tmp = tm.khatri(vecsR, self.vecsImages[:, nt, :].squeeze())
            vecsR = np.tensordot(self.mps.mps[nt], tmp, ([2, 1], [0, 1]))
            norm = np.linalg.norm(vecsR, axis=0)
            vecsR /= norm.repeat(s[2]).reshape(self.numVecSample, s[2]).T
        s = self.mps.mps[nb].shape
        tmp = np.tensordot(self.mps.mps[nb], tm.khatri(vecsL, vecsR), ([0, 2], [0, 1]))
        norm = np.linalg.norm(tmp, axis=0)
        tmp /= norm.repeat(s[2]).reshape(self.numVecSample, s[2]).T
        return tmp

    def calculate_accuracy(self):
        if self.is_label_added is False:
            print('Warning: the accuracy can be calculated only when the labels '
                  'are added to the images')
        else:
            output = self.compute_bond_vectors()
            ref = self.map_to_vectors(np.array(
                range(self.vec_classes.__len__()), dtype=float).reshape(-1, 1))
            output = ref.squeeze().T.dot(output)
            output = np.argmax(abs(output), axis=0)
            label = self.true_label2label_order()
            return np.sum(output == label) / self.numVecSample

    def check_normalization_vecs(self, tol=1e-14):
        tmp = np.linalg.norm(self.vecsImages, axis=0)
        tmp = np.abs(tmp - 1)
        tmp = tmp[tmp > tol]
        if tmp.size > 0.5:
            print('Several vecsImages not well normalized. There are %i vecs nor normalized' % tmp.size)
        else:
            print('All vecs well normalized')


class MachineLearningMPS(MachineLearningFeatureMap):

    def __init__(self, d, chi, dataset):
        MachineLearningFeatureMap.__init__(self, d=d, dataset=dataset)
        self.chi = chi
        self.vecsLeft = list() * self.length
        self.vecsRight = list() * self.length
        self.norms = None

    def initial_mps(self, mps=None, center=0, ini_way='1'):
        if mps is None:
            self.mps = MPS(self.length, self.d, self.chi, is_eco_dims=True, ini_way=ini_way)
        else:
            self.mps = mps
        self.mps.correct_orthogonal_center(center, normalize=True)

    def initialize_virtual_vecs_train(self, way='contract'):
        self.norms = np.ones((self.length, self.numVecSample))
        self.vecsLeft = bf.empty_list(self.length)
        self.vecsRight = bf.empty_list(self.length)
        if way is 'random':
            for n in range(0, self.length):
                self.vecsLeft[n] = np.random.randn(
                    self.mps.virtual_dim[n], self.numVecSample)
                self.vecsRight[n] = np.random.randn(
                    self.mps.virtual_dim[n+1], self.numVecSample)
        elif way is 'ones':
            for n in range(0, self.length):
                self.vecsLeft[n] = np.ones((self.mps.virtual_dim[n], self.numVecSample))
                self.vecsRight[n] = np.ones((self.mps.virtual_dim[n+1], self.numVecSample))
        else:
            self.vecsRight[self.length-1] = np.ones((self.mps.virtual_dim[self.length], self.numVecSample))
            self.update_virtual_vecs_train_all_tensors('right')
            self.vecsLeft[0] = np.ones((self.mps.virtual_dim[0], self.numVecSample))

    def update_virtual_vecs_train(self, which_t, which_side):
        if (which_side is 'left') or (which_side is 'both'):
            tmp = tm.khatri(self.vecsLeft[which_t], self.vecsImages[:, which_t, :])
            self.vecsLeft[which_t + 1] = np.tensordot(
                self.mps.mps[which_t], tmp, ([0, 1], [0, 1]))
            norm = np.linalg.norm(self.vecsLeft[which_t + 1], axis=0)
            self.norms[which_t, :] = norm
            self.vecsLeft[which_t + 1] /= norm.repeat(self.mps.virtual_dim[which_t + 1]).reshape(
                self.numVecSample, self.mps.virtual_dim[which_t + 1]).T
        if (which_side is 'right') or (which_side is 'both'):
            tmp = tm.khatri(self.vecsRight[which_t], self.vecsImages[:, which_t, :])
            self.vecsRight[which_t - 1] = np.tensordot(
                self.mps.mps[which_t], tmp, ([2, 1], [0, 1]))
            norm = np.linalg.norm(self.vecsRight[which_t - 1], axis=0)
            self.norms[which_t, :] = norm
            self.vecsRight[which_t - 1] /= norm.repeat(self.mps.virtual_dim[which_t]).reshape(
                self.numVecSample, self.mps.virtual_dim[which_t]).T

    def update_virtual_vecs_train_all_tensors(self, which_side):
        if (which_side is 'left') or (which_side is 'both'):
            for n in range(self.length-1):
                self.update_virtual_vecs_train(n, 'left')
        if (which_side is 'right') or (which_side is 'both'):
            for n in range(self.length-1, 0, -1):
                self.update_virtual_vecs_train(n, 'right')

    def env_tensor(self, nt, way):
        s = self.mps.mps[nt].shape
        env = tm.khatri(tm.khatri(
            self.vecsLeft[nt], self.vecsImages[:, nt, :]).reshape(
            self.mps.virtual_dim[nt] * self.d, self.numVecSample), self.vecsRight[nt])
        if way is 'mera':
            env = env.dot(np.ones((self.numVecSample, )))
        elif way is 'gradient':
            weight = self.mps.mps[nt].reshape(1, -1).dot(env.reshape(-1, self.numVecSample))
            self.norms[nt, :] = weight
            env = env.dot(1 / weight.T)
        return env.reshape(s)

    def update_tensor_gradient(self, nt, step):
        env = self.env_tensor(nt, way='gradient')
        env = self.mps.mps[nt] - env / self.numVecSample
        # env /= np.linalg.norm(env.reshape(-1, ))
        env /= (np.linalg.norm(env) + 1e-8)
        self.mps.mps[nt] -= (step * env)
        self.mps.mps[nt] /= np.linalg.norm(self.mps.mps[nt])

    def generate_features(self, features, pos=None, gene_order=None, theta_max=np.pi / 2,
                          f_min=-1e20, f_max=1e20, is_display=False, way='eigs'):
        # pos: the positions that the features correspond to
        # gene_order: the order how the unknown features will be generated
        if pos is None:
            pos = list(range(len(features)))
        mps = self.observe_by_features(features, pos)
        if gene_order is None:
            gene_order = list(range(mps.mps.__len__()))
        features_new = self.generate_vecs_features_by_mps(
            mps, gene_order, way=way, theta_max=theta_max)
        if self.is_dct:
            features_new *= self.dct_info['factor']
            features_new += self.dct_info['shift']
            features_new = cv2.idct(features_new.reshape(self.img_size))
            features_new = features_new.reshape(-1, 1)
            features_new = np.max(np.hstack((np.zeros(features_new.shape), features_new)), axis=1)
            features_new = np.min(np.hstack((np.ones(features_new.shape), features_new)), axis=1)
        features_new = self.vecs2images(data=features_new.reshape(features_new.shape + (1,)),
                                        theta_max=theta_max)
        features_new = features_new.reshape(-1, 1)
        features_new = np.max(np.hstack((features_new, f_min * np.ones(features_new.shape))), axis=1)
        features_new = features_new.reshape(-1, 1)
        features_new = np.min(np.hstack((features_new, f_max * np.ones(features_new.shape))), axis=1)
        pos_new = list(range(self.length))
        pos = list(pos)
        feature_all = np.zeros((self.length,))
        for n in range(len(pos)):
            feature_all[pos[n]] = features[n]
            pos_new.remove(pos[n])
        for n in range(len(pos_new)):
            feature_all[pos_new[gene_order[n]]] = features_new[n]
        if is_display is 'colored':
            img_rgb = np.zeros((feature_all.size, 3))
            for n in range(len(pos)):
                img_rgb[pos[n], 0] = features[n]
            for n in range(len(pos_new)):
                img_rgb[pos_new[gene_order[n]], 1] = features_new[n]
            self.show_image(img_rgb.reshape(self.img_size + [3]))
        elif is_display is 'original':
            self.show_image(feature_all.reshape(self.img_size))
        return feature_all

    def observe_by_features(self, features, pos):
        if len(pos) == self.length:
            bf.print_error('Input features cannot be as many as the total features')
        features = np.array(features).reshape(-1, 1)
        features = self.map_to_vectors(features).squeeze()
        data_mps = self.mps.wrap_data()
        mps = MPS(self.length, self.d, self.chi)
        mps.refresh_mps_properties(data_mps)
        for n in range(np.array(pos).size):
            mps.mps[pos[n]] = np.tensordot(mps.mps[pos[n]], features[:, n], [[1], [0]])
        pos = np.sort(pos)
        for n in pos[::-1]:
            if n > 0:
                mps.mps[n - 1] = np.tensordot(mps.mps[n - 1], mps.mps[n], [[mps.mps[n - 1].ndim-1], [0]])
            else:
                mps.mps[n + 1] = np.tensordot(mps.mps[n], mps.mps[n + 1], [[1], [0]])
            mps.mps.__delitem__(n)
        mps.refresh_mps_properties()
        mps.correct_orthogonal_center(0, normalize=True)
        return mps

    @ staticmethod
    def generate_vecs_features_by_mps(mps, order=None, way='eigs', theta_max=np.pi/2):
        # way='eigs': use the dominant eigen-vector to observe; generate grey images
        # way='rand': use rho[0,0] and rho[1,1] as the probabilities to choose [1,0] or [0,1] to observe;
        #             generate black-and-white images
        d = mps.mps[0].shape[1]
        v = np.zeros((d, mps.mps.__len__()))
        if order is None:
            order = range(mps.length)
        order = np.array(order)
        order0 = order.copy()
        now = 0
        while order.size > 0:
            n = order[0]
            mps.correct_orthogonal_center(n, normalize=True)
            rho = np.tensordot(mps.mps[n], mps.mps[n].conj(), [[0, 2], [0, 2]])
            # rho = rho + rho.T.conj()
            if way is 'eigs':
                u = np.abs(eigsh(rho, k=1, which='LM')[1].reshape(-1, 1))
                v[:, order0[now]] = np.min(np.hstack((u, np.ones(u.shape))), axis=1)
            else:
                rho /= np.trace(rho)
                r = np.random.rand()
                if r < rho[0, 0]:
                    v[:, order0[now]] = np.array([1, 0])
                else:
                    v[:, order0[now]] = np.array([np.cos(theta_max), np.sin(theta_max)])
            mat = np.tensordot(mps.mps[n], v[:, order0[now]], [[1], [0]])
            mps.mps.__delitem__(n)
            mps.orthogonality = np.delete(mps.orthogonality, n)
            mps.length -= 1
            order = order[1:]
            order = order - (order > n)
            if mps.length > 0:
                if n != mps.length:
                    mps.mps[n] = np.tensordot(mat, mps.mps[n], [[1], [0]])
                    mps.orthogonality[n] = 0
                    mps.center = n
                else:
                    mps.mps[n-1] = np.tensordot(mps.mps[n-1], mat, [[2], [0]])
                    mps.orthogonality[n-1] = 0
                    mps.center = n-1
            now += 1
        return v

    def compute_nll(self):
        self.env_tensor(self.mps.center, way='gradient')  # to update the norms at the center
        return -np.sum(np.log(self.norms ** 2)) / self.numVecSample - np.log(self.numVecSample)

    def check_normalization_env(self, tol=1e-14):
        num = 0
        for n in self.vecsLeft:
            tmp = np.linalg.norm(n, axis=0)
            tmp = np.abs(tmp - 1)
            tmp = tmp[tmp > tol]
            num += tmp.size
        for n in self.vecsRight:
            tmp = np.linalg.norm(n, axis=0)
            tmp = np.abs(tmp - 1)
            tmp = tmp[tmp > tol]
            num += tmp.size
        if num > 0.5:
            print('Several vecsR not well normalized. There are %i vecs nor normalized' % num)
        else:
            print('All vecs well normalized')

    def clear_before_save(self):
        self.images = np.zeros(0)
        self.labels = np.zeros(0)
        self.vecsLeft = list()
        self.vecsRight = list()
        self.vecsImages = np.zeros(0)
        self.vecsLabels = np.zeros(0)
        self.clear_tmp_data()


class DecisionTensorNetwork(MachineLearningFeatureMap):

    def __init__(self, dataset, d, chi, tn, classes, numbers=None, if_reducing_samples=False):
        MachineLearningFeatureMap.__init__(self, dataset=dataset, d=d)
        self.tn = tn  # 'mps' or 'ttn'
        self.classes = classes
        self.num_classes = classes.__len__()
        self.chi = chi
        self.tensors = list()
        self.num_tensor = 0
        self.if_reducing_samples = if_reducing_samples
        self.initialize_decision_tree()

        self.vLabel = [[] for _ in range(0, self.num_classes)]
        self.images2vecs(classes, numbers)
        # self.train_label2vectors(self.chi)
        self.generate_vector_labels()
        self.remaining_samples_train = list(range(0, self.numVecSample))
        self.remaining_samples_test = list(range(0, self.numSampleTest))
        self.v_ctr_train = [np.ones(1) for _ in range(0, self.numVecSample)]
        self.v_ctr_test = [np.ones(1) for _ in range(0, self.numSampleTest)]
        self.lm = [np.zeros(1) for _ in range(0, self.length)]
        self.intermediate_accuracy_train = np.ones((self.length,))
        self.intermediate_accuracy_test = np.ones((self.length,))

    def initialize_decision_tree(self):
        if self.tn is 'mps':
            self.num_tensor = self.length
            self.tensors = [np.zeros(0) for _ in range(0, self.num_tensor)]
        elif self.tn is 'tree':
            pass  # to be added

    def generate_vector_labels(self):
        dn1 = int(self.chi / self.num_classes)
        dn0 = int((self.chi - dn1) / (self.num_classes - 1))
        for n in range(0, self.num_classes):
            v = np.zeros((self.chi, ))
            n_start = dn0 * n
            v[n_start:n_start + dn1] = np.ones((dn1,)) / np.sqrt(dn1)
            self.vLabel[n] = v.copy()

    def update_tensor_decision_mps_svd(self, nt):
        env = 0
        d0 = self.v_ctr_train[0].shape[0]
        d1 = self.vecsImages[0][:, nt].shape[0]
        for n in self.remaining_samples_train:
            env += np.kron(np.kron(self.v_ctr_train[n], self.vecsImages[n][:, nt]),
                           self.vLabel[self.classes.index(self.LabelNow[n])])
        u, self.lm[nt], v = np.linalg.svd((env / np.linalg.norm(env.reshape(-1, ))).reshape(
            d0 * d1, self.chi), full_matrices=False)
        self.tensors[nt] = u.dot(v).reshape([d0, d1, self.chi])
        self.lm[nt] /= np.linalg.norm(self.lm[nt])

    def update_tensor_decision_mps_svd_threshold_algo(self, nt, time_r=5, threshold=0.9):
        self.update_tensor_decision_mps_svd(nt)
        env = 0
        d0 = self.v_ctr_train[0].shape[0]
        d1 = self.vecsImages[0][:, nt].shape[0]
        for t in range(0, time_r):
            for n in self.remaining_samples_train:
                v1 = tm.absorb_vectors2tensors(
                    self.tensors[nt], (self.v_ctr_train[n], self.vecsImages[n][:, nt]), (0, 1))
                norm = np.linalg.norm(v1)
                fid = self.fun_fidelity(v1 / norm)
                fid_now = fid[self.classes.index(self.LabelNow[n])]
                fid = [fid[nn] / fid_now for nn in range(0, self.num_classes)]
                fid.pop(self.classes.index(self.LabelNow[n]))
                if max(fid) > threshold:
                    env += np.kron(np.kron(self.v_ctr_train[n], self.vecsImages[n][:, nt]),
                                   self.vLabel[self.classes.index(self.LabelNow[n])])
            u, self.lm[nt], v = np.linalg.svd((env / np.linalg.norm(env.reshape(-1, ))).reshape(
                d0 * d1, self.chi), full_matrices=False)
            self.tensors[nt] = u.dot(v).reshape([d0, d1, self.chi])
            self.lm[nt] /= np.linalg.norm(self.lm[nt])

    def update_tensor_decision_mps_gradient_algo(self, nt, time_r=5, threshold=0, step=0.2):
        self.update_tensor_decision_mps_svd(nt)
        for t in range(0, time_r):
            d_tensor = np.zeros(self.tensors[nt].shape)
            for n in self.remaining_samples_train:
                v1 = tm.absorb_vectors2tensors(
                    self.tensors[nt], (self.v_ctr_train[n], self.vecsImages[n][:, nt]), (0, 1))
                norm = np.linalg.norm(v1)
                fid = self.fun_fidelity(v1 / norm)
                fid_now = fid[self.classes.index(self.LabelNow[n])]
                fid = [fid[nn] / fid_now for nn in range(0, self.num_classes)]
                fid.pop(self.classes.index(self.LabelNow[n]))
                if max(fid) > threshold:
                    tmp = np.kron(np.kron(self.v_ctr_train[n], self.vecsImages[n][:, nt]),
                                  self.vLabel[self.classes.index(self.LabelNow[n])]) \
                          / (fid_now * norm)
                    d_tensor += tmp.reshape(self.tensors[nt].shape)
            d_tensor -= self.tensors[nt]
            norm = np.linalg.norm(d_tensor.reshape(-1, ))
            if norm > 1e-10:
                d_tensor /= norm
            self.tensors[nt] = self.tensors[nt] + step * d_tensor
            self.tensors[nt] /= np.linalg.norm(self.tensors[nt].reshape(-1, ))

    def update_v_ctr_train(self, nt):
        for n in self.remaining_samples_train:
            self.v_ctr_train[n] = tm.absorb_vectors2tensors(
                self.tensors[nt], (self.v_ctr_train[n], self.vecsImages[n][:, nt]), (0, 1))
            self.v_ctr_train[n] /= np.linalg.norm(self.v_ctr_train[n])

    def update_v_ctr_test(self, nt):
        for n in self.remaining_samples_test:
            self.v_ctr_test[n] = tm.absorb_vectors2tensors(
                self.tensors[nt], (self.v_ctr_test[n], self.vecsTest[:, nt, n]), (0, 1))
            self.v_ctr_test[n] /= np.linalg.norm(self.v_ctr_test[n])

    def accuracy_from_one_v(self, v, ni, which_set):
        fid = self.fun_fidelity(v)
        if which_set is 'train':
            if np.argmax(fid) == self.classes.index(self.LabelNow[ni]):
                fid_m = max(fid)
                fid.remove(fid_m)
                if self.if_reducing_samples and abs(fid_m/max(fid)) > 2:
                    self.remaining_samples_train.remove(ni)
                return 1
            else:
                return 0
        else:
            if np.argmax(fid) == self.classes.index(self.TestLabelNow[ni]):
                return 1
            else:
                return 0

    def calculate_intermediate_accuracy_train(self, nt):
        n_right = 0
        for n in self.remaining_samples_train:
            n_right += self.accuracy_from_one_v(self.v_ctr_train[n], n, 'train')
        self.intermediate_accuracy_train[nt] = n_right / self.remaining_samples_train.__len__()

    def calculate_intermediate_accuracy_test(self, nt):
        n_right = 0
        for n in self.remaining_samples_train:
            n_right += self.accuracy_from_one_v(self.v_ctr_train[n], n, 'train')
        self.intermediate_accuracy_train[nt] = n_right / self.remaining_samples_train.__len__()

    def fun_fidelity(self, v):
        fid = list()
        for nc in range(0, self.num_classes):
            # fid.append(np.linalg.norm(v - self.vLabel[nc]))
            # fid.append(v.reshape(1, -1).dot(self.vLabel[nc])[0])
            fid.append(abs(v.reshape(1, -1).dot(self.vLabel[nc]))[0])
        return fid


