from library import TNmachineLearning
from library.Parameters import parameters_lazy_learning
import numpy as np
from library.TensorBasicModule import khatri
from library.BasicFunctions import plot
import torch


if_torch = True
num_parts = 20  # to save GPU memory


para = parameters_lazy_learning()
print_dtime = 20

a = TNmachineLearning.MachineLearningFeatureMap(para['d'], para['dataset'])
a.load_data()
a.select_samples(para['classes'], para['num_samples'])
a.images2vecs(theta_max=para['theta']*np.pi)

b = TNmachineLearning.MachineLearningFeatureMap(para['d'], para['dataset'])
b.load_data(data_path='..\\..\\..\\MNIST\\', file_sample='t10k-images.idx3-ubyte',
            file_label='t10k-labels.idx1-ubyte', is_normalize=True)
b.select_samples(para['classes'], para['num_samples'])
b.images2vecs(theta_max=para['theta']*np.pi)

accuracy = list()
num = list()
if if_torch:
    for c2 in para['classes']:
        fid = list()
        imgs2 = b.vecsImages[:, :, b.labels == c2]  # testing images in class c2
        # Calculate the accuracy of the c1-th class of the testing images
        for c1 in para['classes']:
            n_now = 0
            imgs1 = a.vecsImages[:, :, a.labels == c1]  # training images in class c1
            for part in range(num_parts):
                n_end = min(n_now + int(imgs1.shape[2] / num_parts) + 1, imgs1.shape[2])
                if n_now < n_end:
                    if part == 0:
                        fid.append(TNmachineLearning.fidelities_torch(
                            imgs1[:, :, n_now:n_end], imgs2, if_sum=0).reshape(-1, 1))
                    else:
                        fid[-1] += TNmachineLearning.fidelities_torch(
                            imgs1[:, :, n_now:n_end], imgs2, if_sum=0).reshape(-1, 1)
                    n_now += int(imgs1.shape[2] / num_parts) + 1
                else:
                    break
        fid = np.hstack(fid)
        print(fid.shape)
        predict = np.argmin(fid, axis=1)
        plot(predict)
        num.append(predict.size)
        accuracy.append(0)
        for n in range(predict.size):
            accuracy[-1] += (para['classes'][predict[n]] == c2)
        accuracy[-1] /= num[-1]
        print('Accuracies for each testing classes = ' + str(accuracy[-1]))
    accuracy_tot = 0
    for n in range(accuracy.__len__()):
        accuracy_tot += accuracy[n] * num[n]
    accuracy_tot /= np.sum(num)
    print('Average accuracy = ' + str())
else:
    num_correct = 0
    for n in range(b.numVecSample):
        tmp = khatri(b.vecsImages[:, :, n], a.vecsImages.transpose(0, 2, 1).reshape(
            para['d']*a.numVecSample, a.length))
        tmp = tmp.reshape(para['d']*para['d'], a.numVecSample * a.length)
        identity = np.eye(para['d']).reshape(1, -1)
        tmp = np.abs(identity.dot(tmp).reshape(a.numVecSample, a.length))
        tmp = -np.sum(np.log(tmp), axis=1)
        fid = list()
        for nn in para['classes']:
            fid.append(np.sum(tmp[a.labels == nn]))
        predict = np.argmax(fid)
        num_correct += (para['classes'][int(b.labels[n])] == para['classes'][predict])
        if (n+1) % print_dtime == 0:
            print('Current accuracy = ' + str(num_correct / (n+1)))

