from library.TNmachineLearning import MachineLearningFeatureMap
from algorithms.TNmachineLearningAlgo import pca
from library.BasicFunctions import output_txt
import numpy as np
import skimage


# num_f = [1] + list(range(4, 104, 4))
num_f = [80]
psnr = np.zeros(num_f.__len__())
dataset = 'fashion_mnist'

classes = [3]
a = MachineLearningFeatureMap(2, dataset)
a.identify_dataset()
a.load_data()
a.select_samples(classes)

b = MachineLearningFeatureMap(2, dataset)
b.identify_dataset()
b.load_data(file_sample='t10k-images.idx3-ubyte', file_label='t10k-labels.idx1-ubyte')
b.select_samples(classes)
for t in range(num_f.__len__()):
    _, _, u = pca(a.images, num_f[t])
    imgs1 = (b.images.T.dot(u).dot(u.T)).T
    for n in range(imgs1.shape[1]):
        psnr[t] += (skimage.measure.compare_psnr(
            b.images[:, n], imgs1[:, n]) / imgs1.shape[1])
    # a.input_data(images=imgs1, labels=np.arange(imgs1.shape[1]))
    # a.show_image(list(np.random.permutation(imgs1.shape[1])[:25]))
output_txt(psnr, filename='PSNR_pca' + dataset)



