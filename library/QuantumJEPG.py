from skimage import color, io, data
from skimage.viewer import ImageViewer
import numpy as np
import os
import cv2
import copy
from library.BasicFunctions import generate_zigzag_order, rescale_image


class QuantumJEPG:

    def __init__(self, file, b_size, pixel_cutoff=None, is_grey=True):
        self.file = file
        self.image = np.zeros(0)
        self.image0 = np.zeros(0)
        self.original_size = (0, 0)
        self.block_size = b_size
        self.blocks = np.zeros(0)
        self.blocks_dct = np.zeros(0)
        self.is_grey = is_grey
        self.num_channels = 1
        self.num_blocks = [1, 1]
        self.load_image()
        self.order = np.array(range(b_size[0] * b_size[1]))
        self.normalization_factor = [1, 1]
        self.shift = [0, 0]
        if pixel_cutoff is None:
            self.pixel_cutoff = 10 ** 20
        else:
            self.pixel_cutoff = pixel_cutoff
        self.quantization_number = 1

    def load_image(self):
        if os.path.isfile(self.file):
            self.image = io.imread(self.file)
        else:
            self.image = eval('data.' + self.file + '()')
        if self.is_grey:
            self.image = color.rgb2gray(self.image)
        # self.image = transform.resize(self.image, (64, 64))
        self.image = self.image.reshape(self.image.shape[:2] + (-1,))
        self.num_channels = self.image.shape[2]
        self.original_size = self.image.shape
        self.image0 = self.image.copy()

    def show_image(self):
        viewer = ImageViewer(self.image.squeeze())
        viewer.show()

    def show_block_image(self, n, channel=None):
        if channel is None:
            viewer = ImageViewer(self.blocks[:, :, :, n].squeeze())
        else:
            viewer = ImageViewer(self.blocks[:, :, channel, n])
        viewer.show()

    def patch(self, value=None):
        s = self.image.shape[:2]
        x = int(np.ceil(s[0] / self.block_size[0]))
        y = int(np.ceil(s[1] / self.block_size[1]))
        if s[0] != x * self.block_size[0] or s[1] != y * self.block_size[1]:
            image = np.ones((x * self.block_size[0], y * self.block_size[1],
                             self.num_channels))
            for c in range(self.num_channels):  # different channels
                if value is None:  # patching with the last pixel in the image
                    value = self.image[-1, -1, c]
                # image[:, :, c] *= int(value)
                image[:s[0], :s[1], c] = self.image[:, :, c]
            self.image = image
        self.num_blocks = [x, y]

    def cut2blocks(self):
        self.patch()
        self.blocks = np.zeros(tuple(self.block_size) + (
            self.num_channels, self.num_blocks[0] * self.num_blocks[1]))
        n = 0
        for x in range(self.num_blocks[0]):
            for y in range(self.num_blocks[1]):
                self.blocks[:, :, :, n] = \
                    self.image[x * self.block_size[0]:(x + 1) * self.block_size[0],
                    y * self.block_size[1]:(y + 1) * self.block_size[1], :].copy()
                n += 1

    def dct_blocks(self):
        self.blocks_dct = np.zeros(self.blocks.shape)
        for c in range(self.num_channels):
            for n in range(self.blocks.shape[3]):
                self.blocks_dct[:, :, c, n] = cv2.dct(self.blocks[:, :, c, n])

    def pre_process_data_before_ml(self, which=0):
        # Reorder the indexes as [num_pixels, num_channels, num_samples]
        if (which == 0) or (which == 2):
            # Original images
            self.blocks = self.blocks.reshape(self.block_size[0] * self.block_size[1],
                                              self.num_channels, -1)
            self.shift[0] = np.min(self.blocks)
            self.blocks -= self.shift[0]
            self.normalization_factor[0] = np.max(self.blocks)
            # self.normalization_factor[0] = 255
            self.blocks /= self.normalization_factor[0]
        if (which == 1) or (which == 2):
            # DCT images
            # Use zigzag order
            s = self.blocks_dct.shape
            tmp = np.zeros((s[0] * s[1], s[2], s[3]))
            order = generate_zigzag_order(s[0], s[1])
            for n in range(s[0] * s[1]):
                tmp[n, :, :] = self.blocks_dct[order[0][n], order[1][n], :, :]
            self.shift[1] = tmp.min()
            tmp -= self.shift[1]
            self.normalization_factor[1] = tmp.max()
            self.blocks_dct = tmp / self.normalization_factor[1]

    def encode_with_cutoff_dct(self, cutoff):
        # This only applies for the DCT way
        if cutoff is None:
            cutoff = self.blocks_dct.shape[1]
        blocks = dict()
        blocks['q_number'] = self.normalization_factor[1]
        blocks['shift'] = self.shift[1]
        blocks['cutoff'] = cutoff
        blocks['order'] = self.order
        blocks['original_size'] = self.original_size
        blocks['block_size'] = self.block_size
        blocks['num_blocks'] = self.num_blocks
        blocks['for_rescale'] = [self.image.max(), self.image.min(), np.average(self.image)]
        blocks['data'] = self.blocks_dct[:cutoff, :, :].copy()
        # blocks['data'] = blocks['data'].reshape(blocks['data'].shape + (1, ))
        # blocks['data'] = np.round(blocks['data']).astype(int)
        return blocks

    def encode_standard_jpeg(self, cutoff):
        if cutoff is None:
            cutoff = self.blocks_dct.shape[0]
        blocks = dict()
        blocks['q_number'] = self.normalization_factor[1]
        blocks['shift'] = self.shift[1]
        blocks['original_size'] = self.original_size
        blocks['block_size'] = self.block_size
        blocks['num_blocks'] = self.num_blocks
        blocks['for_rescale'] = [self.image.max(), self.image.min(), np.average(self.image)]
        order = np.argsort(self.order)
        blocks['data'] = self.blocks_dct[order, :, :][:cutoff, :, :]
        # blocks['data'] /= self.quantization_number
        return blocks

    def reorder_features(self, order1, channel=0, which=0):
        self.order = self.order[order1]
        if (which == 0) or (which == 2):
            for n in range(self.blocks.shape[2]):
                self.blocks[:, channel, n] = self.blocks[:, channel, n][order1]
        if (which == 1) or (which == 2):
            for n in range(self.blocks_dct.shape[2]):
                self.blocks_dct[:, channel, n] = self.blocks_dct[:, channel, n][order1]


def show_image(image, if_normalize=True):
    if if_normalize:
        c = np.max(image)
        image /= c
    viewer = ImageViewer(image.squeeze())
    viewer.show()


def matrix2vector_zigzag(mat, order=None):
    if order is None:
        order = generate_zigzag_order(mat.shape[0], mat.shape[1])
    return mat[order]


def decode_with_cutoff_dct(blocks0, rescale_way=2):
    blocks = copy.deepcopy(blocks0)
    blocks['data'] *= blocks['q_number']
    blocks['data'] += blocks['shift']
    image0 = np.zeros([blocks['num_blocks'][n] * blocks['block_size'][n] for n in range(2)] + [1])
    num_b_pixels = blocks['block_size'][0] * blocks['block_size'][1]
    # new_blocks: length * n_channel * n_blocks
    new_blocks = np.zeros((num_b_pixels, blocks['data'].shape[1], blocks['data'].shape[2]))
    for z in range(blocks['cutoff']):
        new_blocks[blocks['order'][z], :, :] = blocks['data'][z, :, :]
    order = generate_zigzag_order(blocks['block_size'][0], blocks['block_size'][1])
    tmp = np.zeros((blocks['block_size'][0], blocks['block_size'][1], blocks['data'].shape[1],
                    blocks['data'].shape[2]))
    for z in range(blocks['block_size'][0] * blocks['block_size'][1]):
        tmp[order[0][z], order[1][z], :, :] = new_blocks[z, :, :]
    new_blocks = tmp.copy()
    n = 0
    n_channel = 0
    for x in range(blocks['num_blocks'][0]):
        for y in range(blocks['num_blocks'][1]):
            tmp = cv2.idct(new_blocks[:, :, n_channel, n])
            tmp = tmp.reshape(tmp.shape + (1,))
            image0[x * blocks['block_size'][0]:(x + 1) * blocks['block_size'][0],
            y * blocks['block_size'][1]:(y + 1) * blocks['block_size'][1], :] = tmp.copy()
            n += 1
    image0 = image0[:blocks['original_size'][0], :blocks['original_size'][1], :].squeeze()
    # Rescale image
    image0 = rescale_image(image0, blocks['for_rescale'][0], blocks['for_rescale'][1],
                           blocks['for_rescale'][2], rescale_way)
    return image0


def decode_with_generative_mps(blocks, mps, rescale_way=2, channel=0):
    num_b_pixels = blocks['block_size'][0] * blocks['block_size'][1]  # num pixels in one block
    # new_blocks: length * n_channel * n_blocks
    new_blocks = np.zeros((num_b_pixels, blocks['data'].shape[1], blocks['data'].shape[2]))
    dn = max(int(new_blocks.shape[2] / 20), 1)
    n_print = 1
    print('The order is : ' + str(blocks['order'][:blocks['cutoff']]))
    for n in range(new_blocks.shape[2]):
        image_now = mps.generate_features(features=blocks['data'][:, channel, n], f_max=1, f_min=0)
        if n == n_print * dn:
            print('Norm of the generated part = ' + str(np.linalg.norm(image_now[blocks['data'].shape[1]:])))
        # image_now = blocks['data'][n, :, channel].copy()
        image_now *= blocks['q_number']
        image_now += blocks['shift']
        # image_now = np.hstack((image_now, np.zeros((num_b_pixels - blocks['data'].shape[1]))))
        for z in range(blocks['order'].shape[0]):
            new_blocks[blocks['order'][z], channel, n] = image_now[z]
        if n == n_print * dn:
            # print(np.linalg.norm(image_now[:blocks['cutoff']] - blocks['data'][n, :, channel]))
            # print(np.linalg.norm(image_now[blocks['cutoff']:]))
            print('%.2f%% blocks have been generated' % (n / new_blocks.shape[2] * 100))
            n_print += 1
    print('100%% blocks have been generated')
    order = generate_zigzag_order(blocks['block_size'][0], blocks['block_size'][1])
    tmp = np.zeros((blocks['block_size'][0], blocks['block_size'][1], blocks['data'].shape[1],
                    blocks['num_blocks'][0] * blocks['num_blocks'][1]))
    for z in range(blocks['block_size'][0] * blocks['block_size'][1]):
        tmp[order[0][z], order[1][z], :, :] = new_blocks[z, :, :]
    new_blocks = tmp.copy()
    n = 0
    n_channel = 0
    image0 = np.zeros([blocks['num_blocks'][n] * blocks['block_size'][n] for n in range(2)] + [1])
    for x in range(blocks['num_blocks'][0]):
        for y in range(blocks['num_blocks'][1]):
            tmp = cv2.idct(new_blocks[:, :, n_channel, n])
            tmp = tmp.reshape(tmp.shape + (1,))
            image0[x * blocks['block_size'][0]:(x + 1) * blocks['block_size'][0],
            y * blocks['block_size'][1]:(y + 1) * blocks['block_size'][1], :] = tmp.copy()
            n += 1
    image0 = image0[:blocks['original_size'][0], :blocks['original_size'][1], :].squeeze()
    # Rescale image
    image0 = rescale_image(image0, blocks['for_rescale'][0], blocks['for_rescale'][1],
                           blocks['for_rescale'][2], rescale_way)
    return image0


def decode_jpeg(blocks0, rescale_way=2):
    blocks = copy.deepcopy(blocks0)
    blocks['data'] *= blocks['q_number']
    blocks['data'] += blocks['shift']
    image0 = np.zeros([blocks['num_blocks'][n] * blocks['block_size'][n] for n in range(2)] + [
        blocks['data'].shape[1]])
    new_blocks = np.zeros((blocks['block_size'][0], blocks['block_size'][1],
                           blocks['data'].shape[1], blocks['data'].shape[2]))
    order = generate_zigzag_order(blocks['block_size'][0], blocks['block_size'][1])
    for z in range(blocks['data'].shape[0]):
        new_blocks[order[0][z], order[1][z], :, :] = blocks['data'][z, :, :]
    # new_blocks = new_blocks.transpose([1, 2, 3, 0])
    n = 0
    for x in range(blocks['num_blocks'][0]):
        for y in range(blocks['num_blocks'][1]):
            for nc in range(blocks['data'].shape[1]):
                image0[x * blocks['block_size'][0]:(x + 1) * blocks['block_size'][0],
                y * blocks['block_size'][1]:(y + 1) * blocks['block_size'][1], nc] = \
                    cv2.idct(new_blocks[:, :, nc, n])
            n += 1
    image0 = image0[:blocks['original_size'][0], :blocks['original_size'][1], :].squeeze()
    # Rescale image
    image0 = rescale_image(image0, blocks['for_rescale'][0], blocks['for_rescale'][1],
                           blocks['for_rescale'][2], rescale_way)
    return image0
