from library.QuantumJEPG import show_image, QuantumJEPG as qjpg, \
    decode_with_cutoff_dct, decode_jpeg, decode_with_generative_mps
from library.Parameters import parameters_qjpg, parameters_gtn_one_class
from library.BasicFunctions import plot, save_pr, load_pr, plot_surf, psnr
from algorithms.TNmachineLearningAlgo import gtn_one_class
from skimage import io
import numpy as np
import os


def quantum_jpeg(para_tot=None):
    if para_tot is None:
        para_tot = parameters_qjpg()
    print('Preparing image')
    exp = expression_save(para_tot)

    exp = os.path.join(para_tot['data_path'], exp)
    if os.path.isfile(exp) and para_tot['if_load']:
        a, generator = load_pr(exp, ['a', 'mps'])
    else:
        a = qjpg(file=para_tot['file'], b_size=para_tot['block_size'], is_grey=True)
        a.cut2blocks()
        a.dct_blocks()
        a.pre_process_data_before_ml(which=2)

        para = parameters_gtn_one_class()
        para['chi'] = para_tot['chi']
        para['dataset'] = 'custom'
        para['if_save'] = False
        para['if_load'] = False
        para['dct'] = False

        generator = None
        if 'real' in para_tot['tasks']:
            print('Train in the real space')
            for n in range(para_tot['reorder_time']):
                generator = gtn_one_class(para=para, images=a.blocks.squeeze())[0]
                if n != (para_tot['reorder_time'] - 1):
                    order = generator.mps.markov_measurement(if_restore=False)[0]
                    # ent = generator.mps.calculate_single_entropy()[0]
                    # order = np.argsort(ent)[::-1]
                    a.reorder_features(order, which=0)
        if 'freq' in para_tot['tasks']:
            for n in range(para_tot['reorder_time']):
                print('Train in the frequency space： reorder time = ' + str(n))
                generator = gtn_one_class(para=para, images=a.blocks_dct.squeeze())[0]
                if n != (para_tot['reorder_time'] - 1):
                    order = generator.mps.markov_measurement(if_restore=False)[0]
                    # ent = generator.mps.calculate_single_entropy()[0]
                    # order = np.argsort(ent)[::-1]
                    a.reorder_features(order, which=1)
        save_pr(para_tot['data_path'], exp, [a, generator, para_tot],
                ['a', 'mps', 'para_tot'])
    # a.show_image()
    if 'recover' in para_tot['tasks']:
        blocks = a.encode_with_cutoff_dct(para_tot['pixel_cutoff'])
        image_gtn1 = decode_with_generative_mps(blocks, generator)
        image_cutoff = decode_with_cutoff_dct(blocks)
        blocks_jpg = a.encode_standard_jpeg(para_tot['pixel_cutoff'])
        image_jpg = decode_jpeg(blocks_jpg)
        io.imsave('../data_QJPG/before.jpg', a.image.squeeze())
        io.imsave('../data_QJPG/0cut.jpg', image_cutoff.squeeze())
        io.imsave('../data_QJPG/1JPGway.jpg', image_jpg.squeeze())
        io.imsave('../data_QJPG/2GTNway.jpg', image_gtn1.squeeze())
        p1 = psnr(a.image0, image_jpg)
        p2 = psnr(a.image0, image_cutoff)
        p3 = psnr(a.image0, image_gtn1)
        print('The PSNRs for jpg, cut-off, and gtn are %g, %g, and %g' % (p1, p2, p3))


def expression_save(para):
    exp = 'Block(' + str(para['block_size'][0]) + ',' + str(para['block_size'][1]) + \
          ')_chi' + str(para['chi'])
    return exp
