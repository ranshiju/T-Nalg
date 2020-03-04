from library import Parameters as pm, BasicFunctions as bf, TNmachineLearning
import numpy as np
from library.BasicFunctions import save_pr, load_pr, print_dict
import os
import copy
import time


def gtn_one_class(para=None, images=None, labels=None):
    if 'to_black_and_white' not in para:
        para['to_black_and_white'] = False
    if para is None:
        para = pm.parameters_gtn_one_class()
    para['save_exp'] = save_exp_gtn_one_class(para)
    if para['if_load'] and os.path.isfile(os.path.join(para['data_path'], para['save_exp'])):
        a = bf.load_pr(os.path.join(para['data_path'], para['save_exp']), 'a')
    else:
        a = TNmachineLearning.MachineLearningMPS(para['d'], para['chi'], para['dataset'])
        if para['dataset'] is 'custom':
            a.input_data(copy.deepcopy(images), copy.deepcopy(labels))
        else:
            a.load_data()
        if a.is_there_labels:
            a.select_samples([para['class']])
        if para['to_black_and_white']:
            a.to_black_and_white()
        if para['dct'] is True:
            a.dct(shift=para['shift'], factor=para['factor'])
        a.images2vecs(theta_max=para['theta'] * np.pi / 2)
        a.initial_mps(center=0, ini_way='1')
        a.initialize_virtual_vecs_train()
        a.mps.correct_orthogonal_center(0)
        a.update_tensor_gradient(0, para['step'])
        nll0 = a.compute_nll()
        step = copy.deepcopy(para['step'])
        print('Iniitially, NLL = ' + str(nll0))
        for t in range(0, para['sweep_time']):
            # from left to right
            if para['if_print_detail']:
                print('At the ' + str(t+1) + '-th sweep, from left to right')
                t0 = time.time()
                tt0 = time.clock()
            for nt in range(0, a.length):
                a.mps.correct_orthogonal_center(nt)
                if nt != 0:
                    a.update_virtual_vecs_train(nt-1, 'left')
                a.update_tensor_gradient(nt, step)
            # from right to left
            if para['if_print_detail']:
                print('At the ' + str(t+1) + '-th sweep, from right to left')
            for nt in range(a.length-1, -1, -1):
                a.mps.correct_orthogonal_center(nt)
                if nt != a.length-1:
                    a.update_virtual_vecs_train(nt+1, 'right')
                a.update_tensor_gradient(nt, step)
            if para['if_print_detail']:
                print('Wall time cost for one loop: %s' % (time.time() - t0))
                print('CPU time cost for one loop: %s' % (time.clock() - tt0))

            if t > (para['check_time0'] - 2) and ((t+1) % para['check_time'] == 0 or
                                                  t+1 == para['sweep_time']):
                nll = a.compute_nll()
                print('NLL = ' + str(nll))
                # fid = fidelity_per_site(mps0, a.mps.mps)
                fid = abs(nll - nll0) / nll0
                if fid < (step * para['step_ratio']):
                    print('After ' + str(t+1) + ' sweeps: fid = %g' % fid)
                    step *= para['step_ratio']
                    # mps0 = copy.deepcopy(a.mps.mps)
                    nll0 = nll
                elif t+1 == para['sweep_time']:
                    print('After all ' + str(t+1) + ' sweeps finished, fid = %g. '
                                                    'Consider to increase the sweep times.' % fid)
                else:
                    print('After ' + str(t+1) + ' sweeps, fid = %g.' % fid)
                    # mps0 = copy.deepcopy(a.mps.mps)
                    nll0 = nll
                if step < para['step_min']:
                    print('Now step = ' + str(step) + ' is sufficiently small. Break the loop')
                    break
                else:
                    print('Now step = ' + str(step))
        a.clear_before_save()
        if para['if_save']:
            save_pr(para['data_path'], para['save_exp'], [a, para], ['a', 'para'])
    return a, para


def gtnc(para_tot=None):
    print('Preparing parameters')
    if para_tot is None:
        para_tot = pm.parameters_gcmpm()
    n_class = len(para_tot['classes'])
    paras = bf.empty_list(n_class)
    for n in range(0, n_class):
        paras[n] = copy.deepcopy(para_tot)
        paras[n]['class'] = int(para_tot['classes'][n])
        paras[n]['chi'] = para_tot['chi'][n]
        paras[n]['theta'] = para_tot['theta']
        paras[n]['save_exp'] = save_exp_gtn_one_class(paras[n])
    classifiers = bf.empty_list(n_class)
    for n in range(0, n_class):
        print_dict(paras[n])
        data = para_tot['data_path'] + paras[n]['save_exp']
        if para_tot['if_load'] and os.path.isfile(data):
            print('The classifier already exists. Load directly')
            classifiers[n] = load_pr(data, 'a')
        else:
            print('Training the MPS of ' + str(para_tot['classes'][n]))
            classifiers[n] = gtn_one_class(paras[n])[0]
            # if para_tot['if_save']:
            #     save_pr('../data_tnml/gcmpm/', paras[n]['save_exp'],
            #             [classifiers[n]], ['classifier'])
        # classifiers[n].mps.check_orthogonality_by_tensors(tol=1e-12)
    # ==================== Testing accuracy ====================
    print('Calculating the testing accuracy')
    b = TNmachineLearning.MachineLearningFeatureMap(para_tot['d'])
    b.load_data(data_path='..\\..\\..\\MNIST\\', file_sample='t10k-images.idx3-ubyte',
                file_label='t10k-labels.idx1-ubyte', is_normalize=True)
    b.select_samples(para_tot['classes'])
    if classifiers[0].is_dct:
        b.dct(shift=para_tot['shift'], factor=para_tot['factor'])
    b.images2vecs(para_tot['theta'] * np.pi / 2)
    fid = bf.empty_list(n_class)
    for n in range(0, n_class):
        fid[n] = b.compute_fidelities(classifiers[n].mps.mps)
    max_fid = np.argmin(np.hstack(fid), axis=1)
    predict = np.zeros(max_fid.shape, dtype=int)
    for n in range(0, n_class):
        predict += (max_fid == n) * int(para_tot['classes'][n])
    # plot(predict)
    # plot(b.labels)
    accuracy = np.sum(predict == b.labels, dtype=float) / b.numVecSample
    print(accuracy)


def labeled_gtn(para):
    if para is None:
        para = pm.parameters_labeled_gtn()
    para['save_exp'] = save_exp_labeled_gtn(para)
    if para['parallel'] is True:
        par_pool = para['n_nodes']
    else:
        par_pool = None

    # Preparing testing dataset
    b = TNmachineLearning.MachineLearningFeatureMap(
        para['d'], file_sample='t10k-images.idx3-ubyte', file_label='t10k-labels.idx1-ubyte')
    b.load_data()
    b.select_samples(para['classes'])
    b.add_labels_to_images()
    b.images2vecs(para['theta'])

    data_file = os.path.join(para['data_path'], para['save_exp'])
    if para['if_load'] and os.path.isfile(data_file):
        print('Data exist. Load directly.')
        a = bf.load_pr(data_file, 'a')
    else:
        a = TNmachineLearning.MachineLearningMPS(para['d'], para['chi'], para['dataset'],
                                                 par_pool=par_pool)
        a.load_data()
        a.select_samples(para['classes'])
        a.add_labels_to_images()
        a.images2vecs(para['theta'] * np.pi/2)
        a.initial_mps()
        a.mps.correct_orthogonal_center(0, normalize=True)
        a.initialize_virtual_vecs_train()
        a.update_virtual_vecs_train_all_tensors('both')
        accuracy0 = 0
        for t in range(0, para['sweep_time']):
            # from left to right
            for nt in range(0, a.length):
                a.update_tensor_gradient(nt, para['step'])
                if nt != a.length - 1:
                    a.update_virtual_vecs_train(nt, 'left')
            # from left to right
            for nt in range(a.length - 1, -1, -1):
                a.update_tensor_gradient(nt, para['step'])
                if nt != 0:
                    a.update_virtual_vecs_train(nt, 'right')
            if t > para['check_time0'] and ((t + 1) % para['check_time'] == 0
                                            or t + 1 == para['sweep_time']):
                b.input_mps(a.mps)
                accuracy = b.calculate_accuracy()
                print('After the ' + str(t) + '-th sweep, the testing accuracy = ' + str(accuracy))
                if abs(accuracy - accuracy0) < (para['step'] * para['ratio_step_tol']):
                    para['step'] *= para['step_ratio']
                    accuracy0 = accuracy
                    print('Converged. Reduce the gradient step to ' + str(para['step']))
                elif t + 1 == para['sweep_time']:
                    print('After all ' + str(t + 1) + ' sweeps finished, not converged. '
                                                      'Consider to increase the sweep times.')
                else:
                    accuracy0 = accuracy
                if para['step'] < para['step_min']:
                    print('Now step = ' + str(para['step']) + ' is sufficiently small. Break the loop')
                    break
                else:
                    print('Now step = ' + str(para['step']))
        a.clear_before_save()
        if para['if_save']:
            save_pr(para['data_path'], para['save_exp'], [a, para], ['a', 'para'])
    accuracy = b.calculate_accuracy()
    print('The final testing accuracy = ' + str(accuracy))

    return a, para


def save_exp_gtn_one_class(para):
    if 'to_black_and_white' not in para:
        para['to_black_and_white'] = False
    exp = 'GTN' + str(para['class']) + '_dim(' + str(para['d']) + ',' + str(para['chi']) + ')theta' \
          + str(para['theta']) + para['dataset']
    if para['to_black_and_white']:
        exp += 'B&W'
    return exp


def save_exp_labeled_gtn(para):
    classes = ''
    for n in para['classes']:
        classes += str(n)
    exp = 'LabeledGTN[' + classes + ']_dim(' + str(para['d']) + ',' + str(para['chi']) + ')theta' \
          + str(para['theta']) + para['dataset']
    return exp


def decision_mps(para=None):
    if para is None:
        para = pm.parameters_decision_mps()
    a = TNmachineLearning.DecisionTensorNetwork(para['dataset'], 2, para['chi'], 'mps',
                                                para['classes'], para['numbers'],
                                                if_reducing_samples=para['if_reducing_samples'])
    a.images2vecs_test_samples(para['classes'])
    print(a.vLabel)
    for n in range(0, a.length):
        bf.print_sep()
        print('Calculating the %i-th tensor' % n)
        a.update_tensor_decision_mps_svd(n)
        # a.update_tensor_decision_mps_svd_threshold_algo(n)
        # a.update_tensor_decision_mps_gradient_algo(n)
        a.update_v_ctr_train(n)
        a.calculate_intermediate_accuracy_train(n)
        if a.remaining_samples_train.__len__() == 0:
            print('All samples are classified correctly. Training stopped.')
            break
        print('The current accuracy = %g' % a.intermediate_accuracy_train[n])
        print('Entanglement: ' + str(a.lm[n].reshape(1, -1)))
        if para['if_reducing_samples']:
            print('Number of remaining samples: ' + str(a.remaining_samples_train.__len__()))


def pca(data_mat, top_n_feat=99999999):
    """
    主成分分析：
    输入：矩阵data_mat ，其中该矩阵中存储训练数据，每一行为一条训练数据
         保留前n个特征top_n_feat，默认全保留
    返回：降维后的数据集和原始数据被重构后的矩阵（即降维后反变换回矩阵）
    """

    # 获取数据条数和每条的维数
    data_mat = data_mat.T

    # 数据中心化，即指变量减去它的均值
    mean_vals = data_mat.mean(axis=0)  # shape:(784,)
    data_mat = data_mat - mean_vals  # shape:(100, 784)

    # 计算协方差矩阵（Find covariance matrix）
    cov_mat = np.cov(data_mat, rowvar=0)  # shape：(784, 784)

    # 计算特征值(Find eigenvalues and eigenvectors)
    lm, u = np.linalg.eigh(np.mat(cov_mat))  # 计算特征值和特征向量，shape分别为（784，）和(784, 784)

    eig_val_index = np.argsort(lm)  # 对特征值进行从小到大排序，argsort返回的是索引，即下标

    eig_val_index = eig_val_index[:-(top_n_feat + 1): -1]  # 最大的前top_n_feat个特征的索引
    # 取前top_n_feat个特征后重构的特征向量矩阵reorganize eig vects,
    # shape为(784, top_n_feat)，top_n_feat最大为特征总数
    u = u[:, eig_val_index]

    # 将数据转到新空间
    low_d_data_mat = data_mat.dot(u)  # shape: (100, top_n_feat), top_n_feat最大为特征总数
    recon_mat = (low_d_data_mat * u.T) + mean_vals  # 根据前几个特征向量重构回去的矩阵，shape:(100, 784)

    return np.array(low_d_data_mat.T), np.array(recon_mat.T), np.array(u)
