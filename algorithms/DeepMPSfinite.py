import numpy as np
import copy
import library.TensorBasicModule as T_module
from library.BasicFunctions import empty_list
from library.MPSClass import MpsOpenBoundaryClass as Mob

if_deepcopy = False


def parameters_dmps():
    para = dict()
    para['num_layers'] = 3
    para['chi_overlap'] = 32  # the dimension cut-off when calculating overlap
    para['theta'] = 0  # None: means to optimize over theta
    para['num_theta'] = 1
    return para


def act_umpo_on_mps(mps, mpo, if_conjugate=False):
    mps1 = empty_list(mps.__len__())
    if if_conjugate:  # act on product state (evolve)
        mps1[0] = copy.deepcopy(mpo[0])
        for n in range(1, mps.__len__()-1):
            mps1[n] = T_module.cont([mpo[n], mps[n-1]], [[-1, -3, 1, -4], [-2, 1, -5]])
            s = mps1[n].shape
            mps1[n] = mps1[n].reshape(s[0]*s[1], s[2], s[3]*s[4])
        mps1[-1] = T_module.cont([mpo[-1], mps[-2], mps[-1]], [[-1, -3, 2, 3], [-2, 2, 1], [1, 3, -4]])
        s = mps1[-1].shape
        mps1[-1] = mps1[-1].reshape(s[0]*s[1], s[2], s[3])
    else:  # act on mps (disentangle)
        mps1[0] = T_module.cont([mps[0].squeeze(), mps[1], mpo[0].squeeze(), mpo[1]],
                                [[3, 1], [1, 4, -3], [3, 2], [2, 4, -1, -2]])
        mps1[0] = mps1[0].reshape(1, mpo[1].shape[2], mpo[1].shape[3]*mps[1].shape[2])
        for n in range(2, mps.__len__()-1):
            mps1[n-1] = T_module.cont([mps[n], mpo[n]], [[-2, 1, -5], [-1, 1, -3, -4]])
            s = mps1[n-1].shape
            mps1[n-1] = mps1[n-1].reshape(s[0]*s[1], s[2], s[3]*s[4])
        mps1[-2] = T_module.cont([mps[-1].squeeze(), mpo[-1]], [[-2, 1], [-1, 1, -3, -4]])
        s = mps1[-2].shape
        mps1[-2] = mps1[-2].reshape(s[0]*s[1]*s[2], s[3])
        mps1[-2], lm, mps1[-1] = np.linalg.svd(mps1[-2], full_matrices=False, compute_uv=True)
        mps1[-2] = mps1[-2].reshape(s[0]*s[1], s[2], lm.size)
        mps1[-1] = np.diag(lm).dot(mps1[-1]).reshape(lm.size, s[3], 1)
    return mps1


def deep_mps_qubit(mps, para=None, para_dmrg=None):
    if para is None:
        para = parameters_dmps()
    fid0 = mps.fidelity_log_by_spins_up()
    fid = np.ones((para['num_layers'], ))
    lm_mid = empty_list(para['num_layers'])
    ent = empty_list(para['num_layers'])
    mpo = empty_list(para['num_layers'])
    for n in range(para['num_layers']):
        # mps_chi2 = Mob(length=para_dmrg['l'], d=para_dmrg['d'], chi=para_dmrg['chi'], way='qr', ini_way='r',
        #                operators=para_dmrg['op'], is_parallel=para_dmrg['isParallel'],
        #                is_save_op=para_dmrg['is_save_op'], eig_way=para_dmrg['eigWay'],
        #                is_env_parallel_lmr=para_dmrg['isParallelEnvLMR'])
        # mps_chi2.mps = copy.deepcopy(mps.mps)
        mps_chi2 = copy.deepcopy(mps)
        mps_chi2.truncate_virtual_bonds(chi1=2, center=mps_chi2.length - 1, way='full')
        if para['theta'] is None:
            theta = np.arange(0, 1, 1/para['num_theta'])
            mps_tmp = copy.deepcopy(mps_chi2)
            mps_new = copy.deepcopy(mps_chi2)
            for nn in range(theta.size):
                mpo_tmp = mps_chi2.to_unitary_mpo_qubits(theta[nn], if_trun=False)  # calculate unitary MPO
                mps_data = act_umpo_on_mps(mps.mps, mpo_tmp)  # disentangle MPS to |0...0> by uMPO
                mps_tmp.input_mps(mps_data, if_deepcopy=if_deepcopy)  # calculate fidelity with b|0...0>
                # Normalize MPS (actually not necessary since umpo is unitary)
                mps_tmp.orthogonalize_mps(0, mps_tmp.length-1, normalize=True, is_trun=False)
                fid1 = mps_tmp.fidelity_log_by_spins_up()
                if fid1 < fid[n]:
                    mps_new = copy.deepcopy(mps_tmp)
                    mpo[n] = copy.deepcopy(mpo_tmp)
                    fid[n] = fid1
        else:
            mpo[n] = mps_chi2.to_unitary_mpo_qubits(para['theta'], if_trun=False)  # calculate unitary MPO
            mps_data = act_umpo_on_mps(mps.mps, mpo[n])  # disentangle MPS to |0...0> by uMPO
            mps_new = copy.deepcopy(mps)
            mps_new.input_mps(mps_data, if_deepcopy=if_deepcopy)
            fid[n] = mps_new.fidelity_log_by_spins_up()
        # print('The log-fidelity with |0...0> = ' + str(fid[n]))
        # Truncate the MPS
        mps_new.orthogonalize_mps(mps_new.length - 1, 0, normalize=True, is_trun=False)
        mps_new.center = 0
        mps_new.orthogonalize_mps(0, mps_new.length - 1, normalize=True,
                                  is_trun=True, chi=para['chi_overlap'])
        mps_new.center = mps_new.length - 1
        mps_new.calculate_entanglement_entropy()
        lm_mid[n] = mps_new.lm[round(mps_new.length/2)]
        ent[n] = mps_new.ent.reshape(-1, 1)
        # print('The entanglement entropy of the new MPS =  ')
        # print(ent[n].reshape(1, -1))
        mps = mps_new
    fid1 = np.zeros((fid.size+1, ))
    fid1[0] = fid0
    fid1[1:] = fid
    return fid1, ent, lm_mid, mpo, mps


def fidelities_to_original_state(mps_final, mps_ini, mpos, chi):
    mps0 = copy.deepcopy(mps_final)
    mps0.input_mps(mps_ini, if_deepcopy=if_deepcopy)
    fid = np.zeros((mpos.__len__(), ))
    for n in range(mpos.__len__()):
        mps_ini = act_umpo_on_mps(mps0.mps, mpos[mpos.__len__()-n-1], if_conjugate=True)
        mps0.input_mps(mps_ini, if_deepcopy=if_deepcopy)
        mps0.orthogonalize_mps(0, mps0.length - 1, normalize=True, is_trun=False)
        fid[n] = mps_final.fidelity_per_site(mps0.mps)
        mps0.orthogonalize_mps(0, mps0.length-1, normalize=True, is_trun=True, chi=chi)
    return fid

