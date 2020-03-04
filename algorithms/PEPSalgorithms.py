from library.PEPSClass import PepsInfinite
import numpy as np
from library import HamiltonianModule as hm, Parameters as pm
from library.BasicFunctions import save_pr as save
import time

is_debug = False


def super_orthogonalization_honeycomb(para=None):
    if para is None:
        para = pm.generate_parameters_so_honeycomb()
    peps = PepsInfinite(para['lattice'], para['chi'], para['state_type'],
                        ini_way=para['ini_way'], is_debug=para['is_debug'])
    h = hm.hamiltonian_heisenberg(para['spin'], para['jxy'], para['jxy'], para['jz'],
                                  para['hx'], para['hz'])
    op = hm.spin_operators(para['spin'])
    ob = dict()
    if para['state_type'] is 'pure':
        ob['mx'] = np.zeros((para['tau'].__len__(), peps.nTensor))
        ob['mz'] = np.zeros((para['tau'].__len__(), peps.nTensor))
        ob['eb'] = np.zeros((para['tau'].__len__(), peps.nLm))
    else:
        ob['mx'] = np.zeros((para['beta'].__len__(), peps.nTensor))
        ob['mz'] = np.zeros((para['beta'].__len__(), peps.nTensor))
        ob['eb'] = np.zeros((para['beta'].__len__(), peps.nLm))
        ob['e_site'] = np.zeros((para['beta'].__len__(), ))
    if para['state_type'] is 'pure':
        for n_tau in range(0, para['tau'].__len__()):
            gate_t = hm.hamiltonian2gate_tensors(h, para['tau'][n_tau], 'exp')
            eb0 = np.ones((peps.nLm, ))
            eb = np.zeros((peps.nLm, ))
            for t in range(1, round(para['beta']/para['tau'][n_tau]) + 1):
                for n_lm in range(0, peps.nLm):
                    peps.evolve_once_tensor_and_lm(gate_t[0], gate_t[1], n_lm)
                    if para['so_time'] == 0:
                        peps.super_orthogonalization(n_lm)
                    else:
                        peps.super_orthogonalization('all', it_time=para['so_time'])
                    # if is_debug:
                    #     peps.check_super_orthogonality()
                if t % para['dt_ob'] == 0:
                    rho2 = peps.rho_two_body_simple('all')
                    for nr in range(0, peps.nLm):
                        eb[nr] = rho2[nr].reshape(1, -1).dot(h.reshape(-1, 1))
                    if para['if_print']:
                        print('For tau = %g, t = %g: ' % (para['tau'][n_tau], t) +
                              'bond energy  = ' + str(eb))
                    err = np.linalg.norm(eb0 - eb)
                    if err < para['tol']:
                        ob['eb'][n_tau, :] = eb.copy()
                        rho1 = peps.rho_one_body_simple('all')
                        for nr in range(0, peps.nTensor):
                            ob['mx'][n_tau, nr] = rho1[nr].reshape(1, -1).dot(op['sx'].reshape(-1, 1))
                            ob['mz'][n_tau, nr] = rho1[nr].reshape(1, -1).dot(op['sz'].reshape(-1, 1))
                        if para['if_print']:
                            print('Converged with error = %g' % err)
                        break
                    else:
                        eb0 = eb.copy()
    elif para['state_type'] is 'mixed':
        gate_t = hm.hamiltonian2gate_tensors(h, para['tau'], 'exp')
        beta_now = 0
        t_ob = 0
        for t in range(0, int(1e-6 + para['beta'][-1]/para['tau'])):
            for n_lm in range(0, peps.nLm):
                peps.evolve_once_tensor_and_lm(gate_t[0], gate_t[1], n_lm)
                if para['so_time'] == 0:
                    peps.super_orthogonalization(n_lm)
                else:
                    peps.super_orthogonalization('all', it_time=para['so_time'])
                # if is_debug:
                #     peps.check_super_orthogonality()
            beta_now += para['tau']
            if abs(para['beta'][t_ob] - beta_now) < 1e-8:
                rho2 = peps.rho_two_body_simple('all')
                for nr in range(0, peps.nLm):
                    ob['eb'][t_ob, nr] = rho2[nr].reshape(1, -1).dot(h.reshape(-1, 1))
                ob['e_site'][t_ob] = np.sum(ob['eb'][t_ob, :]) / 2
                if para['if_print']:
                    print('For beta = %g: ' % beta_now + 'energy per site  = ' + str(ob['e_site'][t_ob]))
                rho1 = peps.rho_one_body_simple('all')
                for nr in range(0, peps.nTensor):
                    ob['mx'][t_ob, nr] = rho1[nr].reshape(1, -1).dot(op['sx'].reshape(-1, 1))
                    ob['mz'][t_ob, nr] = rho1[nr].reshape(1, -1).dot(op['sz'].reshape(-1, 1))
                t_ob += 1
    para['data_exp'] = 'HoneycombSO_' + para['state_type'] + '_j(%g,%g)_h(%g,%g)_chi%d' % \
                       (para['jxy'], para['jz'], para['hx'], para['hz'],
                        para['chi'])
    save(para['data_path'], para['data_exp'], [peps, para, ob], ['peps', 'para', 'ob'])


def tree_dmrg_ipeps_kagome_gs(para=None, A=None):
    from library.PEPSClass import TreePepsIdmrgKagome as Peps
    is_print = True

    t_start = time.time()
    info = dict()
    if is_print:
        print('Start tree-DMRG on iPEPS kagome (Husimi) calculation')
    if para is None:
        para = pm.generate_parameters_tree_ipeps_kagome()
        para = pm.make_para_consistent_tree_ipeps_kagome(para)

    if A is None:
        A = Peps(para['chi'], para['spin'])

    ob = dict()
    e1 = 0
    de = 1
    A.update_ort_tensor_kagome()
    A.update_bath_onsite_kagome(para['j1'], para['j2'], para['hx'], para['hz'])
    A.update_effective_ops_kagome()

    # sweep
    for t in range(0, para['sweep_time']):
        A.update_central_tensor_kagome(para['tau'], para['j1'], para['j2'],
                                       para['hx'], para['hz'])

        if t % para['dt_ob'] == 0:
            A.rho_from_central_tensor_kagome()
            ob['eb'], ob['mag'], ob['energy_site'], ob['ent'] = A.observation_kagome(
                para['j1'], para['j2'], para['hx'], para['hz'])
            if is_print:
                print('At the %g-th sweep: Eb = ' % t + str(e1))
            de = sum(abs(ob['eb'] - e1)) / ob['eb'].__len__()
            if de > para['break_tol']:
                e1 = ob['eb']
            elif is_print:
                print('Converged with de = %g' % de)
                break
        if t == para['sweep_time']:
            print('Not sufficiently converged with de = %g' % de)

        A.update_ort_tensor_kagome()
        A.update_bath_onsite_kagome(para['j1'], para['j2'], para['hx'], para['hz'])
        A.update_effective_ops_kagome()
    info['t_cost'] = time.time() - t_start
    print('Energy per site = %g' % ob['energy_site'])
    print('x-magnetization = ' + str(ob['mag']['x']))
    print('z-magnetization = ' + str(ob['mag']['z']))
    print('Entanglement = ' + str(ob['ent']))
    print('Total time cost: %g' % info['t_cost'])
    return A, ob, info

