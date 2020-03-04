from multiprocessing.dummy import Pool as ThreadPool
from library.BasicFunctions import arg_find_array
from library.TensorBasicModule import sort_vectors
from library.HamiltonianModule import hamiltonian_heisenberg, hamiltonian2cell_tensor
from library import Parameters as pm
import matplotlib.pyplot as mp
from termcolor import cprint, colored
import numpy as np
import time

is_debug = False
n_nodes = 4

if is_debug:
    cprint('The DMRG code is in the debug mode', 'cyan')


def dmrg_finite_size(para=None):
    from library.MPSClass import MpsOpenBoundaryClass as Mob
    t_start = time.time()
    info = dict()
    print('Preparing the parameters and MPS')
    if para is None:
        para = pm.generate_parameters_dmrg()
    # Initialize MPS
    is_parallel = para['isParallel']
    if is_parallel or para['isParallelEnvLMR']:
        par_pool = dict()
        par_pool['n'] = n_nodes
        par_pool['pool'] = ThreadPool(n_nodes)
    else:
        par_pool = None

    A = Mob(length=para['l'], d=para['d'], chi=para['chi'], way='qr', ini_way='r',
            operators=para['op'], debug=is_debug, is_parallel=para['isParallel'],
            par_pool=par_pool, is_save_op=para['is_save_op'], eig_way=para['eigWay'],
            is_env_parallel_lmr=para['isParallelEnvLMR'])
    A.correct_orthogonal_center(para['ob_position'])
    print('Starting to sweep ...')
    e0_per_site = 0
    info['convergence'] = 1
    ob = dict()
    for t in range(0, para['sweep_time']):
        if_ob = ((t+1) % para['dt_ob'] == 0) or t == (para['sweep_time'] - 1)
        if if_ob:
            print('In the %d-th round of sweep ...' % (t+1))
        for n in range(para['ob_position']+1, para['l']):
            if para['if_print_detail']:
                print('update the %d-th tensor from left to right...' % n)
            A.update_tensor_eigs(n, para['index1'], para['index2'], para['coeff1'],
                                 para['coeff2'], para['tau'], para['is_real'],
                                 tol=para['eigs_tol'])
        for n in range(para['l']-2, -1, -1):
            if para['if_print_detail']:
                print('update the %d-th tensor from right to left...' % n)
            A.update_tensor_eigs(n, para['index1'], para['index2'], para['coeff1'],
                                 para['coeff2'], para['tau'], para['is_real'],
                                 tol=para['eigs_tol'])
        for n in range(1, para['ob_position']):
            if para['if_print_detail']:
                print('update the %d-th tensor from left to right...' % n)
            A.update_tensor_eigs(n, para['index1'], para['index2'], para['coeff1'],
                                 para['coeff2'], para['tau'], para['is_real'],
                                 tol=para['eigs_tol'])
        if if_ob:
            ob['eb_full'] = A.observe_bond_energy(para['index2'], para['coeff2'])
            ob['mx'] = A.observe_magnetization(1)
            ob['mz'] = A.observe_magnetization(3)
            ob['e_per_site'] = (sum(ob['eb_full']) - para['hx'] * sum(ob['mx']) - para['hz'] *
                                sum(ob['mz'])) / A.length
            # if para['lattice'] in ('square', 'chain'):
            #     ob['e_per_site'] = (sum(ob['eb_full']) - para['hx']*sum(ob['mx']) - para['hz'] *
            #                         sum(ob['mz']))/A.length
            # else:
            #     ob['e_per_site'] = sum(ob['eb_full'])
            #     for n in range(0, para['l']):
            #         ob['e_per_site'] += para['hx'][n] * ob['mx'][n]
            #         ob['e_per_site'] += para['hz'][n] * ob['mz'][n]
            #     ob['e_per_site'] /= A.length
            info['convergence'] = abs(ob['e_per_site'] - e0_per_site)
            if info['convergence'] < para['break_tol']:
                print('Converged at the %d-th sweep with error = %g of energy per site.'
                      % (t+1, info['convergence']))
                break
            else:
                print('Convergence error of energy per site = %g' % info['convergence'])
                e0_per_site = ob['e_per_site']
        if t == para['sweep_time'] - 1 and info['convergence'] > para['break_tol']:
            print('Not converged with error = %g of eb per bond' % info['convergence'])
            print('Consider to increase para[\'sweep_time\']')
    ob['eb'] = get_bond_energies(ob['eb_full'], para['positions_h2'], para['index2'])
    A.calculate_entanglement_spectrum()
    A.calculate_entanglement_entropy()
    ob['corr_x'] = A.observe_correlators_from_middle(1, 1)
    ob['corr_z'] = A.observe_correlators_from_middle(3, 3)
    info['t_cost'] = time.time() - t_start
    print('Simulation finished in %g seconds' % info['t_cost'])
    A.clean_to_save()
    if A._is_parallel:
        par_pool['pool'].close()
    return ob, A, info, para


# ==========================================================================
# Infinite DMRG (one-site or two site)
def dmrg_infinite_size(para=None, A=None, hamilt=None):
    from library.MPSClass import MpsInfinite as Minf
    is_print = True

    t_start = time.time()
    info = dict()
    if is_print:
        print('Start ' + str(para['n_site']) + '-site iDMRG calculation')
    if para is None:
        para = pm.generate_parameters_infinite_dmrg_sawtooth()
    if hamilt is None:
        hamilt = hamiltonian_heisenberg(para['spin'], para['jxy'], para['jxy'], para['jz'],
                                        para['hx']/2, para['hz']/2)
    if A is None:
        if para['dmrg_type'] is 'mpo':
            d = para['d'] ** para['n_site']
        else:
            d = para['d']
        A = Minf(para['form'], d, para['chi'],
                 para['d']**para['n_site'], n_site=para['n_site'],
                 is_symme_env=para['is_symme_env'], dmrg_type=para['dmrg_type'],
                 hamilt_index=para['hamilt_index'])
    if A.dmrg_type is 'mpo':
        tensor = hamiltonian2cell_tensor(hamilt, para['tau'])
    else:
        tensor = np.zeros(0)
    if A.n_site == 1:
        # singe-site iDMRG
        e0 = 0
        e1 = 1
    else:
        # double-site iDMRG (including White's way)
        e0 = np.zeros((1, 3))
        e1 = np.ones((1, 3))
    de = 1
    if A.is_symme_env:
        A.update_ort_tensor_mps('left')
        if A.dmrg_type is 'white':
            A.update_bath_onsite()
            A.update_effective_ops()
        else:
            A.update_left_env(tensor)
    else:
        A.update_ort_tensor_mps('both')
        A.update_left_env(tensor)
        A.update_right_env(tensor)
    # iDMRG sweep
    for t in range(0, para['sweep_time']):
        if A.dmrg_type is 'mpo':
            A.update_central_tensor(tensor)
        else:
            A.update_central_tensor((para['tau'], 'full'))

        if t % para['dt_ob'] == 0:
            A.rho_from_central_tensor()
            e1 = A.observe_energy(hamilt)
            if is_print:
                print('At the %g-th sweep: Eb = ' % t + str(e1))
            de = np.sum(abs(e0-e1))/A.n_site
            if de > para['break_tol']:
                e0 = e1
            elif is_print:
                print('Converged with de = %g' % de)
                break
        if t == para['sweep_time']:
            print('Not sufficiently converged with de = %g' % de)

        if A.is_symme_env:
            A.update_ort_tensor_mps('left')
            if A.dmrg_type is 'mpo':
                A.update_left_env(tensor)
            else:
                A.update_bath_onsite()
                A.update_effective_ops()
        else:
            A.update_ort_tensor_mps('both')
            A.update_left_env(tensor)
            A.update_right_env(tensor)
    ob = {'eb': e1}
    info['t_cost'] = time.time() - t_start
    if is_print:
        print('Total time cost: %g' % info['t_cost'])
    return A, ob, info


def dmrg_infinite_size_sawtooth(para=None, A=None):
    from library.MPSClass import MpsInfiniteSawtooth as Minf
    is_print = True

    t_start = time.time()
    info = dict()
    if is_print:
        print('Start ' + str(para['n_site']) + '-site iDMRG (sawtooth) calculation')
    if para is None:
        para = pm.generate_parameters_infinite_dmrg_sawtooth()
        para = pm.make_para_consistent_idmrg_sawtooth(para)

    if A is None:
        d = para['d']
        A = Minf(para['form'], d, para['chi'], para['d'] ** para['n_site'],
                 n_site=para['n_site'], is_symme_env=para['is_symme_env'],
                 dmrg_type=para['dmrg_type'], spin=para['spin'])

    ob = dict()
    e1 = 0
    de = 1
    A.update_ort_tensor_mps_sawtooth()
    A.update_bath_onsite_sawtooth(para['j1'], para['j2'], para['hx'], para['hz'])
    A.update_effective_ops_sawtooth()

    # iDMRG sweep
    for t in range(0, para['sweep_time']):
        A.update_central_tensor_sawtooth(para['tau'], para['j1'], para['j2'],
                                         para['hx'], para['hz'])

        if t % para['dt_ob'] == 0:
            A.rho_from_central_tensor_sawtooth()
            ob['eb'], ob['mag'], ob['energy_site'], ob['ent'] = A.observation_sawtooth(
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

        A.update_ort_tensor_mps_sawtooth()
        A.update_bath_onsite_sawtooth(para['j1'], para['j2'], para['hx'], para['hz'])
        A.update_effective_ops_sawtooth()
    info['t_cost'] = time.time() - t_start
    print('Energy per site = %g' % ob['energy_site'])
    print('x-magnetization = ' + str(ob['mag']['x']))
    print('z-magnetization = ' + str(ob['mag']['z']))
    print('Entanglement = ' + str(ob['ent']))
    print('Total time cost: %g' % info['t_cost'])
    return A, ob, info


# ======================================================
def deep_dmrg_infinite_size(para=None):
    from library.MPSClass import MpsDeepInfinite as Minf
    is_print = True

    t_start = time.time()
    info = dict()
    if is_print:
        print('Start deep DMRG calculation')
    if para is None:
        para = pm.generate_parameters_deep_mps_infinite()

    hamilt = hamiltonian_heisenberg(para['spin'], para['jxy'], para['jxy'],
                                    para['jz'], -para['hx']/2, -para['hz']/2)
    tensor = hamiltonian2cell_tensor(hamilt, para['tau'])
    A = Minf(para['form'], para['d'], para['chi'], para['d'], para['chib0'], para['chib'],
             para['is_symme_env'], n_site=para['n_site'], is_debug=is_debug)
    # use standard DMRG to get the GS MPS
    A, ob0, info0 = dmrg_infinite_size(para, A, hamilt)
    # get uMPO from the MPS
    A.get_unitary_mpo_from_mps()

    if A.n_site == 1:
        e0 = 0
        e1 = 1
    else:
        e0 = np.zeros((1, 3))
        e1 = np.ones((1, 3))
    de = 1
    for t in range(0, para['sweep_time']):
        A.update_ort_tensor_dmps('left')
        A.update_left_env_dmps_simple(tensor)
        if not A.is_symme_env:
            A.update_ort_tensor_dmps('right')
            A.update_right_env_dmps_simple(tensor)
        A.update_central_tensor_dmps(tensor)
        if t % para['dt_ob'] == 0:
            A.rho_from_central_tensor_dmps()
            e1 = A.observe_energy(hamilt)
            if is_print:
                print('At the %g-th sweep: Eb = ' % t + str(e1))
            de = np.sum(abs(e0-e1))
            if de > para['break_tol']:
                e0 = e1
            elif is_print:
                print('Converged with de = %g' % de)
                break
        if t == para['sweep_time']:
            print('Not sufficiently converged with de = %g' % de)
    ob = {'eb': e1}
    info['t_cost'] = time.time() - t_start
    if is_print:
        print('Total time cost: %g' % info['t_cost'])
    return A, ob, info, ob0, info0


# ======================================================
def positions_set2array(pos_set):
    nh = pos_set.__len__()
    pos_set = list(pos_set)
    pos = np.zeros((nh, 2))
    for n in range(0, nh):
        pos[n, :] = np.array(pos_set[n])
    pos = pos.astype(int)
    pos -= np.min(pos)
    p_max = np.max(pos)
    for i in range(p_max-1, 0, -1):
        number = np.nonzero(pos == i)[0].size
        if number == 0:
            pos -= (pos > i)
    return pos


def sort_positions(pos, which='ascend'):
    # pos must be formed by int
    order = np.argsort(pos[:, 0])
    if which is 'descend':
        order = order[::-1]
    pos = sort_vectors(pos, order, 'row').astype(int)
    l_now = 0
    for n in range(min(pos[:, 0]), max(pos[:, 0])+1):
        ln = arg_find_array(pos[:, 0] == n, 1, 'last')
        if ln.size > 0:
            _tmp = pos[l_now:ln+1, 1:]
            order = np.argsort(_tmp[:, 0])
            pos[l_now:ln+1, 1:] = sort_vectors(_tmp, order, 'row')
            l_now = ln+1
    return pos


def get_bond_energies(eb_full, positions, index2):
    nl = positions.shape[0]
    nh = eb_full.size
    eb = np.zeros((nl, 1))
    for i in range(0, nh):
        p = (index2[i, 0] == positions[:, 0]) * (index2[i, 1] == positions[:, 1])
        p = np.nonzero(p)
        eb[p] += eb_full[i]
    return eb


def plot_finite_dmrg(x, A, para, ob):
    mp.figure()
    if x is 'eb':  # plot bond energies
        if para['lattice'] == 'chain':
            nh1 = ob['eb'].size - (para['bound_cond'] == 'periodic')
            mp.plot(range(0, nh1), ob['eb'][:nh1], 'bo')
            if para['bound_cond'] == 'periodic':
                mp.plot(np.array([0, nh1 - 1]), ob['eb'][-1] * np.ones((2,)), 'r--.', linewidth=0.5)
                mp.text(A.length / 2, ob['eb'][-1] - 0.0002, 'Eb(0, %d) = %g' % (A.length - 1, ob['eb'][-1]),
                        fontsize=10, verticalalignment="top", horizontalalignment="center")
        elif para['lattice'] == 'square' or 'arbitrary':
            nh1 = ob['eb'].size
            f1, = mp.plot(range(1, nh1 + 1), ob['eb'][:nh1], 'bo')
            mp.title('Bond energies (nearest-neighbor correlators)')
        mp.xlabel('lattice bond')
        mp.ylabel(r'$\langle \hat{s}_n \hat{s}_{n+1} \rangle$')
        print('Bond energies = ')
        cprint(str(ob['eb'].T), 'cyan')
        print('Energy per site = ' + colored(str(ob['e_per_site']), 'cyan'))
        if para['lattice'] == 'square':
            print('NOTE: check ' + colored('para[positions_h2]', 'cyan') + 'to see how the bonds are numbered')
    elif x is 'mag':  # plot magnetization
        mp.subplot(2, 1, 1)
        f1, = mp.plot(range(1, A.length + 1), ob['mx'], '-ro')
        mp.ylabel(r'$\langle \hat{s}^x \rangle$')
        # mp.legend(handles=[f1, ], labels=[r'$\langle \hat{s}_n^x \rangle$'], loc='best')
        mp.subplot(2, 1, 2)
        f2, = mp.plot(range(1, A.length + 1), ob['mz'], '--bs')
        mp.xlabel('lattice site')
        mp.ylabel(r'$\langle \hat{s}^z \rangle$')
        # mp.legend(handles=[f2, ], labels=[r'$\langle \hat{s}_n^z \rangle$'], loc='best')
        print('mx = ')
        print(str(ob['mx'].T))
        print('mz = ')
        print(str(ob['mz'].T))
        if para['lattice'] == 'square':
            print('Check the numbers of sites in .\\fig_dmrg\\' + para['lattice']
                  + '(%d,%d).png' % (para['square_width'], para['square_height']))
    elif x is 'ent':
        mp.plot(range(1, A.length), A.ent, '--or')
        mp.xlabel('lattice bond')
        mp.ylabel('entanglement entropy')
        print('entanglement entropy = ')
        print(str(A.ent.T))
    elif x is 'corr':
        mp.subplot(2, 1, 1)
        mp.plot(range(1, ob['corr_x'].__len__() + 1), ob['corr_x'], '-ro')
        mp.ylabel(r'$\langle \hat{s}^x \hat{s}^x \rangle$')
        # mp.legend(handles=[f1, ], labels=[r'$\langle \hat{s}_n^x \rangle$'], loc='best')
        mp.subplot(2, 1, 2)
        mp.plot(range(1, ob['corr_x'].__len__() + 1), ob['corr_x'], '--bs')
        mp.xlabel('lattice site')
        mp.ylabel(r'$\langle \hat{s}^z \hat{s}^z \rangle$')
        # mp.legend(handles=[f2, ], labels=[r'$\langle \hat{s}_n^z \rangle$'], loc='best')
        print('<sx sx> = ')
        print(str(ob['corr_x'].T))
        print('<sz sz> = ')
        print(str(ob['corr_z'].T))
    mp.show()
