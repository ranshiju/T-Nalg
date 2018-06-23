from multiprocessing.dummy import Pool as ThreadPool
from BasicFunctionsSJR import arg_find_array
from TensorBasicModule import sort_vectors
from HamiltonianModule import hamiltonian_heisenberg, hamiltonian2cell_tensor
import Parameters as pm
import numpy as np
import time

is_debug = False
is_parallel = False
n_nodes = 4
is_save_op = True


def dmrg_finite_size(para=None):
    from MPSClass import MpsOpenBoundaryClass as Mob
    t_start = time.time()
    info = dict()
    print('Preparation the parameters and MPS')
    if para is None:
        para = pm.generate_parameters_dmrg()
    # Initialize MPS
    if is_parallel:
        par_pool = ThreadPool(n_nodes)
    else:
        par_pool = None

    A = Mob(length=para['l'], d=para['d'], chi=para['chi'], way='qr', ini_way='r', operators=para['op'],
            debug=is_debug, is_parallel=is_parallel, par_pool=par_pool, is_save_op=is_save_op)
    A.correct_orthogonal_center(para['ob_position'])
    print('Starting to sweep ...')
    e0_total = 0
    info['convergence'] = 1
    ob = dict()
    for t in range(1, para['sweep_time']+1):
        if_ob = ((t % para['dt_ob']) == 0) or t == (para['sweep_time'] - 1)
        if if_ob:
            print('In the %d-th round of sweep ...' % t)
        for n in range(para['ob_position']+1, para['l']):
            if para['if_print_detail']:
                print('update the %d-th tensor from left to right...' % n)
            A.update_tensor_eigs(n, para['index1'], para['index2'], para['coeff1'], para['coeff2'], para['tau'],
                                 para['is_real'], tol=para['eigs_tol'])
        for n in range(para['l']-2, -1, -1):
            if para['if_print_detail']:
                print('update the %d-th tensor from right to left...' % n)
            A.update_tensor_eigs(n, para['index1'], para['index2'], para['coeff1'], para['coeff2'], para['tau'],
                                 para['is_real'], tol=para['eigs_tol'])
        for n in range(1, para['ob_position']):
            if para['if_print_detail']:
                print('update the %d-th tensor from left to right...' % n)
            A.update_tensor_eigs(n, para['index1'], para['index2'], para['coeff1'], para['coeff2'], para['tau'],
                                 para['is_real'], tol=para['eigs_tol'])

        if if_ob:
            ob['eb_full'] = A.observe_bond_energy(para['index2'], para['coeff2'])
            ob['mx'] = A.observe_magnetization(1)
            ob['mz'] = A.observe_magnetization(3)
            if para['lattice'] in ('square', 'chain'):
                ob['e_per_site'] = (sum(ob['eb_full']) - para['hx']*sum(ob['mx']) - para['hz']*
                                    sum(ob['mz']))/A.length
            else:
                ob['e_per_site'] = sum(ob['eb_full'])
                for n in range(0, para['coeff1'].shape[0]):
                    if para['index1'][n, 1] == 1:
                        ob['e_per_site'] += para['coeff1'][n]*ob['mx'][n]
                    elif para['index1'][n, 1] == 3:
                        ob['e_per_site'] += para['coeff1'][n] * ob['mz'][n]
                ob['e_per_site'] /= A.length
            info['convergence'] = abs(ob['e_per_site'] - e0_total)
            if info['convergence'] < para['break_tol']:
                print('Converged at the %d-th sweep with error = %g of energy per site.' % (t, info['convergence']))
                break
            else:
                print('Convergence error of energy per site = %g' % info['convergence'])
                e0_total = ob['e_per_site']
        if t == para['sweep_time'] - 1 and info['convergence'] > para['break_tol']:
            print('Not converged with error = %g of eb per bond' % info['convergence'])
            print('Consider to increase para[\'sweep_time\']')
    ob['eb'] = get_bond_energies(ob['eb_full'], para['positions_h2'], para['index2'])
    A.calculate_entanglement_spectrum()
    A.calculate_entanglement_entropy()
    info['t_cost'] = time.time() - t_start
    print('Simulation finished in %g seconds' % info['t_cost'])
    A.clean_to_save()
    if A._is_parallel:
        par_pool.close()
    return ob, A, info, para


def dmrg_infinite_size(para=None):
    from MPSClass import MpsInfinite as Minf
    t_start = time.time()
    info = dict()
    print('Preparation the parameters and MPS')
    if para is None:
        para = pm.generate_parameters_infinite_dmrg()

    hamilt = hamiltonian_heisenberg(para['jxy'], para['jxy'], para['jz'], para['hx']/2, para['hz']/2)
    tensor = hamiltonian2cell_tensor(hamilt)
    A = Minf(para['form'], para['d'], para['chi'])

    e0 = 0
    for t in range(0, para['sweep_time']):
        A.update_left_env(tensor)
        A.update_right_env(tensor)
        A.update_central_tensor(tensor)
        if t % para['dt_ob'] == 0:
            e1 = A.observe_energy(hamilt)
            if abs(e0-e1) > para['break_tol']:
                e0 = e1
            else:
                break

    info['t_cost'] = time.time() - t_start


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
