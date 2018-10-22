# include the functions that relate to Hamiltonian's and gates
import numpy as np
import scipy.sparse.linalg as la
from TensorBasicModule import cont
from BasicFunctionsSJR import combination


def spin_operators(spin):
    op = dict()
    if spin is 'half':
        op['id'] = np.eye(2)
        op['sx'] = np.zeros((2, 2))
        op['sy'] = np.zeros((2, 2), dtype=np.complex)
        op['sz'] = np.zeros((2, 2))
        op['su'] = np.zeros((2, 2))
        op['sd'] = np.zeros((2, 2))
        op['sx'][0, 1] = 0.5
        op['sx'][1, 0] = 0.5
        op['sy'][0, 1] = 0.5 * 1j
        op['sy'][1, 0] = -0.5 * 1j
        op['sz'][0, 0] = 0.5
        op['sz'][1, 1] = -0.5
        op['su'][0, 1] = 1
        op['sd'][1, 0] = 1
    elif spin is 'one':
        op['id'] = np.eye(3)
        op['sx'] = np.zeros((3, 3))
        op['sy'] = np.zeros((3, 3), dtype=np.complex)
        op['sz'] = np.zeros((3, 3))
        op['sx'][0, 1] = 1
        op['sx'][1, 0] = 1
        op['sx'][1, 2] = 1
        op['sx'][2, 1] = 1
        op['sy'][0, 1] = -1j
        op['sy'][1, 0] = 1j
        op['sy'][1, 2] = -1j
        op['sy'][2, 1] = 1j
        op['sz'][0, 0] = 1
        op['sz'][2, 2] = -1
        op['sx'] /= 2 ** 0.5
        op['sy'] /= 2 ** 0.5
        op['su'] = np.real(op['sx'] + 1j * op['sy'])
        op['sd'] = np.real(op['sx'] - 1j * op['sy'])
    return op


def fermionic_operators(spin):
    op = dict()
    if spin is 'zero':
        op['id'] = np.eye(2)
        op['cu'] = np.zeros((2, 2))
        op['cd'] = np.zeros((2, 2))
        op['n'] = np.zeros((2, 2))
        op['cu'][0, 1] = 1
        op['cd'][1, 0] = 1
        op['n'][1, 1] = 1
    return op


def from_spin2phys_dim(spin):
    if spin is 'half':
        return 2
    elif spin is 'one':
        return 3


def hamiltonian_heisenberg(spin, jx, jy, jz, hx, hz):
    op = spin_operators(spin)
    hamilt = jx*np.kron(op['sx'], op['sx']) + jy*np.kron(op['sy'], op['sy']).real + jz*np.kron(op['sz'], op['sz'])
    hamilt += hx*(np.kron(op['id'], op['sx']) + np.kron(op['sx'], op['id']))
    hamilt += hz*(np.kron(op['id'], op['sz']) + np.kron(op['sz'], op['id']))
    return hamilt


def hamiltonian_spinless_fermion(j, u):
    op = fermionic_operators('zero')
    hamilt = j*(np.kron(op['cu'], op['cd']) + np.kron(op['cd'], op['cu'])) + u/2*(
            np.kron(op['id'], op['n']) + np.kron(op['n'], op['id']))
    return hamilt


def hamiltonian2cell_tensor(h, tau, way='shift'):
    # h has to be a two-body hamiltonian at least
    """ The indexes of the out put tensor are ordered as:
        1
        |
    0 - T - 3
        |
        2
    :param h:
    :param tau:
    :param way:
    :return:
    """
    dd = h.shape[0]
    d = round(dd**0.5)
    if way is 'shift':
        h = np.eye(dd) - tau*h
    else:
        h = la.expm(-tau*h)
    h = h.reshape(d, d, d, d)
    tmp = h.transpose(0, 2, 1, 3).reshape(dd, dd)
    vl, lm, vr = np.linalg.svd(tmp)
    lm = np.diag(lm**0.5)
    vl = vl.dot(lm).reshape(d, d, dd)
    vr = lm.dot(vr).reshape(dd, d, d)
    tensor = cont([h, vr, vl], [[1, 2, -4, -5], [-1, -2, 1], [-3, 2, -6]])
    tensor = tensor.reshape(dd, dd, dd, dd)
    return tensor


def hamiltonian2gate_tensors(h, tau, way='shift'):
    # h has to be a two-body hamiltonian at least
    gate_t = [[], []]
    dd = h.shape[0]
    d = round(dd**0.5)
    if way is 'shift':
        h = np.eye(dd) - tau*h
    else:
        h = la.expm(-tau*h)
    h = h.reshape(d, d, d, d)
    gate_t[0], lm, gate_t[1] = np.linalg.svd(h.transpose(0, 2, 1, 3).reshape(dd, dd))
    lm = np.diag(lm**0.5)
    gate_t[0] = gate_t[0].dot(lm).reshape(d, d, dd).transpose(0, 2, 1)
    gate_t[1] = lm.dot(gate_t[1]).reshape(dd, d, d).transpose(1, 0, 2)
    return gate_t


def environment_tensor_to_bath_hamilt(vl, vr, is_mat=True):
    h = np.tensordot(vl, vr, [[1], [1]]).transpose(0, 2, 1, 3)
    if is_mat:
        s = h.shape
        h = h.reshape(s[0]*s[1], s[2]*s[3])
    return h


def interactions_full_connection_two_body(l):
    # return the interactions of the fully connected two-body Hamiltonian
    # interact: [first_site, second_site]
    from scipy.special import comb
    ni = comb(l, 2)
    interact = np.zeros((int(ni), 2))
    n = 0
    for n1 in range(0, l):
        for n2 in range(n1+1, l):
            interact[n, :] = [n1, n2]
            n += 1
    return interact, ni


def positions_nearest_neighbor_1d(l, bound_cond='open'):
    # return the 1D Hamiltonian with nearest-neighbor interactions
    # index: [first_site, second_site]
    nh = l-1
    index = np.zeros((nh + (bound_cond == 'periodic'), 2))
    for n in range(0, nh):
        index[n, 0] = n
        index[n, 1] = n + 1
    if bound_cond == 'periodic':  # default: open boundary condition
        index[nh, 0] = 0
        index[nh, 1] = l - 1
    return index


def positions_jigsaw_1d(length, bound_cond='open'):
    # return the 1D Hamiltonian with nearest-neighbor interactions
    # index: [first_site, second_site]
    if bound_cond is 'open':
        if length % 2 == 0:
            print('Note: for OBC jigsaw, l has to be odd. Auto-change l = %g to %g'
                  % (length, length + 1))
            length += 1
        n_half = round((length - 1) / 2)
    else:
        if length % 2 == 1:
            print('Note: for PBC jigsaw, l has to be even. Auto-change l = %g to %g'
                  % (length, length + 1))
            length += 1
        n_half = round(length / 2)
    nh = n_half * 3
    pos = np.zeros((nh, 2))
    for n in range(0, length - 1):
        pos[n, 0] = n
        pos[n, 1] = n + 1
    l_now = length - 1
    if bound_cond is 'periodic':
        pos[l_now, 0] = 0
        pos[l_now, 1] = length - 1
        l_now += 1
    if bound_cond is 'open':
        for n in range(0, n_half):
            pos[l_now + n, 0] = n * 2
            pos[l_now + n, 1] = (n + 1) * 2
    elif bound_cond is 'periodic':
        for n in range(0, n_half-1):
            pos[l_now + n, 0] = n * 2
            pos[l_now + n, 1] = (n + 1) * 2
        pos[nh - 1, 0] = 0
        pos[nh - 1, 1] = length - 2
    return pos


def positions_nearest_neighbor_square(width, height, bound_cond='open'):
    pos = np.zeros((width-1, 2))
    for i in range(0, width-1):  # interactions inside the first row
        pos[i, :] = [i, i+1]
    for n in range(1, height):  # interactions inside the n-th row
        tmp = np.zeros((width-1, 2))
        for i in range(0, width-1):
            tmp[i, :] = [n*width + i, n*width + i + 1]
        pos = np.vstack((pos, tmp))
    for n in range(0, width):
        tmp = np.zeros((height-1, 2))
        for i in range(0, height-1):
            tmp[i, :] = [i*width + n, (i + 1)*width + n]
        pos = np.vstack((pos, tmp))
    if bound_cond == 'periodic':
        tmp = np.zeros((height, 2))
        for n in range(0, height):
            tmp[n, :] = [n*width, (n + 1)*width - 1]
        pos = np.vstack((pos, tmp))
        tmp = np.zeros((width, 2))
        for n in range(0, width):
            tmp[n, :] = [n, (height - 1)*width + n]
        pos = np.vstack((pos, tmp))
        pos = pos.astype(int)
    return pos


def positions_fully_connected(n_site):
    n_hamilt = int(combination(n_site, 2))
    pos = np.zeros((n_hamilt, 2), dtype=int)
    n_now = 0
    for n0 in range(0, n_site - 1):
        for n1 in range(n0+1, n_site):
            pos[n_now, 0] = n0
            pos[n_now, 1] = n1
            n_now += 1
    return pos


def interactions_position2full_index_heisenberg_two_body(index_pos):
    # index: [first_site, second_site, first operator, second operator]
    # the operator is ordered as [id, sx, sy, sz, su, sd]
    nh = index_pos.shape[0]
    index = np.zeros((nh*3, 4))
    for n in range(0, nh):
        index[3 * n, 0] = index_pos[n, 0]
        index[3 * n, 1] = index_pos[n, 1]
        index[3 * n, 2] = 4  # su
        index[3 * n, 3] = 5  # sd
        index[3 * n + 1, 0] = index_pos[n, 0]
        index[3 * n + 1, 1] = index_pos[n, 1]
        index[3 * n + 1, 2] = 5  # sd
        index[3 * n + 1, 3] = 4  # su
        index[3 * n + 2, 0] = index_pos[n, 0]
        index[3 * n + 2, 1] = index_pos[n, 1]
        index[3 * n + 2, 2] = 3  # sz
        index[3 * n + 2, 3] = 3  # sz
        index = index.astype(int)
    return index
