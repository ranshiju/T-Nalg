# include the functions that relate to Hamiltonian's and gates
import numpy as np


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
    return op


def hamiltonian_heisenberg(jx, jy, jz, hx, hz):
    op = spin_operators(0.5)
    hamilt = jx*np.kron(op['sx'], op['sx']) + jy*np.kron(op['sy'], op['sy']).real + jz*np.kron(op['sz'], op['sz'])
    hamilt += hx*(np.kron(np.eye(2), op['sx']) + np.kron(op['sx'], np.eye(2)))
    hamilt += hz*(np.kron(np.eye(2), op['sz']) + np.kron(op['sz'], np.eye(2)))
    return hamilt


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


def positions_nearest_neighbor_square(width, height, bound_cond='open'):
    index = np.zeros((width-1, 2))
    for i in range(0, width-1):  # interactions inside the first row
        index[i, :] = [i, i+1]
    for n in range(1, height):  # interactions inside the n-th row
        tmp = np.zeros((width-1, 2))
        for i in range(0, width-1):
            tmp[i, :] = [n*width + i, n*width + i + 1]
        index = np.vstack((index, tmp))
    for n in range(0, width):
        tmp = np.zeros((height-1, 2))
        for i in range(0, height-1):
            tmp[i, :] = [i*width + n, (i + 1)*width + n]
        index = np.vstack((index, tmp))
    if bound_cond == 'periodic':
        tmp = np.zeros((height, 2))
        for n in range(0, height):
            tmp[n, :] = [n*width, (n + 1)*width - 1]
        index = np.vstack((index, tmp))
        tmp = np.zeros((width, 2))
        for n in range(0, width):
            tmp[n, :] = [n, (height - 1)*width + n]
        index = np.vstack((index, tmp))
        index = index.astype(int)
    return index


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
