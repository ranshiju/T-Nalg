import numpy as np
import torch as tc
import time
from library.BasicFunctions import sort_list, trace_stack, print_error, arg_find_array, \
    combination, empty_list
from termcolor import cprint
import copy

is_debug = False
if is_debug:
    cprint('Note: you are in the debug mode of module \'Basic_Functions_SJR\'', 'cyan')


class CONT:

    def __init__(self, tensors, indexes, _is_debug=False, if_cuda=False,
                 device='cuda', dtype=None):
        self._is_debug = _is_debug
        self.if_cuda = if_cuda
        self.device = device
        self.dtype = dtype

        self.tensors = tensors
        self.indexes = indexes
        self.bond_open = list()
        self.bond_ark = list()
        self.find_open_ark_bonds()

        if self._is_debug:
            self.check_consistency()
        if self.if_cuda:
            for n in range(len(self.tensors)):
                if type(self.tensors[n]) == np.ndarray:
                    self.tensors[n] = tc.from_numpy(
                        self.tensors[n]).to(self.device)
                if self.dtype is not None:
                    self.tensors[n].type(self.dtype)

        while self.bond_ark.__len__() > 0:
            pos = self.tensors_and_bonds_in_nth_contraction(self.bond_ark[0])
            self.contract_now(pos)
            if self._is_debug:
                print('Dummy indexes left: ' + str(self.bond_ark))
        ind = sorted(range(len(self.indexes[-1])), key=lambda k: self.indexes[-1][k])
        if if_cuda:
            self.result = self.tensors[-1].permute(ind[::-1])
        else:
            self.result = self.tensors[-1].transpose(ind[::-1])

    def check_consistency(self):
        if len(self.tensors) != len(self.indexes):
            print_error('ErrorNCON: the number of tensors and number of index tuples are not consistent', False)
        if self.bond_open[0] != -1:
            print_error('NumberingError: the starting number of open bonds should be -1', False)
        if self.bond_open.__len__() != (-self.bond_open[-1]):
            print_error('NumberingError: all integers in [-1, -(number of open bonds)] should appear in the '
                        'numbering. Please check.')
        if self.bond_ark[0] != 1:
            print_error('NumberingError: the starting number of open bonds should be 1', False)
        if self.bond_ark.__len__() != self.bond_ark[-1]:
            print_error('NumberingError: all integers in [-1, -(number of ark bonds)] should appear in the '
                        'numbering. Please check.')
        if self.if_cuda:
            if not tc.cuda.is_available():
                print_error('GPU is not available but requested. Set self.if_cuda=False.')
                self.if_cuda = False

    def find_open_ark_bonds(self):
        if self._is_debug:
            self.bond_open = set()
            self.bond_ark = set()
            for n in range(0, self.tensors.__len__()):
                for i in range(0, self.indexes[n].__len__()):
                    if self.indexes[n][i] < 0:
                        self.bond_open.add(self.indexes[n][i])
                    elif self.indexes[n][i] > 0:
                        self.bond_ark.add(self.indexes[n][i])
            self.bond_open = list(self.bond_open)
            self.bond_open.sort(reverse=True)
            self.bond_ark = list(self.bond_ark)
            self.bond_ark.sort()
        else:
            bond_ark_max = 0
            for n in range(0, self.tensors.__len__()):
                bond_ark_max = max(bond_ark_max, max(self.indexes[n]))
            self.bond_ark = list(range(1, bond_ark_max+1))

    def tensors_and_bonds_in_nth_contraction(self, bond):
        # n is the number of the contracted bond
        pos = list()
        for n in range(0, self.tensors.__len__()):
            if bond in self.indexes[n]:
                pos.append(n)
            if pos.__len__() == 2:
                return pos

    def contract_now(self, pos):
        if self._is_debug:
            t0 = time.time()
        ind_now = [self.indexes[pos[0]], self.indexes[pos[1]]]
        ind_con = list(set(ind_now[0]) & set(ind_now[1]))  # indexes to be contracted
        ind_con_pos = [[], []]
        for i in range(0, ind_con.__len__()):
            ind_con_pos[0].append(ind_now[0].index(ind_con[i]))
            ind_con_pos[1].append(ind_now[1].index(ind_con[i]))
        ind_left_pos = [list(range(0, ind_now[0].__len__())), list(range(0, ind_now[1].__len__()))]
        for i in range(0, ind_con.__len__()):
            ind_left_pos[0].remove(ind_con_pos[0][i])
            ind_left_pos[1].remove(ind_con_pos[1][i])
        for i in range(0, ind_con.__len__()):
            ind_now[0].remove(ind_con[i])
            ind_now[1].remove(ind_con[i])
            self.bond_ark.remove(ind_con[i])
        if self._is_debug:
            print('Indexes to be contracted in this stage: ' + str(ind_con))
            print('S1 in contract_now: ')
            print(t0 - time.time())
            t0 = time.time()
        if self.if_cuda:
            self.tensors[min(pos)] = tc.tensordot(
                self.tensors[pos[0]], self.tensors[pos[1]],
                (ind_con_pos[0], ind_con_pos[1]))
        else:
            self.tensors[min(pos)] = np.tensordot(
                self.tensors[pos[0]], self.tensors[pos[1]],
                (ind_con_pos[0], ind_con_pos[1]))
        ind_new = ind_now[0] + ind_now[1]
        if self._is_debug:
            print('S2 in contract_now: ')
            print(t0 - time.time())
            t0 = time.time()

        self.tensors.__delitem__(max(pos))
        self.indexes.__delitem__(max(pos))
        # self.tensors.__delitem__(min(pos))
        # self.indexes.__delitem__(min(pos))
        # self.tensors.append(t_new)
        # self.indexes.append(ind_new)
        self.indexes[min(pos)] = ind_new
        if self._is_debug:
            print('S3 in contract_now: ')
            print(t0 - time.time())


def cont(tensors, indexes):
    """
    Contract tensors sharing the same positive indexes, leave negative indexes open
    :param tensors: tensors
    :param indexes: indexes of tensors,
    :return:  contracted tensor
    Example:
        >>>a = np.array([[[1, 2, 3],[3, 4, 5]],[[3, 4, 5],[6, 7, 8]]])
        >>>b = np.array([[[2, 3, 4,], [4, 5, 6]], [[5, 6, 7], [8, 1, 2]]])
        >>>index1 = [[1, 2, -1], [2, 1, -2]]
        >>>c = cont([a, b], index1)
          [[ 77  96 115]
           [ 42  57  72]
           [ 55  74  93]]
    """
    _tmp = CONT(tensors, indexes)
    return _tmp.result


def gcont(tensors, indexes):
    _tmp = CONT(tensors, indexes, if_cuda=True)
    return _tmp.result


def symmetrical_rand_peps_tensor(d, chi, n_virtual):
    ind = (d, ) + (chi, ) * n_virtual
    tensor = eval('np.random.randn' + str(ind))
    if n_virtual == 2:
        tensor = (tensor + tensor.transpose(0, 2, 1))/2
    elif n_virtual == 3:
        tensor = (tensor + tensor.transpose(0, 2, 3, 1) + tensor.transpose(0, 3, 1, 2))/3
    return tensor


def random_open_mps(l, phys_dim, chi, is_eco=False):
    # Create a random MPS with open boundary condition
    # l: length; d: physical dimension; chi: virtual dimension
    mps = list(range(0, l))
    dims = empty_list(l + 1, 1)
    if is_eco:
        for n in range(0, l):
            if type(phys_dim) is int:
                chi0 = min([phys_dim ** n, chi, phys_dim ** (l - n)])
                chi1 = min([phys_dim ** (n + 1), chi, phys_dim ** (l - n - 1)])
                d = phys_dim
            else:
                chi0 = min([np.prod(phys_dim[:n]), chi, np.prod(phys_dim[n:])])
                chi1 = min([np.prod(phys_dim[:n+1]), chi, np.prod(phys_dim[:n+1])])
                d = phys_dim[n]
            dims[n + 1] = chi1
            mps[n] = np.random.randn(chi0, d, chi1)
    else:
        if type(phys_dim) is int:
            mps[0] = np.random.randn(1, phys_dim, chi)
            mps[l-1] = np.random.randn(chi, phys_dim, 1)
            for n in range(1, l-1):
                mps[n] = np.random.randn(chi, phys_dim, chi)
                dims[n] = chi
            dims[l-1] = chi
        else:
            mps[0] = np.random.randn(1, phys_dim[0], chi)
            mps[l - 1] = np.random.randn(chi, phys_dim[l-1], 1)
            for n in range(1, l - 1):
                mps[n] = np.random.randn(chi, phys_dim[n], chi)
                dims[n] = chi
            dims[l - 1] = chi
    return mps, dims


def ones_open_mps(l, d, chi, is_eco=False):
    """
    Generate an open MPS with all elements are 1
    :param l:  length of MPS
    :param d: dimension on the physical bonds
    :param chi: dimension of inner bonds
    :return:  MPS
    Example:
        >>>M = ones_open_mps(3, 2, 3)
        >>>print(M[0])
          [[[1.  1.  1.]
            [1.  1.  1.]]]
        >>>print(M[1])
          [[[1.  1.  1.]
            [1.  1.  1.]]

            [[1.  1.  1.]
             [1.  1.  1.]]

            [[1.  1.  1.]
             [1.  1.  1.]]]
        >>>print(M[2])
          [[[1.]
            [1.]]

           [[1.]
            [1.]]

           [[1.]
            [1.]]]
    """
    mps = list(range(0, l))
    dims = empty_list(l + 1, 1)
    if is_eco:
        for n in range(0, l):
            chi0 = min([d ** n, chi, d ** (l - n)])
            chi1 = min([d ** (n + 1), chi, d ** (l - n - 1)])
            dims[n + 1] = chi1
            mps[n] = np.ones((chi0, d, chi1))
    else:
        mps[0] = np.ones((1, d, chi))
        mps[l - 1] = np.ones((chi, d, 1))
        for n in range(1, l - 1):
            mps[n] = np.ones((chi, d, chi))
            dims[n] = chi
        dims[l-1] = chi
    return mps, dims


def open_mps_product_state_spin_up(l, d):
    v = np.zeros((d, ))
    v[0] = 1
    v = v.reshape((1, d, 1))
    mps = [copy.deepcopy(v) for _ in range(l)]
    dims = [1] * (l+1)
    return mps, dims


def mps_ghz_state(num):
    d = 2
    mps = empty_list(num)
    mps[0] = np.eye(d) / np.sqrt(2)
    mps[0] = mps[0].reshape(1, d, d)
    for n in range(1, num-1):
        mps[n] = delta_tensor(d, 3)
    mps[num-1] = np.eye(d).reshape(d, d, 1)
    return mps


def mpd_of_ghz(num):
    d = 2
    mpd = empty_list(num)
    mpd[0] = np.eye(d)
    mat = np.zeros((4, 4))
    mat[:, 0] = np.array([1, 0, 0, 0])
    mat[:, 1] = np.array([0, 0, 0, 1])
    mat[:, 2] = np.array([0, 1, 0, 0])
    mat[:, 3] = np.array([0, 0, 1, 0])
    mat = mat.reshape([d, d, d, d])
    for n in range(1, num-1):
        mpd[n] = copy.deepcopy(mat)
    mpd[num-1] = np.zeros((4, 4))
    mpd[num-1][:, 0] = np.array([1, 0, 0, 1]) / np.sqrt(2)
    mpd[num-1][:, 1] = np.array([1, 0, 0, -1]) / np.sqrt(2)
    mpd[num-1][:, 2] = np.array([0, 1, 0, 0])
    mpd[num-1][:, 3] = np.array([0, 0, 1, 0])
    mpd[num - 1] = mpd[num-1].reshape([d, d, d, d])
    return mpd


def delta_tensor(d, nb):
    exp = 'd, ' * nb
    exp = exp[:-2]
    a = eval('np.zeros((' + exp + '))')
    for n in range(nb-1):
        exp = (str(n) + ',') * nb
        exp = exp[:-1]
        exec('a[' + exp + '] = 1')
    return a


def w_state(num, d=2):
    a = np.zeros([d] * num)
    for n in range(num):
        aa = np.ones((1, ))
        for nn in range(num):
            v = np.zeros((d, ))
            if nn == n:
                v[-1] = 1
            else:
                v[0] = 1
            aa = np.kron(aa, v)
        a += aa.reshape(a.shape)
    return a / np.linalg.norm(a)


def tensor_zn(shape, way):
    dim = len(shape)
    print(dim)
    ind = empty_list(dim, 0)
    shape = np.array(shape).reshape(-1, )
    tensor = np.zeros(shape)
    while ind[-1] != shape[-1]:
        if sum(ind) % 2 == 0:
            if way is 'randn':
                tensor[tuple(ind)] = np.random.randn()
            elif way is 'one':
                tensor[tuple(ind)] = 1
        ind[0] += 1
        for n in range(0, dim - 1):
            if ind[n] == shape[n]:
                ind[n] = 0
                ind[n + 1] += 1
    return tensor


def reorder_vectors_in_mat(mat, order, which):
    mat1 = np.zeros(mat.shape)
    for n in range(0, len(order)):
        if which == 0:
            mat1[n, :] = mat[order[n], :]
        else:
            mat1[:, n] = mat[:, order[n]]
    return mat1


def reorder_index_tensor(tensor, orders, bonds):
    ndim = tensor.ndim
    for n in range(0, len(bonds)):
        permute0 = [bonds[n]] + list(range(0, bonds[n])) + list(range(bonds[n] + 1, ndim))
        permute1 = list(range(1, bonds[n] + 1)) + [0] + list(range(bonds[n] + 1, ndim))
        shape0 = tensor.shape
        shape1 = [shape0[x] for x in permute0]
        tensor = tensor.transpose(permute0).reshape(shape1[0], int(np.prod(shape1[1:])))
        tensor = reorder_vectors_in_mat(tensor, orders[n], 0)
        tensor = tensor.reshape(shape1).transpose(permute1)
    return tensor


def decompose_tensor_one_bond(tensor, n, way='qr'):
    """
    Decompose a tensor on the n-th bond to make it orthogonal on n-th bond
    :param tensor: a tensor
    :param n:  the bond to be decomposed
    :param way:  svd decomposition or qr decomposition
    :return:  decomposed tensor, matrix, and dimension of new bond
    Notes: n counts from 0
    Example:
        >>>T = np.random.randn(4, 4, 4)
        >>>[T1, v, d] = decompose_tensor_one_bond(T, 1)
        >>>T2 = cont([T1, T1], [[1, 2, -1], [1, 2, -2]])
        >>>print(T2)
          [[ 1.00000000e+00  5.55111512e-17  6.93889390e-17  4.16333634e-17]
           [ 5.55111512e-17  1.00000000e+00  9.71445147e-17 -1.38777878e-17]
           [ 6.93889390e-17  9.71445147e-17  1.00000000e+00 -2.77555756e-17]
           [ 4.16333634e-17 -1.38777878e-17 -2.77555756e-17  1.00000000e+00]]

    """
    # decompose a matrix from te n-th bond of the tensor
    # the resulting tensor is an isometry
    # the matrix v has the second bond as the new index by the decomposition
    s1 = np.array(tensor.shape)
    dim = tensor.ndim
    index1 = np.append(np.arange(0, n), np.arange(n+1, dim))
    tensor = tensor.transpose(np.append(index1, n))
    d_new = np.prod(s1[index1])
    tensor = tensor.reshape(d_new, s1[n])
    d_min = min(d_new, s1[n])
    if way == 1 or way == "svd":
        # Use SVD decomposition
        tensor, lm, v = np.linalg.svd(tensor)
        v = np.dot(np.diag(lm), v[:d_min, :])
    else:
        # Use QR decomposition
        tensor, v = np.linalg.qr(tensor, mode='reduced')
    tensor = tensor[:, 0:d_min].reshape(np.append(s1[index1], d_min))
    permute_back = np.append(np.append(np.arange(0, n), dim-1), np.arange(n, dim-1))
    tensor = tensor.transpose(permute_back)
    v = v.T  # !!!!!!!! remember this transpose !!!!!!!!
    return tensor, v, d_min


def left2right_decompose_tensor(tensor, way='qr', is_full=False):
    """
    Decompose a rank 3 tensor on the 3rd bond
    :param tensor: a rank 3 tensor
    :param way:  svd decomposition or qr decomposition
    :param is_full:  is svd, is calculating full matrix
    :return:  decomposed tensor, matrix, dimension of new bond, and  singular value spectrum
    Example:
        >>>T = np.random.randn(4, 4, 4)
        >>>T1 = left2right_decompose_tensor(T)
        >>>T2 = cont([T1, T1], [[-1, 1, 2], [-2, 1, 2]])
        >>>print(T2)
          [[ 1.00000000e+00  5.55111512e-17  6.93889390e-17  4.16333634e-17]
           [ 5.55111512e-17  1.00000000e+00  9.71445147e-17 -1.38777878e-17]
           [ 6.93889390e-17  9.71445147e-17  1.00000000e+00 -2.77555756e-17]
           [ 4.16333634e-17 -1.38777878e-17 -2.77555756e-17  1.00000000e+00]]

    """
    # Transform a local tensor to left2right orthogonal form
    # the resulting tensor is an isometry
    # the matrix v has the second bond as the new index by the decomposition
    s1 = tensor.shape
    dim = min(s1[0]*s1[1], s1[2])
    tensor = tensor.reshape(s1[0] * s1[1], s1[2])
    if way == 1 or way == "svd":
        # Use SVD decomposition
        tensor, lm, v = np.linalg.svd(tensor, full_matrices=is_full)
        v = np.diag(lm[:dim]).dot(v[:dim, :])
    else:
        # Use QR decomposition
        tensor, v = np.linalg.qr(tensor)
        lm = np.zeros(0)
    tensor = tensor[:, :dim].reshape(s1[0], s1[1], dim)
    v = v.T
    return tensor, v, dim, lm


def right2left_decompose_tensor(tensor, way='qr', is_full=False):
    """
    Decompose a rank 3 tensor on the 1rd bond
    :param tensor: a rank 3 tensor
    :param way:  svd decomposition or qr decomposition
    :param is_full:  is svd, is calculating full matrix
    :return:  decomposed tensor, matrix, dimension of new bond, and  singular value spectrum
    Example:
        >>>T = np.random.randn(4, 4, 4)
        >>>T1 = right2left_decompose_tensor(T)
        >>>T2 = cont([T1, T1], [[1, 2, -1], [1, 2, -2]])
        >>>print(T2)
          [[ 1.00000000e+00  5.55111512e-17  6.93889390e-17  4.16333634e-17]
           [ 5.55111512e-17  1.00000000e+00  9.71445147e-17 -1.38777878e-17]
           [ 6.93889390e-17  9.71445147e-17  1.00000000e+00 -2.77555756e-17]
           [ 4.16333634e-17 -1.38777878e-17 -2.77555756e-17  1.00000000e+00]]

    """
    # Transform a local tensor to left2right orthogonal form
    # for manipulate MPS
    s1 = np.shape(tensor)
    tensor = tensor.reshape(s1[0], s1[1]*s1[2])
    dim = min(s1[0], s1[1]*s1[2])
    if way == 1 or way == 'svd':
        # Use SVD decomposition
        tensor, lm, v = np.linalg.svd(tensor.T, full_matrices=is_full)
        v = np.dot(np.diag(lm[:dim]), v[:dim, :])
    else:
        # Use QR decomposition
        tensor, v = np.linalg.qr(tensor.T)
        lm = np.zeros(0)
    tensor = tensor[:, :dim].T.reshape(dim, s1[1], s1[2])
    v = v.T
    return tensor, v, dim, lm


def absorb_matrix2tensor(tensor, mat, bond):
    """
    Absorb a matrix to a tensor at desinated bond
    :param tensor: a tensor
    :param mat:  a matrix
    :param bond:  bond the matrix contracted with
    :return:  tensor after absorb the matrix
    Example:
        >>>T = np.array([[[1, 2], [2, 3]],[[3, 4], [4, 5]]])
        >>>M = np.array([[1, 3], [2, 4]])
        >>>print(absorb_matrix2tensor(T, M, 2))
          [[[ 5 11]
            [ 8 18]]

           [[11 25]
            [14 32]]]
    """
    # generally, recommend to use the function 'absorb_matrices2tensor'
    # contract the 1st bond of mat with tensor
    s = np.array(tensor.shape)
    nd = tensor.ndim
    if bond == 0:
        tensor1 = mat.T.dot(tensor.reshape(s[0], np.prod(s[1:nd])))
        s[0] = mat.shape[1]
    elif bond == nd - 1:
        tensor1 = tensor.reshape(np.prod(s[0:nd - 1]), s[nd - 1]).dot(mat)
        s[-1] = mat.shape[1]
    else:
        ind = list(range(0, bond)) + list(range(bond + 1, nd))
        tensor1 = tensor.transpose(ind + [bond]).reshape(np.prod([s[i] for i in ind]),
                                                         s[bond]).dot(mat)
        s[bond] = mat.shape[1]
        tensor1 = tensor1.reshape([s[i] for i in (ind + [bond])])
        ind = list(range(0, bond)) + [nd - 1] + list(range(bond, nd - 1))
        tensor1 = tensor1.transpose(ind)
    if bond == 0 or bond == nd - 1:
        tensor1 = tensor1.reshape(s)
    return tensor1


def absorb_matrices2tensor_full_fast(tensor, mats):
    """
    Absorb tensor with matrices on all bonds
    :param tensor: a tensor
    :param mats: matrices to contracted on all bonds
    :return: tensor after absorb matrices
    Example:
        >>>T = np.array([[[1, 1], [1, 1]],[[1, 1], [1, 1]]])
        >>>M = [np.array([[1, 2], [2, 3]]), np.array([[2, 3], [3, 4]]), np.array([[3, 4], [4, 5]])]
        >>>print(absorb_matrices2tensor_full_fast(T, M))
          [[[105 135]
           [147 189]]

           [[175 225]
            [245 315]]]
    """
    # generally, recommend to use the function 'absorb_matrices2tensor'
    # each bond will have a matrix to contract with
    # the matrices must be in the right order
    # contract the 1st bond of mat with tensor
    nb = tensor.ndim
    s = np.array(tensor.shape)
    is_bug = False
    if is_debug:
        for n in range(0, nb):
            if mats[n].shape[1] != s[n]:
                cprint('Error: the %d-th matrix has inconsistent dimension with the tensor' % n, 'magenta')
                cprint('T.shape = ' + str(s) + ', mat.shape = ' + str(mats[n].shape), 'magenta')
                is_bug = True
    for n in range(nb-1, -1, -1):
        tensor = tensor.reshape(np.prod(s[:nb-1]), s[nb-1]).dot(mats[n])
        s[-1] = mats[n].shape[1]
        ind = [nb-1] + list(range(0, nb-1))
        tensor = tensor.reshape(s).transpose(ind)
        s = s[ind]
    if is_debug and is_bug:
        trace_stack()
    # tensor = CONT([tensor] + mats, [[1, 2, 3], [1, -1], [2, -2], [3, -3]])
    return tensor


def absorb_matrices2tensor(tensor, mats, bonds=np.zeros(0), mat_bond=-1):
    """
    Absorb matrices to tensors on certain bonds
    :param tensor:  a tensor
    :param mats:  matrices
    :param bonds:  which bonds of tensor to absorb matrices, default by from 0 to n
    :param mat_bond:  which bond of matrices to contract to tensor, default as 0 bond
    :return: tensor absorbed matrices
    Example:
        >>>T = np.array([[[1, 1], [1, 1]],[[1, 1], [1, 1]]])
        >>>M = [np.array([[1, 2], [2, 3]]), np.array([[2, 3], [3, 4]])]
        >>>print(absorb_matrices2tensor(T, M))
          [[[15 15]
            [21 21]]

           [[25 25]
            [35 35]]]

    """
    # default: contract the 1st bond of mat with tensor
    nm = len(mats)  # number of matrices to be contracted
    if is_debug:
        if nm != tensor.ndim:
            print_error('InputError: the number of matrices should be equal to the number of indexes of tensor')
    if type(bonds) is list or tuple:
        bonds = np.array(bonds)
    if bonds.size == 0:  # set default of bonds: contract all matrices in order, starting from the 0th bond
        bonds = np.arange(0, nm)
    if mat_bond < 0:  # set default of mat_bond: contract the 1st bond of each matrix
        mat_bond = np.zeros((nm, 1))
    for i in range(0, nm):  # permute if the second bond of a matrix is to be contracted
        if mat_bond[i, 0] == 1:
            mats[i] = mats[i].T
    # check if full_fast function can be used
    if np.array_equiv(np.sort(bonds), np.arange(0, tensor.ndim)):
        order = np.argsort(bonds)
        mats = sort_list(mats, order)
        # this full_fast function can be used when each bond has a matrix which are arranged in the correct order
        tensor = absorb_matrices2tensor_full_fast(tensor, mats)
    else:
        # can be optimized
        for i in range(0, nm):
            tensor = absorb_matrix2tensor(tensor, mats[i], bonds[i])
    return tensor


def absorb_vectors2tensors(tensor, vecs, bonds):
    order = np.argsort(bonds)
    for n in order[::-1]:
        tensor = np.tensordot(tensor, vecs[n], ([bonds[n]], [0]))
    return np.squeeze(tensor)


def scalar2vector(x, dim, theta_max=np.pi/2):
    v = np.zeros((dim, ))
    theta = x * theta_max
    for d in range(1, dim+1):
        v[d-1] = np.sqrt(combination(dim-1, d-1)) * (np.cos(theta)**(dim-d)) \
               * (np.sin(theta)**(d-1))
    return v


def bound_vec_operator_left2right(tensor, op=np.zeros(0), v=np.zeros(0),
                                  normalize=False, symme=False):
    """
    Contract left boundary vector with transfer matrix of MPS
    :param tensor:  a tensor of MPS
    :param op:  operator on physical bonds
    :param v:  left boundary
    :param normalize:  if normalized the outcome vector
    :param symme:  if symmertrized the outcome vector
    :return: the outcome vector
    Notes: 1.if v leaves empty, this function will use identity as default
    Examples:
        >>>T = np.array([[[1, 2, 1], [2, 1, 2]], [[2, 0, 2], [1, 3, 1]], [[3, 1, 0], [2, 2, 1]]])
        >>>print(bound_vec_operator_left2right(T))
          [[23 14 12]
           [14 19  9]
           [12  9 11]]
        >>>print(bound_vec_operator_left2right(T, v = np.array([[1, 1, 1], [1, 2, 1], [2, 2, 1]])))
          [[81 65 58]
           [60 64 45]
           [46 40 33]]
    """
    s = tensor.shape
    if op.size != 0:  # deal with the operator
        tensor1 = absorb_matrix2tensor(tensor, op.T, 1)
    else:  # no operator
        tensor1 = tensor.copy()
    if v.size == 0:  # no input boundary vector v
        tensor = tensor.reshape(s[0]*s[1], s[2]).conj()
        tensor1 = tensor1.reshape(s[0]*s[1], s[2])
        v1 = tensor.T.dot(tensor1)
    else:  # there is an input boundary vector v
        if is_debug:
            if v.shape[1] != s[0]:
                cprint('BondDimError: the v_left has inconsistent dimension with the tensor', 'magenta')
                cprint('v.shape = ' + str(v.shape) + '; T.shape = ' + str(s))
                trace_stack()
        tensor1 = v.dot(tensor1.reshape(s[0], s[1]*s[2]))
        v1 = tensor.conj().reshape(s[0]*s[1], s[2]).T.dot(tensor1.reshape(s[0]*s[1], s[2]))
    if normalize:
        v1 = normalize_tensor(v1)[0]
    if symme:
        v1 = (v1 + v1.conj().T)/2
    return v1


def bound_vec_operator_right2left(tensor, op=np.zeros(0), v=np.zeros(0), normalize=False,
                                  symme=False):
    """
    Contract right boundary vector with transfer matrix of MPS
    :param tensor:  a tensor of MPS
    :param op:  operator on physical bonds
    :param v:  left boundary
    :param normalize:  if normalized the outcome vector
    :param symme:  if symmertrized the outcome vector
    :return: the outcome vector
    Notes: 1.if v leaves empty, this function will use identity as default
    Examples:
        >>>T = np.array([[[1, 2, 1], [2, 1, 2]], [[2, 0, 2], [1, 3, 1]], [[3, 1, 0], [2, 2, 1]]])
        >>>print(bound_vec_operator_right2left(T))
          [[15 11 13]
           [11 19 15]
           [13 15 19]]
        >>>print(bound_vec_operator_right2left(T, v = np.array([[1, 1, 1], [1, 2, 1], [2, 2, 1]])))
          [[55 54 57]
           [53 58 59]
           [48 51 50]]
    """
    s = tensor.shape
    if op.size != 0:  # deal with the operator
        tensor1 = absorb_matrix2tensor(tensor, op.T, 1)
    else:  # no operator
        tensor1 = tensor.copy()
    if v.size == 0:  # no input boundary vector v
        tensor = tensor.reshape(s[0], s[1]*s[2]).conj()
        tensor1 = tensor1.reshape(s[0], s[1]*s[2])
        v1 = tensor.dot(tensor1.T)
    else:  # there is an input boundary vector v
        if is_debug:
            if v.shape[0] != s[2]:
                cprint('BondDimError: the v_right has inconsistent dimension with the tensor', 'magenta')
                cprint('v.shape = ' + str(v.shape) + '; T.shape = ' + str(s))
                trace_stack()
        tensor = tensor.reshape(s[0]*s[1], s[2]).conj().dot(v)
        v1 = tensor.reshape(s[0], s[1]*s[2]).dot(tensor1.reshape(s[0], s[1]*s[2]).T)
    if normalize:
        v1 = normalize_tensor(v1)[0]
    if symme:
        v1 = (v1 + v1.conj().T)/2
    return v1


def bound_vec_with_phys_left2right(tensor, v=np.zeros(0), normalize=False):
    s = tensor.shape
    if v.size == 0:
        tmp = tensor.reshape(s[0], s[1] * s[2])
        v = tmp.T.conj().dot(tmp).reshape(s[1], s[2], s[1], s[2]).transpose(0, 2, 1, 3)
    elif v.ndim == 2:
        v = np.tensordot(v, tensor, ([1], [0]))
        v = np.tensordot(tensor.conj(), v, ([0], [0])).transpose(0, 2, 1, 3)
    elif v.ndim == 4:
        v = cont([tensor, tensor.conj(), v], [[2, 3, -4], [1, 3, -3], [-1, -2, 1, 2]])
    if normalize:
        v /= np.linalg.norm(v.reshape(-1, ))
    return v


def bound_vec_with_phys_right2left(tensor, v=np.zeros(0), normalize=False):
    s = tensor.shape
    if v.size == 0:
        tmp = tensor.reshape(s[0] * s[1], s[2])
        v = tmp.conj().dot(tmp.T).reshape(s[0], s[1], s[0], s[1]).transpose(1, 3, 0, 2)
    elif v.ndim == 2:
        v = np.tensordot(tensor, v, ([2], [1]))
        v = np.tensordot(tensor.conj(), v, ([2], [2])).transpose(1, 3, 0, 2)
    elif v.ndim == 4:
        v = cont([tensor, tensor.conj(), v], [[-4, 3, 2], [-3, 3, 1], [-1, -2, 1, 2]])
    if normalize:
        v /= np.linalg.norm(v.reshape(-1, ))
    return v


def transfer_matrix_mps(tensor):
    """
    Obtain a transfer matrix of MPS
    :param tensor:  tensor of MPS
    :return:  transfer matrix
    Example:
        >>>T = np.array([[[1, 2, 1], [2, 1, 2]], [[2, 0, 2], [1, 3, 1]], [[3, 1, 0], [2, 2, 1]]])
        >>>print(transfer_matrix_mps(T))
         [[ 5  4  5  4  5  4  5  4  5]
          [ 4  6  4  5  3  5  4  6  4]
          [ 7  5  2  8  4  1  7  5  2]
          [ 4  5  4  6  3  6  4  5  4]
          [ 5  3  5  3  9  3  5  3  5]
          [ 8  4  1  6  6  3  8  4  1]
          [ 7  8  7  5  4  5  2  1  2]
          [ 8  6  8  4  6  4  1  3  1]
          [13  7  2  7  5  2  2  2  1]]
    """
    s = tensor.shape
    tmp = tensor.transpose(0, 2, 1).reshape(s[0]*s[2], s[1])
    tm = tmp.conj().dot(tmp.T).reshape(s[0], s[2], s[0], s[2])
    tm = tm.transpose(0, 2, 1, 3).reshape(s[0]*s[0], s[2]*s[2])
    return tm


def transformation_from_env_mats(ml, mr, lmm=None, dc=None, norm_way=1):
    # lmm: lm in the middle bond
    # dc: dimension cut-off (None means no truncation)
    lml, ul = np.linalg.eigh(ml)
    lmr, ur = np.linalg.eigh(mr)
    lml = lml ** 0.5
    lmr = lmr ** 0.5
    # lml /= np.linalg.norm(lml)
    # lmr /= np.linalg.norm(lmr)
    if lmm is None:
        m_mid = np.diag(lml).dot(ul.conj().T).dot(ur.conj()).dot(np.diag(lmr))
    else:
        m_mid = np.diag(lml).dot(ul.conj().T).dot(np.diag(lmm)).dot(ur.conj()).dot(np.diag(lmr))
    u, lm, v = np.linalg.svd(m_mid)
    if dc is not None:
        dc = min(dc, lm.shape[0])
    else:
        dc = lm.shape[0]
    ul = ul.dot(np.linalg.pinv(np.diag(lml))).dot(u[:, :dc])
    ur = ur.dot(np.linalg.pinv(np.diag(lmr))).dot(v[:dc, :].T)
    lm = lm[:dc]
    if norm_way == 1:
        norm = np.linalg.norm(lm)
        lm /= norm
        norm = norm ** 0.5
        ul *= norm
        ur *= norm
    elif norm_way == 2:
        lm = normalize_tensor(lm)[0]
        ul = normalize_tensor(ul)[0]
        ur = normalize_tensor(ur)[0]
    return ul, ur, lm, dc


def bond_permutation_transformation(order):
    # Contracting the first bond of u, i.e., T.dot(u)
    dim = len(order)
    u = np.zeros((dim, dim))
    for n in range(0, dim):
        u[n, order[n]] = 1
    return u


def operate_tensor_slice(tensor, nb, slice, data):
    """
    (for the nb-th bond) tensor[:, ..., slice[0]:slice[1], ..., :] = data[:, :, ..., :, :]
    :param tensor:
    :param nb:
    :param slice:
    :param data:
    :return:
    Example:
    >>> x = np.random.randn(4, 6, 4)
    >>> y = np.random.randn(4, 2, 4)
    >>> x1 = x.copy()
    >>> x1[:, 0:2, :] = y
    >>> x2 = x.copy()
    >>> x2 = operate_tensor_slice(x2, 1, [0, 2], y)
    >>> err = np.linalg.norm((x1 - x2).reshape(-1, ))
    >>> print(err)
    """
    exp = ''
    for n in range(0, tensor.ndim):
        exp += ':'
        if n != tensor.ndim-1:
            exp += ','
    if type(slice) is str:
        exp1 = exp[:nb * 2] + slice + exp[nb * 2 + 1:]
    else:
        exp1 = exp[:nb * 2] + str(slice[0]) + ':' + str(slice[1]) + exp[nb * 2 + 1:]
    exec('tensor[' + exp1 + '] = data[' + exp + ']')
    return tensor


def off_diagonal_mat(mat):
    return mat - np.diag(np.diag(mat))


def normalize_tensor(tensor, if_flatten=False, is_enforce=False):
    """
    Normalize a tensor
    :param tensor:  a tensor
    :param if_flatten:  if flat the tensor into a vector
    :return:  a tensor or a vector, and the norm
    Example:
        >>>T = np.array([[[1, 1], [1, 1]],[[1, 1], [1, 1]]])
        >>>print(normalize_tensor(T))
          (array([[[0.35355339, 0.35355339],
                  [0.35355339, 0.35355339]],

                 [[0.35355339, 0.35355339],
                  [0.35355339, 0.35355339]]]), 2.8284271247461903)
    """
    v = tensor.reshape(-1, )
    norm = np.linalg.norm(v)
    if norm < 1e-30 and not is_enforce:
        cprint('InfWarning: norm is too small to normalize', 'magenta')
        trace_stack()
        if if_flatten:
            return v, norm
        else:
            return tensor, norm
    else:
        if if_flatten:
            return v/norm, norm
        else:
            return tensor/norm, norm


def entanglement_entropy(lm, tol=1e-20):
    """
    Calculate the engtanglement entropy from spectrum lambda
    :param lm:  a spectrum
    :param tol:   minimal set of values.
    :return:  entanglement entropy
        >>>lm = np.array([2, 1, 0.5, 0.3, 0])
        >>>print(entanglement_entropy(lm))
          -4.981888749420921
    """
    lm = np.sort(lm.reshape(-1, ))
    lm = lm[::-1]
    ind = arg_find_array(lm > tol, 1, 'last')
    lm = lm[:ind + 1]
    lm /= np.linalg.norm(lm)
    ent = -2*(lm**2).T.dot(np.log(lm))
    return ent


def q_sparsity(a):
    qs = list()
    # print('------------------' + str(a.ndim))
    while a.ndim > 0:
        see = list()
        v = list()
        for n in range(a.ndim):
            ind = list(range(a.ndim))
            ind.pop(n)
            rho = np.tensordot(a, a.conj(), [ind, ind])
            lm, u = np.linalg.eigh(rho)
            lm = lm[lm > 1e-20]
            if lm.size == 0 or np.max(np.abs(lm)) < 1e-20:
                see.append(0)
            else:
                lm /= np.sum(lm)
                see.append(-np.inner(lm, np.log(lm)) / np.log(a.shape[0]))
            v.append(u[:, np.argmax(np.abs(lm))])
        qs.append(np.average(see)-1)
        pos = np.argmax(see).item()
        a = np.tensordot(a, v[pos], [[pos], [0]])
        if a.ndim > 0:
            a /= np.linalg.norm(a)

        # d = a.ndim + 1
        # print('d = ' + str(d))
        # print(qs[-1])
        # if d > 1:
        #     s = -(d-1)/d*np.log((d-1)/d) + 1/d*np.log(d)
        # else:
        #     s = 0
        # print(s / np.log(2) - 1)
        # print('---------------')
    return sum(qs)


def khatri(mat1, mat2):
    d1, d = mat1.shape
    d2 = mat2.shape[0]
    mat1 = mat1.repeat(d2, 0).reshape(d1, d2, d)
    mat2 = mat2.repeat(d1, 1).reshape(d2, d, d1).transpose(2, 0, 1)
    return mat1 * mat2


def kron(x, *y):
    t = np.kron(x, y[0])
    for n in range(1, y.__len__()):
        t = np.kron(t, y[n])
    return t


def eye_tensor(k, dim):
    # k: tensor order
    # dim: bond dimension
    t = np.zeros(dim * np.ones((k, ), dtype=int))
    for n in range(dim):
        ind = str(n) + (',' + str(n)) * (k-1)
        exec('t[' + ind + '] = 1')
    return t


def is_identity_by_norm(mat, tol=1e-20):
    """
    Check if matrix is identity matrix by compare the norm
    :param mat:  matrix
    :param tol:  error tolerant
    :return:   true or false
    """
    c = mat[0, 0]
    s = mat.shape
    is_id = True
    if abs(c) < tol or mat.ndim != 2:
        is_id = False
    elif s[0] != s[1]:
        is_id = False
    else:
        mat = mat/mat[0, 0] - np.eye(mat.shape[0])
        norm = np.linalg.norm(mat.reshape(-1, 1))/s[0]/s[1]
        if norm > tol:
            is_id = False
    return is_id


def is_identity(mat, tol=1e-15, sample_t=10):
    """
    Check if matrix is identity matrix by sampling
    :param mat:  matrix
    :param tol:  error tolerant
    :param sample_t:  sample time
    :return:  true or false
    """
    # if mat is an identity, return c with mat = c*I
    # if not, return False
    c = mat[0, 0]
    s = mat.shape
    if abs(c) < tol or mat.ndim != 2:
        is_id = False
    elif s[0] != s[1]:
        is_id = False
    else:
        is_id = True
        sample_t1 = min(sample_t, s[0])
        ind = np.random.permutation(s[0]-1)[:sample_t1]+1
        for i in ind:
            if abs(mat[i, i] - mat[i-1, i-1]) > tol:
                is_id = False
                break
        if is_id:
            ind1 = np.random.permutation(s[0]-1)[:sample_t1]+1
            ind2 = np.random.permutation(s[0]-1)[:sample_t1]+1
            for i1 in ind1:
                for i2 in ind2:
                    if i1 != i2 and abs(mat[i1, i2]) > tol:
                        is_id = False
                        break
                if not is_id:
                    break
        if is_id:
            is_id = is_identity_by_norm(mat, tol)
    return is_id


def is_zero(mat, tol=1e-20):
    """
    Check if matrix are zero matrix
    :param mat:  matrix
    :param tol:  error tolerant
    :return:  True or False
    """
    norm = np.prod(abs(mat) < tol)
    return norm


def check_orthogonality(tensor, ind0, tol=1e-20):
    """
    Check a tensor if it's orthogonal on one index or some indexes
    :param tensor:  a tensor
    :param ind0:  indexes
    :param tol:  error tolerant of orthogonality
    :return:  True or False
    Example:
        >>>T = np.array([[[1, 1], [1, 1]],[[1, 1], [1, 1]]])
        >>>print(check_orthogonality(T, [2]))
          False
    """
    s = list(tensor.shape)
    ind1 = list(range(0, len(s)))
    dim0 = 1
    dim1 = 1
    for n in range(0, len(ind0)):
        ind1.remove(ind0[n])
        dim0 *= s[ind0[n]]
    for n in range(0, len(ind1)):
        dim1 *= s[ind1[n]]
    tensor = tensor.transpose(ind0 + ind1).reshape(dim0, dim1)
    rm = tensor.conj().dot(tensor.T)
    return is_identity(rm, tol=tol)


def sort_vectors(mat, order, way='column'):
    """
    Sort vectors in a matrix at designated order
    :param mat: matrix
    :param order: designated order
    :param way: sort column vectors or row vectors
    :return: sorted matrix
    Example:
        >>>M = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])
        >>>print(sort_vectors(M, (2, 0, 1)))
          [[3. 1. 2.]
           [6. 4. 5.]
           [9. 7. 8.]]
        >>>print(sort_vectors(M, (2, 0, 1), 'row'))
          [[7. 8. 9.]
           [1. 2. 3.]
           [4. 5. 6.]]
    """
    mat1 = np.zeros(mat.shape)
    if way == 'row':
        nv = mat.shape[0]
        for n in range(0, nv):
            mat1[n, :] = mat[order[n], :]
        return mat1
    elif way == 'column':
        nv = mat.shape[1]
        for n in range(0, nv):
            mat1[:, n] = mat[:, order[n]]
        return mat1
    else:
        return mat


def tensor3_to_unitary4(tensor, theta):
    # tensor should be 3rd-order
    # for UMPO idea
    s = tensor.shape
    mat = np.array([[np.cos(theta * np.pi/2), np.sin(theta * np.pi/2)],
                    [np.sin(theta * np.pi/2), -np.cos(theta * np.pi/2)]])
    tensor = tensor.reshape(s[0]*s[1], s[1])
    lm, v = np.linalg.eigh(tensor.dot(tensor.transpose(1, 0).conj()))
    tensor1 = np.zeros((s[0], s[1], s[1], s[2]))
    tensor1[:, :, 0, :] = tensor.reshape(s[0], s[1], -1)
    tensor1[:, :, 1, :] = v[:, lm < 1e-14].dot(mat).reshape(s[0], s[1], s[1])
    return tensor1


def get_orthogonal_vecs(v, k=None):
    v = v.reshape(-1, )
    v /= np.linalg.norm(v)
    if k is None:
        k = v.size-1
    vecs = np.zeros((v.size, k+1))
    vecs[:, 0] = v
    for n in range(1, k+1):
        err = 1
        vv = 0
        while err > 1e-15:
            vv = np.random.randn(v.size, )
            vv /= np.linalg.norm(vv)
            for nn in range(n):
                vv -= np.inner(vv, vecs[:, nn]) * vecs[:, nn]
                norm = np.linalg.norm(vv)
                vv /= max(norm, 1e-20)
            err = 0
            for nn in range(n):
                err += abs(np.inner(vv, vecs[:, nn]))
        vecs[:, n] = vv
    return vecs


# ========================================================
# Some special functions
def embed_list_into_matrix(v_list):
    # v is a list of length nv
    # Each element of v, say v[n], is a list that consists of integers or floats
    nv = v_list.__len__()
    dim = np.zeros((nv, )).astype(int)
    for n in range(0, nv):
        dim[n] = v_list[n].__len__()
    mat = np.zeros((nv, max(dim)))
    for n in range(0, nv):
        mat[n, :dim[n]] = np.array(v_list[n]).reshape(1, -1)
    return mat, dim
