import numpy as np
from BasicFunctionsSJR import sort_list, trace_stack, print_error, arg_find_array
from termcolor import cprint
is_debug = False
if is_debug:
    cprint('Note: you are in the debug mode of module \'Basic_Functions_SJR\'', 'cyan')


class CONT:

    def __init__(self, tensors, indexes, _is_debug=False):
        self._is_debug = _is_debug
        self.tensors = tensors
        self.indexes = indexes
        self.n_tensor = self.tensors.__len__()
        self.bond_open = list()
        self.bond_ark = list()
        self.find_open_ark_bonds()

        if self._is_debug:
            self.check_consistency()

        while self.bond_ark.__len__() > 0:
            tensor_now, index_now, pos = self.tensors_and_bonds_in_nth_contraction(self.bond_ark[0])
            t_new, ind_new = self.contract_now(tensor_now, index_now)
            self.update_tensors_and_indexes(pos, t_new, ind_new)
        ind = sorted(range(len(indexes[0])), key=lambda k: indexes[0][k])
        self.result = self.tensors[0].transpose(ind)

    def check_consistency(self):
        if self.n_tensor != len(self.indexes):
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

    def find_open_ark_bonds(self):
        self.bond_open = set()
        self.bond_ark = set()
        for n in range(0, self.n_tensor):
            for i in range(0, self.indexes[n].__len__()):
                if self.indexes[n][i] < 0:
                    self.bond_open.add(self.indexes[n][i])
                elif self.indexes[n][i] > 0:
                    self.bond_ark.add(self.indexes[n][i])
        self.bond_open = list(self.bond_open)
        self.bond_open.sort(reverse=True)
        self.bond_ark = list(self.bond_ark)
        self.bond_ark.sort()

    def tensors_and_bonds_in_nth_contraction(self, bond):
        # n is the number of the contracted bond
        n_found = 0
        tensors_now = list()
        index_now = list()
        pos = list()
        for n in range(0, self.n_tensor):
            if bond in self.indexes[n]:
                n_found += 1
                tensors_now.append(self.tensors[n])
                index_now.append(self.indexes[n])
                pos.append(n)
            if n_found == 2:
                return tensors_now, index_now, pos

    def contract_now(self, t_now, ind_now):
        # print(t_now[0].shape)
        # print(ind_now[0])
        # print(t_now[1].shape)
        # print(ind_now[1])
        ind_con = list(set(ind_now[0]) & set(ind_now[1]))
        ind_con_pos = [list(range(0, ind_con.__len__())), list(range(0, ind_con.__len__()))]
        for i in range(0, ind_con.__len__()):
            ind_con_pos[0][i] = ind_now[0].index(ind_con[i])
            ind_con_pos[1][i] = ind_now[1].index(ind_con[i])
        ind_left_pos = [list(range(0, ind_now[0].__len__())), list(range(0, ind_now[1].__len__()))]
        for i in range(0, ind_con.__len__()):
            ind_left_pos[0].remove(ind_con_pos[0][i])
            ind_left_pos[1].remove(ind_con_pos[1][i])
        for i in range(0, ind_con.__len__()):
            ind_now[0].remove(ind_con[i])
            ind_now[1].remove(ind_con[i])
            self.bond_ark.remove(ind_con[i])

        pos0 = ind_left_pos[0] + ind_con_pos[0]
        pos1 = ind_con_pos[1] + ind_left_pos[1]
        s0 = t_now[0].shape
        s1 = t_now[1].shape
        t_new = t_now[0].transpose(pos0).reshape(np.prod([s0[i] for i in ind_left_pos[0]]),
                                                 np.prod([s0[i] for i in ind_con_pos[0]]))
        t_new = t_new.dot(t_now[1].transpose(pos1).reshape(np.prod([s1[i] for i in ind_con_pos[1]]),
                                                           np.prod([s1[i] for i in ind_left_pos[1]])))
        s_new0 = [s0[i] for i in ind_left_pos[0]]
        s_new1 = [s1[i] for i in ind_left_pos[1]]
        t_new = t_new.reshape(s_new0 + s_new1)
        ind_new = ind_now[0] + ind_now[1]
        return t_new, ind_new

    def update_tensors_and_indexes(self, pos, t_new, ind_new):
        self.tensors.__delitem__(max(pos))
        self.tensors.__delitem__(min(pos))
        self.indexes.__delitem__(max(pos))
        self.indexes.__delitem__(min(pos))
        self.tensors.append(t_new)
        self.indexes.append(ind_new)
        self.n_tensor -= 1


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


def random_open_mps(l, d, chi):
    """
    Generate an open MPS randomly
    :param l:  length of MPS
    :param d: dimension on the physical bonds
    :param chi: dimension of inner bonds
    :return:  MPS
    Example:
        >>>M = random_open_mps(3, 2, 3)
        >>>print(M[0])
          [[[-2.5679364  -1.12846181  0.12026503]
            [ 1.23217514  0.58611443  0.66304506]]]
        >>>print(M[1])
          [[[ 0.02140644 -0.2359836   1.41704847]
            [ 0.42441594  0.11000762 -0.23754322]]

            [[ 0.42759564  0.32495413 -0.81798019]
             [-0.54115541  0.63275244 -0.31163543]]

            [[-1.11706565  0.36694417 -1.67561183]
             [-0.37247627  0.85373283  0.99919477]]]
        >>>print(M[2])
          [[[-1.01253065]
            [-0.82606855]]

           [[ 0.524324  ]
            [ 0.52489905]]

           [[ 0.68181659]
            [-0.62746486]]]
    """
    # Create a random MPS with open boundary condition
    # l: length; d: physical dimension; chi: virtual dimension
    mps = list(range(0, l))
    mps[0] = np.random.randn(1, d, chi)
    mps[l-1] = np.random.randn(chi, d, 1)
    for n in range(1, l-1):
        mps[n] = np.random.randn(chi, d, chi)
    return mps


def ones_open_mps(l, d, chi):
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
    mps[0] = np.ones((1, d, chi))
    mps[l - 1] = np.ones((chi, d, 1))
    for n in range(1, l - 1):
        mps[n] = np.ones((chi, d, chi))
    return mps


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


def left2right_decompose_tensor(tensor, way='qr'):
    """
    Decompose a rank 3 tensor on the 3rd bond
    :param tensor: a rank 3 tensor
    :param way:  svd decomposition or qr decomposition
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
        tensor, lm, v = np.linalg.svd(tensor)
        v = np.dot(np.diag(lm), v[:dim, :])
    else:
        # Use QR decomposition
        tensor, v = np.linalg.qr(tensor)
        lm = np.zeros(0)
    tensor = tensor[:, :dim].reshape(s1[0], s1[1], dim)
    v = v.T
    return tensor, v, dim, lm


def right2left_decompose_tensor(tensor, way='qr'):
    """
    Decompose a rank 3 tensor on the 1rd bond
    :param tensor: a rank 3 tensor
    :param way:  svd decomposition or qr decomposition
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
        tensor, lm, v = np.linalg.svd(tensor.T)
        v = np.dot(np.diag(lm), v[:dim, :])
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
        ind = np.arange(0, bond)
        ind = np.append(ind, np.arange(bond + 1, nd))
        tensor1 = tensor.transpose(np.append(ind, bond)).reshape(np.prod(s[ind]), s[bond]).dot(mat)
        s[bond] = mat.shape[1]
        tensor1 = tensor1.reshape(s[np.append(ind, bond)])
        ind = np.arange(0, bond)
        ind = np.append(np.append(ind, nd - 1), np.arange(bond, nd - 1))
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
        ind = np.append(nb-1, np.arange(0, nb-1))
        tensor = tensor.reshape(s).transpose(ind)
        s = s[ind]
    if is_debug and is_bug:
        trace_stack()
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

    """
    # default: contract the 1st bond of mat with tensor
    nm = len(mats)  # number of matrices to be contracted
    if is_debug:
        if nm != tensor.ndim:
            print_error('InputError: the number of matrices should be equal to the number of indexes of tensor')
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


def bound_vec_operator_left2right(tensor, op=np.zeros(0), v=np.zeros(0), normalize=False, symme=False):
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


def bound_vec_operator_right2left(tensor, op=np.zeros(0), v=np.zeros(0), normalize=False, symme=False):
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


def normalize_tensor(tensor, if_flatten=False):
    v = tensor.reshape(-1, )
    norm = np.linalg.norm(v)
    if norm < 1e-30:
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
    lm = np.sort(lm.reshape(-1, ))
    lm = lm[::-1]
    ind = arg_find_array(lm > tol, 1, 'last')
    lm = lm[:ind + 1]
    ent = -2*(lm**2).T.dot(np.log(lm))
    return ent


def is_identity_by_norm(mat, tol=1e-20):
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


def is_identity(mat, tol=1e-20, sample_t=10):
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
    norm = np.prod(abs(mat) < tol)
    return norm


def check_orthogonality(tensor, ind0=0, ind1=1, tol=1e-20):
    s = np.array(tensor.shape)
    dim1 = np.prod(s[ind0])
    dim2 = np.prod(s[ind1])
    tensor = tensor.transpose(np.hstack((ind0, ind1))).reshape(dim1, dim2)
    rm = tensor.conj().dot(tensor.T)
    return is_identity(rm, tol=tol)


def sort_vectors(mat, order, way='column'):
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