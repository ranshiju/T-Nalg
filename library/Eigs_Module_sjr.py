import numpy as np
from scipy.linalg import eigh_tridiagonal
'''
d: linear map dimension
n: number of eigen vectors
k: number of Lanczos basis; k = d by default
v0: initial vector
tol: tolerance
max_it: maximal iteration time
by Shi-Ju Ran

Notes:
the first eigenvalue and vector are computed perfectly
the higher eigenvalues and vectors are computed badly
'''


def eigs_fh(lin_map, d, n=6, k=-1, v0=np.zeros(0), tol=1e-15, max_it=1, which='lm'):
    if k < 0:
        k = d
    else:
        k = min(max(k, n + 1), d)  # set minimal number of Lanczos basis
    if k == d:
        max_it = 1  # when constructing the full tri-diagonal matrix, iterate only once
    v0 = set_initial_v(v0, d)  # handle initial vector
    vs = set_random_initial_vectors(v0, k)
    alpha = np.zeros((k, ))
    beta = np.zeros((k-1, ))
    w = 1
    error = np.ones((1, n))
    info = dict()
    lm = np.zeros((k, 1))
    v0_new = np.zeros((d, n))
    ind = np.zeros((d, n))

    for t in range(0, max_it):
        for k_now in range(0, k):
            if k_now == 0:
                w, alpha[0] = update_w(lin_map, vs[:, [0]])
            else:
                vs[:, [k_now]], beta[k_now-1] = update_v(w, vs[:, :k_now], vs[:, [k_now]])
                w, alpha[k_now] = update_w(lin_map, vs[:, [k_now]], beta[k_now-1], vs[:, [k_now - 1]])
        lm, u = eigh_tridiagonal(d=alpha, e=beta)  # diagonalize the k-by-k tri-diagonal matrix
        ind = handle_which(which, lm, n)
        v0_new = vs.dot(u[:, ind])
        error = np.linalg.norm(x=vs[:, 0:n]-v0_new, axis=0)
        error_tot = np.sum(error)
        if error_tot > tol:
            if t == max_it-1 and t != 0:
                print('Not convergent! error = %g' % error_tot)
                info['it_time'] = max_it
            else:
                vs[:, [0]] = v0_new[:, [0]]
        else:
            info['it_time'] = t+1
            break
    info['error'] = error
    return lm[ind], v0_new, info


##################################################################
def update_w(lin_map, v, beta=0, v_former=np.zeros(0)):
    w1 = lin_map(v)
    alpha = w1.conj().T.dot(v)
    w = w1 - alpha*v
    if v_former.size > 0:
        w -= beta * v_former
    return w, alpha


def update_v(w, vs, v_now=np.zeros(0), tol=1e-15):
    # vs only the first x columns, with x the updated vs
    # v_now: in case beta = 0, reset v as v_now
    beta = np.linalg.norm(w)
    if beta < tol:
        if v_now.size == 0:
            v_now = np.random.randn(w.shape)
        v = orthogonal_v1_from_v0(v_now, vs, tol)
    else:
        v = w/beta
        v = orthogonal_v1_from_v0(v, vs, tol)
    return v, beta


def orthogonal_v1_from_v0(v1, vs, tol=1e-16):
    # v0 matrix formed by multiple column orthogonal vectors
    length = vs.shape[1]
    for l in range(0, length):
        norm = -1
        while norm < tol:
            v2 = v1 - (vs[:, [l]].conj().T.dot(v1)) * vs[:, [l]]
            norm = np.linalg.norm(v2)
            if norm >= tol:
                v1 = v2 / norm
            else:
                # note: recursive process used here
                # if residual is too small, perturb v1
                v1 += np.random.randn(v1.shape)*tol*2
                v1 = v1/np.linalg.norm(v1)
                if l > 0:
                    v1 = orthogonal_v1_from_v0(v1+np.random.randn(v1.shape)*tol*2, vs[:, :l])
    return v1


def set_random_initial_vectors(v0, k):
    vs = np.random.randn(v0.size, k)
    vs[:, [0]] = v0
    for i in range(1, k):
        vs[:, [i]] = orthogonal_v1_from_v0(vs[:, [i]], vs[:, :i])
    return vs


def set_ones_initial_vectors(v0, k):
    vs = np.ones((v0.size, k)) / (v0.size**0.5)
    vs[:, [0]] = v0
    return vs


def set_initial_v(v0, d):
    if v0.size == 0:
        v0 = np.random.randn(d, 1)
    v0 = v0/np.linalg.norm(v0)
    return v0


def handle_which(which, lm, n):
    which = which.lower()
    if which == 'sm':
        p = np.argpartition(abs(lm), n)[:n]
    elif which == 'la':
        s = lm.size
        p = range(s-1, s-n-1, -1)
    elif which == 'sa':
        p = range(0, n)
    else:  # if which is not standard, use 'lm'
        p = np.argpartition(abs(lm), -n)[-n:]
        p = p[range(n-1, -1, -1)]
    return p


def formulate_t_matrix(alpha, beta):
    k = alpha.size
    t_mat = np.zeros((k, k))
    for n in range(0, k):
        t_mat[n, n] = alpha[n]
    for n in range(0, k-1):
        t_mat[n, n+1] = beta[n]
        t_mat[n+1, n] = beta[n]
    return t_mat


def check_matrix_orthogonality(u):
    s = u.shape
    err_ort = 0
    for n1 in range(0, s[1]-1):
        for n2 in range(n1+1, s[1]):
            err_ort += np.dot(u[:, n1].conj().T, u[:, n2])
    return err_ort
