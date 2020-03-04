import numpy as np
import time
import scipy.sparse.linalg as la


def rank1_decomp(tensor, v=None, it_time=500, tol=1e-12, method='power', show_error=False):
    # way = 'power' or 'eigs' or 'svds'
    # 'eigs' way only applies for real symmetrical tensor
    info = dict()
    ndim = tensor.ndim
    s = tensor.shape
    if v is None:
        v = [np.random.randn(s[n], ) for n in range(ndim)]
        v = [x / np.linalg.norm(x) for x in v]
    norm0 = -1
    norm = 1
    info['it_time'] = it_time
    z = np.linalg.norm(tensor)
    for t in range(it_time):
        if method is 'power':
            for n in range(ndim):  # update the n-th vector
                tmp = tensor.copy()
                for nb in range(n):
                    tmp = np.tensordot(tmp, v[nb], ([0], [0]))
                for nb in range(ndim-1, n, -1):
                    tmp = np.tensordot(tmp, v[nb], ([-1], [0]))
                norm = np.linalg.norm(tmp)
                v[n] = tmp.copy() / norm
        else:
            # only for even ndim
            nh = int(ndim / 2)
            for n in range(nh):
                tmp = tensor.copy()
                nb_now = 0
                for nb in range(ndim):
                    if nb in [n, n + nh]:
                        nb_now += 1
                    else:
                        tmp = np.tensordot(tmp, v[nb].reshape(-1, ), ([nb_now], [0]))
                if method is 'svds':
                    v[n], norm, v[n + nh] = la.svds(tmp, k=1)
                else:
                    norm, v[n] = la.eigsh(tmp, k=1)
                    v[n + nh] = v[n].copy()
                norm = norm[0]
        if show_error:
            tmp = np.kron(np.kron(np.kron(v[0], v[1]), v[2]), v[3]) * norm
            info['error'] = np.linalg.norm(tensor.reshape(-1, ) - tmp.reshape(-1, )) / z
            print('For t = ' + str(t) + ', error = ' + str(info['error']))
        if abs(norm - norm0) < tol:
            info['it_time'] = t
            break
        else:
            norm0 = norm
    if not show_error:
        tmp = np.kron(np.kron(np.kron(v[0], v[1]), v[2]), v[3]) * norm
        info['error'] = np.linalg.norm(tensor.reshape(-1, ) - tmp.reshape(-1, )) / z
    return v, info


dim = 2
ndim = 4

np.random.seed(0)
tensor = eval('np.random.randn' + str((dim, ) * ndim))
tensor = tensor + tensor.transpose(0, 3, 2, 1)
tensor = tensor + tensor.transpose(2, 1, 0, 3)
t0 = time.time()
v, info = rank1_decomp(tensor, method='power', show_error=False)
print('Time cost = ' + str(time.time() - t0))
print('Error of rank-1 decomposition = ' + str(info['error']))
t0 = time.time()
v, info = rank1_decomp(tensor, method='eigs', show_error=False)
print('Time cost = ' + str(time.time() - t0))
print('Error of rank-1 decomposition = ' + str(info['error']))
t0 = time.time()
v, info = rank1_decomp(tensor, method='svds', show_error=False)
print('Time cost = ' + str(time.time() - t0))
print('Error of rank-1 decomposition = ' + str(info['error']))
