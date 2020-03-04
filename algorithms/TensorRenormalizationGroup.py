import numpy as np
import time


def one_rg_step_svd(tensor, chi):
    u1, s1, v1 = np.linalg.svd(tensor.reshape(chi*chi, chi*chi))
    u1 = u1[:, :chi].dot(np.diag(s1[:chi]**0.5))
    v1 = np.diag(s1[:chi]**0.5).dot(v1[:chi, :])
    u1 = u1.reshape(chi, chi, chi)
    v1 = v1.reshape(chi, chi, chi)

    u2, s2, v2 = np.linalg.svd(tensor.transpose(1, 2, 3, 0).reshape(chi * chi, chi * chi))
    u2 = u2[:, :chi].dot(np.diag(s2[:chi]**0.5))
    v2 = np.diag(s2[:chi]**0.5).dot(v2[:chi, :])
    u2 = u2.reshape(chi, chi, chi)
    v2 = v2.reshape(chi, chi, chi)

    tmp1 = np.tensordot(u2, u1, ([1], [0]))
    tmp2 = np.tensordot(v1, v2, ([1], [2]))
    return np.tensordot(tmp1, tmp2, ([0, 2], [1, 3])).transpose(2, 3, 1, 0)


def one_rg_step_eig(tensor, chi):
    tmp = tensor.reshape(chi * chi, chi * chi)
    lm, u = np.linalg.eigh(tmp.dot(tmp.transpose(1, 0)))
    u = truncate_u(lm, u, chi)
    u1 = u.reshape(chi, chi, chi)
    v1 = u.transpose(1, 0).conj().dot(tmp).reshape(chi, chi, chi)
    u1 = u1.reshape(chi, chi, chi)
    v1 = v1.reshape(chi, chi, chi)

    tmp = tensor.transpose(1, 2, 3, 0).reshape(chi * chi, chi * chi)
    lm, u = np.linalg.eigh(tmp.dot(tmp.transpose(1, 0)))
    u = truncate_u(lm, u, chi)
    u2 = u.reshape(chi, chi, chi)
    v2 = u.transpose(1, 0).conj().dot(tmp).reshape(chi, chi, chi)
    u2 = u2.reshape(chi, chi, chi)
    v2 = v2.reshape(chi, chi, chi)

    tmp1 = np.tensordot(u2, u1, ([1], [0]))
    tmp2 = np.tensordot(v1, v2, ([1], [2]))
    return np.tensordot(tmp1, tmp2, ([0, 2], [1, 3])).transpose(2, 3, 1, 0)


def truncate_u(lm, u, chi):
    # This function is not favorable to GPU
    ord = np.argsort(lm)[::-1]
    u1 = np.zeros((u.shape[0], chi))
    for n in range(chi):
        u1[:, n] = u[:, ord[n]]
    return u1


# Parameters
chi = 10  # bond dimension of the tensor
it_time = 100  # Total iteration time

t0 = time.time()
c = np.zeros((it_time, ))
tensor = np.random.rand(chi, chi, chi, chi)
for t in range(it_time):
    # tensor = one_rg_step_svd(tensor, chi)
    tensor = one_rg_step_eig(tensor, chi)
    c[t] = np.linalg.norm(tensor.reshape(-1, ))
    tensor /= c[t]
time_cost = time.time() - t0
print('Dim = ' + str(chi) + ', it_time = ' + str(it_time) + ': cost ' + str(time_cost) + ' seconds')
