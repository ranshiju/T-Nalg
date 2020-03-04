import numpy as np
import time


def contract_tensors(tensor):
    return np.tensordot(np.tensordot(tensor, tensor, ([1], [2])), tensor, ([1, 3], [1, 2]))


# Parameters
chi = 10  # bond dimension of the tensor
it_time = 10  # Total iteration time

tensor = np.random.randn(chi, chi, chi)
c = np.zeros((it_time, ))
t0 = time.time()
for t in range(it_time):
    tensor = contract_tensors(tensor)
    c[t] = np.linalg.norm(tensor.reshape(-1, ))
    tensor /= c[t]
time_cost = time.time() - t0
print('Dim = ' + str(chi) + ', it_time = ' + str(it_time) + ': cost ' + str(time_cost) + ' seconds')
