import os
import sys
import numpy as np
import library.Parameters as pm
import library.HamiltonianModule as hm
import library.TensorBasicModule as tm
import library.BasicFunctions as bf
import scipy.linalg as la
import pickle
import math
import os.path as opath
import time
import torch
import library.TNmachineLearning as TNML


num = list(range(3, 12))
qs = np.zeros((num.__len__(), ))
qqs = np.zeros((num.__len__(), ))
qqqs = np.zeros((num.__len__(), ))
y = np.zeros((num.__len__(), ))
for n in num:
    x = tm.w_state(n)
    qs[n-num[0]] = tm.q_sparsity(x)

    qqs[n - num[0]] = -1
    for nn in range(2, n+1):
        qqs[n-num[0]] += (-(nn-1)/nn*np.log((nn-1)/nn) + 1/nn*np.log(nn)) / np.log(2) - 1

    qqqs[n - num[0]] = -n + np.log(n) / np.log(2)
    for nn in range(3, n+1):
        qqqs[n-num[0]] += 1 / nn * np.log(nn - 1) / np.log(2)
print(qs)
print(qqs)
print(qqqs)
bf.plot(num, qs, qqs, qqqs)
# bf.plot(num, y)


