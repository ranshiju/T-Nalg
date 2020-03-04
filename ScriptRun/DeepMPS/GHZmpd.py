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
from library.MPSClass import MpsOpenBoundaryClass
from algorithms.DeepMPSfinite import act_umpo_on_mps


num = 4
mps = tm.mps_ghz_state(num)
a = MpsOpenBoundaryClass(num, 2, 2)
mpd = tm.mpd_of_ghz(num)
mps = act_umpo_on_mps(mps, mpd)
a.input_mps(mps, if_deepcopy=False)
a.correct_orthogonal_center(a.length-1)
a.check_mps_norm1()

f0 = a.fidelity_log_by_spins_up()
print(f0)


