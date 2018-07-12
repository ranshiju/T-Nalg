import numpy as np
from matplotlib import pyplot as mpy
import BasicFunctionsSJR as bfr


# intel = bfr.load_pr('.\\Intels\\linear_intel.pr', 'intel')
# # This environment only allow spotty to move to west
# env = np.array([[-1, -1, 0, -1, -1, -1, -1, -1]])
# # The linear intel should only output 3, but instead it output -1, which is meaningless
# print(linear_intel(intel, env))

def plot_square_map(width, height, _is_show=False):
    figure = mpy.figure()
    for i in range(0, height):
        pos0 = [i, i]
        pos1 = [0, width - 1]
        mpy.plot(pos0, pos1, 'black')
    for j in range(0, width):
        pos0 = [0, height - 1]
        pos1 = [j, j]
        mpy.plot(pos0, pos1, 'black')
    if _is_show:
        mpy.show()
    return figure


fig1 = plot_square_map(5, 4)

# mpy.figure(1)
# mpy.clf()
fig2.show()