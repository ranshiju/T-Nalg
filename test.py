import numpy as np
from matplotlib import pyplot as mpy
import BasicFunctionsSJR as bfr


x = 0
y = np.random.randn(2, 2, 2)
print(x + y)


# tensors = [np.random.randn(2, 2, 2), np.random.randn(2, 2, 2), np.random.randn(2, 2, 2)]
# indexes = [[1, 2, -1], [2, 3, -2], [3, 1, -3]]
# a = tbm.CONT(tensors, indexes)
# print(dir(a))

# x = np.array(range(0, 4))
# y = np.array(range(0, 2))
# print(x[y])

# x = {-1, -3, -6, -2, -4, -5}
# print(x.add('a'))
# print(x)
# print(list(x))
# y = tbm.embed_list_into_matrix(x)
# print(y)
# pos = np.nonzero(y == 1)
# print(pos)

# x = {1, 2, 3, 4}
# print(2 in x)

# x = [
#     [4, 6, 13, 14],
#     [0, 1, 3, 4],
#     [1, 2, 5, 6],
#     [0, 1, 1, 2],
#     [1, 1, 9, 10],
#     [4, 2, 11, 12],
#     [4, 2, 11, 12],
#     [4, 2, 11, 12],
#     [1, 3, 7, 8]
# ]
# x = np.array(x)
# y = from_index2_to_positions_h2(x)
# print(y)

# x = np.array(x)
# x = x[:, :2]
# y = sort_positions(x)
# print(y)

# x = np.random.random_integers(low=0, high=1, size=(20, 2))
# print(x)
# y = map(np.unique, x)
# print(y)

# L = 100000
# a = np.random.randn(L, 1)
#
# t0 = time.time()
# pool = ThreadPool(4)
# results1 = pool.map(plus, a)
# pool.close()
# pool.join()
# print(time.time() - t0)
# print(type(results1))
#
# t0 = time.time()
# results2 = np.zeros((L, 1))
# for n in range(0, L):
#     results2[n] = plus(a[n])
# print(time.time() - t0)
#
# print(np.linalg.norm(results2-results1))


# x = np.random.randn(3, 3, 3)
# print(x)
# y = np.nonzero(x > 0)
# print(y)
# z = Bf.arg_find_array(x > 0, 2, 'first')
# print(z)

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

