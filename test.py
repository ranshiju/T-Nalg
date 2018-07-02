import numpy as np
from matplotlib import pyplot as mpy


def extend_map(the_map, depth=1):
    size_0 = np.shape(the_map)
    map_extended = np.vstack((the_map, - np.ones((depth, size_0[1]), dtype=int)))
    map_extended = np.vstack((- np.ones((depth, size_0[1]), dtype=int), map_extended))
    map_extended = np.hstack((map_extended, - np.ones((size_0[0] + 2*depth, depth), dtype=int)))
    map_extended = np.hstack((- np.ones((size_0[0] + 2*depth, depth), dtype=int), map_extended))
    return map_extended


the_map = np.ones((4, 5))
print(extend_map(np.ones((4, 5))))


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
