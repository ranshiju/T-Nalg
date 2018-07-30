import numpy as np
import matplotlib.pyplot as mpy
from SpottiesGame import find_position_from_movement

marker_list = ('o', 's', 'p', '*', 'h', 'x', 'D')
list_color = ('C1', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9')
list_plot = {'tribe': marker_list, 'nation': list_color}


class Universe:

    def __init__(self, info, lx=10, ly=10, _is_debug=False):
        self._is_debug = _is_debug
        self.size = [lx, ly]
        self._depth = 1
        self.fig = mpy.figure()
        self.map = dict()
        for p in info:
            self.map[p] = np.zeros((lx, ly), dtype=int)

    def read_map(self, spotty, length=2):
        neighbour = dict()
        for p in spotty.info:
            neighbour[p] = self.read_neighbour(spotty.position, length)
        return neighbour

    def move_universe(self, spotty, decision):
        self.delete_universe(spotty)
        spotty.position = find_position_from_movement(spotty.position, decision)
        self.put_universe(spotty)
        # if is_figure:
        #     self.fig = self.plot_universe()

    def put_universe(self, spotty):
        for p in spotty.info:
            self.map[p][spotty.position[0], spotty.position[1]] = spotty.info[p]

    def delete_universe(self, spotty):
        for p in spotty.info:
            self.map[p][spotty.position[0], spotty.position[1]] = 0

    def plot_universe(self, _show_time=1, _is_show=True):
        ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlim(0, self.size[1] - 1)
        ax.set_ylim(self.size[0] - 1, 0)
        ax.set_xticks(np.arange(0, self.size[1], 1))
        ax.set_yticks(np.arange(0, self.size[0], 1))
        ax.grid(color='black', linestyle='-')
        mpy.ion()
        mpy.show()
        pos = np.nonzero(self.map['tribe'])
        for i in range(0, len(pos[1])):
            if 'nation' in self.map:
                map_color = self.map['nation'] - 1
            else:
                map_color = np.zeros([self.size[0], self.size[1]], dtype=int)
            ax.plot(pos[1][i], pos[0][i], color=list_color[map_color[pos[0][i], pos[1][i]]],
                    marker=marker_list[self.map['tribe'][pos[0][i], pos[1][i]]])
        if _is_show:
            mpy.draw()
            mpy.pause(_show_time)
            mpy.show()
        mpy.clf()

    def read_neighbour(self, pos, length=2, depth=1):
        # 0: east, 1: south, 2: west, 3:north 4:NE 5:SE 6:SW 7:NW
        _map = extend_map(self.map['tribe'], depth)
        neighbour = np.zeros((1, 4), dtype=int)
        neighbour[0, 0] = _map[pos[0] + depth, pos[1] + depth + 1]
        neighbour[0, 1] = _map[pos[0] + depth + 1, pos[1] + depth]
        neighbour[0, 2] = _map[pos[0] + depth, pos[1] + depth - 1]
        neighbour[0, 3] = _map[pos[0] + depth - 1, pos[1] + depth]
        if length == 2:
            neighbour = np.hstack((neighbour, np.zeros((1, 4), dtype=int)))
            neighbour[0, 4] = _map[pos[0] + depth - 1, pos[1] + depth + 1]
            neighbour[0, 5] = _map[pos[0] + depth + 1, pos[1] + depth + 1]
            neighbour[0, 6] = _map[pos[0] + depth + 1, pos[1] + depth - 1]
            neighbour[0, 7] = _map[pos[0] + depth - 1, pos[1] + depth - 1]
        return neighbour


def extend_map(the_map, depth=1):
    size_0 = np.shape(the_map)
    map_extended = -np.ones((size_0[0]+2*depth, size_0[1]+2*depth), dtype=int)
    map_extended[depth:size_0[0]+depth, depth:size_0[1]+depth] = the_map[:, :]
    return map_extended
################################################
# plot function


def plot_square_map(width, height, _is_show=False):
    figure = mpy.figure('grid')
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

