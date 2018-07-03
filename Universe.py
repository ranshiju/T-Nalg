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
        self.figure = plot_square_map(lx, ly)
        self.map = dict()
        for p in info:
            self.map[p] = np.zeros((lx, ly), dtype=int)

    def read_map(self, spotty, length=2):
        neighbour = dict()
        for p in spotty.info:
            neighbour[p] = self.read_neighbour(spotty.position, length)
        return neighbour

    def move_universe(self, spotty, decision, is_figure=False):
        self.delete_universe(spotty, is_figure)
        spotty.position = find_position_from_movement(spotty.position, decision)
        self.put_universe(spotty, is_figure)
        if is_figure:
            self.figure = self.plot_universe()

    def put_universe(self, spotty, is_figure=False):
        for p in spotty.info:
            self.map[p][spotty.position[0], spotty.position[1]] = spotty.info[p]
        if is_figure:
            self.figure = self.plot_universe()

    def delete_universe(self, spotty, is_figure=False):
        for p in spotty.info:
            self.map[p][spotty.position[0], spotty.position[1]] = 0
        if is_figure:
            self.figure = self.plot_universe()

    def plot_universe(self, _show_time=0.1, _is_show=True):
        mpy.clf()
        mpy.ion()
        mpy.show()
        figure = plot_square_map(self.size[0], self.size[1])
        pos = np.nonzero(self.map['tribe'])
        for i in range(0, len(pos[1])):
            if 'nation' in self.map:
                map_color = self.map['nation'] - 1
            else:
                map_color = np.zeros([self.size[0], self.size[1]], dtype=int)
            figure = mpy.plot(pos[0][i], pos[1][i], color=list_color[map_color[pos[0][i], pos[1][i]]],
                              marker=marker_list[self.map['tribe'][pos[0][i], pos[1][i]]])
        if _is_show:
            mpy.draw()
            mpy.pause(_show_time)
            mpy.show()
        return figure

    def read_neighbour(self, pos, length=2):
        # 0: east, 1: south, 2: west, 3:north 4:NE 5:SE 6:SW 7:NW
        _map = extend_map(self.map['tribe'], 1)
        neighbour = np.zeros((1, 4), dtype=int)
        neighbour[0, 0] = _map[pos[0] + 2, pos[1] + 1]
        neighbour[0, 1] = _map[pos[0] + 1, pos[1]]
        neighbour[0, 2] = _map[pos[0], pos[1] + 1]
        neighbour[0, 3] = _map[pos[0] + 1, pos[1] + 2]
        if length == 2:
            neighbour = np.hstack((neighbour, np.zeros((1, 4), dtype=int)))
            neighbour[0, 4] = _map[pos[0] + 2, pos[1] + 2]
            neighbour[0, 5] = _map[pos[0] + 2, pos[1]]
            neighbour[0, 6] = _map[pos[0], pos[1]]
            neighbour[0, 7] = _map[pos[0], pos[1] + 1]
        return neighbour


def extend_map(the_map, depth=1):
    size_0 = np.shape(the_map)
    map_extended = -np.ones(size_0[0]+2*depth, size_0[1]+2*depth)
    map_extended[depth:size_0[0]+depth, depth:size_0[1]+depth] = the_map[:, :]
    return map_extended
################################################
# plot function


def plot_square_map(width, height, _is_show=False):
    figure = mpy.plot()
    for i in range(0, height):
        pos0 = [i, i]
        pos1 = [0, width - 1]
        figure = mpy.plot(pos0, pos1, 'black')
    for j in range(0, width):
        pos0 = [0, height - 1]
        pos1 = [j, j]
        mpy.plot(pos0, pos1, 'black')
    if _is_show:
        mpy.show()
    return figure


def plot_put(position, info, _is_show=False):
    try:
        nation = info['nation'] - 1
    except KeyError:
        nation = 0
    figure = mpy.plot(position[0], position[1], color=list_color[nation], marker=marker_list[info['tribe'] - 1])
    if _is_show:
        mpy.show(figure)
    return figure
