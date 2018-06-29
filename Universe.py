import numpy as np
import matplotlib.pyplot as mpy
from SpottiesGame import find_position_from_movement


class Universe:

    def __init__(self, info, lx=10, ly=10, _is_debug=False):
        self._is_debug = _is_debug
        self.size = [lx, ly]
        self.map = dict()
        self.figure = plot_square_map(lx, ly)
        for p in info:
            self.map[p] = np.zeros([lx, ly], dtype=int)

    def read_map(self, spotty, length=2):
        neighbour = dict()
        for p in spotty.info:
            neighbour[p] = read_neighbour(self.map[p], self.size, spotty, length)
        return neighbour

    def move_universe(self, spotty, move):
        self.delete_universe(spotty)
        spotty.position = find_position_from_movement(spotty.position, move)
        self.put_universe(spotty)
        self.figure = self.plot_universe()

    def put_universe(self, spotty):
        for p in spotty.info:
            self.map[p][spotty.position[0], spotty.position[1]] = spotty.info[p]
        self.figure = self.plot_universe()

    def delete_universe(self, spotty):
        for p in spotty.info:
            self.map[p][spotty.position[0], spotty.position[1]] = 0
        self.figure = self.plot_universe()

    def plot_universe(self, _is_show=False):
        mpy.clf()
        figure = plot_square_map(self.size[0], self.size[1])
        occupation = np.zeros([self.size[0], self.size[1]])
        for p in self.map:
            occupation = occupation + self.map[p]
        pos = np.nonzero(occupation)
        for i in range(0, len(pos[1])):
            try:
                map_color = self.map['nation']
            except KeyError:
                map_color = np.zeros([self.size[0], self.size[1]])
            figure = mpy.plot(pos[0][i], pos[1][i], color=list_color()[map_color[pos[0][i], pos[1][i]]],
                              marker=list_marker()[self.map['tribe'][pos[0][i], pos[1][i]]])
        if _is_show:
            mpy.show(figure)
        return figure


def read_nearest_neighbour(the_map, the_size, spotty, cont=5):
    # 0: origin, 1: east, 2: south, 3: west, 4:north
    pos_x = spotty.position[0]
    pos_y = spotty.position[1]
    nneighbour = np.zeros(cont, dtype=int)
    if pos_x == 0:
        nneighbour[3] = -1
    else:
        nneighbour[3] = the_map[pos_x - 1, pos_y]
    if pos_x == the_size[0] - 1:
        nneighbour[1] = -1
    else:
        nneighbour[1] = the_map[pos_x + 1, pos_y]
    if pos_y == 0:
        nneighbour[4] = -1
    else:
        nneighbour[4] = the_map[pos_x, pos_y - 1]
    if pos_y == the_size[1] - 1:
        nneighbour[2] = -1
    else:
        nneighbour[2] = the_map[pos_x, pos_y + 1]
    return nneighbour


def read_neighbour(the_map, the_size, spotty, length=2):
    nneighbour = read_nearest_neighbour(the_map, the_size, spotty)
    neighbour = np.zeros(5)
    pos_x = spotty.position[0]
    pos_y = spotty.position[1]
    if length == 1:
        neighbour = nneighbour
    elif length == 2:
        neighbour = np.hstack((nneighbour, -np.ones(4, dtype=int)))
        if pos_x == 0 and pos_y == 0:
            neighbour[6] = the_map[pos_x + 1, pos_y + 1]
        elif pos_x == 0 and pos_y == the_size[1] - 1:
            neighbour[5] = the_map[pos_x + 1, pos_y - 1]
        elif pos_x == the_size[0] - 1 and pos_y == 0:
            neighbour[7] = the_map[pos_x - 1, pos_y + 1]
        elif pos_x == the_size[0] - 1 and pos_y == the_size[1] - 1:
            neighbour[8] = the_map[pos_x - 1, pos_y + 1]
        elif pos_x == 0:
            neighbour[6] = the_map[pos_x + 1, pos_y + 1]
            neighbour[5] = the_map[pos_x + 1, pos_y - 1]
        elif pos_x == the_size[0] - 1:
            neighbour[7] = the_map[pos_x - 1, pos_y + 1]
            neighbour[8] = the_map[pos_x - 1, pos_y + 1]
        elif pos_y == 0:
            neighbour[6] = the_map[pos_x + 1, pos_y + 1]
            neighbour[7] = the_map[pos_x - 1, pos_y + 1]
        elif pos_y == the_size[1] - 1:
            neighbour[5] = the_map[pos_x + 1, pos_y - 1]
            neighbour[8] = the_map[pos_x - 1, pos_y - 1]
        else:
            neighbour[5] = the_map[pos_x + 1, pos_y - 1]
            neighbour[6] = the_map[pos_x + 1, pos_y + 1]
            neighbour[7] = the_map[pos_x - 1, pos_y + 1]
            neighbour[8] = the_map[pos_x - 1, pos_y - 1]
    return neighbour


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


def plot_list():
    list_plot = {'tribe': list_marker(), 'nation': list_color()}
    return list_plot


def list_color():
    color_list = ('C1', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9')
    return color_list


def list_marker():
    marker_list = ('o', 's', 'p', '*', 'h', 'x', 'D')
    return marker_list


# def plot_map(the_map, the_size, the_list):
#     for i in range(0, the_size[0]):
#         for j in range(0, the_size[1]):
#             figure = plot()


def plot_put(position, info, _is_show=False):
    try:
        nation = info['nation'] - 1
    except KeyError:
        nation = 0
    figure = mpy.plot(position[0], position[1], color=list_color()[nation], marker=list_marker()[info['tribe'] - 1])
    if _is_show:
        mpy.show(figure)
    return figure


# def plot_delete(position, _is_show=False):
#     mpy.clf(position)