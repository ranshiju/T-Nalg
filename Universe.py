import numpy as np


class Universe:

    def __init__(self, lx=10, ly=10, _is_debug=False):
        self._is_debug = _is_debug
        self.size = [lx, ly]
        self.map_tribe = np.zeros((lx, ly), dtype=int, )
        self.map_nation = np.zeros((lx, ly), dtype=int, )

    def read_map(self, spotty, length=2):
        neighbour = {'tribe': np.zeros(1), 'nation': np.zeros(1)}
        is_nation = check_nation(spotty)
        neighbour['tribe'] = read_neighbour(self.map_tribe, self.size, spotty, length)
        if not is_nation == 0:
            neighbour['nation'] = read_neighbour(self.map_nation, self.size, spotty, length)
        return neighbour

    def update_universe(self, spotty, move):
        is_nation = check_nation(spotty)
        self.map_tribe = update_map(self.map_tribe, spotty.position, spotty.age, spotty.max_age,
                                    spotty.split_energy, spotty.tribe, move)
        if not is_nation == 0:
            self.map_nation = update_map(self.map_nation, spotty.position, spotty.age, spotty.max_age,
                                         spotty.split_energy, spotty.nation, move)


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


def check_nation(spotty):
    try:
        is_nation = spotty.nation
    except KeyError:
        is_nation = 0
    return is_nation


def update_map(the_map, position, age, max_age, split_energy, info, move):
    pos_x = position[0]
    pos_y = position[1]
    if age == max_age:
        the_map[pos_x, pos_y] = 0
    elif not move == 0:
        the_map[pos_x, pos_y] = 0
        if split_energy == 1:
            the_map[pos_x, pos_y] = info
        if move == 1:
            the_map[pos_x + 1, pos_y] = info
        if move == 2:
            the_map[pos_x, pos_y + 1] = info
        if move == 3:
            the_map[pos_x - 1, pos_y] = info
        if move == 4:
            the_map[pos_x, pos_y - 1] = info
    return the_map
