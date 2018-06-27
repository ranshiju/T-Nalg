import numpy as np


class Universe:

    def __init__(self, lx=10, ly=10, _is_debug=False):
        self._is_debug = _is_debug
        self.size = [lx, ly]
        self.map = np.zeros((lx, ly))

    def read_neighbour(self, spotty, connect=5):
        # 0: origin, 1: east, 2: south, 3: west, 4:north
        pos_x = spotty.position[0]
        pos_y = spotty.position[1]
        neighbour = np.zeros(connect)
        if pos_x == 0:
            neighbour[3] = 1
        elif not self.map[pos_x - 1, pos_y] == 0:
                neighbour[3] = 1
        if pos_x == self.size[0] - 1:
            neighbour[1] = 1
        elif not self.map[pos_x + 1, pos_y] == 0:
                neighbour[1] = 1
        if pos_y == 0:
            neighbour[4] = 1
        elif not self.map[pos_x, pos_y - 1] == 0:
                neighbour[4] = 1
        if pos_y == self.size[1] - 1:
            neighbour[2] = 1
        elif not self.map[pos_x, pos_y + 1] == 0:
                neighbour[2] = 1
        return neighbour

    def update_universe(self, spotty, move):
        pos_x = spotty.position[0]
        pos_y = spotty.position[1]
        if spotty.age == spotty.max_age:
            self.map[pos_x, pos_y] = 0
        elif not move == 0:
            self.map[pos_x, pos_y] = 0
            if spotty.split_energy == 1:
                self.map[pos_x, pos_y] = spotty.tribe
            if move == 1:
                self.map[pos_x + 1, pos_y] = spotty.tribe
            if move == 2:
                self.map[pos_x, pos_y + 1] = spotty.tribe
            if move == 3:
                self.map[pos_x - 1, pos_y] = spotty.tribe
            if move == 4:
                self.map[pos_x, pos_y - 1] = spotty.tribe