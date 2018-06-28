import numpy as np


class Universe:

    def __init__(self, lx=10, ly=10, _is_debug=False):
        self._is_debug = _is_debug
        self.size = [lx, ly]
        self.map = np.zeros((lx, ly), dtype=int, )
        self.map = np.zeros((lx, ly), dtype=int, )
        # self.map = list([np.zeros((lx, ly), dtype=int,)])

    # def add_sppoty_to_map(self, spotty):
    #     self.map[spotty.position[0], spotty.position[1]] = spotty.tribe

    # def add_map(self, info):
    #     nation = self.read_nation(info)
    #     if nation + 1 > self.map.__len__():
    #         self.map = self.map + list([np.zeros((self.size[0], self.size[1]), dtype=int,)])

    def read_nearest_neighbour(self, spotty, cont=5):
        # 0: origin, 1: east, 2: south, 3: west, 4:north
        pos_x = spotty.position[0]
        pos_y = spotty.position[1]
        nneighbour = np.zeros(cont, dtype=int)
        if pos_x == 0:
            nneighbour[3] = -1
        else:
            nneighbour[3] = self.map[pos_x - 1, pos_y]
        if pos_x == self.size[0] - 1:
            nneighbour[1] = -1
        else:
            nneighbour[1] = self.map[pos_x + 1, pos_y]
        if pos_y == 0:
            nneighbour[4] = -1
        else:
            nneighbour[4] = self.map[pos_x, pos_y - 1]
        if pos_y == self.size[1] - 1:
            nneighbour[2] = -1
        else:
            nneighbour[2] = self.map[pos_x, pos_y + 1]
        return nneighbour

    def read_neighbour(self, spotty, length=2):
        nneighbour = self.read_nearest_neighbour(spotty)
        neighbour = np.zeros(5)
        pos_x = spotty.position[0]
        pos_y = spotty.position[1]
        if length == 1:
            neighbour = nneighbour
        elif length == 2:
            neighbour = np.hstack((nneighbour, -np.ones(4, dtype=int)))
            if pos_x == 0 and pos_y == 0:
                neighbour[6] = self.map[pos_x + 1, pos_y + 1]
            elif pos_x ==0 and pos_y == self.size[1] - 1:
                neighbour[5] = self.map[pos_x + 1, pos_y - 1]
            elif pos_x == self.size[0] - 1 and pos_y == 0:
                neighbour[7] = self.map[pos_x - 1, pos_y + 1]
            elif pos_x == self.size[0] - 1 and pos_y == self.size[1] - 1:
                neighbour[8] = self.map[pos_x - 1, pos_y + 1]
            elif pos_x == 0:
                neighbour[6] = self.map[pos_x + 1, pos_y + 1]
                neighbour[5] = self.map[pos_x + 1, pos_y - 1]
            elif pos_x == self.size[0] - 1:
                neighbour[7] = self.map[pos_x - 1, pos_y + 1]
                neighbour[8] = self.map[pos_x - 1, pos_y + 1]
            elif pos_y == 0:
                neighbour[6] = self.map[pos_x + 1, pos_y + 1]
                neighbour[7] = self.map[pos_x - 1, pos_y + 1]
            elif pos_y == self.size[1] - 1:
                neighbour[5] = self.map[pos_x + 1, pos_y - 1]
                neighbour[8] = self.map[pos_x - 1, pos_y - 1]
            else:
                neighbour[5] = self.map[pos_x + 1, pos_y - 1]
                neighbour[6] = self.map[pos_x + 1, pos_y + 1]
                neighbour[7] = self.map[pos_x - 1, pos_y + 1]
                neighbour[8] = self.map[pos_x - 1, pos_y - 1]
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

    # @staticmethod
    # def read_nation(info):
    #     nation = list()
    #     try:
    #         nation = nation + list([info['nation']])
    #     except KeyError:
    #         nation = list([0])
    #     return nation
