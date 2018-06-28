import numpy as np
from SpottiesGame import find_position_from_movement
import BasicFunctionsSJR as bfr
_is_debug = False


class Spotty:

    def __init__(self, info, max_age, split_energy):
        # in info: tribe, nation, etc.
        self.age = 0
        self.max_age = max_age
        self.info = info
        self.position = [-1, -1]
        self.energy = 0
        self.split_energy = split_energy

    def enter_map_random(self, universe_map):
        poses = np.nonzero(universe_map == 0)
        if len(poses[0]) > 0:
            rand_pos = np.random.randint(0, len(poses[0]), 1)
            self.position = [poses[0][rand_pos[0]], poses[1][rand_pos[0]]]
            return self.position
        else:
            bfr.print_error('Cannot put more into the map since it is full')

    def enter_map_position(self, position):
        self.position = position

    def move(self, movement):
        if self.position[0] > 0:
            pos = find_position_from_movement(self.position, movement)
            self.position = pos
        else:
            bfr.print_error('Cannot move before entering the map')


def save_intel_linear_random(input_len, output_len, file_name):
    intel = np.random.randn(input_len, output_len)
    bfr.save_pr('.\\Intels\\', file_name, [lambda v: linear_intel(intel, v)], ['intel'])


def linear_intel(intel, env):
    output = env.dot(intel)
    output = output**2
    output /= np.sum(output)
    v = np.zeros((output.size-1, ))
    v[0] = output[0]
    for n in range(1, output.size):
        v[n] = v[n-1] + output[n]
    rand = np.random.random()
    decision = bfr.arg_find_array(v > rand, 1, 'first')
    return decision
