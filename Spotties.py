import numpy as np
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

    def move(self, decision):
        if self.position[0] > 0:
            pos = find_position_from_movement(self.position, decision)
            self.position = pos
        else:
            bfr.print_error('Cannot move before entering the map')

    def report_yourself(self):
        print('Age : ' + str(self.age))
        print('Energy: ' + str(self.energy))
        print('Position: ' + str(self.position))


def decide_by_intel(intel, env):
    env = env[:].astype(np.float64)
    if intel['type'] is 'linear':
        return linear_intel(intel['data'], env)


def save_intel_linear_random(input_len, output_len, file_name):
    intel = np.random.randn(input_len, output_len)
    bfr.save_pr('.\\Intels\\', file_name, [intel], ['intel'])


def linear_intel(intel, env):
    output = (env+1).dot(intel)
    output[0, 1:] = output[0, 1:] * (env[0, :4] == 0)
    output = output**2
    output /= np.sum(output)
    return output2decision(output)


def output2decision(output):
    v = np.zeros((output.size, ))
    v[0] = output[0, 0]
    for n in range(1, output.size):
        v[n] = v[n - 1] + output[0, n]
    rand = np.random.random()
    return bfr.arg_find_array(v > rand, 1, 'first') - 1


def spotty_copy(spotty):
    new = Spotty(spotty.info, spotty.max_age, spotty.split_energy)
    return new


def find_position_from_movement(pos, movement):
    if movement == 1:
        pos[0] += 1
    elif movement == 2:
        pos[1] -= 1
    elif movement == 3:
        pos[0] -= 1
    elif movement == 4:
        pos[1] += 1
    return pos
