import numpy as np
import BasicFunctionsSJR as bfr
from Spotties import Spotty, save_intel_linear_random
from Universe import Universe
_is_debug = 0


class GameBasic:

    def __init__(self, lx, ly, max_age, split_energy, is_debug=False):
        self.universe = Universe(lx, ly)
        self.spotties = list()
        self.population = [0]
        self.max_age = max_age
        self.split_energy = split_energy
        self._is_debug = is_debug

    def add_spotties_random(self, new_pop, info):
        if self.population.__len__() < info['tribe'] + 0.5:
            self.population.append(0)
        for n in range(0, new_pop):
            self.spotties.append(Spotty(info, self.max_age, self.split_energy))
            pos = self.spotties[n + self.population[0]].enter_map_random(self.universe.map_tribe)
            self.universe.map_tribe[pos[0], pos[1]] = info['tribe']
        self.population[info['tribe']] += new_pop
        self.population[0] += new_pop

    def add_spotty_positions(self, info, positions):
        if self.population.__len__() < info['tribe'] + 0.5:
            self.population.append(0)
        new_pop = len(positions)
        for n in range(0, new_pop):
            self.spotties.append(Spotty(info, self.max_age, self.split_energy))
            self.spotties[n + self.population[0]].enter_map_position(positions[n])
            self.universe.map_tribe[positions[n][0], positions[n][1]] = info['tribe']
        self.population[info['tribe']] += new_pop
        self.population[0] += new_pop

    def spotty_split_random(self, nth):
        # split the nth spotty
        neighbor = self.universe.read_map(self.spotties[nth], 1)
        movement = np.nonzero(neighbor['tribe'] == 0)
        rand_pos = np.random.randint(0, len(movement[0]), 1)
        movement = movement[rand_pos]
        new_pos = find_position_from_movement(self.spotties[nth].position, movement)
        new = spotty_copy(self.spotties[nth])
        new.enter_map_position(new_pos)
        self.universe.map_tribe[new_pos[0], new_pos[1]] = self.spotties[nth].info['tribe']
        self.spotties.append(new)
        self.population[self.spotties[nth].info['tribe']] += 1
        self.population[0] += 1
        self.spotties[nth].energy = 0

    def spotty_move(self, nth, decision):
        pos0 = self.spotties[nth].position
        self.spotties[nth].move(decision)
        pos1 = self.spotties[nth].position
        self.universe.map_tribe[pos0[0], pos0[1]] = 0
        self.universe.map_tribe[pos1[0], pos1[1]] = self.spotties[nth].info['tribe']

    def spotty_die(self, nth):
        pos = self.spotties[nth].position
        self.universe.map_tribe[pos[0], pos[1]] = 0
        self.spotties.__delitem__(nth)


class SpottiesGameV0(GameBasic):
    # One tribe
    # empty sites > self.cond_gain_energy: gain energy

    def __init__(self, cond_gain_energy, lx, ly, max_age, split_energy, is_debug=False):
        super(SpottiesGameV0, self).__init__(lx, ly, max_age, split_energy, is_debug)
        self.cond_gain_energy = cond_gain_energy

    def update_one_spotty(self, nth, intel):
        if self.spotties[nth].age > self.max_age:
            self.spotty_die(nth)
        elif self.spotties[nth].energy == self.split_energy:
            self.spotty_split_random(nth)
        else:
            env = self.universe.read_map(self.spotties[nth], 2)
            decision = intel(env['tribe'])
            self.spotty_move(nth, decision)
            self.spotties[nth].age += 1
            if self.if_gain_energy(env, self.cond_gain_energy):
                self.spotties[nth].energy += 1

    @ staticmethod
    def if_gain_energy(env2, cond_energy):
        return not (sum(env2 == 0) < cond_energy)


def game_v0():
    # iteration time
    time = 200
    # map size
    lx = 9
    ly = 9
    # properties of spotties
    ini_pos = [4, 4]
    max_age = 15
    split_energy = 5
    cond_gain_energy = 3
    # initial game
    game = SpottiesGameV0(cond_gain_energy, lx, ly, max_age, split_energy)
    info = {'tribe': 1}
    game.add_spotty_positions(info, ini_pos)

    save_intel_linear_random(8, 5, 'linear_intel.pr')
    intel = [bfr.load_pr('.\\Intels\\linear_intel.pr', 'intel')]

    for t in range(0, time):
        for n in range(game.population[0]-1, -1, -1):
            game.update_one_spotty(n, intel[game.spotties[n].info['tribe']-1])
        # show map


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

