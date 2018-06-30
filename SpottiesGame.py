import numpy as np
import BasicFunctionsSJR as bfr
from Spotties import Spotty, save_intel_linear_random, spotty_copy, find_position_from_movement, decide_by_intel
from Universe import Universe
_is_debug = True


class GameBasic:

    def __init__(self, info, lx, ly, max_age, split_energy, is_debug=False):
        self.universe = Universe(info, lx, ly)
        self.spotties = list()
        self.population = dict()
        self.population_tot = 0
        self.max_age = max_age
        self.split_energy = split_energy
        self._is_debug = is_debug

    def add_spotties_random(self, new_pop, info):
        pop0 = self.spotties.__len__()
        for n in range(0, new_pop):
            self.spotties.append(Spotty(info, self.max_age, self.split_energy))
            self.spotties[n + pop0].enter_map_random(self.universe.map['tribe'])
            self.universe.put_universe(self.spotties[n + pop0])

    def add_spotty_positions(self, info, positions):
        new_pop = len(positions)
        pop0 = self.spotties.__len__()
        for n in range(0, new_pop):
            self.spotties.append(Spotty(info, self.max_age, self.split_energy))
            self.spotties[n + pop0].enter_map_position(positions[n])
            self.universe.put_universe(self.spotties[n + pop0])

    def spotty_split_random(self, nth):
        # split the nth spotty
        neighbor1 = self.universe.read_map(self.spotties[nth], 1)
        movement = np.nonzero(neighbor1['tribe'] == 0)
        rand_pos = np.random.randint(0, len(movement[1]), 1)
        movement = movement[1][rand_pos[0]] + 1
        new_pos = find_position_from_movement(self.spotties[nth].position, movement)
        new = spotty_copy(self.spotties[nth])
        new.enter_map_position(new_pos)
        self.universe.put_universe(new)
        self.spotties.append(new)
        self.spotties[nth].energy = 0
        self.spotties[nth].age += 1

    def spotty_move(self, nth, decision):
        self.universe.move_universe(self.spotties[nth], decision)
        # self.spotties[nth].move(decision)
        self.spotties[nth].age += 1

    def spotty_die(self, nth):
        self.universe.delete_universe(self.spotties[nth])
        self.spotties.__delitem__(nth)

    def update_population_info(self):
        self.population_tot = self.spotties.__len__()
        self.population = dict()
        for s in self.spotties:
            for p in s.info:
                if p in self.population:
                    self.population[p] += 1
                else:
                    self.population[p] = 1


class SpottiesGameV0(GameBasic):
    # One tribe
    # empty sites > self.cond_gain_energy: gain energy

    def __init__(self, info, cond_gain_energy, lx, ly, max_age, split_energy, is_debug=False):
        super(SpottiesGameV0, self).__init__(info, lx, ly, max_age, split_energy, is_debug)
        self.cond_gain_energy = cond_gain_energy

    def update_one_spotty(self, nth, intel):
        if _is_debug:
            self.spotties[nth].report_yourself()
        if self.spotties[nth].age > self.max_age:
            self.spotty_die(nth)
        else:
            env = self.universe.read_map(self.spotties[nth], 2)
            if self.if_gain_energy(env, self.cond_gain_energy):
                self.spotties[nth].energy += 1
            if self.spotties[nth].energy == self.split_energy:
                self.spotty_split_random(nth)
            else:
                decision = decide_by_intel(intel, env['tribe'])
                self.spotty_move(nth, decision)

    @ staticmethod
    def if_gain_energy(env2, cond_energy):
        return np.count_nonzero(env2 == 0) < cond_energy + 0.1


def game_v0():
    # iteration time
    time = 200
    # map size
    lx = 9
    ly = 9
    # properties of spotties
    ini_pos = [[4, 4]]
    max_age = 15
    split_energy = 5
    cond_gain_energy = 3
    # initial game
    info = {'tribe': 1}
    game = SpottiesGameV0(info, cond_gain_energy, lx, ly, max_age, split_energy)
    game.add_spotty_positions(info, ini_pos)

    save_intel_linear_random(8, 5, 'linear_intel.pr')
    intel1 = dict()
    intel1['type'] = 'linear'
    intel1['data'] = bfr.load_pr('.\\Intels\\linear_intel.pr', 'intel')
    intel = [intel1]

    for t in range(0, time):
        for n in range(game.spotties.__len__() - 1, -1, -1):
            game.update_one_spotty(n, intel[game.spotties[n].info['tribe'] - 1])
        game.universe.plot_universe()


