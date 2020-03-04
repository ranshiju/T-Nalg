from algorithms.DMRG_anyH import dmrg_finite_size
from library import Parameters as Pm
import numpy as np
from os import path
from library.BasicFunctions import load_pr, save_pr, plot, output_txt
from algorithms.DeepMPSfinite import parameters_dmps, deep_mps_qubit, fidelities_to_original_state
from library.TensorBasicModule import open_mps_product_state_spin_up


# ====================== Test ==========================
lattice = 'chain'
para_dmrg = Pm.generate_parameters_dmrg(lattice)
para_dmrg['spin'] = 'half'
para_dmrg['bound_cond'] = 'open'
para_dmrg['chi'] = 3
para_dmrg['l'] = 32
para_dmrg['jxy'] = 0
para_dmrg['jz'] = 1
para_dmrg['hx'] = 0.5
para_dmrg['hz'] = 0
para_dmrg = Pm.make_consistent_parameter_dmrg(para_dmrg)

para_dmps = parameters_dmps()
para_dmps['num_layers'] = 5
para_dmps['chi_overlap'] = 64
para_dmps['theta'] = None
para_dmps['num_theta'] = 100


if path.isfile(path.join(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr')):
    print('Load existing data ...')
    a, ob = load_pr(path.join(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr'), ['a', 'ob'])
else:
    ob, a, info, para = dmrg_finite_size(para_dmrg)
    save_pr(para['data_path'], para['data_exp'] + '.pr', (ob, a, info, para),
            ('ob', 'a', 'info', 'para'))
print('Energy per site = ' + str(ob['e_per_site']))
a.calculate_entanglement_spectrum()
a.calculate_entanglement_entropy()

fid, ent, lm_mid, mpo, mps = deep_mps_qubit(a, para_dmps)

lm = [a.lm[round(a.length/2)].reshape(-1, 1)]
lm_mat = np.ones((lm_mid[-1].size, lm_mid.__len__()+1))
lm_mat[:a.lm[round(a.length/2)].size, 0] = a.lm[round(a.length/2)]
for n in range(lm_mid.__len__()):
    lm_mat[:lm_mid[n].size, n+1] = lm_mid[n]
output_txt(np.log10(lm_mat), 'lm')
gap = np.log(lm_mat[1, :]) - np.log(lm_mat[0, :])
output_txt(gap, 'gap')

ent_av = [np.average(a.ent)]
for n in range(ent.__len__()):
    ent_av.append(np.average(ent[n]))
output_txt(ent_av, 'ent_av')

pos_mid = round(a.length/2)
ent_mid = [a.ent[pos_mid]]
for n in range(ent.__len__()):
    ent_mid.append(ent[n][pos_mid])
output_txt(ent_mid, 'ent_mid')

fid = fid[1:]
plot(fid)
output_txt(fid.reshape(-1, ), 'fid')

mps_ini = open_mps_product_state_spin_up(a.length, a.mps[0].shape[1])[0]
fid1 = fidelities_to_original_state(a, mps_ini, mpo, para_dmps['chi_overlap'])
plot(fid1)
output_txt(fid1.reshape(-1, ), 'fid1')

fid2 = fidelities_to_original_state(a, mps.mps, mpo, para_dmps['chi_overlap'])
plot(fid2)
output_txt(fid2.reshape(-1, ), 'fid2')

# exp = 'plot(np.arange(ent[0].size),'
# for n in range(0, ent.__len__(), 2):
#     exp += 'ent[' + str(n) + '],'
# exp = exp[:-1] + ')'
# exec(exp)

# ent = np.hstack(ent)
# output_txt(ent, 'ent')
