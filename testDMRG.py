from DMRG_anyH import dmrg_finite_size, plot_finite_dmrg
import Parameters as Pm
import BasicFunctionsSJR as Bf


para = Pm.generate_parameters_dmrg('chain')
ob, A, info, para = dmrg_finite_size(para)
Bf.save_pr('.\\data_dmrg', para['data_exp']+'.pr', (ob, A, info, para), ('ob', 'A', 'info', 'para'))
plot_finite_dmrg('eb', A, para, ob)
plot_finite_dmrg('mag', A, para, ob)
plot_finite_dmrg('ent', A, para, ob)
