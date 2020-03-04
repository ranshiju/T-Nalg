from algorithms.DMRG_anyH import dmrg_finite_size
from library import Parameters as Pm


para = Pm.generate_parameters_dmrg('husimi')
para['spin'] = 'half'
para['depth'] = 2
# The interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
para['jxy'] = 1
para['jz'] = 1
para['hx'] = 0
para['hz'] = 0
para['chi'] = 32  # Virtual bond dimension cut-off
para['eigWay'] = 1
para = Pm.make_consistent_parameter_dmrg(para)

ob, A, info1, para1 = dmrg_finite_size(para)
print(ob['e_per_site'])
