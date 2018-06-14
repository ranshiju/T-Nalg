def parameter_dmrg():
    para = dict()
    # Physical parameters
    para['lattice'] = 'chain'
    para['spin'] = 'half'
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['bound_cond'] = 'open'
    # Calculation parameters
    para['l'] = 14  # Length of MPS
    para['chi'] = 32  # Virtual bond dimension cut-off
    para['d'] = 2  # Physical bond dimension
    para['sweep_time'] = 200  # sweep time
    # Fixed parameters
    para['if_print_detail'] = False
    para['tau'] = 1e-3  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-3
    para['break_tol'] = 1e-9  # tolerance for breaking the loop
    para['is_real'] = True
    para['dt_ob'] = 2  # in how many sweeps, observe to check the convergence
    para['ob_position'] = (para['l'] / 2).__int__()  # to check the convergence, chose a position to observe
    para['data_path'] = '.\\data_dmrg\\'
    para['data_exp'] = 'N%d_j(%g,%g)_h(%g,%g)_chi%d' % \
                       (para['l'], para['jxy'], para['jz'], para['hx'],
                        para['hz'], para['chi']) + para['bound_cond']
    if para['lattice'] == 'square':
        para['data_exp'] = para['lattice'] + '(%d,%d)' % \
                           (para['square_width'], para['square_height']) + para['data_exp']
    else:
        para['data_exp'] = para['lattice'] + para['data_exp']
    return para
