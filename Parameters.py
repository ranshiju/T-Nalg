import numpy as np
import Hamiltonian_Module as hm
from Basic_Functions_SJR import print_error, print_options, print_dict


def generate_parameters_dmrg():
    lattice = 'arbitrary'

    # =======================================================
    # No further changes are needed for these codes
    model = ['chain', 'square', 'arbitrary']
    if lattice is 'chain':
        para = parameter_dmrg_chain()
    elif lattice is 'square':
        para = parameter_dmrg_square()
    elif lattice is 'arbitrary':
        para = parameter_dmrg_arbitrary()
    else:
        para = None
        print_error('Wrong input of lattice!')
        print_options(model, welcome='Set lattice as one of the following:\t', quote='\'')
    para['nh'] = para['index2'].shape[0]  # number of two-body interactions
    return para
    # =======================================================


def common_parameters_dmrg():
    para = dict()
    para['chi'] = 16  # Virtual bond dimension cut-off
    para['d'] = 2  # Physical bond dimension
    para['sweep_time'] = 200  # sweep time
    # Fixed parameters
    para['if_print_detail'] = False
    para['tau'] = 1e-3  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-3
    para['break_tol'] = 1e-9  # tolerance for breaking the loop
    para['is_real'] = True
    para['dt_ob'] = 2  # in how many sweeps, observe to check the convergence
    para['ob_position'] = 0  # to check the convergence, chose a position to observe
    para['data_path'] = '.\\data_dmrg\\'
    return para


def parameter_dmrg_arbitrary():
    para = dict()
    para['lattice'] = 'choose_a_name'
    para['spin'] = 'half'

    # Do NOT change the first 6 elements of para['op'] (as well as the order), unless you know what you are doing
    # Otherwise, some problem may happen for computing the observables (such as magnetization or energy per site)
    # If you want more operators, just add them in the list
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]

    # The numbers of the spin operators are defined in op
    # For each row (say the m-th), it means one one-body terms where the site index1[m, 0] is acted by the
    #   operator op[index1[m, 1]]
    # The example below means the sz terms on the sites 0, 1, and 2
    para['index1'] = [
        [0, 1],
        [1, 1],
        [2, 1]
    ]
    para['coeff1'] = [0.1, 0.1, 0.1]

    # The numbers of the spin operators are defined in op
    # For each row (say the m-th), it means one two-body terms, where the site index1[m, 0] is with the
    #   operator op[index1[m, 2]], and the site index1[m, 1] is with the op[index1[m, 3]]
    # The example below means the sz.sz interactions between the 1st and second, as well as the
    #   second and third spins.
    para['index2'] = [
        [0, 1, 3, 3],
        [1, 2, 3, 3]
    ]
    para['coeff2'] = [1, 1, 1]
    para['data_exp'] = 'Put_here_your_file_name_to_save_data'

    # ====================================================================
    # No further changes are needed for these codes
    para['coeff1'] = np.array(para['coeff1']).reshape(-1, 1)
    para['coeff2'] = np.array(para['coeff2']).reshape(-1, 1)
    para['index1'] = np.array(para['index1'])
    para['index2'] = np.array(para['index2'])
    para['l'] = max(max(para['index1'][:, 0]), max(para['index2'][:, 0]), max(para['index2'][:, 1])) + 1
    para['positions_h2'] = from_index2_to_positions_h2(para['index2'])
    check_continuity_pos_h2(pos_h2=para['positions_h2'])
    para1 = common_parameters_dmrg()
    para = dict(para, **para1)  # combine with the common parameters
    return para
    # ====================================================================


def parameter_dmrg_chain():
    para = dict()
    para['spin'] = 'half'
    para['lattice'] = 'chain'
    para['bound_cond'] = 'open'  # open or periodic
    para['l'] = 12  # Length of MPS and chain
    para['spin'] = 'half'
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
    # in chain, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0

    # ====================================================================
    # No further changes are needed for these codes
    para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
    para['index1'] = np.mat(np.arange(0, para['l']))
    para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
    para['positions_h2'] = hm.positions_nearest_neighbor_1d(para['l'], para['bound_cond'])
    para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
    para['data_exp'] = 'chainN%d_j(%g,%g)_h(%g,%g)_chi%d' % \
                       (para['l'], para['jxy'], para['jz'], para['hx'],
                        para['hz'], para['chi']) + para['bound_cond']
    para1 = common_parameters_dmrg()
    para = dict(para, **para1)  # combine with the common parameters
    return para
    # ====================================================================


def parameter_dmrg_square():
    para = dict()
    para['lattice'] = 'square'
    para['bound_cond'] = 'open'
    para['square_width'] = 4  # width of the square lattice
    para['square_height'] = 4  # height of the square lattice
    para['l'] = para['square_width'] * para['square_height']
    para['spin'] = 'half'
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
    # in square, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0

    # ====================================================================
    # No further changes are needed for these codes
    para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
    para['index1'] = np.mat(np.arange(0, para['l']))
    para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
    para['positions_h2'] = hm.positions_nearest_neighbor_square(
        para['square_width'], para['square_height'], para['bound_cond'])
    para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
    para['data_exp'] = 'square' + '(%d,%d)' % (para['square_width'], para['square_height']) + \
                       'N%d_j(%g,%g)_h(%g,%g)_chi%d' % (para['l'], para['jxy'], para['jz'], para['hx'],
                                                        para['hz'], para['chi']) + para['bound_cond']
    para1 = common_parameters_dmrg()
    para = dict(para, **para1)  # combine with the common parameters
    return para
    # ====================================================================


# =================================================================
# Some function used here
def from_index2_to_positions_h2(index2):
    from DMRG_anyH import sort_positions
    pos_h2 = index2[:, :2]
    pos_h2 = sort_positions(pos_h2)
    new_pos = pos_h2[0, :].reshape(1, -1)
    for n in range(1, pos_h2.shape[0]):
        if not (pos_h2[n, 0] == new_pos[-1, 0] and pos_h2[n, 1] == new_pos[-1, 1]):
            new_pos = np.vstack((new_pos, pos_h2[n, :]))
    return new_pos


def check_continuity_pos_h2(pos_h2):
    p0 = np.min(pos_h2)
    if p0 != 0:
        exit('The numbering of sites should start with 0, not %d. Please revise the numbering.' % p0)
    p1 = np.max(pos_h2)
    missing_number = list()
    for n in range(p0+1, p1):
        if n not in pos_h2:
            missing_number.append(n)
    if missing_number.__len__() > 0:
        print_error('The pos_h2 is expected to contain all numbers from 0 to %d. The following numbers are missing:' % p1)
        print(str(missing_number))
        exit('Please check and revise you numbering')


def show_parameters(para):
    print_dict(para, welcome='The parameters are: \n', style_sep=':\n')
