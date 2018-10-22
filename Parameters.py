import numpy as np
import HamiltonianModule as hm
from BasicFunctionsSJR import print_error, print_options, print_dict


def common_parameters_dmrg():
    # common parameters for finite-size DMRG
    para = dict()
    para['chi'] = 30  # Virtual bond dimension cut-off
    para['sweep_time'] = 100  # sweep time
    # Fixed parameters
    para['if_print_detail'] = False
    para['tau'] = 1e-4  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-5
    para['break_tol'] = 1e-8  # tolerance for breaking the loop
    para['is_real'] = True
    para['dt_ob'] = 4  # in how many sweeps, observe to check the convergence
    para['ob_position'] = 0  # to check the convergence, chose a position to observe

    para['eigWay'] = 1
    para['isParallel'] = False
    para['isParallelEnvLMR'] = False
    para['is_save_op'] = True
    para['data_path'] = '.\\data_dmrg\\'
    return para


def parameter_dmrg_arbitrary():
    para = dict()
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
        [0, 6],
        [1, 6],
        [2, 6]
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
    para['coeff2'] = [1, 1]
    para['lattice'] = 'arbitrary'
    para['data_exp'] = 'Put_here_your_file_name_to_save_data'
    return para
    # ====================================================================


def parameter_dmrg_chain():
    para = dict()
    para['spin'] = 'half'
    para['bound_cond'] = 'open'  # open or periodic
    para['l'] = 18  # Length of MPS and chain
    para['spin'] = 'half'
    # in chain, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'chain'
    return para
    # ====================================================================


def parameter_dmrg_jigsaw():
    para = dict()
    para['spin'] = 'one'
    para['bound_cond'] = 'open'  # open or periodic
    para['l'] = 21  # Length of MPS; odd for open boundary and even for periodic boundary
    # The interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 1
    para['jz'] = 1
    para['jxy1'] = 1
    para['jz1'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'jigsaw'
    return para
    # ====================================================================


def parameter_dmrg_square():
    para = dict()
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
    para['lattice'] = 'square'
    return para
# ====================================================================


def parameter_dmrg_full():
    para = dict()
    para['spin'] = 'half'
    para['l'] = 6  # Length of MPS and chain
    # in chain, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0
    para['lattice'] = 'full'
    return para


def parameter_dmrg_long_range():
    para = dict()
    para['spin'] = 'half'
    para['alpha'] = 1
    para['l'] = 6  # Length of MPS and chain
    # in chain, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0
    para['lattice'] = 'longRange'
    return para
    # ====================================================================


def generate_parameters_dmrg(lattice='chain'):
    # =======================================================
    # No further changes are needed for these codes
    model = ['chain', 'square', 'arbitrary', 'jigsaw', 'full', 'longRange']
    if lattice is 'chain':
        para = parameter_dmrg_chain()
    elif lattice is 'square':
        para = parameter_dmrg_square()
    elif lattice is 'arbitrary':
        para = parameter_dmrg_arbitrary()
    elif lattice is 'jigsaw':
        para = parameter_dmrg_jigsaw()
    elif lattice is 'full':
        para = parameter_dmrg_full()
    elif lattice is 'longRange':
        para = parameter_dmrg_long_range()
    else:
        para = dict()
        print_error('Wrong input of lattice!')
        print_options(model, welcome='Set lattice as one of the following:\t', quote='\'')
    para1 = common_parameters_dmrg()
    para = dict(para, **para1)  # combine with the common parameters
    para = make_consistent_parameter_dmrg(para)
    return para
# =======================================================


def make_consistent_parameter_dmrg(para):
    if para['lattice'] is 'chain':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_nearest_neighbor_1d(para['l'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'chainN%d_j(%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'], para['jxy'], para['jz'], para['hx'],
                            para['hz'], para['chi']) + para['bound_cond']
        para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            para['coeff2'][n * 3] = para['jxy'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2
            para['coeff2'][n * 3 + 2] = para['jz']
    elif para['lattice'] is 'square':
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_nearest_neighbor_square(
            para['square_width'], para['square_height'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'square' + '(%d,%d)' % (para['square_width'], para['square_height']) + \
                           'N%d_j(%g,%g)_h(%g,%g)_chi%d' % (para['l'], para['jxy'], para['jz'], para['hx'],
                                                            para['hz'], para['chi']) + para['bound_cond']
        para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            para['coeff2'][n * 3] = para['jxy'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2
            para['coeff2'][n * 3 + 2] = para['jz']
    elif para['lattice'] is 'arbitrary':
        para['coeff1'] = np.array(para['coeff1']).reshape(-1, 1)
        para['coeff2'] = np.array(para['coeff2']).reshape(-1, 1)
        para['index1'] = np.array(para['index1'])
        para['index2'] = np.array(para['index2'])
        para['l'] = max(max(para['index1'][:, 0]), max(para['index2'][:, 0]), max(para['index2'][:, 1])) + 1
        para['positions_h2'] = from_index2_to_positions_h2(para['index2'])
        check_continuity_pos_h2(pos_h2=para['positions_h2'])
    elif para['lattice'] is 'jigsaw':
        op = hm.spin_operators(para['spin'])
        if para['bound_cond'] is 'open':
            if para['l'] % 2 == 0:
                print('Note: for OBC jigsaw, l has to be odd. Auto-change l = %g to %g'
                      % (para['l'], para['l'] + 1))
                para['l'] += 1
        else:
            if para['l'] % 2 == 1:
                print('Note: for PBC jigsaw, l has to be even. Auto-change l = %g to %g'
                      % (para['l'], para['l'] + 1))
                para['l'] += 1
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_jigsaw_1d(para['l'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['l'] - (para['bound_cond'] is 'open')):
            para['coeff2'][n * 3] = para['jxy'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2
            para['coeff2'][n * 3 + 2] = para['jz']
        for n in range(para['l'] - (para['bound_cond'] is 'open'), para['positions_h2'].shape[0]):
            para['coeff2'][n * 3] = para['jxy1'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy1'] / 2
            para['coeff2'][n * 3 + 2] = para['jz1']
        para['data_exp'] = 'JigsawN%d_j(%g,%g,%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'], para['jxy'], para['jz'], para['jxy1'], para['jz1'], para['hx'],
                            para['hz'], para['chi']) + para['bound_cond']
    elif para['lattice'] is 'full':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_fully_connected(para['l'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'fullConnectedN%d_j(%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'], para['jxy'], para['jz'], para['hx'],
                            para['hz'], para['chi'])
        para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            para['coeff2'][n * 3] = para['jxy'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2
            para['coeff2'][n * 3 + 2] = para['jz']
    elif para['lattice'] is 'longRange':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_fully_connected(para['l'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'longRangeN%d_j(%g,%g)_h(%g,%g)_chi%d_alpha%g' % \
                           (para['l'], para['jxy'], para['jz'], para['hx'],
                            para['hz'], para['chi'], para['alpha'])
        para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            const = abs(para['positions_h2'][n, 0] - para['positions_h2'][n, 1])**(para['alpha'])
            para['coeff2'][n * 3] = para['jxy'] / 2 / const
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2 / const
            para['coeff2'][n * 3 + 2] = para['jz'] / const
    para['d'] = physical_dim_from_spin(para['spin'])
    para['nh'] = para['index2'].shape[0]  # number of two-body interactions
    return para


# =================================================================
# Parameters of infinite DMRG
def generate_parameters_infinite_dmrg():
    para = dict()
    para['spin'] = 'half'
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0

    para['n_site'] = 2
    para['chi'] = 16  # Virtual bond dimension cut-off
    para['d'] = 4  # Physical bond dimension (2-sites in one tensor)
    para['sweep_time'] = 200  # sweep time
    # Fixed parameters
    para['tau'] = 1e-3  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-10
    para['break_tol'] = 1e-8  # tolerance for breaking the loop
    para['is_symme_env'] = False
    para['is_real'] = True
    para['form'] = 'center_ort'
    para['dt_ob'] = 5  # in how many sweeps, observe to check the convergence
    para['data_path'] = '.\\data_idmrg\\'
    return para


# =================================================================
# Parameters of infinite DMRG
def generate_parameters_deep_mps_infinite():
    para = dict()
    para['spin'] = 'half'
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0

    para['n_site'] = 2 # n-site DMRG algorithm
    para['chi'] = 8  # Virtual bond dimension cut-off
    para['chib0'] = 4  # dimension cut-off of the uMPO (maximal para['chi'])
    para['chib'] = 4  # Virtual bond dimension for the secondary MPS
    para['d'] = 4  # Physical bond dimension (2-sites in one tensor)
    # Fixed parameters
    para['sweep_time'] = 200  # sweep time
    para['tau'] = 1e-3  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-12
    para['break_tol'] = 1e-9  # tolerance for breaking the loop
    para['is_symme_env'] = False
    para['is_real'] = True
    para['dt_ob'] = 5  # in how many sweeps, observe to check the convergence
    para['form'] = 'center_ort'
    para['data_path'] = '.\\data_idmrg\\'

    para['chib0'] = min(para['chi'], para['chib0'])
    return para


# =================================================================
# Parameters of super-orthogonalization of honeycomb model
def generate_parameters_so_honeycomb():
    para = dict()
    para['lattice'] = 'honeycomb0'
    para['state_type'] = 'mixed'
    para['spin'] = 'half'
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0

    para['chi'] = 12
    para['so_time'] = 20
    if para['state_type'] is 'pure':
        para['tau'] = [1e-1, 1e-2, 1e-3]
        para['beta'] = 10
        para['tol'] = 1e-7
        para['dt_ob'] = 10
        para['ini_way'] = 'random'
    elif para['state_type'] is 'mixed':
        para['tau'] = 1e-2
        para['beta'] = np.arange(0.1, 1.1, 0.1)
        para['ini_way'] = 'id'

    para['d'] = physical_dim_from_spin(para['spin'])
    if para['state_type'] is 'mixed':
        para['d'] *= 2
    para['if_print'] = True
    para['is_debug'] = False
    para['data_path'] = '.\\data_ipeps\\'
    return para


# =================================================================
# Parameters of tree DMRG of honeycomb lattice (square TN)
def generate_parameters_tre_dmrg_honeycomb_lattice():
    para = dict()
    para['lattice'] = 'honeycomb0'
    para['state_type'] = 'pure'
    para['spin'] = 'half'
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0

    para['chi'] = 12
    para['sweep_time'] = 20
    para['dt_ob'] = 4
    para['tau'] = [1e-1, 1e-2, 1e-3]
    para['tol'] = 1e-7

    para['d'] = physical_dim_from_spin(para['spin'])
    para['if_print'] = True
    para['data_path'] = '.\\data_ipeps\\'
    return para


def generate_parameters_mps_ml():
    para = dict()
    para['dataset'] = 'mnist'
    para['d'] = 2
    para['chi'] = 8
    para['sweep_time'] = 100
    para['tol'] = 1e-4
    return para


# =================================================================
# Some function used here that need not be modified
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


def physical_dim_from_spin(spin):
    if spin is 'half':
        return 2
    elif spin is 'one':
        return 3
    else:
        return False
