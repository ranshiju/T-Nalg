import numpy as np
from library import HamiltonianModule as hm
from library.BasicFunctions import print_error, print_options, print_dict, \
    project_path
from os.path import join


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
    return para


def parameter_dmrg_arbitrary():
    para = dict()
    para['spin'] = 'half'
    para['jxy'] = 0
    para['jz'] = 0
    para['hx'] = 0
    para['hz'] = 0

    # The numbers of the spin operators are defined in op
    # For each row (say the m-th), it means one one-body terms where the site index1[m, 0] is acted by the
    #   operator op[index1[m, 1]]
    # The example below means the sz terms on the sites 0, 1, and 2
    para['index1'] = [
        [0, 6],
        [1, 6],
        [2, 6]
    ]
    para['index1'] = np.array(para['index1'])
    para['coeff1'] = [1, 1, 1]
    para['coeff1'] = np.array(para['coeff1']).reshape(-1, 1)

    # The numbers of the spin operators are defined in op
    # For each row (say the m-th), it means one two-body terms, where the site index1[m, 0] is with the
    #   operator op[index1[m, 2]], and the site index1[m, 1] is with the op[index1[m, 3]]
    # The example below means the sz.sz interactions between the 1st and second, as well as the
    #   second and third spins.
    para['index2'] = [
        [0, 1, 3, 3],
        [1, 2, 3, 3]
    ]
    para['index2'] = np.array(para['index2'])
    para['coeff2'] = [1, 1]
    para['coeff2'] = np.array(para['coeff2']).reshape(-1, 1)
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
    para['bound_cond'] = 'open'
    para['is_pauli'] = True
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
    para['bound_cond'] = 'open'
    para['is_pauli'] = True
    para['lattice'] = 'longRange'
    return para


def parameter_dmrg_husimi():
    para = dict()
    para['spin'] = 'half'
    para['depth'] = 2  # Depth of three branches
    # in chain, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0
    para['is_pauli'] = True
    para['lattice'] = 'husimi'
    return para
# ====================================================================


def generate_parameters_dmrg(lattice='chain'):
    # =======================================================
    # No further changes are needed for these codes
    model = ['chain', 'square', 'arbitrary', 'jigsaw', 'full', 'longRange', 'husimi']
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
    elif lattice is 'husimi':
        para = parameter_dmrg_husimi()
    else:
        para = dict()
        print_error('Wrong input of lattice!')
        print_options(model, welcome='Set lattice as one of the following:\t', quote='\'')
    para1 = common_parameters_dmrg()
    para = dict(para, **para1)  # combine with the common parameters
    para = make_consistent_parameter_dmrg(para)
    para['project_path'] = project_path()
    para['data_path'] = para['project_path'] + '\\data_dmrg\\'
    return para
# =======================================================


def make_consistent_parameter_dmrg(para):
    if para['lattice'] is 'chain':
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
        para['l'] = max(max(para['index1'][:, 0]), max(para['index2'][:, 0]), max(para['index2'][:, 1])) + 1
        para['positions_h2'] = from_index2_to_positions_h2(para['index2'])
        check_continuity_pos_h2(pos_h2=para['positions_h2'])
    elif para['lattice'] is 'jigsaw':
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
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_fully_connected(para['l'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'fullConnectedN%d_j(%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'], para['jxy'], para['jz'], para['hx'],
                            para['hz'], para['chi'])
        if para['is_pauli']:
            para['coeff1'] = np.ones((para['l'], 1)) * 2
        else:
            para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            if para['is_pauli']:
                para['coeff2'][n * 3] = para['jxy'] * 2
                para['coeff2'][n * 3 + 1] = para['jxy'] * 2
                para['coeff2'][n * 3 + 2] = para['jz'] * 4
            else:
                para['coeff2'][n * 3] = para['jxy'] / 2
                para['coeff2'][n * 3 + 1] = para['jxy'] / 2
                para['coeff2'][n * 3 + 2] = para['jz']
    elif para['lattice'] is 'longRange':
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_fully_connected(para['l'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'longRange' + para['bound_cond'] + 'N%d_j(%g,%g)_h(%g,%g)_chi%d_alpha%g' % \
                           (para['l'], para['jxy'], para['jz'], para['hx'],
                            para['hz'], para['chi'], para['alpha'])
        if para['is_pauli']:
            para['coeff1'] = np.ones((para['l'], 1)) * 2
        else:
            para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            if para['bound_cond'] is 'open':
                dist = abs(para['positions_h2'][n, 0] - para['positions_h2'][n, 1])
            else:  # periodic
                dist = min(abs(para['positions_h2'][n, 0] - para['positions_h2'][n, 1]),
                           para['l'] - abs(para['positions_h2'][n, 0] - para['positions_h2'][n, 1]))
            const = dist**(para['alpha'])
            if para['is_pauli']:
                para['coeff2'][n * 3] = para['jxy'] * 2 / const
                para['coeff2'][n * 3 + 1] = para['jxy'] * 2 / const
                para['coeff2'][n * 3 + 2] = para['jz'] * 4 / const
            else:
                para['coeff2'][n * 3] = para['jxy'] / 2 / const
                para['coeff2'][n * 3 + 1] = para['jxy'] / 2 / const
                para['coeff2'][n * 3 + 2] = para['jz'] / const
    elif para['lattice'] is 'husimi':
        para['positions_h2'] = hm.positions_husimi(para['depth'])
        para['l'] = para['positions_h2'].max() + 1
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'HusimiDepth%d_j(%g,%g)_h(%g,%g)_chi%d' % \
                           (para['depth'], para['jxy'], para['jz'], para['hx'],
                            para['hz'], para['chi'])
        para['coeff1'] = np.ones((para['l'], ))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, ))
        for n in range(0, para['positions_h2'].shape[0]):
            para['coeff2'][n * 3] = para['jxy'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2
            para['coeff2'][n * 3 + 2] = para['jz']
    para['d'] = physical_dim_from_spin(para['spin'])
    para['nh'] = para['index2'].shape[0]  # number of two-body interactions
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
    para['op'].append(-para['hx'] * op['sx'] - para['hz'] * op['sz'])
    return para


# =================================================================
# Parameters of infinite DMRG
def generate_parameters_standard_tebd(lattice='chain'):
    para = dict()
    para['spin'] = 'half'
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['l'] = 12
    para['chi'] = 32
    para['bound_cond'] = 'open'

    para['tau0'] = 1e-1
    para['dtau'] = 0.1
    para['taut'] = 3
    para['dt_ob'] = 10
    para['iterate_time'] = 5000

    para['lattice'] = lattice
    para['save_mode'] = 'final'  # 'final': only save the converged result; 'all': save all results
    para['if_break'] = True  # if break with certain tolerance
    para['break_tol'] = 1e-7

    para['data_path'] = '.\\data_tebd\\'
    return make_para_consistent_tebd(para)


def make_para_consistent_tebd(para):
    para['positions_h2'] = hm.positions_nearest_neighbor_1d(para['l'], para['bound_cond'])
    para['num_h2'] = para['positions_h2'].shape[0]
    para['d'] = physical_dim_from_spin(para['spin'])
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
    para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
    para['ob_time'] = int(para['iterate_time'] / para['dt_ob'])
    return para


def generate_parameters_tebd_any_h(lattice='any', dims=None):
    para = dict()
    # this corresponds to the function "interactions_tebd_any_h". Define more cases as you want.
    para['case'] = 'QES_1D'
    # Parameters of physical Hamiltonian
    para['spin'] = 'half'
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    # total length and virtual bond dimensions
    para['l'] = 12
    para['chi'] = 32

    para['tau0'] = 1e-1
    para['dtau'] = 0.1
    para['taut'] = 3
    para['dt_ob'] = 10
    para['iterate_time'] = 5000

    para['lattice'] = lattice
    para['save_mode'] = 'final'  # 'final': only save the converged result; 'all': save all results
    para['if_break'] = True  # if break with certain tolerance
    para['break_tol'] = 1e-7

    para['d'] = physical_dim_from_spin(para['spin'])
    para['dbath'] = para['d']
    # dims: the dimensions of the physical bonds of the MPS
    if dims is None:
        para['phys_dims'] = dims
    else:
        para['phys_dims'] = [para['d']] * para['l']
    para['data_path'] = '.\\data_tebd\\'
    return interactions_tebd_any_h(para)


def interactions_tebd_any_h(para):
    if para['case'] is 'QES_1D':
        para['phys_dims'] = [para['dbath']] + list(np.ones((para['l']-2,))) + [para['dbath']]
        # Note: hamilt = [h_phys, h_bath_left, h_bath_right]
        para['MPO_pos'] = list()
        para['MPO_coup'] = list()
        # Physical parts
        para['MPO_pos'].append(list(np.arange(1, para['l'] - 1)))
        para['MPO_coup'].append(list(np.zeros((para['l-3'], ))))
        # Bath parts
        para['MPO_pos'].append([0, 1])
        para['MPO_coup'].append([1])
        para['MPO_pos'].append([para['l-2'], para['l-1']])
        para['MPO_coup'].append([2])
        para['phys_sites'] = list(range(1, para['l'] + 1))
        para['phys_coup'] = np.hstack((np.arange(1, para['l']).reshape(-1, 1),
                                       np.arange(2, para['l']+1).reshape(-1, 1),
                                       np.zeros((para['l']-2, 1))))
    return para


# =================================================================
# Parameters of infinite DMRG
def generate_parameters_infinite_dmrg():
    para = dict()
    para['dmrg_type'] = 'mpo'

    para['spin'] = 'half'
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0

    para['n_site'] = 2
    para['chi'] = 16  # Virtual bond dimension cut-off
    para['sweep_time'] = 1000  # sweep time
    # Fixed parameters
    para['tau'] = 1e-4  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-15
    para['break_tol'] = 2e-10  # tolerance for breaking the loop
    para['is_symme_env'] = False
    para['is_real'] = True
    para['form'] = 'center_ort'
    para['dt_ob'] = 10  # in how many sweeps, observe to check the convergence
    para['data_path'] = '.\\data_idmrg\\'
    return make_para_consistent_idmrg(para)


def make_para_consistent_idmrg(para):
    if para['dmrg_type'] not in ['mpo', 'white']:
        print('Bad para[\'d,rg_type\']. Set to \'white\'')
        para['dmrg_type'] = 'white'
    if para['dmrg_type'] is 'white':
        print('Warning: dmrg_type==\'white\' only suits for nearest-neighbor chains')
        print('In this mode, self.is_symme_env is set as True')
        para['is_symme_env'] = True
    para['model'] = 'heisenberg'
    para['hamilt_index'] = hm.hamiltonian_indexes(
        para['model'], (para['jxy'], para['jz'], -para['hx']/2, -para['hz']/2))
    para['d'] = physical_dim_from_spin(para['spin'])
    return para


def generate_parameters_infinite_dmrg_sawtooth():
    para = dict()
    para['dmrg_type'] = 'white'

    para['spin'] = 'half'
    para['j1'] = 0
    para['j2'] = 1
    para['hx'] = 0.5
    para['hz'] = 0

    para['n_site'] = 3
    para['chi'] = 16  # Virtual bond dimension cut-off
    para['sweep_time'] = 1000  # sweep time
    # Fixed parameters
    para['tau'] = 1e-4  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-15
    para['break_tol'] = 2e-10  # tolerance for breaking the loop
    para['is_symme_env'] = False
    para['is_real'] = True
    para['form'] = 'center_ort'
    para['dt_ob'] = 10  # in how many sweeps, observe to check the convergence
    para['data_path'] = '.\\data_idmrg\\'
    return para


def make_para_consistent_idmrg_sawtooth(para):
    para['model'] = 'heisenberg'
    para['d'] = physical_dim_from_spin(para['spin'])
    return para


def generate_parameters_tree_ipeps_kagome():
    para = dict()

    para['spin'] = 'half'
    para['j1'] = 0  # up-triangle
    para['j2'] = 1  # down-triangle
    para['hx'] = 0.5
    para['hz'] = 0

    para['chi'] = 16  # Virtual bond dimension cut-off
    para['sweep_time'] = 1000  # sweep time
    # Fixed parameters
    para['tau'] = 1e-4  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-15
    para['break_tol'] = 2e-10  # tolerance for breaking the loop
    para['is_symme_env'] = True
    para['is_real'] = True
    para['form'] = 'center_ort'
    para['dt_ob'] = 4  # in how many sweeps, observe to check the convergence
    para['data_path'] = '.\\data_idmrg\\'
    return para


def make_para_consistent_tree_ipeps_kagome(para):
    para['model'] = 'heisenberg'
    para['d'] = physical_dim_from_spin(para['spin'])
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


def parameters_lazy_learning():
    para = dict()
    para['dataset'] = 'mnist'
    para['classes'] = list(range(10))
    para['num_samples'] = ['all'] * para['classes'].__len__()
    para['d'] = 2
    para['theta'] = 1  # the unit is pi/2
    return para


def parameters_gcmpm():
    para = dict()
    para['dataset'] = 'mnist'
    para['classes'] = [0, 1]
    para['dct'] = False

    para['chi'] = [8, 8]
    para['d'] = 2
    para['theta'] = 1  # unit is pi/2
    para['step'] = 0.1  # gradient step
    para['step_ratio'] = 0.2  # how the step is reduced
    para['step_min'] = 1e-5  # minimal gradient step
    para['ratio_step_tol'] = 5  # Suggested value: 1
    # para['break_tol'] = para['step'] * para['step_ratio']
    para['sweep_time'] = 60
    para['check_time0'] = 1
    para['check_time'] = 1

    # shift and normalization factor for DCT
    para['shift'] = -10
    para['factor'] = 25

    para['if_save'] = True
    para['if_load'] = True
    para['data_path'] = '..\\data_tnml\\gcmpm\\'
    para['if_print_detail'] = True
    return para


def parameters_gtn_one_class():
    para = dict()
    para['dataset'] = 'mnist'
    para['class'] = 0
    para['dct'] = False
    para['d'] = 2
    para['chi'] = 8
    para['theta'] = 1  # unit is pi/2
    para['step'] = 1e-2  # gradient step
    para['step_ratio'] = 0.5  # how the step is reduced
    para['step_min'] = 1e-4  # minimal gradient step
    para['ratio_step_tol'] = 5
    # para['break_tol'] = para['step'] * para['ratio_step_tol']
    para['sweep_time'] = 20
    para['check_time0'] = 0
    para['check_time'] = 2
    para['break_tol'] = 1e-6
    para['project_path'] = project_path()
    para['data_path'] = join(para['project_path'], 'data_tnml\\gcmpm\\')

    # For para['dataset'] is 'custom'

    para['if_save'] = True
    para['if_load'] = True
    para['if_print_detail'] = True
    return para


def parameters_labeled_gtn():
    para = dict()
    para['dataset'] = 'mnist'
    para['class'] = [0, 1]
    para['d'] = 2
    para['chi'] = 8
    para['theta'] = 1  # unit is pi/2
    para['step'] = 1e-2  # gradient step
    para['step_ratio'] = 0.5  # how the step is reduced
    para['step_min'] = 1e-4  # minimal gradient step
    para['ratio_step_tol'] = 5
    # para['break_tol'] = para['step'] * para['ratio_step_tol']
    para['sweep_time'] = 20
    para['check_time0'] = 0
    para['check_time'] = 2
    para['break_tol'] = 1e-6
    para['data_path'] = '..\\data_tnml\\gcmpm\\'

    para['parallel'] = False
    para['n_nodes'] = 4

    para['if_save'] = True
    para['if_load'] = True
    para['if_print_detail'] = True
    return para


def parameters_decision_mps():
    para = dict()
    para['dataset'] = 'mnist'
    para['classes'] = [0, 1]
    para['numbers'] = [5, 5]
    para['chi'] = 2
    para['if_reducing_samples'] = False

    para['n_local'] = 1  # how many pixels are input in each tensor
    para['order'] = 'normal'  # 'normal' or 'random'
    return para


# =================================================================
# Parameters of ED algorithms
def parameters_ed_time_evolution(lattice):
    para = dict()
    para['spin'] = 'half'
    para['jx'] = 1
    para['jy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['l'] = 12
    para['tau'] = 1e-2
    para['time'] = 10
    para['dt_ob'] = 0.2
    para['bound_cond'] = 'open'

    para['lattice'] = lattice
    para['task'] = 'TE'  # Time Evolution
    para = make_para_consistent_ed(para)
    return para


def parameters_ed_ground_state(lattice):
    para = dict()
    para['spin'] = 'half'
    para['jx'] = 1
    para['jy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['l'] = 12
    para['tau'] = 1e-4  # Hamiltonian will be shifted as I-tau*H
    para['bound_cond'] = 'open'

    para['task'] = 'GS'
    para['lattice'] = lattice
    para = make_para_consistent_ed(para)
    return para


def make_para_consistent_ed(para):
    para['positions_h2'] = hm.positions_nearest_neighbor_1d(para['l'], para['bound_cond'])
    para['num_h2'] = para['positions_h2'].shape[0]
    tmp = np.zeros((para['num_h2'], 1), dtype=int)
    para['couplings'] = np.hstack((para['positions_h2'], tmp))
    para['d'] = physical_dim_from_spin(para['spin'])
    para['pos4corr'] = np.hstack((np.zeros((para['l'] - 1, 1), dtype=int), np.arange(
        1, para['l'], dtype=int).reshape(-1, 1)))
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
    para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
    if para['task'] is not 'GS':
        para['iterate_time'] = int(para['time'] / para['tau'])
    return para


def parameter_qes_gs_by_ed():
    para = generate_parameters_infinite_dmrg()
    para['task'] = 'gs'
    para['spin'] = 'half'
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0
    para['l_phys'] = 6  # number of physical sites in the bulk

    para['chi'] = 16  # Virtual bond dimension cut-off
    para['if_load_bath'] = True
    para = make_para_consistent_qes(para)
    return para


def parameter_qes_ft_by_ed():
    para = generate_parameters_infinite_dmrg()
    para['task'] = 'ft'
    para['spin'] = 'half'
    para['beta'] = np.arange(0.1, 1, 0.1)
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0
    para['l_phys'] = 6  # number of physical sites in the bulk

    para['chi'] = 16  # Virtual bond dimension cut-off
    para['if_load_bath'] = True
    para = make_para_consistent_qes(para)
    return para


def make_para_consistent_qes(para):
    para = make_para_consistent_idmrg(para)
    para['phys_sites'] = list(range(1, para['l_phys']+1))
    para['bath_sites'] = [0, para['l_phys']+1]
    para['positions_h2'] = hm.positions_nearest_neighbor_1d(para['l_phys']+2, 'open')
    para['num_h2'] = para['positions_h2'].shape[0]
    tmp = np.zeros((para['num_h2'], 1), dtype=int)
    tmp[0] = 1
    tmp[-1] = 2
    para['couplings'] = np.hstack((para['positions_h2'], tmp))
    para['d'] = physical_dim_from_spin(para['spin'])
    para['pos4corr'] = np.hstack((np.ones((para['l_phys']-1, 1), dtype=int), np.arange(
        2, para['l_phys']+1, dtype=int).reshape(-1, 1)))
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
    para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
    para['data_path'] = '../data_qes/results/'
    para['data_exp'] = para['task'] + 'QES_ED_chainL(%d,2)_j(%g,%g)_h(%g,%g)_chi%d' % (
        para['l_phys'], para['jxy'], para['jz'], para['hx'], para['hz'], para['chi'])
    para['bath_path'] = '../data_qes/bath/'
    para['bath_exp'] = 'bath_chain_j(%g,%g)_h(%g,%g)_chi%d' % (
        para['jxy'], para['jz'], para['hx'], para['hz'], para['chi'])
    if para['dmrg_type'] is 'white':
        para['data_exp'] += '_white'
        para['bath_exp'] += '_white'
    para['data_exp'] += '.pr'
    para['bath_exp'] += '.pr'
    return para


def parameters_qjpg():
    para = dict()
    para['file'] = 'hubble_deep_field'
    para['tasks'] = ['real', 'freq']  # ['real', 'freq', 'recover']
    para['recover_way'] = 'generative'  # 'normal' or 'generative'
    para['block_size'] = [8, 8]
    para['chi'] = 16
    para['pixel_cutoff'] = 20
    para['reorder_time'] = 2
    para['if_load'] = True
    para['data_path'] = '../data_QJPG/'
    return para


# =================================================================
# Some function used here that need not be modified
def from_index2_to_positions_h2(index2):
    from algorithms.DMRG_anyH import sort_positions
    pos_h2 = index2[:, :2]
    pos_h2 = sort_positions(pos_h2)
    new_pos = pos_h2[0, :].reshape(1, -1)
    for n in range(1, pos_h2.shape[0]):
        if not (pos_h2[n, 0] == new_pos[-1, 0] and pos_h2[n, 1] == new_pos[-1, 1]):
            new_pos = np.vstack((new_pos, pos_h2[n, :]))
    return np.array(new_pos, dtype=int)


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
