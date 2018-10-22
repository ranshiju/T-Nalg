import TensorBasicModule as T_module
import numpy as np
from ipdb import set_trace
from scipy.sparse.linalg import eigsh as eigs
from scipy.sparse.linalg import LinearOperator as LinearOp
from HamiltonianModule import spin_operators
from BasicFunctionsSJR import empty_list, trace_stack, sort_list, print_error, print_sep, \
    print_options, print_dict, info_contact
from termcolor import colored, cprint


class MpsBasic:

    def __init__(self):
        self.version = '2018-06-3'
        self.operators = list()

    def append_operators(self, op_new):
        if type(op_new) is np.ndarray:
            self.operators.append(op_new)
        else:
            for n in range(0, len(op_new)):
                self.operators.append(op_new[n])


class MpsOpenBoundaryClass(MpsBasic):
    """ Create an open-boundary MPS
    Example: create an MPS with 8 sites
        >>> length = 8  # number of sites/tensors
        >>> d = 2  # physical bond dimension
        >>> chi = 10  # virtual bond dimension
        >>> a = MpsOpenBoundaryClass(length, d, chi, way='svd', ini_way='r')
    * Other inputs:
        way: 'svd' (default) or 'qr'. When decomposing the tensors, use SVD or QR decomposition
        ini_way: 'r' or 'q'. When initializing the MPS, use numpy.random.randn or numpy.ones
        debug: True or False. Whether or not (default) in the debug mode.
    For some general information, use self.print_general_info()
        >>> a.print_general_info()
    The help documentation for the member functions are to be added
    Note: with large size of lattice, or large bond dimenion cut-off, you may play with the following parameters
    to reach the optimal efficiency (for small parameter, the default is the optimal choice):
        1. is_parallel = True or False (default). If it is true, parallel computing will be used. It costs slightly
        more memory. Parallel computing will in theory improve the efficiency with a large number of interaction
        terms (e.g., long-range or fully-connected models)
        2. is_save_op = True or False (default).  If it is true, effective operators will be save for reusing. It
        costs more memory. Saving effective operators will in theory improve the efficiency with a large lattice
        size (e.g., long chains)
        3. eig_way = 0 (default) or 1. With eig_way = 0, the full effective Hamiltonians will be calculated to put
        in eigs function. With eig_way = 1, the full effective Hamiltonians is avoided to (possibly) reduce the memory
        cost, and an operator is defined to put in eigs function. Recommend to use the first with relatively a large
        number of interaction terms, and use the second with a relatively large bond dimension cut-off
    """
    def __init__(self, length, d, chi, spin='half', way='qr', ini_way='r', operators=None,
                 debug=False, is_parallel=False, is_save_op=False, eig_way=0, par_pool=None,
                 is_env_parallel_lmr=True):
        MpsBasic.__init__(self)
        self.spin = spin
        self.phys_dim = d
        self.decomp_way = way  # 'svd' or 'qr'
        self.length = length
        # self.orthogonality:  -1: left2right; 0: not orthogonal or center; 1: right2left
        self.orthogonality = np.zeros((length, 1))
        self.center = -1  # orthogonal center; -1 means no center
        self.lm = empty_list(length-1, np.zeros(0))
        self.ent = np.zeros((self.length-1, 1))
        if ini_way == 'r':  # randomly initialize MPS
            self.mps = T_module.random_open_mps(length, d, chi)
        elif ini_way == '1':  # initialize MPS as eyes
            self.mps = T_module.ones_open_mps(length, d, chi)
        self.virtual_dim = np.ones((length + 1,)).astype(int) * chi
        self.virtual_dim[0] = 1
        self.virtual_dim[-1] = 1
        if operators is None:
            op_half = spin_operators(spin)
            self.operators = [op_half['id'], op_half['sx'], op_half['sy'], op_half['sz'], op_half['su'],
                              op_half['sd']]
        else:
            self.operators = operators

        self._is_save_op = is_save_op  # whether saving all effective operators to accelerate the code
        self.effect_s = {'none': np.zeros(0)}
        self.pos_effect_s = np.zeros((0, 3)).astype(int)
        self.effect_ss = {'none': np.zeros(0)}
        self.pos_effect_ss = np.zeros((0, 5)).astype(int)
        self.effective_id = {'none': np.zeros(0)}
        self.opt_env = dict()

        self._is_parallel = is_parallel
        self.pool = par_pool
        self._debug = debug  # if in debug mode
        self.eig_way = eig_way
        self._is_env_parallel_lmr = is_env_parallel_lmr

        if self._is_parallel and self._is_save_op:
            cprint('Note: this version forbids to use parallel computing while in the is_save_op mode', 'cyan')
            cprint('The is_save_op mode has been automatically switched off', 'magenta')
            cprint('This issue will be fixed in the next version', 'cyan')
            self._is_save_op = False
        if debug:
            cprint('Note: you are in the debug mode', 'cyan')
        if self._is_save_op:
            cprint('Note: you are in the is_save_op mode. The code will save intermediate results to accelerate '
                   'the computation', 'cyan')
        if self._is_parallel:
            cprint('Note: you are using parallel computing. The parallel computing will be used when '
                   'computing the environment for different different coupling terms', 'cyan')

    def print_general_info(self):
        print_sep('DMRG & MPS Documentation (%s)' % self.version, style='#')
        print('Install the following modules/libs before using: ')
        print_options(['numpy', 'scipy', 'matplotlib'], welcome='\t', style_sep='.', end='\n\t', color='magenta')
        print_sep('For using EasyStartDMRG (v2018.06-1)')
        print('* To use EasyStartDMRG, you only need to know three things:')
        print_options(['What you are simulating (e.g., Heisenber model, entanglement, ect.)', 'How to run a Python code',
                       'English'], welcome='\t', style_sep='.', end='\n\t')
        cprint('\t* It is ok if you may not know how DMRG works')
        print('* Steps to use EasyStartDMRG: ')
        print_options(['Run \'EasyStartDMRG\'', 'Input the parameters by following the instructions',
                       'Choose the quantities you are interested in'], welcome='\t', style_sep='.', end='\n\t',)
        print('Some notes:')
        print_options(['Your parameters are saved in \'.\\para_dmrg\_para.pr\'',
                       'To read *.pr, use function \'load_pr\' in \'Basic_functions_SJR.py\'',
                       'The results including the MPS will be save in \'.\\data_dmrg\''
                       ], welcome='\t', style_sep='.', end='\n\t',)
        print_sep('Contact Information')
        print_dict(info_contact(), ['name', 'affiliation', 'email'])

    def report_yourself(self):
        """
        Print some relevant information of the current MPS
        Example:
            >>> a = MpsOpenBoundaryClass(8, 2, 4)
            >>> a.report_yourself()
        """
        print('center: ' + str(self.center))
        print('orthogonality:' + str(self.orthogonality.T))
        print('virtual bond dimensions: ' + str(self.virtual_dim))
        for n in range(0, self.length-1):
            print('lm[%d] = ' % n + str(self.lm[n]))
        for n in range(0, self.length-1):
            print('ent[%d] = ' % n + str(self.ent[n]))

    def orthogonalize_mps(self, l0, l1):
        """
        Orthogonalization of the MPS from l0-th to l1-th tensors. Note that from l0-th to (l1-1)-th, the tensor will
        be left-to-right orthogonal (l0<l1) or right-to-left orthogonal (l0>l1)
        :param l0: starting pointing of the orthogonalization
        :param l1: ending pointing of the orthogonalization
        """
        if l0 < l1:  # Orthogonalize MPS from left to rigth
            for n in range(l0, l1):
                self.mps[n], mat, self.virtual_dim[n+1], lm = \
                    T_module.left2right_decompose_tensor(self.mps[n], self.decomp_way)
                if lm.size > 0 and self.center > -1:
                    self.lm[n] = lm.copy()
                self.mps[n+1] = T_module.absorb_matrix2tensor(self.mps[n + 1], mat, 0)
            self.orthogonality[l0:l1] = -1
            self.orthogonality[l1] = 0
        elif l0 > l1:  # Orthogonalize MPS from right to left
            for n in range(l0, l1, -1):
                self.mps[n], mat, self.virtual_dim[n], lm =\
                    T_module.right2left_decompose_tensor(self.mps[n], self.decomp_way)
                if lm.size > 0 and self.center > -1:
                    self.lm[n-1] = lm.copy()
                self.mps[n-1] = T_module.absorb_matrix2tensor(self.mps[n - 1], mat, 2)
            self.orthogonality[l0:l1:-1] = 1
            self.orthogonality[l1] = 0

    # transfer the MPS into the central orthogonal form with the center lc
    def central_orthogonalization(self, lc, l0=0, l1=-1):
        """
        Transform the MPS into central orthogonal form, with lc the center. Note you can specify the starting
        and ending point of the orthogonalization process
        :param lc: new center
        :param l0: starting point (0 as default)
        :param l1: ending point (self.length-1 as default)
        Warning: * if the MPS is not central orthogonal, this function may not transform it into a orthogonal form
                 * To central-orthogonalize the MPS or change the center, recommend to use correct_orthogonal_center
        Example:
        """
        if l1 == -1:
            l1 = self.length-1
        self.orthogonalize_mps(l0, lc)
        self.orthogonalize_mps(l1, lc)
        self.center = lc

    # move the orthogonal center at p
    def correct_orthogonal_center(self, p=-1):
        # if p<0 (default) and there is no center, automatically find a new center
        if p < -0.5 and self.center < -0.5:
            p = self.check_orthogonal_center(if_print=False)
        elif p < -0.5:
            p = self.center
        if self.center < -0.5:
            self.central_orthogonalization(p)
        elif self.center != p:
            self.orthogonalize_mps(self.center, p)
        self.center = p

# ===========================================================
# For handling effective operators in the fast mode
    @ staticmethod
    def key_effective_operators(info):
        # generate the key of one-body or two-body effective operator
        # info = (sn, ssn, p0, q0, p1)
        x = ''
        for n in range(0, info.__len__() - 1):
            x += str(info[n]) + '_'
        x += str(info[-1])
        return x

    @ staticmethod
    def key_restore_info(key):
        return key.split('_')

    def add_key_and_pos(self, which_op, key_info, op):
        key = self.key_effective_operators(key_info)
        if which_op is 'one':
            if key not in self.effect_s:
                self.pos_effect_s = np.vstack((self.pos_effect_s, np.array(key_info)))
            self.effect_s[key] = op
        elif which_op is 'two':
            if key not in self.effect_ss:
                self.pos_effect_ss = np.vstack((self.pos_effect_ss, np.array(key_info)))
            self.effect_ss[key] = op

    def find_nearest_key_one_body(self, sn, p0, p1):
        pos = self.pos_effect_s[self.pos_effect_s[:, 0] == sn, :]
        pos = pos[pos[:, 1] == p0, :]
        if p0 < p1:  # RG flow: left to right
            pos = pos[pos[:, 2] < p1, :]
            pos = pos[pos[:, 2] > p0, :]
            if pos.size == 0:
                p_before = None
                key_info = (sn, p0, p0+1)
            else:
                n = np.argmax(pos[:, 2])
                p_before = pos[n, 2]
                key_info = tuple(pos[n, :])
        else:
            pos = pos[pos[:, 2] > p1, :]
            pos = pos[pos[:, 2] <= p0, :]
            if pos.size == 0:
                p_before = None
                key_info = (sn, p0, p0)
            else:
                n = np.argmin(pos[:, 2])
                p_before = pos[n, 2]
                key_info = tuple(pos[n, :])
        return key_info, p_before

    def get_effective_operators_one_body(self, sn, p0, p1, is_update_op=True):
        # sn: which operator
        # p0: original position of the operator (site)
        # p1: position of the target effective operator (bond)
        key = self.key_effective_operators((sn, p0, p1))
        if key in self.effect_s:
            return self.effect_s[key]
        else:
            key_info, p_before = self.find_nearest_key_one_body(sn, p0, p1)
            if p_before is None:
                if p0 < p1:
                    if self.center < p0:
                        v = self.effective_id[str(self.center) + '_' + str(p0)]
                        v = T_module.bound_vec_operator_left2right(self.mps[p0], self.operators[sn], v=v)
                    else:
                        v = T_module.bound_vec_operator_left2right(self.mps[p0], self.operators[sn])
                    if is_update_op:
                        self.add_key_and_pos('one', key_info, v)
                    v = self.update_effect_op_l0_to_l1(p0+1, p1, v, sn, p0, is_update_op=is_update_op)
                else:
                    if self.center > p0:
                        v = self.effective_id[str(self.center) + '_' + str(p0+1)]
                        v = T_module.bound_vec_operator_right2left(self.mps[p0], self.operators[sn], v=v)
                    else:
                        v = T_module.bound_vec_operator_right2left(self.mps[p0], self.operators[sn])
                    if is_update_op:
                        self.add_key_and_pos('one', key_info, v)
                    v = self.update_effect_op_l0_to_l1(p0-1, p1-1, v, sn, p0, is_update_op=is_update_op)
            else:
                key_before = self.key_effective_operators(key_info)
                if p0 < p1:
                    v = self.update_effect_op_l0_to_l1(p_before, p1, self.effect_s[key_before],
                                                       sn, p0, is_update_op=is_update_op)
                else:
                    v = self.update_effect_op_l0_to_l1(p_before-1, p1-1, self.effect_s[key_before],
                                                       sn, p0, is_update_op=is_update_op)
            return v

    def get_effective_operator_two_body(self, sn, snn, p0, q0, p1, is_update_op=True):
        # the self.operators[sn]  is originally at p0-th site
        # the self.operators[ssn] is originally at q0-th site
        # the effective two-body operator is at the p1-th bond
        # here, we have p0 < q0 <= p1, or p1 >= q0 > p0 (on the same side of the RG endpoint)
        if p0 > q0:  # make sure p0 < q0
            p0, q0 = q0, p0
            sn, snn = snn, sn
        key2 = self.key_effective_operators((sn, snn, p0, q0, p1))
        if key2 in self.effect_ss:
            return self.effect_ss[key2]
        elif q0 == p1:
            print_error('LogicBug detected: please check')
        else:
            return self.update_effect_from_op1_to_op2(sn, snn, p0, q0, p1, is_update_op=is_update_op)

    def del_bad_effective_operators(self, p):
        # delete the badly defined effective operators due to the change of the p-th tensor
        if self.pos_effect_s.shape[0] > 0:
            ind = (self.pos_effect_s[:, 1] < p) * (self.pos_effect_s[:, 2] <= p)
            ind += (self.pos_effect_s[:, 1] > p) * (self.pos_effect_s[:, 2] > p)
            ind_del = (~ ind)
            pos_del = self.pos_effect_s[ind_del, :]
            for n in range(0, pos_del.shape[0]):
                key = self.key_effective_operators(tuple(pos_del[n, :]))
                self.effect_s.__delitem__(key)
            self.pos_effect_s = self.pos_effect_s[ind, :]
        if self.pos_effect_ss.shape[0] > 0:
            ind = (self.pos_effect_ss[:, 3] < p) * (self.pos_effect_ss[:, 4] <= p)
            ind += (self.pos_effect_ss[:, 2] > p) * (self.pos_effect_ss[:, 4] > p)
            ind_del = (~ ind)
            pos_del = self.pos_effect_ss[ind_del, :]
            for n in range(0, pos_del.shape[0]):
                key = self.key_effective_operators(tuple(pos_del[n, :]))
                self.effect_ss.__delitem__(key)
            self.pos_effect_ss = self.pos_effect_ss[ind, :]

# ===========================================================================
# DMRG related functions
    @ staticmethod
    def calculate_environment_for_parallel(results, dim):
        x = np.zeros((dim, dim))
        for n in range(0, len(results)):
            x += np.kron(np.kron(results[n][0], results[n][1]), results[n][2])
        return x

    def environment_s1_s2(self, inputs):
        # p is the center and the position of the tensor to be updated
        # the two operators are at positions[0] and positions[1]
        p, sn, positions = inputs
        if self._debug:
            self.check_orthogonal_center(p)
        operators = [self.operators[sn[0]], self.operators[sn[1]]]
        v_left = np.zeros(0)
        v_right = np.zeros(0)
        if positions[0] > positions[1]:
            positions = sort_list(positions, [1, 0])
            operators = sort_list(operators, [1, 0])
        if p < positions[0]:
            v_left = np.eye(self.virtual_dim[p])
            v_middle = np.eye(self.mps[p].shape[1])
            if self._is_save_op:
                v_right = self.get_effective_operator_two_body(sn[0], sn[1], positions[0],
                                                               positions[1], p+1)
            else:
                v_right = T_module.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
                v_right = self.contract_v_l0_to_l1(positions[1]-1, positions[0], v_right)
                v_right = T_module.bound_vec_operator_right2left(self.mps[positions[0]], operators[0], v_right)
                v_right = self.contract_v_l0_to_l1(positions[0] - 1, p, v_right)
        elif p > positions[1]:
            if self._is_save_op:
                v_left = self.get_effective_operator_two_body(sn[0], sn[1], positions[0],
                                                              positions[1], p)
            else:
                v_left = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
                v_left = self.contract_v_l0_to_l1(positions[0]+1, positions[1], v_left)
                v_left = T_module.bound_vec_operator_left2right(self.mps[positions[1]], operators[1], v_left)
                v_left = self.contract_v_l0_to_l1(positions[1] + 1, p, v_left)
            v_middle = np.eye(self.mps[p].shape[1])
            v_right = np.eye(self.virtual_dim[p + 1])
        elif p == positions[0]:
            v_left = np.eye(self.virtual_dim[p])
            v_middle = operators[0]
            if self._is_save_op:
                v_right = self.get_effective_operators_one_body(sn[1], positions[1], p+1)
            else:
                v_right = T_module.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
                v_right = self.contract_v_l0_to_l1(positions[1] - 1, p, v_right)
        elif p == positions[1]:
            if self._is_save_op:
                v_left = self.get_effective_operators_one_body(sn[0], positions[0], p)
            else:
                v_left = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
                v_left = self.contract_v_l0_to_l1(positions[0] + 1, p, v_left)
            v_right = np.eye(self.virtual_dim[p + 1])
            v_middle = operators[1]
        else:
            if self._is_save_op:
                v_left = self.get_effective_operators_one_body(sn[0], positions[0], p)
                v_right = self.get_effective_operators_one_body(sn[1], positions[1], p+1)
            else:
                v_left = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
                v_left = self.contract_v_l0_to_l1(positions[0] + 1, p, v_left)
                v_right = T_module.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
                v_right = self.contract_v_l0_to_l1(positions[1] - 1, p, v_right)
            v_middle = np.eye(self.mps[p].shape[1])
        # if self.virtual_dim[p] != v_left.shape[0] or self.virtual_dim[p+1] != v_right.shape[0]:
        #     print('Wrong dimension: ' + str((sn[0], sn[1], positions[0], positions[1], p+1)))
        return v_left, v_middle, v_right

    def environment_s1_parallel(self, inputs):
        env2 = 0
        for n in range(0, inputs.__len__()):
            v_left, v_middle, v_right = self.environment_s1(inputs[n][:3])
            env2 += inputs[n][3] * np.kron(np.kron(v_left, v_middle), v_right)
        return env2

    # calculate the environment (one-body terms)
    def environment_s1(self, inputs):
        # p is the position of the tensor to be updated
        # the operator[sn] is at position
        p, sn, position = inputs
        if self._debug:
            self.check_orthogonal_center(p)
            self.check_virtual_bond_dimensions()

        operator = self.operators[sn]
        v_left = np.zeros(0)
        v_right = np.zeros(0)
        if p < position:
            v_left = np.eye(self.virtual_dim[p])
            v_middle = np.eye(self.mps[p].shape[1])
            if self._is_save_op:
                v_right = self.get_effective_operators_one_body(sn, position, p+1)
            else:
                v_right = T_module.bound_vec_operator_right2left(self.mps[position],
                                                                 operator, v_right)
                v_right = self.contract_v_l0_to_l1(position - 1, p, v_right)
        elif p > position:
            if self._is_save_op:
                v_left = self.get_effective_operators_one_body(sn, position, p)
            else:
                v_left = T_module.bound_vec_operator_left2right(self.mps[position],
                                                                operator, v_left)
                v_left = self.contract_v_l0_to_l1(position + 1, p, v_left)
            v_right = np.eye(self.virtual_dim[p + 1])
            v_middle = np.eye(self.mps[p].shape[1])
        else:  # p == position
            v_left = np.eye(self.virtual_dim[p])
            v_right = np.eye(self.virtual_dim[p + 1])
            v_middle = operator
        return v_left, v_middle, v_right

    def environment_s1_s2_parallel(self, inputs):
        env2 = 0
        for n in range(0, inputs.__len__()):
            v_left, v_middle, v_right = self.environment_s1_s2(inputs[n])
            env2 += np.kron(np.kron(v_left, v_middle), v_right)
        return env2

    # update the boundary vector v by contracting from l0 to l1 without operators
    def contract_v_l0_to_l1(self, l0, l1, v=np.zeros(0)):
        if l0 < l1:
            for n in range(l0, l1):
                v = T_module.bound_vec_operator_left2right(tensor=self.mps[n], v=v)
        elif l0 > l1:
            for n in range(l0, l1, -1):
                v = T_module.bound_vec_operator_right2left(tensor=self.mps[n], v=v)
        return v

    def contract_v_with_phys_l0_to_l1(self, l0, lp, l1, v=np.zeros(0)):
        # Note: lp is the position of remained physical bond, and l0<lp<l1, or l0>lp>l1
        if l0 < l1:  # left to right
            for n in range(l0, lp):
                v = T_module.bound_vec_operator_left2right(tensor=self.mps[n], v=v)
            for n in range(lp, l1):
                v = T_module.bound_vec_with_phys_left2right(self.mps[n], v)
        elif l0 > l1:
            for n in range(l0, lp, -1):
                v = T_module.bound_vec_operator_right2left(tensor=self.mps[n], v=v)
            for n in range(lp, l1, -1):
                v = T_module.bound_vec_with_phys_right2left(self.mps[n], v)
        return v

    def update_effect_op_l0_to_l1(self, l0, l1, v, sn=-1, pos0=-1, is_update_op=True):
        # l0: starting site
        # l1: before the ending site
        # sn is the number of the operator
        # pos0 is the original position of v (effective operator)
        if l0 < l1:
            for n in range(l0, l1):
                v = T_module.bound_vec_operator_left2right(tensor=self.mps[n], v=v)
                if is_update_op:
                    self.add_key_and_pos('one', (sn, pos0, n+1), v)
        elif l0 > l1:
            for n in range(l0, l1, -1):
                v = T_module.bound_vec_operator_right2left(tensor=self.mps[n], v=v)
                if is_update_op:
                    self.add_key_and_pos('one', (sn, pos0, n), v)
        return v

    def update_all_effective_id(self):
        self.effective_id = dict()
        v = np.eye(self.virtual_dim[self.center])
        for n in range(self.center, self.length):
            key = str(self.center) + '_' + str(n+1)
            v = T_module.bound_vec_operator_left2right(self.mps[n], v=v)
            self.effective_id[key] = v
        v = np.eye(self.virtual_dim[self.center+1])
        for n in range(self.center, 0, -1):
            key = str(self.center) + '_' + str(n)
            v = T_module.bound_vec_operator_right2left(self.mps[n], v=v)
            self.effective_id[key] = v

    def update_effect_from_op1_to_op2(self, sn, snn, p0, q0, p1, is_update_op=True):
        # here, we have p0 < q0 < p1, or p1 >= q0 > p0 (on the same side of the RG endpoint)
        if q0 < p1:  # left to right
            v = self.get_effective_operators_one_body(sn, p0, q0)
            v = T_module.bound_vec_operator_left2right(self.mps[q0], self.operators[snn], v)
            if is_update_op:
                self.add_key_and_pos('two', (sn, snn, p0, q0, q0+1), v)
            for n in range(q0+1, p1):
                v = T_module.bound_vec_operator_left2right(tensor=self.mps[n], v=v)
                # v = self.update_effect_op_l0_to_l1(n, n+1, v, is_update_op=False)
                if is_update_op:
                    self.add_key_and_pos('two', (sn, snn, p0, q0, n+1), v)
        elif p1 <= p0:
            v = self.get_effective_operators_one_body(snn, q0, p0+1)
            v = T_module.bound_vec_operator_right2left(self.mps[p0], self.operators[sn], v)
            if is_update_op:
                self.add_key_and_pos('two', (sn, snn, p0, q0, p0), v)
            for n in range(p0-1, p1-1, -1):
                v = T_module.bound_vec_operator_right2left(tensor=self.mps[n], v=v)
                # v = self.update_effect_op_l0_to_l1(n, n - 1, v, is_update_op=False)
                if is_update_op:
                    self.add_key_and_pos('two', (sn, snn, p0, q0, n), v)
        else:
            # if this happen, there must be a logic bug
            v = None
            print_error('LogicBug detected. Please check')
        return v

    def effective_hamiltonian_dmrg(self, p, index1, index2, coeff1, coeff2, tol=1e-12):
        if self._debug and p != self.center:
            print_error('CenterError: the tensor must be at the orthogonal center before '
                        'defining the function handle', 'magenta')
        nh1 = index1.shape[0]  # number of one-body Hamiltonians
        nh2 = index2.shape[0]  # number of two-body Hamiltonians
        s = [self.virtual_dim[p], self.phys_dim, self.virtual_dim[p+1]]
        dim = np.prod(s)
        if not self._is_parallel:
            h_effect = np.zeros((dim, dim))
            for n in range(0, nh1):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    v_left, v_middle, v_right = \
                        self.environment_s1((p, index1[n, 1], index1[n, 0]))
                    if self._debug:
                        self.check_environments(v_left, v_middle, v_right, p)
                    h_effect += coeff1[n] * np.kron(np.kron(v_left, v_middle), v_right)
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    v_left, v_middle, v_right = \
                        self.environment_s1_s2((p, index2[n, 2:4], index2[n, :2]))
                    if self._debug:
                        self.check_environments(v_left, v_middle, v_right, p)
                    h_effect += coeff2[n] * np.kron(np.kron(v_left, v_middle), v_right)
        else:  # parallel computations
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh1):
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    inputs[n_now % self.pool['n']].append((p, index1[n, 1], index1[n, 0], coeff1[n]))
                    n_now += 1
            tmp = self.pool['pool'].map(self.environment_s1_parallel, inputs)
            h_effect = 0
            for n in range(0, tmp.__len__()):
                h_effect += tmp[n]
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    inputs[n_now % self.pool['n']].append((p, index2[n, 2:4], index2[n, :2]))
                    n_now += 1
            tmp = self.pool['pool'].map(self.environment_s1_s2_parallel, inputs)
            for n in range(0, tmp.__len__()):
                h_effect += tmp[n]
        # h_effect = (h_effect + h_effect.conj().T) / 2
        return h_effect, s

    def all_environments(self, p, index1, index2, coeff1, coeff2, tol=1e-12):
        # for 'update_tensor_eigs' while mapping the effective H to a linear operator
        if self._debug and p != self.center:
            print_error('CenterError: the tensor must be at the orthogonal center before '
                        'defining the function handle', 'magenta')
        nh1 = index1.shape[0]
        nh2 = index2.shape[0]
        s = [self.virtual_dim[p], self.phys_dim, self.virtual_dim[p+1]]
        if not self._is_parallel and not self._is_env_parallel_lmr:
            # No parallel computing
            env1 = list()
            env2 = list()
            for n in range(0, nh1):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    env1.append(self.environment_s1((p, index1[n, 1], index1[n, 0])))
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    env2.append(self.environment_s1_s2((p, index2[n, 2:4], index2[n, :2])))
        elif (not self._is_parallel) and self._is_env_parallel_lmr:
                inputs = list()
                for n in range(0, nh1):
                    if abs(coeff1[n]) > tol and np.linalg.norm(
                            self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                        inputs.append((p, index1[n, 1], index1[n, 0]))
                env1 = self.pool['pool'].map(self.environment_s1, inputs)
                inputs = list()
                for n in range(0, nh2):
                    if abs(coeff2[n]) > tol:
                        inputs.append((p, index2[n, 2:4], index2[n, :2]))
                env2 = self.pool['pool'].map(self.environment_s1_s2, inputs)
        else:
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh1):
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    inputs[n_now % self.pool['n']].append((p, index1[n, 1], index1[n, 0]))
                    n_now += 1
            env1 = self.pool['pool'].map(self.environment_s1_parallel, inputs)
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    inputs[n_now % self.pool['n']].append((p, index2[n, 2:4], index2[n, :2]))
                    n_now += 1
            env2 = self.pool['pool'].map(self.environment_s1_s2_parallel, inputs)
            self.pool['pool'].join()
        return env1, env2, s

    def all_environments_optimized(self, p, index1, index2, coeff1, coeff2, tol=1e-12):
        # for 'update_tensor_eigs' while mapping the effective H to a linear operator
        if self._debug and p != self.center:
            print_error('CenterError: the tensor must be at the orthogonal center before '
                        'defining the function handle', 'magenta')
        nh1 = index1.shape[0]
        nh2 = index2.shape[0]
        s = [self.virtual_dim[p], self.phys_dim, self.virtual_dim[p+1]]
        if not self._is_parallel and not self._is_env_parallel_lmr:
            # No parallel computing
            for n in range(0, nh1):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    env1 = self.environment_s1((p, index1[n, 1], index1[n, 0]))
                    self.classify_and_update_env(env1, coeff1[n], 'one', index1[n, 0], index1[n, 1], p)
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    env2 = self.environment_s1_s2((p, index2[n, 2:4], index2[n, :2]))
                    self.classify_and_update_env(env2, coeff2[n], 'two', index2[n, :2], index2[n, 2:4], p)
        elif (not self._is_parallel) and self._is_env_parallel_lmr:
                inputs = list()
                for n in range(0, nh1):
                    if abs(coeff1[n]) > tol and np.linalg.norm(
                            self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                        inputs.append((p, index1[n, 1], index1[n, 0]))
                env1 = self.pool['pool'].map(self.environment_s1, inputs)
                inputs = list()
                for n in range(0, nh2):
                    if abs(coeff2[n]) > tol:
                        inputs.append((p, index2[n, 2:4], index2[n, :2]))
                env2 = self.pool['pool'].map(self.environment_s1_s2, inputs)
        else:
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh1):
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    inputs[n_now % self.pool['n']].append((p, index1[n, 1], index1[n, 0]))
                    n_now += 1
            env1 = self.pool['pool'].map(self.environment_s1_parallel, inputs)
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    inputs[n_now % self.pool['n']].append((p, index2[n, 2:4], index2[n, :2]))
                    n_now += 1
            env2 = self.pool['pool'].map(self.environment_s1_s2_parallel, inputs)
            self.pool['pool'].join()
        return s

    def classify_and_update_env(self, envs, coeff, which_env, pos, sn, p):
        if which_env is 'one':
            if pos < p:
                key = '1_0_0'
                if key in self.opt_env:
                    self.opt_env[key] += envs[0] * coeff
                else:
                    self.opt_env[key] = envs[0] * coeff
            elif pos == p:
                key = '0_' + str(sn) + '_0'
                if key in self.opt_env:
                    self.opt_env[key] += envs[1] * coeff
                else:
                    self.opt_env[key] = envs[1] * coeff
            elif pos > p:
                key = '0_0_1'
                if key in self.opt_env:
                    self.opt_env[key] += envs[2] * coeff
                else:
                    self.opt_env[key] = envs[2] * coeff
        elif which_env is 'two':
            if max(pos) < p:
                key = '1_0_0'
                if key in self.opt_env:
                    self.opt_env[key] += envs[0] * coeff
                else:
                    self.opt_env[key] = envs[0] * coeff
            elif min(pos) > p:
                key = '0_0_1'
                if key in self.opt_env:
                    self.opt_env[key] += envs[2] * coeff
                else:
                    self.opt_env[key] = envs[2] * coeff
            elif max(pos) == p:
                key = '1_' + str(sn[1]) + '_0'
                if key in self.opt_env:
                    self.opt_env[key] += envs[0] * coeff
                else:
                    self.opt_env[key] = envs[0] * coeff
            elif min(pos) == p:
                key = '0_' + str(sn[0]) + '_1'
                if key in self.opt_env:
                    self.opt_env[key] += envs[2] * coeff
                else:
                    self.opt_env[key] = envs[2] * coeff
            else:
                key = '1_0_1'
                if key not in self.opt_env:
                    self.opt_env[key] = list()
                self.opt_env[key].append([coeff, envs[0], envs[2]])

    @staticmethod
    def update_tensor_eigs_f_handle(tensor, env1, env2, coeff1, coeff2, s, tau):
        nh1 = len(env1)
        nh2 = len(env2)
        tensor = tensor.reshape(s)
        tensor1 = tensor.copy()
        if type(env1) is list:
            for n in range(0, nh1):
                tensor1 -= tau * coeff1[n] * T_module.absorb_matrices2tensor_full_fast(
                    tensor, [x.T for x in env1[n]])
            for n in range(0, nh2):
                tensor1 -= tau * coeff2[n] * T_module.absorb_matrices2tensor_full_fast(
                    tensor, [x.T for x in env2[n]])
        else:
            for n in range(0, nh1):
                tensor1 -= tau * env1[n].dot(tensor)
            for n in range(0, nh2):
                tensor1 -= tau * env2[n].dot(tensor)
        return tensor1.reshape(-1, 1)

    def update_tensor_eigs_f_handle_optimized(self, tensor, s, tau):
        tensor = tensor.reshape(s)
        tensor1 = tensor.copy()
        for key in self.opt_env:
            x = key.split('_')
            if x[0] is '1' and x[1] is '0' and x[2] is '0':  # left
                tensor1 -= tau * T_module.absorb_matrix2tensor(tensor, self.opt_env[key].T, 0)
            elif x[0] is '0' and x[1] is '0' and x[2] is '1':  # right
                tensor1 -= tau * T_module.absorb_matrix2tensor(tensor, self.opt_env[key].T, 2)
            elif x[0] is '0' and x[1] is not '0' and x[2] is '0':  # middle
                tensor1 -= tau * T_module.absorb_matrix2tensor(tensor, self.opt_env[key].T, 1)
            elif x[0] is '1' and x[1] is not '0' and x[2] is '0':
                tensor1 -= tau * T_module.absorb_matrices2tensor(
                    tensor, [self.opt_env[key].T, self.operators[int(x[1])].T], [0, 1])
            elif x[0] is '0' and x[1] is not '0' and x[2] is '1':
                tensor1 -= tau * T_module.absorb_matrices2tensor(
                    tensor, [self.opt_env[key].T, self.operators[int(x[1])].T], [2, 1])
            elif key is '1_0_1':
                for env in self.opt_env['1_0_1']:
                    tensor1 -= tau * env[0] * T_module.absorb_matrices2tensor(
                        tensor, [x.T for x in env[1:]], [0, 2])
        return tensor1.reshape(-1, )

    def update_tensor_eigs(self, p, index1, index2, coeff1, coeff2, tau, is_real, tol=1e-16):
        _center = self.center
        self.correct_orthogonal_center(p)  # move the orthogonal tensor to p
        if self._is_save_op:
            if -0.5 < _center < p:
                for n in range(_center, p+1):
                    self.del_bad_effective_operators(n)
            elif _center >= p:
                for n in range(p, _center+1):
                    self.del_bad_effective_operators(n)
            else:
                cprint('CenterError: central-orthogonalize MPS before updating the tensor', 'magenta')
                set_trace()
            self.update_all_effective_id()
        if self.eig_way == 0:
            h_effect, s = self.effective_hamiltonian_dmrg(p, index1, index2, coeff1, coeff2)
            h_effect = np.eye(h_effect.shape[0]) - tau * h_effect
        else:
            # env1, env2, s = self.all_environments(p, index1, index2, coeff1, coeff2, tol=tol)
            # dim = np.prod(s)
            # h_effect = LinearOp((dim, dim), lambda a: self.update_tensor_eigs_f_handle(
            #     a, env1, env2, coeff1, coeff2, s, tau))
            s = self.all_environments_optimized(p, index1, index2, coeff1, coeff2, tol=tol)
            dim = np.prod(s)
            h_effect = LinearOp((dim, dim), lambda a: self.update_tensor_eigs_f_handle_optimized(
                a, s, tau))
        self.mps[p] = eigs(h_effect, k=1, which='LM', v0=self.mps[p].reshape(-1, 1),
                           tol=tol)[1].reshape(s)
        if is_real:
            self.mps[p] = self.mps[p].real
        if self.eig_way == 1:
            self.opt_env = dict()

# ========================================================
    def calculate_entanglement_spectrum(self, if_fast=True):
        # NOTE: this function will central orthogonalize the MPS
        _way = self.decomp_way
        _center = self.center
        self.decomp_way = 'svd'
        if if_fast and _center > -0.5:
            p0 = self.length - 1
            p1 = 0
            for n in range(0, self.length - 1):
                if self.lm[n].size == 0:
                    p0 = min(p0, n)
                    p1 = max(p1, n)
            self.correct_orthogonal_center(p0)
            self.correct_orthogonal_center(p1+1)
            self.correct_orthogonal_center(_center)
        else:
            self.correct_orthogonal_center(0)
            self.correct_orthogonal_center(self.length-1)
            if _center > 0:
                self.correct_orthogonal_center(_center)
        self.decomp_way = _way

    def calculate_entanglement_entropy(self):
        for i in range(0, self.length - 1):
            if self.lm[i].size == 0:
                self.ent[i] = -1
            else:
                self.ent[i] = T_module.entanglement_entropy(self.lm[i])

    def reduced_density_matrix_two_body(self, p1, p2):
        # p1 < p2
        if self.center < p1:
            v1 = self.contract_v_with_phys_l0_to_l1(self.center, p1, p2)
            v2 = self.contract_v_with_phys_l0_to_l1(p2, p2, p2 - 1)
        elif self.center > p2:
            v1 = self.contract_v_with_phys_l0_to_l1(p1, p1, p1 + 1)
            v2 = self.contract_v_with_phys_l0_to_l1(self.center, p2, p1)
        else:
            v1 = self.contract_v_with_phys_l0_to_l1(p1, p1, p2)
            v2 = self.contract_v_with_phys_l0_to_l1(p2, p2, p2 - 1)
        rho = np.tensordot(v1, v2, ([2, 3], [2, 3])).transpose(0, 2, 1, 3)
        d = rho.shape[0]
        rho = rho.reshape([d*d, d*d])
        return rho/np.trace(rho)

    def observation_s1(self, inputs):
        sn, position = inputs
        operator = self.operators[sn]
        if self._is_save_op:
            if position >= self.center:
                v = self.get_effective_operators_one_body(sn, position, self.center, is_update_op=False)
            else:
                v = self.get_effective_operators_one_body(sn, position, self.center+1, is_update_op=False)
        else:
            if position > self.center:
                v = T_module.bound_vec_operator_right2left(self.mps[position], operator)
                v = self.contract_v_l0_to_l1(position - 1, self.center - 1, v)
            else:
                v = T_module.bound_vec_operator_left2right(self.mps[position], operator)
                v = self.contract_v_l0_to_l1(position + 1, self.center + 1, v)
        return np.trace(v)

    def observation_s1_s2(self, inputs):
        ssn, positions = inputs
        if self._debug:
            self.check_mps_norm1()
        if positions[0] > positions[1]:
            ssn = sort_list(ssn, [1, 0])
            positions = sort_list(positions, [1, 0])
        operators = [self.operators[ssn[0]], self.operators[ssn[1]]]
        if self._is_save_op:
            if self.center <= positions[0]:
                v = self.get_effective_operator_two_body(ssn[0], ssn[1], positions[0],
                                                         positions[1], self.center,
                                                         is_update_op=False)
                return np.trace(v)
            elif self.center > positions[1]:
                v = self.get_effective_operator_two_body(ssn[0], ssn[1], positions[0],
                                                         positions[1], self.center+1,
                                                         is_update_op=False)
                return np.trace(v)
            else:
                vl = self.get_effective_operators_one_body(ssn[0], positions[0],
                                                           self.center, is_update_op=False)
                vr = self.get_effective_operators_one_body(ssn[1], positions[1],
                                                           self.center, is_update_op=False)
                return np.trace(vl.dot(vr.T))
        else:
            if self.center < positions[0]:
                v = self.contract_v_l0_to_l1(self.center, positions[0])
            else:
                v = np.zeros(0)
            v = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v)
            v = self.contract_v_l0_to_l1(positions[0] + 1, positions[1], v)
            v = T_module.bound_vec_operator_left2right(self.mps[positions[1]], operators[1], v)
            if positions[1] < self.center:
                v = self.contract_v_l0_to_l1(positions[1] + 1, self.center + 1, v)
            return np.trace(v)

    def observe_magnetization(self, sn):
        mag = np.zeros((self.length, 1))
        inputs = list()
        for i in range(0, self.length):
            if self._is_parallel:
                inputs.append((sn, i))
            else:
                mag[i] = self.observation_s1((sn, i))
        if self._is_parallel:
            mag = np.array(self.pool['pool'].map(self.observation_s1, inputs))
        return mag

    def observe_bond_energy(self, index2, coeff2):
        nh = index2.shape[0]
        eb = np.zeros((nh, 1))
        inputs = list()
        for n in range(0, nh):
            if self._is_parallel:
                inputs.append((index2[n, 2:], index2[n, :2]))
            else:
                eb[n] = coeff2[n] * self.observation_s1_s2((index2[n, 2:], index2[n, :2]))
        if self._is_parallel:
            eb = np.array(self.pool['pool'].map(self.observation_s1_s2, inputs))
        return eb

    def observe_correlators_from_middle(self, op1, op2, ob_len=None):
        if ob_len is None:
            ob_len = self.length
        corr = list()
        pos_mid = round(self.length/2)
        pos1 = pos_mid
        pos2 = pos_mid + 1
        n_control = 0
        while pos1 > -0.1 and pos2 < self.length and ob_len > -0.1:
            corr.append(self.observation_s1_s2(([op1, op2], [pos1, pos2])))
            if n_control % 2 == 0:
                pos1 -= 1
            else:
                pos2 += 1
            n_control += 1
            ob_len -= 1
        return np.array(corr)

    def norm_mps(self):
        # calculate the norm of an MPS
        if self._debug:
            lc = self.check_orthogonal_center()
            if lc != self.center:
                cprint('CenterError: center should be at %d but at %d' % self.center, lc, 'magenta')
                trace_stack()
        if self.center < -0.5:
            v = self.contract_v_l0_to_l1(0, self.length)
            norm = v[0, 0]
        else:
            norm = np.linalg.norm(self.mps[self.center].reshape(1, -1))
        return norm

    def full_coefficients_mps(self, tol_memory=20):
        cprint('Warning: full_coefficients_mps is used to calculate the full coefficients of the MPS', 'magenta')
        tot_size_log2 = self.length * np.log2(self.phys_dim) - 5
        if tot_size_log2 > tol_memory:
            cprint('The memory cost of the total coefficients is too large (a lot more than %d Mb). '
                   'Stop calculation' % tot_size_log2, 'magenta')
            cprint('If you want to calculate anyway, please input a larger \'tol_memory\'', 'cyan')
            x = None
        else:
            s = self.mps[0].shape
            x = self.mps[0].reshape(s[0]*s[1], s[2])
            d0 = s[2]
            for n in range(1, self.length):
                s = self.mps[n].shape
                x = x.dot(self.mps[n].reshape(s[0], s[1]*s[2]))
                x.reshape(d0*s[1], s[2])
                d0 = s[2]
        return x.reshape(-1, 1)

# ===========================================================
# Checking functions
    def check_orthogonal_center(self, expected_center=-2, if_print=True):
        # Check if MPS has the correct center, or at the expected center
        # if not, find the correct center, or recommend a new center while it is not central orthogonal
        # NOTE: no central-orthogonalization in this function, only recommendation
        if self.center > -0.5:
            left = self.orthogonality[:self.center]
            right = self.orthogonality[self.center+1:]
            if not(np.prod(left == -1) and np.prod(right == 1)):
                if if_print:
                    cprint(colored('self.center is incorrect. Change it to -1', 'magenta'))
                    trace_stack()
                self.center = -1
        if self.center < -0.5:
            left = np.nonzero(self.orthogonality == -1)
            left = left[0][-1]
            right = np.nonzero(self.orthogonality == 1)
            right = right[0][0]
            if np.prod(self.orthogonality[:left+1]) and np.prod(self.orthogonality[right:]) and left + 2 == right:
                self.center = left+1
                if if_print:
                    cprint(colored('self.center is corrected to %g' % self.center, 'cyan'))
            else:
                if if_print:
                    cprint(colored('MPS is not central orthogonal. self.center remains -1', 'cyan'))
        else:
            left = self.center - 1
        if self.center > -0.5:
            if expected_center > -0.5 and expected_center != self.center:
                cprint('The center is at %d, not the expected position at %d' % (self.center, expected_center))
            recommend_center = self.center
        else:
            # if not central-orthogonal, recommend the tensor after the last left-orthogonal one as the new center
            recommend_center = left + 1
        return recommend_center

    def check_orthogonality_by_tensors(self, tol=1e-20, is_print=True):
        incorrect_ort = list()
        for n in range(0, self.length):
            if self.orthogonality[n] == -1:
                is_ort = T_module.check_orthogonality(self.mps[n], [2], tol=tol)
            elif self.orthogonality[n] == 1:
                is_ort = T_module.check_orthogonality(self.mps[n], [0], tol=tol)
            else:
                is_ort = True
            if not is_ort:
                incorrect_ort.append(n)
        if is_print:
            if incorrect_ort.__len__() == 0:
                print('The orthogonality of all tensors are marked correctly by self.orthogonality')
            else:
                cprint('In self.orthogonality, the orthogonality of the following tensors is incorrect:', 'magenta')
                cprint(str(incorrect_ort), 'cyan')
        return incorrect_ort

    def check_environments(self, vl, vm, vr, n):
        # check if the environments of the n-th tensor have consistent dimensions
        is_bug0 = False
        bond = str()
        if vl.shape[0] != vl.shape[1]:
            is_bug0 = True
            bond = 'LEFT'
        if vm.shape[0] != vm.shape[1]:
            is_bug0 = True
            bond = 'MIDDLE'
        if vr.shape[0] != vr.shape[1]:
            is_bug0 = True
            bond = 'RIGHT'
        if is_bug0:
            cprint('EnvError: for the %d-th tensor, the ' % n + bond + ' v is not square', 'magenta')
        is_bug = False
        if vl.shape[0] != self.virtual_dim[n]:
            is_bug = True
            bond = 'LEFT'
        if vr.shape[0] != self.virtual_dim[n + 1]:
            is_bug = True
            bond = 'RIGHT'
        if vm.shape[0] != self.mps[n].shape[1]:
            bond = 'MIDDLE'
            is_bug = True
        if is_bug:
            cprint('EnvError: for the %d-th tensor, the ' % n + bond + ' v has inconsistent dimension', 'magenta')
        if is_bug0 or is_bug:
            trace_stack()

    def check_virtual_bond_dimensions(self):
        is_error = False
        for n in range(1, self.length):
            if self.virtual_dim[n] != self.mps[n].shape[0] or self.virtual_dim[n] != self.mps[n-1].shape[2]:
                cprint('BondDimError: inconsistent dimension detected for the %d-th virtual bond' % n, 'magenta')
                is_error = True
        if is_error:
            trace_stack(2)

    def check_mps_norm1(self, if_print=False):
        # check if the MPS is norm-1
        norm = self.norm_mps()
        if abs(norm - 1) > 1e-14:
            print_error('The norm is MPS is %g away from 1' % abs(norm - 1))
        if if_print:
            cprint('The norm of MPS is %g' % norm, 'cyan')

    def clean_to_save(self):
        self.__delattr__('effect_s')
        self.__delattr__('effect_ss')
        self.effect_s = {'none': np.zeros(0)}
        self.effect_ss = {'none': np.zeros(0)}
        self.effective_id = {'none': np.zeros(0)}
        self.pos_effect_s = np.zeros((0, 3)).astype(int)
        self.pos_effect_ss = np.zeros((0, 5)).astype(int)
        self.pool = None


def ln_fidelity_per_site(mps1, mps2):
    """
    :param mps1: one MPS
    :param mps2: the other MPS
    :return: ln fidelity per site
    Example:
    >>> f = ln_fidelity_per_site(mps1.mps, mps2.mps)
    """
    length = mps1.__len__()
    v = np.eye(mps1[0].shape[0])
    f = 0
    for n in range(0, length):
        v = T_module.cont([mps1[n], mps2[n], v], [[2, 3, -2], [1, 3, -1], [1, 2]])
        norm = np.linalg.norm(v.reshape(-1, ))
        v /= norm
        f += np.log(norm)/length
    return f


# ========================================================================
# class for infinite-size MPS
# relevant algorithms: iDMRG, iTEBD, AOP (1D)
class MpsInfinite(MpsBasic):
    """The tensors and bonds for central orthogonal MPS are arranged as:
            |        |        |
        - mps[0] - mps[1] - mps[2] -

             1                    1                    1
             |                    |                    |
        0 - mps[0] - 2       0 - mps[1] - 2       2 - mps[2] - 0
      (left-to-right ort)     (normalized)      (right-to-left ort)

    The tensors and bonds for MPO are arranged as:
            1
            |
        0 - T - 3
            |
            2
    """
    def __init__(self, form, d, chi, D, n_tensor=3, n_site=1, spin='half', way='qr',
                 operators=None, is_symme_env=True, is_real=True, debug=False):
        # form = 'center_ort': central orthogonal form; self.n_tensor fixed to 3 and self.lm[n] = zeros(0)
        # form = 'translation_invariant': central orthogonal form; self.n_tensor is flexible, self.lm[n]
        # must be initialized
        MpsBasic.__init__(self)
        self.spin = spin
        self.n_tensor = n_tensor  # number of tensors to give the iMPS
        self.n_site = n_site  # n-site DMRG algorithm
        self.mps = empty_list(self.n_tensor)
        self.lm = empty_list(self.n_tensor, np.zeros(0))
        self.env = empty_list(2, np.zeros(0))
        if self.n_site == 1:
            self.rho = np.ndarray([])
        else:
            self.rho = empty_list(self.n_site*2-1)

        self.orthogonality = np.zeros((self.n_tensor, 1)).astype(int)
        self.is_center_ort = False
        self.is_canonical = False
        self.d = d
        self.chi = chi
        self.D = D  # dimension of the "physical" bond of the time MPS
        self.form = form

        self.decomp_way = way
        self.is_symme_env = is_symme_env
        self.initialize_imps()
        if is_symme_env:
            self.env[1] = self.env[0]
        self.is_real = is_real
        if operators is None:
            op_half = spin_operators(spin)
            self.op = [op_half['id'], op_half['sx'], op_half['sy'], op_half['sz'],
                       op_half['su'], op_half['sd']]
        else:
            self.op = operators

        self._debug = debug

    def initialize_imps(self):
        if self.form is 'center_ort':
            # randomly initialize central orthogonal tensor
            self.n_tensor = 3
            if self.n_site == 1:
                self.mps[1] = np.random.randn(self.chi, self.d, self.chi)
            elif self.n_site == 2:
                self.mps[1] = np.random.randn(self.chi, self.d, self.d, self.chi)
            self.mps[1] /= np.linalg.norm(self.mps[1].reshape(-1, 1))
            self.mps[0] = np.ndarray([])
            self.mps[2] = np.ndarray([])
            self.lm = empty_list(self.n_tensor, np.zeros(0))
            self.update_ort_tensor_mps('both')
            self.env[0] = np.ones((self.chi, self.D, self.chi))
            self.env[1] = np.ones((self.chi, self.D, self.chi))
            self.orthogonality = np.array([-1, 0, 1])
            self.is_center_ort = True
        elif self.form is 'translation_invariant':
            # randomly initialize translational invariant tensor
            # the ordering is: - lm[0] - mps[0] - lm[1] - mps[1] - ... - lm[n] - mps[n]
            self.mps[0] = np.random.randn(self.chi, self.d, self.chi)
            self.lm[0] = np.ones((self.chi, 1)) / (self.chi ** (-0.5))
            for n in range(1, self.n_tensor):
                self.mps[n] = self.mps[0].copy()
                self.lm[n] = self.lm[0].copy()

    def update_left_env(self, tensor):
        # compatible to one-site and two-site iDMRG
        self.env[0] = T_module.cont([self.mps[0].conj(), tensor, self.mps[0], self.env[0]],
                                    [[4, 5, -1], [1, 5, 3, -2], [2, 3, -3], [4, 1, 2]])
        self.env[0] /= np.linalg.norm(self.env[0].reshape(1, -1))
        self.env[0] = (self.env[0] + self.env[0].transpose(2, 1, 0)) / 2

    def update_right_env(self, tensor):
        # compatible to one-site and two-site iDMRG
        self.env[1] = T_module.cont([self.mps[2].conj(), tensor, self.mps[2], self.env[1]],
                                    [[4, 5, -1], [-2, 5, 3, 1], [2, 3, -3], [4, 1, 2]])
        self.env[1] /= np.linalg.norm(self.env[1].reshape(1, -1))
        self.env[1] = (self.env[1] + self.env[1].transpose(2, 1, 0)) / 2

    def update_ort_tensor_mps(self, which, dc=None):
        if self.n_site == 1:
            if which is 'left' or 'both':
                tmp = T_module.left2right_decompose_tensor(self.mps[1], self.decomp_way)
                self.mps[0] = tmp[0]
                if self.decomp_way is 'svd':
                    self.lm[0] = tmp[3]
            if which is 'right' or 'both':
                tmp = T_module.left2right_decompose_tensor(self.mps[1].transpose(2, 1, 0),
                                                           self.decomp_way)
                self.mps[2] = tmp[0]
                if self.decomp_way is 'svd':
                    self.lm[1] = tmp[3]
        elif self.n_site == 2:
            s = self.mps[1].shape
            self.mps[0], self.lm[0], self.mps[2] = np.linalg.svd(
                self.mps[1].reshape(s[0] * s[1], s[2] * s[3]))
            if dc is None:
                dc = min(self.chi, s[0] * s[1])
            else:
                dc = min(dc, s[0] * s[1])
            self.mps[0] = self.mps[0][:, :dc].reshape(s[0], s[1], dc)
            self.mps[2] = self.mps[2][:dc, :].reshape(dc, s[2], s[3]).transpose(2, 1, 0)
            self.lm[0] = self.lm[0][:dc]

    def central_orthogonalize_imps_without_h(self, mps='mps', iter_time=200, tol=1e-12):
        cprint('Warning: this function has not been testified!', 'red')
        """ Only for the MPS in the following form:
                    1                1
                    |                |
                - mps[0] - lm[0] - mps[2] -
               0         2       2         0
            (NOTE: this function will change self.mps; the original mps can deviate
            from central orthogonal form)
        """
        if mps is 'mps':
            for t in range(0, iter_time):
                vl = T_module.bound_vec_operator_left2right(
                    self.mps[0], normalize=True, symme=True)
                vr = T_module.bound_vec_operator_left2right(
                    self.mps[2], normalize=True, symme=True)
                ul, ur, lm = T_module.transformation_from_env_mats(
                    vl, vr, self.lm[0])[:3]
                self.mps[0] = np.tensordot(np.tensordot(self.mps[0], vl, ([2], [0])),
                                           np.linalg.pinv(vl), ([0], [0]))
                self.mps[2] = np.tensordot(np.tensordot(self.mps[2], vr, ([2], [0])),
                                           np.linalg.pinv(vr), ([0], [0]))
                if np.linalg.norm(lm - self.lm[0]) < tol:
                    self.lm[0] = lm
                    break
                else:
                    self.lm[0] = lm

    def effective_hamiltonian(self, tensor):
        # compatible to one-site and two-site iDMRG
        if self.n_site == 1:
            if self.is_symme_env:
                h = T_module.cont([self.env[0], tensor, self.env[0]],
                                  [[-1, 1, -4], [1, -2, -5, 2], [-3, 2, -6]])
            else:
                h = T_module.cont([self.env[0], tensor, self.env[1]],
                                  [[-1, 1, -4], [1, -2, -5, 2], [-3, 2, -6]])
            s = h.shape
            h = h.reshape(s[0] * s[1] * s[2], s[3] * s[4] * s[5])
            h = (h + h.transpose(1, 0)) / 2
            return h
        elif self.n_site == 2:
            if self.is_symme_env:
                h = T_module.cont([self.env[0], tensor, tensor, self.env[0]],
                                  [[-1, 1, -5], [1, -2, -6, 3], [3, -3, -7, 2], [-4, 2, -8]])
            else:
                h = T_module.cont([self.env[0], tensor, tensor, self.env[1]],
                                  [[-1, 1, -5], [1, -2, -6, 3], [3, -3, -7, 2], [-4, 2, -8]])

            s = h.shape
            h = h.reshape(s[0] * s[1] * s[2] * s[3], s[4] * s[5] * s[6] * s[7])
            h = (h + h.transpose(1, 0)) / 2
            return h

    def update_central_tensor(self, tensor):
        # compatible to one-site and two-site iDMRG
        s = self.mps[1].shape
        h = self.effective_hamiltonian(tensor)
        self.mps[1] = eigs(h, 1, v0=self.mps[1].reshape(-1, 1))[1].reshape(s)
        if self.is_real:
            self.mps[1] = self.mps[1].real

    def rho_from_central_tensor(self):
        if self.n_site == 1:
            self.rho = np.tensordot(self.mps[1].conj(), self.mps[1], ([0, 2], [0, 2]))
        elif self.n_site == 2:
            d0 = round(self.d**0.5)
            tmp = self.mps[1].reshape(self.chi, d0, d0, d0, d0, self.chi)
            nb = 6
            self.rho = empty_list(nb - 3)
            for n in range(1, nb - 2):
                bonds_con = list(range(0, n)) + list(range(n + 2, nb))
                self.rho[n - 1] = np.tensordot(tmp, tmp, (bonds_con, bonds_con))\
                    .reshape(self.d, self.d)

    def observe_energy(self, h):
        # compatible to one-site iDMRG and one-site deep iDMRG
        if self.form is 'center_ort':
            if self.n_site == 1:
                energy = np.trace(self.rho.dot(h))
                return energy
            elif self.n_site == 2:
                nr = self.rho.__len__()
                energy = np.ndarray(nr, )
                for n in range(0, nr):
                    energy[n] = np.trace(self.rho[n].dot(h))
                return energy

    def check_orthogonality_mps(self):
        no_bug = True
        tol = 1e-12
        tmp = np.eye(self.chi).reshape(-1, 1)
        tmp1 = np.tensordot(self.mps[0].conj(), self.mps[0], ([0, 1], [0, 1])).reshape(-1, 1)
        if np.linalg.norm(tmp - tmp1) > tol:
            no_bug = False
            print('The left part of the MPS is not orthogonal')
        tmp1 = np.tensordot(self.mps[2].conj(), self.mps[2], ([0, 1], [0, 1])).reshape(-1, 1)
        if np.linalg.norm(tmp - tmp1) > tol:
            no_bug = False
            print('The right part of the MPS is not orthogonal')
        if no_bug:
            print('The MPS satisfies the orthogonality')


# ========================================================================
# class for infinite-size deep MPS
class MpsDeepInfinite(MpsInfinite):

    def __init__(self, form, d, chi, D, chib0, chib, is_symme_env, n_site, is_debug=False):
        MpsInfinite.__init__(self, form, d, chi, D, n_tensor=3, spin='half', way='svd',
                             operators=None, n_site=n_site, is_symme_env=is_symme_env,
                             is_real=True, debug=is_debug)
        self.chib0 = chib0
        self.chib = chib
        self.dlm = empty_list(2)
        self.umpo = empty_list(self.n_tensor)
        self.dmps = empty_list(self.n_tensor)
        self.envB = empty_list(2)
        self.initialize_deep_mps()

    def initialize_deep_mps(self):
        if self.form is 'center_ort':
            # randomly initialize central orthogonal tensor
            self.n_tensor = 3
            if self.n_site == 1:
                self.dmps[1] = np.random.randn(self.chib, self.d, self.chib)
            elif self.n_site == 2:
                self.dmps[1] = np.random.randn(self.chib, self.d, self.d, self.chib)
            self.dmps[1] /= np.linalg.norm(self.mps[1].reshape(-1, 1))
            self.update_ort_tensor_dmps('both')
            self.envB[0] = np.random.randn(self.chib, self.chib0, self.D, self.chib0, self.chib)
            self.envB[1] = np.random.randn(self.chib, self.chib0, self.D, self.chib0, self.chib)
            self.orthogonality = np.array([-1, 0, 1])
            self.is_center_ort = True

    def get_unitary_mpo_from_mps(self):
        # !!! NOTE: self.mps will be changed in this function
        # Preparation
        if self._debug:
            self.check_orthogonality_mps()
        db = self.chib0
        if self.chib0 < self.chi:
            if self.n_site == 1:
                self.mps[1] = self.mps[1][:db, :, :db]
            elif self.n_site == 2:
                # self.mps[0] = self.mps[0][:self.chib0, :, :self.chib0]
                self.mps[1] = self.mps[1][:db, :, :db]
                # self.mps[2] = self.mps[2][:self.chib0, :, :self.chib0]
                # self.lm[0] = self.lm[0][:self.chib0]
                # self.central_orthogonalize_imps_without_h()
        if self.is_symme_env:
            self.update_ort_tensor_mps('left')
        else:
            self.update_ort_tensor_mps('both', db)
        # Update unitary mpo[0]
        # !!! NOTE: self.mps has been changed in the following codes
        self.umpo[0] = np.zeros((db, self.d, db, self.d))
        self.umpo[0][:, :, :, 0] = self.mps[0][:db, :, :db]
        s = self.mps[0].shape
        tmp = np.linalg.svd(self.mps[0][:db, :, :db].reshape(db*s[1], db), True)[0].\
            reshape(db, s[1], db*s[1])
        for n in range(1, s[1]):
            self.umpo[0][:, :, :, n] = tmp[:, :, db * n:db * (n + 1)]
        self.umpo[0] = self.umpo[0].transpose(0, 1, 3, 2)
        # Update unitary mpo[2] if not env_symmetrical
        if not self.is_symme_env:
            s = self.mps[2].shape
            self.umpo[2] = np.zeros((db, self.d, db, self.d))
            self.umpo[2][:, :, :, 0] = self.mps[2][:db, :, :db]
            tmp = np.linalg.svd(self.mps[2][:db, :, :db].reshape(db * s[1], db), True)[0].\
                reshape(db, s[1], db * s[1])
            for n in range(1, s[1]):
                self.umpo[2][:, :, :, n] = tmp[:, :, db * n:db * (n + 1)]
            self.umpo[2] = self.umpo[2].transpose(0, 1, 3, 2)

        if self.n_site == 1:
            # Update central unitary mpo[1]; Only for one-site algorithm
            s = self.mps[1].shape
            self.umpo[1] = np.zeros((db, self.d, db, self.d))
            self.umpo[1][:, :, :, 0] = self.mps[1][:db, :, :db]
            tmp1, lm, v = np.linalg.svd(self.mps[1][:db, :, :db].reshape(db*s[1], db), True)
            v = np.diag(lm).dot(v)
            for n in range(1, s[1]):
                self.umpo[1][:, :, :, n] = tmp1[:, db*n:db*(n+1)].dot(v).reshape(
                    db, s[1], db)
            self.umpo[1] = self.umpo[1].transpose(0, 1, 3, 2)
        if self.n_site == 2:
            self.lm[0] = self.lm[0][:db]
        if self._debug:
            self.check_unitary_umpo()

    def update_left_env_dmps_simple(self, tensor):
        # print(self.dmps[0].shape)
        # print(self.umpo[0].shape)
        # print(tensor.shape)
        # print(self.envB[0].shape)
        self.envB[0] = T_module.cont([self.dmps[0].conj(), self.umpo[0].conj(),
                                      tensor, self.umpo[0], self.dmps[0],
                                      self.envB[0]], [[8, 6, -1], [7, 9, 6, -2],
                                                      [3, 9, 1, -3], [4, 1, 2, -4],
                                                      [5, 2, -5], [8, 7, 3, 4, 5]])
        self.envB[0] /= np.linalg.norm(self.envB[0].reshape(-1, 1))
        self.envB[0] = (self.envB[0] + self.envB[0].transpose(4, 3, 2, 1, 0)) / 2

    def update_right_env_dmps_simple(self, tensor):
        # print(self.dmps[2].shape)
        # print(self.umpo[2].shape)
        # print(tensor.shape)
        # print(self.envB[1].shape)
        self.envB[1] = T_module.cont([self.dmps[2].conj(), self.umpo[2].conj(),
                                      tensor, self.umpo[2], self.dmps[2],
                                      self.envB[1]], [[8, 6, -1], [7, 9, 6, -2],
                                                      [-3, 9, 1, 3], [4, 1, 2, -4],
                                                      [5, 2, -5], [8, 7, 3, 4, 5]])
        self.envB[1] /= np.linalg.norm(self.envB[1].reshape(1, -1))
        self.envB[1] = (self.envB[1] + self.envB[1].transpose(4, 3, 2, 1, 0)) / 2

    def effective_hamiltonian_dmps_simple(self, tensor):
        if self.n_site == 1:
            if self.is_symme_env:
                h = T_module.cont([self.umpo[1].conj(), tensor, self.umpo[1], self.envB[0],
                                   self.envB[0]], [[3, 1, -2, 6], [4, 1, 2, 7], [5, 2, -5, 8],
                                                   [-1, 3, 4, 5, -4], [-3, 6, 7, 8, -6]])
            else:
                h = T_module.cont([self.umpo[1].conj(), tensor, self.umpo[1], self.envB[0],
                                   self.envB[1]], [[3, 1, -2, 6], [4, 1, 2, 7], [5, 2, -5, 8],
                                                   [-1, 3, 4, 5, -4], [-3, 6, 7, 8, -6]])
            s = h.shape
            h = h.reshape(s[0]*s[1]*s[2], s[3]*s[4]*s[5])
            h = (h + h.transpose(1, 0))/2
            return h
        elif self.n_site == 2:
            if self.is_symme_env:
                h = T_module.cont([self.envB[0], self.umpo[0].conj(), tensor, self.umpo[0],
                                   np.diag(self.lm[0]), np.diag(self.lm[0]),
                                   self.umpo[0].conj(), tensor, self.umpo[0], self.envB[0]],
                                  [[-1, 10, 11, 12, -5], [10, 3, -2, 1], [11, 3, 4, 8],
                                   [12, 4, -6, 2], [1, 7], [2, 9], [13, 5, -3, 7],
                                   [8, 5, 6, 14], [15, 6, -7, 9], [-4, 13, 14, 15, -8]])
            else:
                h = T_module.cont([self.envB[0], self.umpo[0].conj(), tensor, self.umpo[0],
                                   np.diag(self.lm[0]), np.diag(self.lm[0]),
                                   self.umpo[2].conj(), tensor, self.umpo[2], self.envB[1]],
                                  [[-1, 10, 11, 12, -5], [10, 3, -2, 1], [11, 3, 4, 8],
                                   [12, 4, -6, 2], [1, 7], [2, 9], [13, 5, -3, 7],
                                   [8, 5, 6, 14], [15, 6, -7, 9], [-4, 13, 14, 15, -8]])
            s = h.shape
            h = h.reshape(s[0]*s[1]*s[2]*s[4], s[4]*s[5]*s[6]*s[7])
            h = (h + h.transpose(1, 0))/2
            return h

    def update_ort_tensor_dmps(self, which):
        if self.n_site == 1:
            if which is 'left' or 'both':
                tmp = T_module.left2right_decompose_tensor(self.dmps[1], self.decomp_way)
                self.dmps[0] = tmp[0]
                if self.decomp_way is 'svd':
                    self.dlm[0] = tmp[3]
            if which is 'right' or 'both':
                tmp = T_module.left2right_decompose_tensor(self.dmps[1].transpose(2, 1, 0),
                                                           self.decomp_way)
                self.dmps[2] = tmp[0]
                if self.decomp_way is 'svd':
                    self.dlm[1] = tmp[3]
        elif self.n_site == 2:
            s = self.dmps[1].shape
            self.dmps[0], self.dlm[0], self.dmps[2] = np.linalg.svd(
                self.dmps[1].reshape(s[0] * s[1], s[2] * s[3]))
            self.dmps[0] = self.dmps[0][:, :self.chib].reshape(s[0], s[1], self.chib)
            self.dmps[2] = self.dmps[2][:self.chib, :].reshape(self.chib, s[2], s[3])\
                .transpose(2, 1, 0)
            self.dlm[0] = self.dlm[0][:self.chib]

    def update_central_tensor_dmps(self, tensor):
        s = self.dmps[1].shape
        h = self.effective_hamiltonian_dmps_simple(tensor)
        self.dmps[1] = eigs(h, 1, v0=self.dmps[1].reshape(-1, 1))[1].reshape(s)
        if self.is_real:
            self.dmps[1] = self.dmps[1].real

    def rho_from_central_tensor_dmps(self):
        if self.n_site == 1:
            self.rho = T_module.cont([self.dmps[1].conj(), self.umpo[1].conj(),
                                      self.umpo[1], self.dmps[1]], [[3, 5, 4], [1, -1, 5, 2],
                                                                    [1, -2, 6, 2], [3, 6, 4]])
        elif self.n_site == 2:
            if self.is_symme_env:
                tmp = T_module.cont([self.dmps[0].conj(), self.umpo[0].conj(), self.umpo[0],
                                     self.dmps[0], np.diag(self.dlm[0]), np.diag(self.lm[0]),
                                     np.diag(self.lm[0]), np.diag(self.dlm[0]), self.dmps[2].conj(),
                                     self.umpo[0].conj(), self.umpo[0], self.dmps[2]],
                                    [[9, 13, 7], [3, -1, 13, 1], [3, -3, 15, 2], [9, 15, 8],
                                     [7, 11], [1, 5], [2, 6], [8, 12], [10, 14, 11], [4, -2, 14, 5],
                                    [4, -4, 16, 6], [10, 16, 12]])
            else:
                tmp = T_module.cont([self.dmps[0].conj(), self.umpo[0].conj(), self.umpo[0],
                                     self.dmps[0], np.diag(self.dlm[0]), np.diag(self.lm[0]),
                                     np.diag(self.lm[0]), np.diag(self.dlm[0]), self.dmps[2].conj(),
                                     self.umpo[2].conj(), self.umpo[2], self.dmps[2]],
                                    [[9, 13, 7], [3, -1, 13, 1], [3, -3, 15, 2], [9, 15, 8],
                                     [7, 11], [1, 5], [2, 6], [8, 12], [10, 14, 11], [4, -2, 14, 5],
                                     [4, -4, 16, 6], [10, 16, 12]])
            d0 = round(self.d**0.5)
            tmp = tmp.reshape(d0, d0, d0, d0, d0, d0, d0, d0)
            iden = np.eye(d0**2).reshape(d0, d0, d0, d0)
            for n in range(0, 3):
                bond_con = list(range(0, n)) + list(range(n+2, 4))
                bond_con += list(range(4, n+4)) + list(range(n+6, 8))
                self.rho[n] = np.tensordot(tmp, iden, (bond_con, [0, 1, 2, 3])).\
                    reshape(self.d, self.d)

    # ========================================================================
    # functions for checking
    def check_unitary_umpo(self):
        no_bug = True
        s = self.umpo[0].shape
        tmp1 = np.tensordot(self.umpo[0].conj(), self.umpo[0], ([0, 1], [0, 1])).reshape(-1, )
        tmp2 = np.kron(np.eye(s[2]), np.eye(s[3])).reshape(-1,)
        err = np.linalg.norm(tmp1 - tmp2) / tmp1.size
        # print(err)
        if err > 1e-10:
            no_bug = False
            print_error('The left part of the UMPO is not unitary, with err = %g' % err, False)
        if not self.is_symme_env:
            tmp1 = np.tensordot(self.umpo[2].conj(), self.umpo[2],
                                ([0, 1], [0, 1])).reshape(-1, )
            err = np.linalg.norm(tmp1 - tmp2) / tmp1.size
            if err > 1e-10:
                no_bug = False
                print_error('The right part of the UMPO is not unitary, with err = %g'
                            % err, False)
        if self.n_site == 1:
            tmp1 = np.tensordot(self.umpo[1].conj(), self.umpo[1], ([0, 1, 3],
                                                                    [0, 1, 3])).reshape(-1, )
            tmp2 = np.eye(s[2]).reshape(-1, )
            err = np.linalg.norm(tmp1 - tmp2) / tmp1.size
            # print(err)
            if err > 1e-10:
                no_bug = False
                print_error('The middle part of the UMPO is not unitary, with err = %g'
                            % err, False)
        if no_bug:
            print('The uMPO is unitary.')
