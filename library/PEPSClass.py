from library.HamiltonianModule import from_spin2phys_dim, environment_tensor_to_bath_hamilt, \
    hamiltonian_heisenberg, hamiltonian2gate_tensors, spin_operators
from scipy.sparse.linalg import LinearOperator as LinearOp
from scipy.sparse.linalg import eigsh as eigs
from scipy.linalg import expm
import numpy as np
from library import BasicFunctions as bf, TensorBasicModule as tm


class PepsBasic:

    def __init__(self):
        self.version = '2018-08-22'
        self.operators = list()


class PepsInfinite(PepsBasic):

    def __init__(self, lattice, chi, state_type='pure', spin='half', ini_way='random',
                 operators=None, is_symme_env=True, is_debug=False):
        PepsBasic.__init__(self)
        """
        Order of bonds: (phys, virtual, virtual, ...)
        Lattices: 
        'honeycomb0' lattice: n-th virtual bond, n-th lambda
        0,lm0          1,lm1
          \            /
           -  2,lm2  -
         /            \
        1,lm1          0,lm0
        """
        self.lattice = lattice
        self.stateType = state_type  # pure or mixed
        self.spin = spin
        self.d0 = from_spin2phys_dim(spin)
        self.chi = chi
        self.nTensor = 0
        self.nLm = 0
        self.nVirtual = 0
        self.pos_lm = list()
        self.lm_ten_bond = None
        self.iniWay = ini_way
        self._is_debug = is_debug
        if self.stateType is 'mixed':
            self.d = self.d0 ** 2
        else:
            self.d = self.d0

        self.initial_lattice()
        self.tensors = bf.empty_list(self.nTensor)
        self.lm = bf.empty_list(self.nLm)
        self.initial_ipeps()
        self.initial_lm()

        if operators is None:
            op_half = spin_operators(spin)
            self.operators = [op_half['id'], op_half['sx'], op_half['sy'], op_half['sz'],
                              op_half['su'], op_half['sd']]
        else:
            self.operators = operators

        # Below is for tree DMRG
        self.is_symme_env = is_symme_env
        self.env = None
        self.model_related = dict()

    def initial_lattice(self):
        # pos_lm[nt][nb]=x denotes the x-th lm locates at the nb-th bond of the nt-th tensor
        if self.lattice is 'honeycomb0':
            self.nTensor = 2
            self.nLm = 3
            self.nVirtual = 3
            # self.pos_lm[x] means for the x-th tensor, it is the 0, 1, and 2-th lm on the three virtual bonds
            self.pos_lm = [[], []]
            self.pos_lm[0] = [0, 1, 2]
            self.pos_lm[1] = [0, 1, 2]
        elif self.lattice is 'honeycombTreeDMRG':
            # for tree DMRG of honeycomb lattice; the TN is square (one tensor with two spins)
            self.nTensor = 5
            self.nVirtual = 4
            self.d = self.d0 ** 2
        elif self.lattice in ['kagome', 'husimi']:
            self.nTensor = 2  # use symmetrical environment: only one orthogonal tensor
            self.nVirtual = 3
            self.d = self.d0
            self.stateType = 'pure'
        else:
            bf.print_error('Incorrect input of the lattice')
        self.find_pos_of_lm()

    def initial_ipeps(self):
        dim = (self.d,) + (self.chi,) * self.nVirtual
        if self.iniWay is 'random':
            tensor = tm.symmetrical_rand_peps_tensor(self.d, self.chi, self.nVirtual)
            if self.stateType is 'mixed':
                bond = (self.d0, self.d0) + (self.chi,) * self.nVirtual
                tensor = tensor.reshape(bond)
                ind = (1, 0) + tuple(range(2, self.nVirtual + 2))
                tensor = (tensor + tensor.transpose(ind)) / 2
                bond = (self.d,) + (self.chi,) * self.nVirtual
                tensor = tensor.reshape(bond)
            for n in range(0, self.nTensor):
                self.tensors[n] = tensor.copy()
        elif self.iniWay is 'ones':
            for n in range(0, self.nTensor):
                self.tensors[n] = np.ones(dim)
        elif self.iniWay is 'id':
            if self.stateType is 'mixed':
                if self._is_debug:
                    if abs(self.d0**2 - self.d) > 1e-10:
                        bf.print_error('For mixed state, d should be as d0^2. '
                                       'Check self.d or self.stateType')
                for n in range(0, self.nTensor):
                    self.tensors[n] = np.eye(self.d0).reshape((self.d,) + (1,) * self.nVirtual)
            else:
                bf.print_error('Initial way "id" is only for thermal states')

    def initial_lm(self):
        lm = np.random.rand(self.chi, )
        lm /= np.linalg.norm(lm)
        if self.iniWay is 'random':
            for n in range(0, self.nLm):
                self.lm[n] = lm.copy()
        elif self.iniWay is 'ones':
            for n in range(0, self.nLm):
                self.tensors[n] = np.ones((self.chi, ))/(self.chi**0.5)
        elif self.iniWay is 'id':
            if self.stateType is 'mixed':
                for n in range(0, self.nLm):
                    self.lm[n] = np.ones(1, )
            else:
                bf.print_error('Initial way "id" is only for thermal states')

    def find_pos_of_lm(self):
        self.lm_ten_bond = np.zeros((self.nLm, 2, 2), dtype=int)
        for n_lm in range(0, self.nLm):
            n_found = 0
            for n in range(0, self.nTensor):
                if n_lm in self.pos_lm[n]:
                    self.lm_ten_bond[n_lm, n_found, 0] = n
                    self.lm_ten_bond[n_lm, n_found, 1] = self.pos_lm[n].index(n_lm)
                    n_found += 1
                if n_found == 2:
                    break
            if self._is_debug and n_found < 2:
                bf.print_error('In "find_pos_of_one_lm", n_found is ony %g. It should 2'
                               % n_found)

    def absorb_lm(self, nt, if_sqrt, which_vb):
        # which_vb does NOT count physical bond
        tensor = self.tensors[nt].copy()
        if if_sqrt:
            if which_vb is 'all':
                tensor = tm.absorb_matrices2tensor(tensor, [np.diag(np.sqrt(
                    self.lm[self.pos_lm[nt][n]])) for n in range(0, self.nVirtual)],
                                                   [n + 1 for n in range(0, self.nVirtual)])
            elif type(which_vb) is int:
                tensor = tm.absorb_matrix2tensor(tensor, np.diag(
                    np.sqrt(self.lm[self.pos_lm[nt][which_vb]])), which_vb + 1)
            else:
                tensor = tm.absorb_matrices2tensor(tensor, [np.diag(np.sqrt(
                    self.lm[self.pos_lm[nt][n]])) for n in which_vb], [n+1 for n in which_vb])
        else:
            if which_vb is 'all':
                tensor = tm.absorb_matrices2tensor(
                    tensor, [np.diag(self.lm[self.pos_lm[nt][n]]) for n in range(0, self.nVirtual)],
                    [n + 1 for n in range(0, self.nVirtual)])
            elif type(which_vb) is int:
                tensor = tm.absorb_matrix2tensor(
                    tensor, np.diag(self.lm[self.pos_lm[nt][which_vb]]), which_vb + 1)
            else:
                tensor = tm.absorb_matrices2tensor(
                    tensor, [np.diag(self.lm[self.pos_lm[nt][n]]) for n in which_vb],
                    [n+1 for n in which_vb])
        return tensor

    def one_bond_so_transformation(self, nt1, vb1, nt2, vb2):
        # Super-orthogonal transformation on one virtual bond
        # vb does NOT count the physical bond
        if self._is_debug:
            if self.pos_lm[nt1][vb1] != self.pos_lm[nt2][vb2]:
                bf.print_error('In one_bond_so_transformation, the two virtual bonds must'
                               'correspond to the same lambda')
        m1 = self.bond_env_matrix_simple(nt1, vb1)
        m2 = self.bond_env_matrix_simple(nt2, vb2)

        flag = False
        if self._is_debug:
            _lm = self.lm[self.pos_lm[nt1][vb1]].copy()
            flag = (self.chi == self.tensors[nt1].shape[vb1+1])
        u1, u2, self.lm[self.pos_lm[nt1][vb1]] = tm.transformation_from_env_mats(
            m1, m2, self.lm[self.pos_lm[nt1][vb1]], self.chi, norm_way=1)[:3]
        if self._is_debug and flag:
            _tmp = u1.dot(np.diag(self.lm[self.pos_lm[nt1][vb1]])).dot(u2.T)
            err = np.linalg.norm(tm.off_diagonal_mat(_tmp).reshape(-1, ))
            if err > 1e-10:
                print('Warning of the transformations from environment: not diagonal (%g)' % err)
            _tmp = np.diag(_tmp)
            _tmp = _tmp / np.linalg.norm(_tmp)
            err = np.linalg.norm(_tmp - self.lm[self.pos_lm[nt1][vb1]])
            if err > 1e-10:
                print('Warning of the transformations from environment: not recover lm (%g)' % err)
            print(self.lm[self.pos_lm[nt1][vb1]])
        self.tensors[nt1] = tm.absorb_matrix2tensor(self.tensors[nt1], u1, vb1 + 1)
        self.tensors[nt2] = tm.absorb_matrix2tensor(self.tensors[nt2], u2, vb2 + 1)
        self.tensors[nt1] /= max(abs(self.tensors[nt1].reshape(-1, 1)))
        self.tensors[nt2] /= max(abs(self.tensors[nt2].reshape(-1, 1)))
        # self.lm[self.pos_lm[nt1][vb1]] = tm.normalize_tensor(self.lm[self.pos_lm[nt1][vb1]])[0]
        return m1, m2

    def bond_env_matrix_simple(self, nt, vb, is_symme=True, is_normalize=True):
        # the nb-th bond matrix of the nt-th tensor (including the phys bond)
        bonds = list(range(0, self.nVirtual))
        bonds.remove(vb)
        tmp = self.absorb_lm(nt, False, bonds)
        bonds = list(range(0, self.nVirtual+1))
        bonds.remove(vb+1)
        tmp = np.tensordot(tmp, tmp, (bonds, bonds))
        if is_symme:
            tmp = (tmp + tmp.conj().T) / 2
        if is_normalize:
            tmp = tm.normalize_tensor(tmp)[0]
        return tmp

    def evolve_tensor_one_bond(self, gate_t, nt, vb):
        # vb: the vb-th virtual bond (do NOT including the phys bond)
        """
         0
         |
        G -1
         |
         2
        """
        s = self.tensors[nt].shape
        if self.stateType is 'pure':
            self.tensors[nt] = np.tensordot(gate_t, self.tensors[nt], ([2], [0]))
            ind = (0,) + tuple(range(2, vb + 2)) + (1,) + tuple(range(vb + 2, self.nVirtual + 2))
            ind_shape = [s[0]] + [s[n] for n in range(2, vb + 2)] + [s[1] * s[vb + 2]] + \
                        [s[n] for n in range(vb + 3, self.nVirtual + 2)]
            self.tensors[nt] = self.tensors[nt].transpose(ind).reshape(ind_shape)
        elif self.stateType is 'mixed':
            self.tensors[nt] = self.tensors[nt].reshape((self.d0, self.d0) + s[1:])
            ind = (0, 2) + tuple(range(3, vb + 3)) + (1,) + tuple(range(vb + 3, self.nVirtual + 3))
            s = list(s)
            s[vb+1] *= gate_t.shape[1]
            self.tensors[nt] = np.tensordot(gate_t, self.tensors[nt],
                                            ([2], [0])).transpose(ind).reshape(s)

    def evolve_lm(self, n_lm, gate_lm):
        self.lm[n_lm] = np.kron(gate_lm, self.lm[n_lm])

    def evolve_once_tensor_and_lm(self, gate_t1, gate_t2, n_lm):
        dg = gate_t1.shape[1]
        self.evolve_lm(n_lm, np.ones((dg, )))
        self.evolve_tensor_one_bond(gate_t1, self.lm_ten_bond[n_lm, 0, 0],
                                    self.lm_ten_bond[n_lm, 0, 1])
        self.evolve_tensor_one_bond(gate_t2, self.lm_ten_bond[n_lm, 1, 0],
                                    self.lm_ten_bond[n_lm, 1, 1])

    def super_orthogonalization(self, which_lm, it_time=200, tol=1e-10):
        if which_lm is 'all':
            err = 1
            for t in range(0, it_time):
                for n_lm in range(0, self.nLm):
                    nt1 = self.lm_ten_bond[n_lm, 0, 0]
                    vb1 = self.lm_ten_bond[n_lm, 0, 1]
                    nt2 = self.lm_ten_bond[n_lm, 1, 0]
                    vb2 = self.lm_ten_bond[n_lm, 1, 1]
                    m1, m2 = self.one_bond_so_transformation(nt1, vb1, nt2, vb2)
                    m1 /= m1[0, 0]
                    m2 /= m2[0, 0]
                    err = np.linalg.norm((m1 - np.eye(m1.shape[0])).reshape(-1, ))
                    err += np.linalg.norm((m2 - np.eye(m2.shape[0])).reshape(-1, ))
                    err /= (2 * m1.shape[0])
                if err < tol:
                    break
            # print('SO time = ' + str(t))
        elif type(which_lm) is int:
            nt1 = self.lm_ten_bond[which_lm, 0, 0]
            vb1 = self.lm_ten_bond[which_lm, 0, 1]
            nt2 = self.lm_ten_bond[which_lm, 1, 0]
            vb2 = self.lm_ten_bond[which_lm, 1, 1]
            self.one_bond_so_transformation(nt1, vb1, nt2, vb2)
        else:
            for n_lm in which_lm:
                nt1 = self.lm_ten_bond[n_lm, 0, 0]
                vb1 = self.lm_ten_bond[n_lm, 0, 1]
                nt2 = self.lm_ten_bond[n_lm, 1, 0]
                vb2 = self.lm_ten_bond[n_lm, 1, 1]
                self.one_bond_so_transformation(nt1, vb1, nt2, vb2)

    # =======================================================
    # For tree DMRG on honeycomb lattice (square TN)
    def get_model_related_tree_dmrg(self, jx, jy, jz, hx, hz, tau=1e-6):
        if 'h2phys' not in self.model_related:
            self.model_related['h2phys'] = hamiltonian_heisenberg(self.spin, jx, jy, jz, hx, hz)
            self.model_related['h2_gate'] = expm(-tau / 2 * self.model_related['h2phys'])
        if 'tensor_gate' not in self.model_related:
            self.model_related['tensor_gate'] = hamiltonian2gate_tensors(self.model_related['h2phys'], tau)
        if 'hbath' not in self.model_related:
            self.model_related['hbath'] = bf.empty_list(self.nVirtual)

    def initialize_env(self):
        self.env = bf.empty_list(self.nVirtual)
        tmp = np.random.randn(self.chi, self.d0**2, self.chi)
        tmp = (tmp + tmp.transpose([2, 1, 0])) / 2
        tmp /= np.linalg.norm(tmp.reshape(-1, ))
        for n in range(0, self.nVirtual):
            self.env[n] = tmp.copy()

    def update_bath_h_tree_dmrg(self, which):
        for n in list(which):
            if n == 0:
                self.model_related['hbath'][0] = environment_tensor_to_bath_hamilt(
                    self.model_related['tensor_gate'][0], self.env[0])
            elif n == 1:
                self.model_related['hbath'][1] = environment_tensor_to_bath_hamilt(
                    self.model_related['tensor_gate'][0], self.env[1])
            elif n == 2:
                self.model_related['hbath'][2] = environment_tensor_to_bath_hamilt(
                    self.model_related['tensor_gate'][1], self.env[2 - self.is_symme_env * 2])
            elif n == 3:
                self.model_related['hbath'][3] = environment_tensor_to_bath_hamilt(
                    self.model_related['tensor_gate'][1], self.env[3 - self.is_symme_env * 2])

    @ staticmethod
    def evolve_central_tensor(tensor, h2, ind, if_permute_back=False):
        s = list(tensor.shape)
        ind1 = list(range(0, tensor.ndim))
        for x in ind:
            ind1.remove(x)
        s0 = [s[n] for n in ind]
        s1 = [s[n] for n in ind1]
        tensor = tensor.transpose(ind + ind1).reshape(np.prod(s0), np.prod(s1))
        tensor = h2.dot(tensor).reshape(s0 + s1)
        if if_permute_back:
            ind_back = np.argsort(ind + ind1)
            tensor = tensor.transpose(ind_back)
        return tensor

    def update_central_tensor_honeycomb_tree_dmrg_eigs(self, tensor, s):
        # tensor_phys is obtained by "hamiltonian2gate_tensors" in "HamiltonianModule.py"
        d0 = np.round(np.sqrt(s[0]))
        tensor = tensor.reshape([d0, d0] + s[1:])
        tensor = self.evolve_central_tensor(
            tensor, self.model_related['h2_gate'], [0, 1])  # [0, 1, 2, 3, 4, 5]
        tensor = self.evolve_central_tensor(
            tensor, self.model_related['hbath'][0], [0, 2])  # [0, 2, 1, 3, 4, 5]
        tensor = self.evolve_central_tensor(
            tensor, self.model_related['hbath'][1], [0, 3])  # [0, 3, 2, 1, 4, 5]
        tensor = self.evolve_central_tensor(
            tensor, self.model_related['hbath'][2], [3, 4])  # [1, 4, 0, 3, 2, 5]
        tensor = self.evolve_central_tensor(
            tensor, self.model_related['hbath'][3], [0, 5])  # [1, 5, 4, 0, 3, 2]
        tensor = tensor.transpose(3, 0, 5, 4, 3, 2)
        tensor = self.evolve_central_tensor(
            tensor, self.model_related['h2_gate'], [0, 1])  # [0, 1, 2, 3, 4, 5]
        return tensor.reshape(-1, )

    def update_central_tensor_tree_dmrg(self, nt=0, tol=1e-15):
        s = self.tensors[nt].shape
        dim = np.prod(s)
        h_effect = LinearOp((dim, dim),
                            lambda a: self.update_central_tensor_honeycomb_tree_dmrg_eigs(a, s))
        self.tensors[nt] = eigs(h_effect, k=1, which='LM', v0=self.tensors[nt].reshape(-1, ),
                                tol=tol)[1].reshape(s)

    def calculate_orthogonal_tensor(self, ne, decomp='qr'):
        self.tensors[ne + 1] = self.tensors[0].transpose(
            list(range(0, ne+1)) + list(range(ne+2, self.nVirtual+1)) + [ne+1])
        self.tensors[ne + 1] = self.tensors[ne + 1].reshape(
            self.d * (self.chi ** (self.nVirtual - 1)), self.chi)
        if decomp is 'qr':
            self.tensors[ne + 1] = np.linalg.qr(self.tensors[ne + 1])[0]
        else:
            self.tensors[ne + 1], self.lm[ne] = np.linalg.svd(
                self.tensors[ne + 1], full_matrices=False)[:2]
        self.tensors[ne+1] = self.tensors[ne+1].reshape(
            [self.d] + [self.chi]*self.nVirtual).transpose(
            list(range(0, ne+1)) + [self.nVirtual+1] + list(range(ne+1, self.nVirtual)))

    def update_env_tree_dmrg(self, ne, decomp='qr'):
        # ne: which environment (0, 1, 2, 3)
        self.calculate_orthogonal_tensor(ne, decomp)
        s = self.tensors[ne+1].shape
        d0 = np.round(np.sqrt(s[0]))
        tensor = self.tensors[ne+1].reshape([d0, d0] + s[1:])
        tensor = self.evolve_central_tensor(
            tensor, self.model_related['h2_gate'].T, [0, 1])  # [0, 1, 2, 3, 4, 5]
        tensor1 = tensor.conj()
        if ne == 3:
            tensor1 = self.evolve_central_tensor(
                tensor1, self.model_related['hbath'][3].T, [1, 5])  # [1, 5, 0, 2, 3, 4]
            tensor1 = self.evolve_central_tensor(
                tensor1, self.model_related['hbath'][2].T, [1, 5])  # [1, 4, 5, 0, 2, 3]
            tensor1 = tensor1.transpose(3, 0, 4, 5, 1, 2)
            tensor = self.evolve_central_tensor(
                tensor, self.model_related['hbath'][0], [0, 2], if_permute_back=True)
            self.env[ne] = tm.cont([tensor, tensor1, self.model_related['tensor_gate'][0]],
                                   [[6, 4, 1, -3, 2, 3], [5, 4, 1, -1, 2, 3], [5, -2, 6]])
        elif ne == 2:
            tensor1 = self.evolve_central_tensor(
                tensor1, self.model_related['hbath'][3].T, [1, 5])  # [1, 5, 0, 2, 3, 4]
            tensor1 = self.evolve_central_tensor(
                tensor1, self.model_related['hbath'][2].T, [1, 5])  # [1, 4, 5, 0, 2, 3]
            tensor1 = self.evolve_central_tensor(
                tensor1, self.model_related['hbath'][1].T, [3, 5])  # [0, 3, 1, 4, 5, 2]
            tensor1 = tensor1.transpose(0, 2, 5, 1, 3, 4)
            self.env[ne] = tm.cont([tensor, tensor1, self.model_related['tensor_gate'][0]],
                                   [[6, 4, -3, 3, 1, 2], [5, 4, -1, 3, 1, 2], [5, -2, 6]])
        elif ne == 1:
            tensor = self.evolve_central_tensor(
                tensor, self.model_related['hbath'][0], [0, 2])  # [0, 2, 1, 3, 4, 5]
            tensor = self.evolve_central_tensor(
                tensor, self.model_related['hbath'][1], [0, 3])  # [0, 3, 2, 1, 4, 5]
            tensor = self.evolve_central_tensor(
                tensor, self.model_related['hbath'][2], [0, 4])  # [0, 4, 3, 2, 1, 5]
            tensor = tensor.transpose(0, 4, 3, 2, 1, 5)
            self.env[ne] = tm.cont([tensor, tensor1, self.model_related['tensor_gate'][1]],
                                   [[4, 6, 1, 2, 3, -3], [4, 5, 1, 2, 3, -1], [5, -2, 6]])
        elif ne == 0:
            tensor = self.evolve_central_tensor(
                tensor, self.model_related['hbath'][0], [0, 2])  # [0, 2, 1, 3, 4, 5]
            tensor = self.evolve_central_tensor(
                tensor, self.model_related['hbath'][1], [0, 3])  # [0, 3, 2, 1, 4, 5]
            tensor = tensor.transpose(0, 3, 2, 1, 4, 5)
            tensor1 = self.evolve_central_tensor(
                tensor1, self.model_related['hbath'][3].T, [1, 5], if_permute_back=True)
            self.env[ne] = tm.cont([tensor, tensor1, self.model_related['tensor_gate'][1]],
                                   [[4, 6, 1, 2, -3, 3], [4, 5, 1, 2, -1, 3], [5, -2, 6]])
        self.env[ne] = (self.env[ne] + self.env[ne].transpose(2, 1, 0)) / 2
        self.env[ne] /= np.linalg.norm(self.env[ne].reshape(-1, ))

    def update_all_tree_dmrg(self):
        for ne in range(0, round(self.nVirtual/(self.is_symme_env+1))):
            self.update_env_tree_dmrg(ne)
            self.update_bath_h_tree_dmrg(ne)
            if self.is_symme_env:
                self.update_bath_h_tree_dmrg(ne+round(self.nVirtual/2))
        self.update_central_tensor_tree_dmrg()

    # =======================================================
    # rho and observations
    def rho_one_body_simple(self, which_t):
        if which_t is 'all':
            rho = bf.empty_list(self.nTensor)
            for nt in range(0, self.nTensor):
                rho[nt] = self.rho_one_body_simple_nt_tensor(nt)
        elif type(which_t) is int:
            rho = self.rho_one_body_simple_nt_tensor(which_t)
        else:
            rho = bf.empty_list(len(which_t))
            n = 0
            for nt in which_t:
                rho[n] = self.rho_one_body_simple_nt_tensor(nt)
                n += 1
        return rho

    def rho_one_body_simple_nt_tensor(self, nt):
        tensor = self.absorb_lm(nt, False, 'all')
        if self.stateType is 'pure':
            ind = list(range(1, self.nVirtual + 1))
            rho = np.tensordot(tensor.conj(), tensor, (ind, ind))
        else:
            d0 = round(self.d**0.5)
            s = tensor.shape
            tensor = tensor.reshape((d0,d0) + s[1:])
            ind = list(range(1, self.nVirtual + 2))
            rho = np.tensordot(tensor.conj(), tensor, (ind, ind))
        rho = (rho.conj().T + rho) / 2
        rho /= np.trace(rho)
        return rho

    def rho_two_body_simple(self, which_lm):
        if which_lm is 'all':
            rho = bf.empty_list(self.nLm)
            for n_lm in range(0, self.nLm):
                rho[n_lm] = self.rho_two_body_nlm_simple(n_lm)
        elif type(which_lm) is int:
            rho = self.rho_two_body_nlm_simple(which_lm)
        else:
            rho = bf.empty_list(len(which_lm))
            n = 0
            for n_lm in which_lm:
                rho[n] = self.rho_two_body_nlm_simple(n_lm)
                n += 1
        return rho

    def rho_two_body_nlm_simple(self, n_lm):
        nt1 = self.lm_ten_bond[n_lm, 0, 0]
        vb1 = self.lm_ten_bond[n_lm, 0, 1]
        nt2 = self.lm_ten_bond[n_lm, 1, 0]
        vb2 = self.lm_ten_bond[n_lm, 1, 1]
        if self._is_debug:
            if n_lm != self.pos_lm[nt2][vb2]:
                bf.print_error('In rho_two_body_simple, the two virtual bonds must'
                               'correspond to the same lambda')
        bonds = list(range(0, self.nVirtual))
        bonds.remove(vb1)
        tmp1 = self.absorb_lm(nt1, False, bonds)
        tmp2 = self.absorb_lm(nt2, False, 'all')
        if self.stateType is 'pure':
            bonds = list(range(1, self.nVirtual+1))
            bonds.remove(vb1 + 1)
            tmp1 = np.tensordot(tmp1.conj(), tmp1, (bonds, bonds))
            bonds = list(range(1, self.nVirtual + 1))
            bonds.remove(vb2 + 1)
            tmp2 = np.tensordot(tmp2.conj(), tmp2, (bonds, bonds))
        elif self.stateType is 'mixed':
            s = tmp1.shape
            bonds = list(range(1, self.nVirtual + 2))
            bonds.remove(vb1 + 2)
            tmp1 = tmp1.reshape((self.d0, self.d0) + s[1:])
            tmp1 = np.tensordot(tmp1.conj(), tmp1, (bonds, bonds))
            s = tmp2.shape
            bonds = list(range(1, self.nVirtual + 2))
            bonds.remove(vb2 + 2)
            tmp2 = tmp2.reshape((self.d0, self.d0) + s[1:])
            tmp2 = np.tensordot(tmp2.conj(), tmp2, (bonds, bonds))
        rho = tm.cont([tmp1, tmp2], [[-1, 1, -3, 2], [-2, 1, -4, 2]])
        rho = rho.reshape(self.d0*self.d0, self.d0*self.d0)
        rho = (rho + rho.conj().T)/2
        rho /= np.trace(rho)
        return rho

    def check_super_orthogonality(self, tol=1e-8):
        for n in range(0, self.nTensor):
            for vb in range(0, self.nVirtual):
                tmp = self.bond_env_matrix_simple(n, vb)
                tmp /= tmp[0, 0]
                tmp -= np.eye(tmp.shape[0])
                err = np.linalg.norm(tmp.reshape(-1, 1))
                if err > tol:
                    print('The ' + str(n) + '-th tensor ' + str(vb) +
                          '-th virtual bond is NOT super-orthogonal with err = ' + str(err))


class TreePepsIdmrgKagome(PepsInfinite):

    def __init__(self, chi, spin):
        PepsInfinite.__init__(self, chi=chi, lattice='kagome', state_type='pure', spin=spin,
                              ini_way='random', is_symme_env=True, is_debug=False)
        # note: self.tensors[0] - central tensor; self.tensors[1] - orthogonal tensor
        self.effective_ops = [np.eye(chi) for _ in range(8)]
        self.bath_op_onsite = np.eye(chi)
        self.rho = list()
        self.initial_ipeps_kagome()

    def initial_ipeps_kagome(self):
        self.tensors[0] = np.random.randn(self.d * self.chi, self.d * self.chi, self.d * self.chi)
        self.tensors[0] = self.tensors[0] + self.tensors[0].transpose(
            1, 2, 0) + self.tensors[0].transpose(2, 0, 1)
        self.tensors[0] = self.tensors[0].reshape(
            (self.d, self.chi) * 3).transpose(0, 2, 4, 1, 3, 5)
        self.tensors[0] /= np.linalg.norm(self.tensors[0])
        self.update_ort_tensor_kagome()

    def update_ort_tensor_kagome(self):
        tmp = self.tensors[0].transpose(0, 1, 3, 4, 2, 5).reshape(
            self.d**2 * self.chi**2, self.d * self.chi)
        self.tensors[1] = np.linalg.svd(tmp, full_matrices=False)[0]
        self.tensors[1] = self.tensors[1][:, :self.chi].reshape(
            self.d, self.d, self.chi, self.chi, self.chi)

    @staticmethod
    def update_by_given_effective_ops(psi, ops, bonds):
        indexes = bf.empty_list(1 + bonds.__len__())
        indexes[0] = list(range(psi.ndim))
        x = 1
        for n in range(psi.ndim):
            if n in bonds:
                indexes[0][n] = x
                indexes[bonds.index(n) + 1] = [-n - 1, x]
                x += 1
            else:
                indexes[0][n] = -n - 1
        return tm.cont([psi] + ops, indexes)

    def update_effective_ops_kagome(self):
        # Only use mps[0] assuming left and right parts are symmetrical
        op_ind = [1, 3, 4, 5]  # x, z, u, d
        for p in [0, 1]:  # iterate on two physical bonds
            for n in range(op_ind.__len__()):
                tmp = self.update_by_given_effective_ops(
                    self.tensors[1], [self.operators[op_ind[n]]], [p])
                self.effective_ops[n + p*4] = tm.cont(
                    [self.tensors[1].conj(), tmp], [[1, 2, 3, 4, -1], [1, 2, 3, 4, -2]])

    def update_bath_onsite_kagome(self, j1, j2, hx, hz):
        # baths on the two branches
        op1 = tm.cont([self.tensors[1].conj(), self.tensors[1], self.bath_op_onsite],
                      [[4, 5, 1, 3, -1], [4, 5, 2, 3, -2], [1, 2]])
        op2 = tm.cont([self.tensors[1].conj(), self.tensors[1], self.bath_op_onsite],
                      [[4, 5, 2, 1, -1], [4, 5, 2, 3, -2], [1, 3]])
        self.bath_op_onsite += op1 + op2

        opp = [self.operators[5], self.operators[4], self.operators[3]]  # sd, su, sz on the physical site
        opb = [self.effective_ops[2], self.effective_ops[3], self.effective_ops[1]]  # su, sd, sz on the 1st bath site
        for n in range(3):
            # 1st physical - 1st bath
            self.bath_op_onsite += j2 * (0.5 + 0.5 * (n == 2)) * tm.cont(
                [self.tensors[1].conj(), self.tensors[1], opb[n], opp[n]],
                [[2, 4, 1, 6, -1], [3, 4, 5, 6, -2], [1, 5], [2, 3]])
            # 2nd physical - 1st bath
            self.bath_op_onsite += j2 * (0.5 + 0.5 * (n == 2)) * tm.cont(
                [self.tensors[1].conj(), self.tensors[1], opb[n], opp[n]],
                [[3, 2, 5, 1, -1], [3, 4, 5, 6, -2], [1, 6], [2, 4]])
            # 1st physical - 2nd bath
            # self.bath_op_onsite += j2 * (0.5 + 0.5 * (n == 2)) * tm.cont(
            #     [self.tensors[1].conj(), self.tensors[1], opb[n], opp[n]],
            #     [[2, 6, 3, 1, -1], [5, 6, 3, 4, -2], [1, 4], [2, 5]])

        opp = [self.operators[5], self.operators[4], self.operators[3]]  # sd, su, sz on the physical site
        opb = [self.effective_ops[6], self.effective_ops[7], self.effective_ops[5]]  # su, sd, sz on the 2nd bath site
        for n in range(3):
            # 1st physical - 2nd bath
            self.bath_op_onsite += j2 * (0.5 + 0.5 * (n == 2)) * tm.cont(
                [self.tensors[1].conj(), self.tensors[1], opb[n], opp[n]],
                [[2, 4, 1, 6, -1], [3, 4, 5, 6, -2], [1, 5], [2, 3]])
            # 2nd physical - 2nd bath
            self.bath_op_onsite += j2 * (0.5 + 0.5 * (n == 2)) * tm.cont(
                [self.tensors[1].conj(), self.tensors[1], opb[n], opp[n]],
                [[3, 2, 5, 1, -1], [3, 4, 5, 6, -2], [1, 6], [2, 4]])
            # self.bath_op_onsite += j2 * (0.5 + 0.5 * (n == 2)) * tm.cont(
            #     [self.tensors[1].conj(), self.tensors[1], opb[n], opp[n]],
            #     [[5, 2, 1, 4, -1], [5, 6, 3, 4, -2], [1, 3], [2, 6]])
            # 2nd physical - 2nd bath

        op1 = [self.operators[4], self.operators[5], self.operators[3]]  # su, sd, sz on the 1st physical site
        op2 = [self.operators[5], self.operators[4], self.operators[3]]  # sd, su, sz on the 2nd physical site
        for n in range(3):
            self.bath_op_onsite += j1 * (0.5 + 0.5 * (n == 2)) * tm.cont(
                [self.tensors[1].conj(), self.tensors[1], op1[n], op2[n]],
                [[1, 2, 3, 4, -1], [5, 6, 3, 4, -2], [1, 5], [2, 6]])

        if abs(hx) > 1e-15:
            op1 = self.operators[1]  # sx
            self.bath_op_onsite -= hx * tm.cont(
                [self.tensors[1].conj(), self.tensors[1], op1],
                [[1, 5, 2, 3, -1], [4, 5, 2, 3, -2], [1, 4]])
            self.bath_op_onsite -= hx * tm.cont(
                [self.tensors[1].conj(), self.tensors[1], op1],
                [[4, 1, 2, 3, -1], [4, 5, 2, 3, -2], [1, 5]])
        if abs(hz) > 1e-15:
            op1 = self.operators[3]  # sz
            self.bath_op_onsite -= hz * tm.cont(
                [self.tensors[1].conj(), self.tensors[1], op1],
                [[1, 5, 2, 3, -1], [4, 5, 2, 3, -2], [1, 4]])
            self.bath_op_onsite -= hz * tm.cont(
                [self.tensors[1].conj(), self.tensors[1], op1],
                [[4, 1, 2, 3, -1], [4, 5, 2, 3, -2], [1, 5]])
        self.bath_op_onsite = (self.bath_op_onsite + self.bath_op_onsite.conj().T) / 2

    def update_central_tensor_kagome_effective_ops_fh(self, psi, tau, j1, j2, hx, hz):
        psi = psi.reshape((self.d, ) * 3 + (self.chi,) * 3)
        psi1 = psi.copy()
        # Projecting physical parts
        coup = [[4, 5, 0.5], [5, 4, 0.5], [3, 3, 1]]
        for cp in coup:
            op1 = self.operators[cp[0]]
            op2 = self.operators[cp[1]]
            psi1 -= tau * j1 * cp[2] * self.update_by_given_effective_ops(
                psi, [op1, op2], [0, 1])
            psi1 -= tau * j1 * cp[2] * self.update_by_given_effective_ops(
                psi, [op1, op2], [1, 2])
            psi1 -= tau * j1 * cp[2] * self.update_by_given_effective_ops(
                psi, [op1, op2], [0, 2])
        if abs(hx) > 1e-15:
            for ns in range(3):
                psi1 += tau * hx * self.update_by_given_effective_ops(
                    psi, [self.operators[1]], [ns])
        if abs(hz) > 1e-15:
            for ns in range(3):
                psi1 += tau * hz * self.update_by_given_effective_ops(
                    psi, [self.operators[3]], [ns])
        # Projecting on-site bath parts
        for n in range(3, 6):
            psi1 -= tau * self.update_by_given_effective_ops(psi, [self.bath_op_onsite], [n])

        # Projecting physical-bath parts [environments are symmetrical]
        #  - The interactions of the physical and the 1st bath sites
        opp = [self.operators[5], self.operators[4], self.operators[3]]  # sd, su, sz on the physical site
        opb = [self.effective_ops[2], self.effective_ops[3], self.effective_ops[1]]  # su, sd, sz on the 1st bath site
        coff = [0.5, 0.5, 1]
        for n in range(3):
            psi1 -= tau * j2 * coff[n] * self.update_by_given_effective_ops(
                psi, [opb[n], opp[n]], [3, 0])
            psi1 -= tau * j2 * coff[n] * self.update_by_given_effective_ops(
                psi, [opb[n], opp[n]], [4, 1])
            psi1 -= tau * j2 * coff[n] * self.update_by_given_effective_ops(
                psi, [opb[n], opp[n]], [5, 2])

        #  - The interactions of the physical and the 2nd bath sites
        opp = [self.operators[5], self.operators[4], self.operators[3]]  # sd, su, sz on the physical site
        opb = [self.effective_ops[6], self.effective_ops[7], self.effective_ops[5]]  # su, sd, sz on the 2nd bath site
        coff = [0.5, 0.5, 1]
        for n in range(3):
            psi1 -= tau * j2 * coff[n] * self.update_by_given_effective_ops(
                psi, [opb[n], opp[n]], [3, 0])
            psi1 -= tau * j2 * coff[n] * self.update_by_given_effective_ops(
                psi, [opb[n], opp[n]], [4, 1])
            psi1 -= tau * j2 * coff[n] * self.update_by_given_effective_ops(
                psi, [opb[n], opp[n]], [5, 2])
        return psi1.reshape(-1, )

    def update_central_tensor_kagome(self, tau, j1, j2, hx, hz, is_real=True):
        # compatible to one-site and two-site iDMRG
        s = self.tensors[0].shape
        dim = (self.chi ** 3) * (self.d ** 3)
        h = LinearOp((dim, dim),
                     lambda v: self.update_central_tensor_kagome_effective_ops_fh(
                         v, tau, j1, j2, hx, hz))
        self.tensors[0] = eigs(h, 1, which='LM', v0=self.tensors[0].reshape(-1, 1))[1].reshape(s)
        if is_real:
            self.tensors[0] = self.tensors[0].real

    def rho_from_central_tensor_kagome(self):
        self.rho = list()
        z = np.linalg.norm(self.tensors[0])
        tmp = self.tensors[0].reshape((self.d,) * 3 + (self.chi,) * 3) / z
        self.rho.append(np.tensordot(tmp.conj(), tmp, [(2, 3, 4, 5), (
            2, 3, 4, 5)]).reshape(self.d * self.d, self.d * self.d))
        self.rho.append(np.tensordot(tmp.conj(), tmp, [(1, 3, 4, 5), (
            1, 3, 4, 5)]).reshape(self.d * self.d, self.d * self.d))
        self.rho.append(np.tensordot(tmp.conj(), tmp, [(0, 3, 4, 5), (
            0, 3, 4, 5)]).reshape(self.d * self.d, self.d * self.d))

    def observation_kagome(self, j1, j2, hx, hz):
        # compatible to one-site iDMRG and one-site deep iDMRG
        z = np.linalg.norm(self.tensors[0].reshape(-1, ))
        tmp = self.tensors[0].reshape((self.d,) * 3 + (self.chi,) * 3) / z
        mag = dict()
        mag['x'] = np.zeros((3, ))
        mag['z'] = np.zeros((3, ))
        for n in range(3):
            ind = list(range(6))
            ind.pop(n)
            rho = np.tensordot(tmp.conj(), tmp, [ind, ind])
            mag['x'][n] = np.trace(rho.dot(self.operators[1]))
            mag['z'][n] = np.trace(rho.dot(self.operators[3]))

        energy = np.zeros(self.rho.__len__(), )
        j_coup = [j1, j1, j2]
        for n in range(self.rho.__len__()):
            h = j_coup[n] * (np.kron(self.operators[4], self.operators[5]) / 2 + np.kron(
                self.operators[5], self.operators[4]) / 2 + np.kron(
                self.operators[3], self.operators[3]))
            energy[n] = np.trace(self.rho[n].dot(h))
        energy_site = np.sum(energy) - hx * (
                mag['x'][0] / 2 + mag['x'][1] + mag['x'][2] / 2) - hz * (
                              mag['z'][0] / 2 + mag['z'][1] + mag['z'][2] / 2)
        energy_site /= 2

        lm = list()
        lm.append(np.linalg.svd(self.tensors[0].transpose(0, 3, 1, 2, 4, 5).reshape(
            self.d * self.chi, self.d * self.d * self.chi * self.chi), compute_uv=False))
        lm.append(np.linalg.svd(self.tensors[0].transpose(1, 4, 0, 2, 3, 5).reshape(
            self.d * self.chi, self.d * self.d * self.chi * self.chi), compute_uv=False))
        lm.append(np.linalg.svd(self.tensors[0].transpose(2, 5, 0, 1, 3, 4).reshape(
            self.d * self.chi, self.d * self.d * self.chi * self.chi), compute_uv=False))
        ent = [tm.entanglement_entropy(x) for x in lm]
        return energy, mag, energy_site, ent


