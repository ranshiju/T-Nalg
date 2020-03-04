import numpy as np
import library.TensorBasicModule as tm
from library.HamiltonianModule import spin_operators


class EDbasic:
    """
    The main step to use this function is to prepare:
     1. hamilts: all inequivalent Hamiltonians
     2. para['couplings']: it has three columns - [site-1, site-2, num_h], meaning the site-1 and site-2 interact with
        the hamiltonian of num_h in hamilts
    """

    def __init__(self, dims, state_type='pure', ini='random', operators=None):
        self.dims = dims
        self.dim_tot = int(np.prod(self.dims))
        self.l = len(self.dims)

        self.is_vec = True  # If self.v is saved as a vector or tensor
        self.state_type = state_type
        if type(ini) is np.ndarray:
            self.v = ini
        elif ini is 'random':
            if state_type is 'pure':
                self.v = np.random.randn(self.dim_tot, )
                self.v = tm.normalize_tensor(self.v)[0]
            else:
                self.v = np.random.randn(self.dim_tot, self.dim_tot)
                self.v = self.v + self.v.transpose((1, 0))
        else:
            if state_type is 'pure':
                self.v = np.ones((self.dim_tot, 1)) / (self.dim_tot ** 0.5)
            else:
                self.v = np.eye(self.dim_tot)
        if operators is None:
            op = spin_operators('half')
            self.operators = [op['id'], op['sx'], op['sy'], op['sz'], op['su'],
                              op['sd']]
        else:
            self.operators = operators

    def reshape_v_tensor(self):
        if self.is_vec:
            self.v = self.v.reshape(self.dims)
            self.is_vec = False

    def reshape_v_vector(self):
        if not self.is_vec:
            self.v = self.v.reshape(-1, )
            self.is_vec = True

    def contract_with_local_matrix(self, v, mat, bonds):
        # self.reshape_v_tensor()
        ind = list(range(0, self.l))
        for b in bonds:
            ind.remove(b)
        dim1 = list()
        s = v.shape
        for n in bonds:
            dim1.append(s[n])
        dim2 = list()
        for n in ind:
            dim2.append(s[n])
        v = v.transpose(bonds + ind).reshape(np.prod(dim1), np.prod(dim2))
        v = mat.dot(v).reshape(dim1 + dim2)
        v = v.transpose(np.argsort(bonds + ind))
        return v

    def project_all_hamilt(self, v0, hamilts, tau, couplings):
        # This function is usually used as the function handle for the GS simulation
        v0 = v0.reshape(self.dims)
        v = v0.copy()
        nh = couplings.shape[0]
        for n in range(nh):
            p1, p2, c = tuple(couplings[n, :])
            v -= tau * self.contract_with_local_matrix(v0, hamilts[c], [p1, p2])
        return v.reshape(-1, )

    def reduced_matrix(self, bonds1):
        self.reshape_v_tensor()
        bonds2 = list(range(self.l))
        for b in bonds1:
            bonds2.remove(b)
        dim1 = list()
        for n in bonds1:
            dim1.append(self.dims[n])
        dim2 = list()
        for n in bonds2:
            dim2.append(self.dims[n])
        mat = self.v.transpose(bonds1 + bonds2).reshape(np.prod(dim1), np.prod(dim2))
        mat = mat.conj().dot(mat.transpose(1, 0))  # reduced density matrix
        return mat

    def observe_operator(self, op, bonds):
        mat = self.reduced_matrix(bonds)
        return np.trace(mat.dot(op)) / np.trace(mat)

    def observe_magnetizations(self, pos):
        mx = list()
        mz = list()
        for n in pos:
            mat = self.reduced_matrix([n])
            z = np.trace(mat)
            mat /= z
            mx.append(np.trace(mat.dot(self.operators[1])))
            mz.append(np.trace(mat.dot(self.operators[3])))
        return mx, mz

    def observe_bond_energies(self, hamilt, pos2):
        eb = list()
        for n in range(pos2.shape[0]):
            eb.append(self.observe_operator(hamilt, list(pos2[n, :2])))
        return eb

    def calculate_entanglement(self, position=-1):
        if position < 0:
            position = int(self.l / 2 + 0.1)  # in the middle
        lm = np.linalg.svd(self.v.reshape(int(np.prod(self.dims[:position])),
                                          int(np.prod(self.dims[position:]))), compute_uv=False)
        return lm

    def observe_correlations(self, pos2, op):
        corr = list()
        for n in range(pos2.shape[0]):
            mat = self.reduced_matrix(list(pos2[n, :2]))
            z = np.trace(mat)
            mat /= z
            corr.append(np.trace(mat.dot(np.kron(op, op))))
        return corr


