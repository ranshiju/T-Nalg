import numpy as np
from library.HamiltonianModule import spin_operators


class QES_1D:

    def __init__(self, d, chi, D, l_phys, tau, spin='half', operators=None):
        # l_phys: number of sites in the bulk to simulate the bath
        self.d = d
        self.chi = chi  # bath dimension
        self.D = D
        self.l_phys = l_phys
        self.tau = tau
        self.gate_phys = np.zeros(0)
        self.tensors_gate_phys = [np.zeros(0), np.zeros(0)]  # two tensors of the physical gates, by SVD or QR
        self.gate_bath = [np.zeros(0), np.zeros(0)]  # the two physical-bath Hamiltonians
        self.hamilt_bath = [np.zeros(0), np.zeros(0)]
        if operators is None:
            op_half = spin_operators(spin)
            self.operators = [op_half['id'], op_half['sx'], op_half['sy'], op_half['sz'],
                              op_half['su'], op_half['sd']]
        else:
            self.operators = operators

    def obtain_physical_gate_tensors(self, hamilt):
        """
        gate_phys: physical gate (or shifted physical Hamiltonian) is 4-th tensor
         0     1
          \   /
           G
         /  \
        2    3
        """
        self.gate_phys = np.eye(self.d ** 2) - self.tau * hamilt
        self.gate_phys = self.gate_phys.reshape(self.d, self.d, self.d, self.d)
        u, s, v = np.linalg.svd(self.gate_phys.transpose(0, 2, 1, 3).reshape(self.d**2, self.d**2))
        s = np.diag(s ** 0.5)
        self.tensors_gate_phys[0] = u.dot(s).reshape(self.d, self.d, self.d**2).transpose(0, 2, 1)
        self.tensors_gate_phys[1] = s.dot(v).reshape(self.d**2, self.d, self.d).transpose(1, 0, 2)

    def obtain_bath_h(self, env, which, way='shift'):
        """
        h_bath is 4-th tensor
         0     1
          \   /
           G
         /  \
        2    3
        """
        if (which is 'left') or (which is 'both'):
            self.gate_bath[0] = np.tensordot(env[0], self.tensors_gate_phys[1], ([1], [1]))
            self.gate_bath[0] = self.gate_bath[0].transpose(0, 2, 1, 3)
            s = self.gate_bath[0].shape
            self.hamilt_bath[0] = self.gate_bath[0].reshape(s[0] * s[1], s[2] * s[3]).copy()
            lm, u = np.linalg.eigh(self.hamilt_bath[0])
            lm /= np.max(lm)
            if way is 'shift':
                self.hamilt_bath[0] = u.dot(np.diag((np.ones((
                    s[0] * s[1],)) - lm) / self.tau)).dot(u.T.conj())
            else:
                self.hamilt_bath[0] = u.dot(np.diag(-np.log(abs(lm))/self.tau)).dot(u.T.conj())
            self.hamilt_bath[0] = self.hamilt_bath[0] - np.trace(self.hamilt_bath[0]) * np.eye(
                s[0]*s[1]) / (s[0]*s[1])
            self.hamilt_bath[0] = (self.hamilt_bath[0] + self.hamilt_bath[0].T.conj())/2
        if (which is 'right') or (which is 'both'):
            self.gate_bath[1] = np.tensordot(self.tensors_gate_phys[0], env[1], ([1], [1]))
            self.gate_bath[1] = self.gate_bath[1].transpose(0, 2, 1, 3)
            s = self.gate_bath[1].shape
            self.hamilt_bath[1] = self.gate_bath[1].reshape(s[0] * s[1], s[2] * s[3]).copy()
            lm, u = np.linalg.eigh(self.hamilt_bath[1])
            lm /= np.max(lm)
            if way is 'shift':
                self.hamilt_bath[1] = u.dot(np.diag((np.ones((
                    s[0] * s[1],)) - lm) / self.tau)).dot(u.T.conj())
            else:
                self.hamilt_bath[1] = u.dot(np.diag(-np.log(abs(lm))/self.tau)).dot(u.T.conj())
            self.hamilt_bath[1] = self.hamilt_bath[1] - np.trace(self.hamilt_bath[1]) * np.eye(
                s[0] * s[1]) / (s[0] * s[1])
            self.hamilt_bath[1] = (self.hamilt_bath[1] + self.hamilt_bath[1].T.conj()) / 2

    def obtain_bath_h_by_effective_ops_1d(self, hb_onsite, op_effective, h_index):
        self.hamilt_bath[0] = np.kron(hb_onsite, np.eye(self.d))
        for n in range(h_index.shape[0]):
            op1 = op_effective[int(h_index[n, 0])]
            op2 = self.operators[int(h_index[n, 1])]
            j = h_index[n, 2]
            self.hamilt_bath[0] += j * np.kron(op1, op2)
        self.hamilt_bath[1] = self.hamilt_bath[0].reshape(self.chi, self.d, self.chi, self.d).transpose(
            1, 0, 3, 2).reshape(self.d * self.chi, self.d * self.chi)

