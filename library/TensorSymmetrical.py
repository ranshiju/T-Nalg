import numpy as np
from library import BasicFunctions as bf, TensorBasicModule as tm


class TensorZ2:

    def __init__(self, data, parity=0):
        self.parity = parity
        if type(data) is dict:
            self.data = data
            one_data = data['00']
            self.ndim = one_data.ndim
            self.shape = [one_data.shape[n] * 2 for n in range(0, self.ndim)]
        else:
            self.ndim = data.ndim
            self.shape = list(data.shape)
            self.data = dict()
            self.normal_to_z2(data)
        for n in range(0, self.ndim):
            if self.shape[n] % 2 == 1:
                bf.print_error('DimError: for a Z2 tensor, all bond dimensions should be even')

    def normal_to_z2(self, data):
        data = self.permutation_transformation(data)
        ind_key = bf.get_z2_indexes(self.ndim, self.parity)
        for key in ind_key:
            ind_now = self.interpret_key2slice(self.shape, key)
            self.data[key] = eval('data[' + ind_now + ']')

    def reshape_combine_z2(self, indexes):
        # reshape the indexes from indexes[0] to indexes[1] as one large bond
        for n in range(min(indexes), max(indexes)):
            self.reshape_combine_two_bonds_z2(min(indexes))

    def permute_z2(self, indexes):
        for key in self.data:
            self.data[key] = self.data[key].transpose(indexes)

    def reshape_combine_two_bonds_z2(self, nb):
        # reshape nb and nb+1 together
        data = dict()
        for key in self.data:
            dim0 = self.data[key].shape[:nb]
            dim1 = self.data[key].shape[nb + 2:]
            dim_now = self.data[key].shape[nb] * self.data[key].shape[nb + 1]
            tmp = self.data[key].reshape(dim0 + (dim_now,) + dim1)
            key_num = [int(x) for x in key]
            key_new = key[:nb] + str((key_num[nb] + key_num[nb + 1]) % 2) + key[nb + 2:]
            if key_new not in data:
                s = list(tmp.shape)
                s[nb] *= 2
                data[key_new] = np.zeros(s)
            if (key_num[nb] == 0 and key_num[nb + 1] == 0) or \
                    (key_num[nb] == 0 and key_num[nb + 1] == 1):
                data[key_new] = tm.operate_tensor_slice(data[key_new], nb, ':'+str(dim_now), tmp)
            else:
                data[key_new] = tm.operate_tensor_slice(data[key_new], nb, str(dim_now)+':', tmp)
        self.data = data
        self.ndim -= 1
        self.shape[nb] = self.shape[nb] * self.shape[nb + 1]
        self.shape.pop(nb + 1)

    def reshape_split_two_z2(self, bond, dims):
        data = dict()
        dims_ = [int(n/2) for n in dims]
        for key in self.data:
            s = list(self.data[key].shape)
            s = s[:bond] + list(dims_) + s[bond+1:]
            key_num = [int(x) for x in key]
            if key_num[bond] == 0:
                key_now = key[:bond] + str(0) + str(0) + key[bond+1:]
                slice_exp = self.interpret_full_slice(self.data[key].ndim)
                slice_exp = slice_exp[:bond * 2] + ':' + str(dims_[0] * dims_[1]) + slice_exp[bond * 2 + 1:]
                data[key_now] = eval('self.data[key][' + slice_exp + ']').reshape(s)

                key_now = key[:bond] + str(1) + str(1) + key[bond + 1:]
                slice_exp = self.interpret_full_slice(self.data[key].ndim)
                slice_exp = slice_exp[:bond * 2] + str(dims_[0] * dims_[1]) + ':' + slice_exp[bond * 2 + 1:]
                data[key_now] = eval('self.data[key][' + slice_exp + ']').reshape(s)
            if key_num[bond] == 1:
                key_now = key[:bond] + str(0) + str(1) + key[bond + 1:]
                slice_exp = self.interpret_full_slice(self.data[key].ndim)
                slice_exp = slice_exp[:bond * 2] + ':' + str(dims_[0] * dims_[1]) + slice_exp[bond * 2 + 1:]
                data[key_now] = eval('self.data[key][' + slice_exp + ']').reshape(s)

                key_now = key[:bond] + str(1) + str(0) + key[bond + 1:]
                slice_exp = self.interpret_full_slice(self.data[key].ndim)
                slice_exp = slice_exp[:bond * 2] + str(dims_[0] * dims_[1]) + ':' + slice_exp[bond * 2 + 1:]
                data[key_now] = eval('self.data[key][' + slice_exp + ']').reshape(s)
        self.data = data
        self.shape = self.shape[:bond] + list(dims) + self.shape[bond+1:]
        self.ndim += 1

    def reshape_split_z2(self, bond, dims):
        for n in range(len(dims)-1, 0, -1):
            self.reshape_split_two_z2(bond, [np.prod(dims[:n], dtype=int), dims[n]])

    def calculate_original_tensor(self):
        data = np.zeros(self.shape)
        for key in self.data:
            exp1 = self.interpret_key2slice(self.shape, key)
            exp2 = self.interpret_full_slice(self.ndim)
            exec('data[' + exp1 + '] = self.data[\'' + key + '\'][' + exp2 + ']')
        return self.permutation_transformation(data, if_inverse=True)

    def decomp_z2(self, index1, index2, decomp):
        if decomp is 'svd':
            tensor1 = dict()
            lm = dict()
            tensor2 = dict()
            nb1 = len(index1)
            nb2 = len(index2)
            shape1 = [int(self.shape[n] / 2) for n in index1]
            shape2 = [int(self.shape[n] / 2) for n in index2]
            self.permute_z2(list(index1) + list(index2))
            self.reshape_combine_z2([nb1, nb1+nb2-1])
            self.reshape_combine_z2([0, nb1-1])
            for key in self.data:
                tensor1[key], lm0, tensor2[key] = np.linalg.svd(self.data[key])
                lm[key] = np.diag(lm0)
            tensor1 = TensorZ2(tensor1)
            lm = TensorZ2(lm)
            tensor2 = TensorZ2(tensor2)
            tensor1.reshape_split_z2([0], shape1)
            tensor2.reshape_split_z2([1], shape2)
            return tensor1, lm, tensor2

    def mat_dot_z2(self, mat1):
        # mat1 should be a TensorZ2 object (Z2 matrix)
        mat = dict()
        if self.parity == 0 and mat1.parity == 0:
            mat['00'] = self.data['00'].dot(mat1.data['00'])
            mat['11'] = self.data['11'].dot(mat1.data['11'])
        elif self.parity == 0 and mat1.parity == 1:
            mat['01'] = self.data['00'].dot(mat1.data['01'])
            mat['10'] = self.data['11'].dot(mat1.data['10'])
        elif self.parity == 1 and mat1.parity == 0:
            mat['01'] = self.data['01'].dot(mat1.data['11'])
            mat['10'] = self.data['10'].dot(mat1.data['00'])
        elif self.parity == 1 and mat1.parity == 1:
            mat['00'] = self.data['01'].dot(mat1.data['10'])
            mat['11'] = self.data['10'].dot(mat1.data['01'])
        return TensorZ2(mat, parity=self.parity*mat1.parity)

    @staticmethod
    def interpret_key2slice(shape, key):
        ind_now = ''
        shape_ = [int(m / 2) for m in shape]
        even_odd = [int(m) for m in key]
        for n in range(0, even_odd.__len__()):
            if even_odd[n] == 0:
                ind_now += str(0) + ':' + str(shape_[n])
            else:
                ind_now += str(shape_[n]) + ':' + str(shape[n])
            if n != even_odd.__len__() - 1:
                ind_now += ','
        return ind_now

    @staticmethod
    def interpret_full_slice(ndim):
        exp = ''
        for n in range(0, ndim):
            exp += ':'
            if n != ndim - 1:
                exp += ','
        return exp

    @staticmethod
    def permutation_transformation(data, if_inverse=False):
        ind_con = [list(range(1, data.ndim + 1))]
        u = list()
        for n in range(0, data.ndim):
            order = list(range(0, data.shape[n], 2)) + list(range(1, data.shape[n], 2))
            u.append(tm.bond_permutation_transformation(order))
            if if_inverse:
                ind_con.append([n + 1, -n - 1])
            else:
                ind_con.append([-n-1, n+1])
        return tm.cont([data] + u, ind_con)
