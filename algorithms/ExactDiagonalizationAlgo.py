from library.EDspinClass import EDbasic
from library.TensorBasicModule import entanglement_entropy
from library.Parameters import parameters_ed_time_evolution, parameters_ed_ground_state
from library.HamiltonianModule import hamiltonian_heisenberg
from scipy.sparse.linalg import eigsh as eigs
import scipy.sparse.linalg as la
from scipy.sparse.linalg import LinearOperator as LinearOp


# !!!: ED class has been modified to fit the QES simulation;
# But this algorithm code has NOT been modified accordingly
def exact_ground_state(para=None):
    if para is None:
        para = parameters_ed_ground_state('chain')
    hamilt = hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                    para['hx']/2, para['hz']/2)
    dims = [para['d'] for _ in range(para['l'])]
    a = EDbasic(dims)
    ob = dict()
    dim = para['d'] ** para['l']
    heff = LinearOp((dim, dim), lambda x: a.project_all_hamilt(x, [hamilt], para['tau'],
                                                               para['couplings']))
    ob['e_eig'], a.v = eigs(heff, k=1, which='LM', v0=a.v.reshape(-1, ).copy())
    a.is_vec = True
    ob['e_eig'] = (1 - ob['e_eig']) / para['tau']
    ob['mx'], ob['mz'] = a.observe_magnetizations(list(range(para['l'])))
    ob['eb'] = a.observe_bond_energies(hamilt, para['positions_h2'])
    ob['lm'] = a.calculate_entanglement()
    ob['ent'] = entanglement_entropy(ob['lm'][-1])
    ob['e_site'] = sum(ob['eb']) / para['l']
    # ob['e_site'] = sum(ob['eb']) / para['l'] - sum(ob['mx']) * para['hx'] / para['l'] - \
    #                sum(ob['mz']) * para['hz'] / para['l']
    return a, ob


def exact_time_evolution(v0, para=None):
    if para is None:
        para = parameters_ed_time_evolution('chain')
    a = EDbasic(para['d'], para['l'], ini=v0)

    hamilt = hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                    para['hx']/2, para['hz']/2)
    gate = la.expm(1j * para['tau'] * hamilt)

    ob = dict()
    ob['lm'] = list()
    ob['ent'] = list()
    ob['mx'] = list()
    ob['mz'] = list()
    ob['eb'] = list()
    ob['e_site'] = list()

    time_now = 0
    for t in range(para['iterate_time']):
        for n in range(para['num_h2']):
            a.contract_with_local_matrix(gate, list(para['positions_h2'][n, :]))
        time_now += para['tau']
        tmp = time_now / para['ob_dt']
        if abs(tmp - round(tmp)) < 1e-9:
            mx, mz = a.observe_magnetizations()
            ob['mx'].append(mx)
            ob['mz'].append(mz)
            ob['eb'].append(a.observe_bond_energies(hamilt, para['positions_h2']))
            ob['lm'].append(a.calculate_entanglement())
            ob['ent'].append(entanglement_entropy(ob['lm'][-1]))
            ob['e_site'].append(sum(ob['eb'][-1]) / para['l'] + sum(
                    ob['mx'][-1]) * para['hx'] / para['l'] + sum(
                    ob['mz'][-1]) * para['hz'] / para['l'])
            print('t = ' + str(time_now) + ', Mz = ' + str(sum(ob['mz'])/a.l))
