from library.BasicFunctions import save_pr, load_pr
from algorithms.DMRG_anyH import dmrg_infinite_size
from library.QESclass import QES_1D
from library.EDspinClass import EDbasic
from library.Parameters import parameter_qes_gs_by_ed, parameter_qes_ft_by_ed
from library.HamiltonianModule import hamiltonian_heisenberg
from library.TensorBasicModule import entanglement_entropy
from scipy.sparse.linalg import LinearOperator as LinearOp
from scipy.sparse.linalg import eigsh as eigs
import os.path as opath


def prepare_bath_hamilts(para):
    print('Starting iDMRG for the entanglement bath')
    bath_data = opath.join(para['bath_path'], para['bath_exp'])
    if para['if_load_bath'] and opath.isfile(bath_data):
        print('Bath data found. Load the bath.')
        bath, ob0, hamilt = load_pr(bath_data, ['A', 'ob0', 'hamilt'])
    else:
        print('Bath data not found. Calculate bath by iDMRG.')
        hamilt = hamiltonian_heisenberg(para['spin'], para['jxy'], para['jxy'], para['jz'],
                                        para['hx'] / 2, para['hz'] / 2)
        bath, ob0 = dmrg_infinite_size(para, hamilt=hamilt)[:2]
        save_pr(para['bath_path'], para['bath_exp'], [bath, ob0, hamilt], ['A', 'ob0', 'hamilt'])
    if (bath.is_symme_env is True) and (bath.dmrg_type is 'mpo'):
        bath.env[1] = bath.env[0]

    print('Preparing the physical-bath Hamiltonians')
    qes = QES_1D(para['d'], para['chi'], para['d'] * para['d'], para['l_phys'], para['tau'])
    if bath.dmrg_type is 'mpo':
        qes.obtain_physical_gate_tensors(hamilt)
        qes.obtain_bath_h(bath.env, 'both')
    else:
        qes.obtain_bath_h_by_effective_ops_1d(
            bath.bath_op_onsite, bath.effective_ops, bath.hamilt_index)
    hamilts = [hamilt] + qes.hamilt_bath
    return hamilts, bath, ob0


def qes_gs_1d_ed(para=None):
    if para is None:
        para = parameter_qes_ft_by_ed()
    hamilts, bath, ob0 = prepare_bath_hamilts(para)
    print('Starting ED for the entanglement bath')
    dims = [para['d'] for _ in range(para['l_phys'])]
    dims = [para['chi']] + dims + [para['chi']]
    ob = dict()
    solver = EDbasic(dims)
    heff = LinearOp((solver.dim_tot, solver.dim_tot),
                    lambda x: solver.project_all_hamilt(
                        x, hamilts, para['tau'], para['couplings']))
    ob['e_eig'], solver.v = eigs(heff, k=1, which='LM', v0=solver.v.reshape(-1, ).copy())
    solver.is_vec = True
    ob['e_eig'] = (1 - ob['e_eig']) / para['tau']
    ob['mx'], ob['mz'] = solver.observe_magnetizations(para['phys_sites'])
    ob['eb'] = solver.observe_bond_energies(hamilts[0], para['positions_h2'][1:para['num_h2']-1, :])
    ob['lm'] = solver.calculate_entanglement()
    ob['ent'] = entanglement_entropy(ob['lm'])
    ob['e_site'] = sum(ob['eb']) / (para['l_phys'] - 1)
    ob['corr_xx'] = solver.observe_correlations(para['pos4corr'], para['op'][1])
    ob['corr_zz'] = solver.observe_correlations(para['pos4corr'], para['op'][3])
    for n in range(para['pos4corr'].shape[0]):
        p1 = para['pos4corr'][n, 0] - 1
        p2 = para['pos4corr'][n, 1] - 1
        ob['corr_xx'][n] -= ob['mx'][p1] * ob['mx'][p2]
        ob['corr_zz'][n] -= ob['mz'][p1] * ob['mz'][p2]
    return bath, solver, ob0, ob


def qes_ft_1d_ltrg(para=None):
    if para is None:
        para = parameter_qes_gs_by_ed()
    hamilts, bath, ob0 = prepare_bath_hamilts(para)
    print('Starting ED for the entanglement bath')
    dims = [para['d'] for _ in range(para['l_phys'])]
    dims = [para['chi']] + dims + [para['chi']]
    ob = dict()
    solver = EDbasic(dims)
    heff = LinearOp((solver.dim_tot, solver.dim_tot),
                    lambda x: solver.project_all_hamilt(
                        x, hamilts, para['tau'], para['couplings']))
    ob['e_eig'], solver.v = eigs(heff, k=1, which='LM', v0=solver.v.reshape(-1, ).copy())
    solver.is_vec = True
    ob['e_eig'] = (1 - ob['e_eig']) / para['tau']
    ob['mx'], ob['mz'] = solver.observe_magnetizations(para['phys_sites'])
    ob['eb'] = solver.observe_bond_energies(hamilts[0], para['positions_h2'][1:para['num_h2']-1, :])
    ob['lm'] = solver.calculate_entanglement()
    ob['ent'] = entanglement_entropy(ob['lm'])
    ob['e_site'] = sum(ob['eb']) / (para['l_phys'] - 1)
    ob['corr_xx'] = solver.observe_correlations(para['pos4corr'], para['op'][1])
    ob['corr_zz'] = solver.observe_correlations(para['pos4corr'], para['op'][3])
    for n in range(para['pos4corr'].shape[0]):
        p1 = para['pos4corr'][n, 0] - 1
        p2 = para['pos4corr'][n, 1] - 1
        ob['corr_xx'][n] -= ob['mx'][p1] * ob['mx'][p2]
        ob['corr_zz'][n] -= ob['mz'][p1] * ob['mz'][p2]
    return bath, solver, ob0, ob