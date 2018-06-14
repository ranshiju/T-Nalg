import numpy as np
import os.path as opath
import matplotlib.pyplot as mp
from termcolor import cprint, colored
from Parameters import parameter_dmrg
from Basic_Functions_SJR import input_and_check_type, input_and_check_value, \
    save_pr, load_pr, print_options, print_sep, plot_square_lattice, input_and_check_type_multiple_items, \
    plot_connections_polar
from Tensor_Basic_Module import sort_vectors
from DMRG_anyH import dmrg_finite_size

is_from_input = False
is_load_data = False


def print_info_to_be_added(model, method):
    print(colored(model, 'cyan') + ' by ' + colored(method, 'cyan') + ' are to be added')


def print_info_ref(topic, ref, url_ref):
    print(colored(topic + ': ' + ref, 'cyan') + ' (' + url_ref + ')')


def positions_set2array(pos_set):
    nh = pos_set.__len__()
    pos_set = list(pos_set)
    pos = np.zeros((nh, 2))
    for n in range(0, nh):
        pos[n, :] = np.array(pos_set[n])
    pos = pos.astype(int)
    pos -= np.min(pos)
    p_max = np.max(pos)
    for i in range(p_max-1, 0, -1):
        number = np.nonzero(pos == i)[0].size
        if number == 0:
            pos -= (pos > i)
    return pos


def sort_positions(pos, which='ascend'):
    order = np.argsort(pos[:, 0])
    if which is 'descend':
        order = order[::-1]
    return sort_vectors(pos, order, 'row')


# =========================================================
print('Thanks for using EasyStart_DMRG!' + colored('(v2018.06-1, by S.J Ran)', 'cyan'))
print_sep()
print('For any questions or comments, please leave us messages on GitHub '
      'or ResearchGate, \n or email to ' + colored('ranshiju10@mails.ucas.ac.cn', 'blue'))
print('Important: you need to install ' + colored('numpy, scipy, and matplotlib', 'magenta') + ' in your python')
print('You are more than welcome to use/modify our code for any simulations.')
print('GitHub: https://github.com/ranshiju/TensorNetworkClassLibary')
print('ResearchGate: www.researchgate.net/project/TN-Encoding-and-Applications')
print('It would greatly appreciated to ' +
      colored('cite our review paper (https://arxiv.org/abs/1708.09213)', 'cyan') + ' if possible')

print_sep('Info: coming updates')
print_info_to_be_added('Finite-size square lattices', 'DMRG')
print_info_to_be_added('Finite-size kagome lattices', 'DMRG')
print_info_to_be_added('Infinite-size chains', 'iDMRG/AOP(1D)')
print_info_to_be_added('Infinite square lattice', 'simple-update/AOP(2D)')
print_info_to_be_added('Finite-temperature simulations of 1D/2D systems', 'LTRG/super-orthogonalization/AOP')
print_sep('Info: references')
print_info_ref('DMRG', 'Phys. Rev. Lett. 69, 2863 (1992)',
               'https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.69.2863')
print_info_ref('Simple update', 'Phys. Rev. Lett. 101, 090603 (2008)',
               'https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.090603')
print_info_ref('AOP (1D)', 'Phys. Rev. E 93, 053310 (2016)',
               'https://journals.aps.org/pre/abstract/10.1103/PhysRevE.93.053310')
print_info_ref('AOP (2D&3D)', 'Phys. Rev. B 96, 155120 (2017)',
               'https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.155120')
cprint('If you want to know more about tensor networks, we recommend our review '
       '(https://arxiv.org/abs/1708.09213)', 'cyan')
print_sep()

if is_from_input:
    para = dict()
    print_sep('Start inputting parameters')
    print('Note: you can always set all parameters in \"Parameters.py\" and run \"dmrg_finite_size\"')
    print('For the models, currently only spin-1/2 Heisenberg models can be simulated')
    para['model'] = 'heisenberg'
    para['jxy'] = input_and_check_type((int, float), 'jxy')
    para['jz'] = input_and_check_type((int, float), 'jz')
    para['hx'] = input_and_check_type((int, float), 'hx')
    para['hz'] = input_and_check_type((int, float), 'hz')

    print_options(('chain', 'square', 'arbitrary'), welcome='For the lattice, please chose: ')
    tmp = input_and_check_value([1, 2, 3], ('chain', 'square', 'arbitrary'), 'lattice', 'para')
    para['lattice'] = ['chain', 'square', 'arbitrary'][tmp-1]
    if para['lattice'] == 'chain':
        print('For the length of the chain')
        para['l'] = input_and_check_type(int, 'l')
    elif para['lattice'] == 'square':
        para['square_width'] = input_and_check_type(int, 'square_width')
        para['square_height'] = input_and_check_type(int, 'square_height')
        para['l'] = para['square_width'] * para['square_height']
        plot_square_lattice(para['square_width'], para['square_height'], True,
                            'The sites are numbered in this way (close this figure to continue ...)',
                            '.\\fig_dmrg\\')
    elif para['lattice'] == 'arbitrary':
        print('Setting the positions of the two-body interactions')
        print('Each time, if you input ' + colored('n1,n2', 'cyan') + ' or ' + colored('(n1,n2)', 'cyan') +
              ', it means to add an interaction between the n1-th and n2-th sites')
        print('For example: ' + colored('1,2', 'cyan'))
        print('NOTE: input ' + colored('integers', 'cyan') + ' and make sure ' +
              colored('0 $\leq$ n1 < n2', 'cyan') + ', or the input will be ' + colored('invalid', 'magenta'))
        cprint('These numbers determine how the sites are located in the MPS, '
               'so optimize the numbering if possible', 'cyan')
        position_set = input_and_check_type_multiple_items(tuple, lambda values: values[0] < values[1],
                                                           'positions_h2 (input ' + colored('-1', 'cyan') +
                                                           ' to finish)', is_print=True)
        para['positions_h2'] = positions_set2array(position_set)
        para['positions_h2'] = sort_positions(para['positions_h2']).astype(int)
        print('The interactions terms you have input are (invalid sites are automatically removed):')
        cprint(str(para['positions_h2']), 'cyan')
        plot_connections_polar(para['positions_h2'], True, 'Close the figure to continue', '.\\fig_dmrg')
        para['l'] = np.max(para['positions_h2']) + 1
        para['bound_cond'] = ''
    if para['lattice'] != 'arbitrary':
        print_options(('open BC', 'periodic BC'), welcome='For the boundary condition (BC), please chose: ')
        tmp = input_and_check_value([1, 2], ('open', 'periodic'), 'bound_cond', 'para')
        para['bound_cond'] = (('open', 'periodic')[tmp-1])
    print('For the dimension cut-off (if you are not sure what to input, input -1)')
    para['chi'] = input_and_check_type(int, 'chi')
    if para['chi'] < 0:
        para['chi'] = max(min(para['l']*2, 128), 16)
        print('You have set ' + colored('para[\'chi\'] = ' + str(para['chi']), 'cyan'))
    elif para['chi'] > 128:
        cprint('Note: chi is quite large (=%d), it can be slow. '
               'Unless you are confident, suggest to choose a chi no larger than 128', 'cyan')

    print('Other parameters are set as default ...')
    para['spin'] = 'half'
    para['d'] = 2  # Physical bond dimension
    para['sweep_time'] = 500  # sweep time
    # Fixed parameters
    para['if_print_detail'] = False
    para['tau'] = 1e-3  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-10
    para['break_tol'] = 1e-8  # tolerance for breaking the loop
    para['is_real'] = True
    para['dt_ob'] = 5  # in how many sweeps, observe to check the convergence
    para['ob_position'] = (para['l']/2).__int__()  # to check the convergence, chose a position to observe
    para['data_path'] = '.\\data_dmrg\\'
    para['data_exp'] = 'N%d_j(%g,%g)_h(%g,%g)_chi%d' % \
                       (para['l'], para['jxy'], para['jz'], para['hx'],
                        para['hz'], para['chi']) + para['bound_cond']
    if para['lattice'] == 'square':
        para['data_exp'] = para['lattice'] + '(%d,%d)' % \
                           (para['square_width'], para['square_height']) + para['data_exp']
    else:
        para['data_exp'] = para['lattice'] + para['data_exp']
else:
    para = parameter_dmrg()

data_full_name = para['data_path'] + para['data_exp'] + '.pr'
save_pr('.\\para_dmrg\\', '_para.pr', (para,), ('para',))
print('The parameter have been saved as ' + colored('.\\para_dmrg\_para.pr', 'green'))

print_sep('Start DMRG simulation')
if is_load_data and opath.isfile(data_full_name) and (para['lattice'] is not 'arbitrary'):
    print('The data exists in ' + para['data_path'].rstrip("\\") + '. Load directly...')
    ob, A = load_pr(data_full_name, ('ob', 'A'))
elif (not is_load_data) or (not opath.isfile(data_full_name)) or (para['lattice'] is 'arbitrary'):
    ob, A, info, para = dmrg_finite_size(para)
if (not opath.isfile(data_full_name)) or (para['lattice'] is 'arbitrary'):
    if A._is_parallel:
        A.pool.close()
        A.pool.join()
    save_pr(para['data_path'], para['data_exp'] + '.pr', (ob, A, info, para), ('ob', 'A', 'info', 'para'))
    print('The data have been saved in ' + colored(para['data_path'].rstrip("\\"), 'green'))

print_sep('DMRG simulation finished')
if is_from_input:
    end_plot = False
    while not end_plot:
        print('Which property are you interested in (' + colored('to exit, input 0', 'cyan') + '):')
        options = ('bond energies', 'magnetization', 'entanglement entropy')
        print_options(options)
        x = input_and_check_value(np.vstack(((np.arange(1, len(options)+1)).reshape(-1, 1), 0)).T)
        if x == 0:
            end_plot = True
        elif x == 1:  # plot bond energies
            if para['lattice'] == 'chain':
                nh1 = ob['eb'].size - (para['bound_cond'] == 'periodic')
                mp.plot(range(0, nh1), ob['eb'][:nh1], 'bo')
                if para['bound_cond'] == 'periodic':
                    mp.plot(np.array([0, nh1-1]), ob['eb'][-1] * np.ones((2,)), 'r--.', linewidth=0.5)
                    mp.text(A.length / 2, ob['eb'][-1] - 0.0002, 'Eb(0, %d) = %g' % (A.length - 1, ob['eb'][-1]),
                            fontsize=10, verticalalignment="top", horizontalalignment="center")
            elif para['lattice'] == 'square' or 'arbitrary':
                nh1 = ob['eb'].size
                f1, = mp.plot(range(1, nh1 + 1), ob['eb'][:nh1], 'bo')
                mp.title('Bond energies (nearest-neighbor correlators)')
            mp.xlabel('lattice bond')
            mp.ylabel(r'$\langle \hat{s}_n \hat{s}_{n+1} \rangle$')
            print('Bond energies = ')
            cprint(str(ob['eb'].T), 'cyan')
            print('Energy per site = ' + colored(str(ob['e_per_site']), 'cyan'))
            if para['lattice'] == 'square':
                print('NOTE: check ' + colored('para[positions_h2]', 'cyan') + 'to see how the bonds are numbered')
        elif x == 2:  # plot magnetization
            mp.subplot(2, 1, 1)
            f1, = mp.plot(range(1, A.length + 1), ob['mx'], '-ro')
            mp.ylabel(r'$\langle \hat{s}^x \rangle$')
            # mp.legend(handles=[f1, ], labels=[r'$\langle \hat{s}_n^x \rangle$'], loc='best')
            mp.subplot(2, 1, 2)
            f2, = mp.plot(range(1, A.length + 1), ob['mz'], '--bs')
            mp.xlabel('lattice site')
            mp.ylabel(r'$\langle \hat{s}^z \rangle$')
            # mp.legend(handles=[f2, ], labels=[r'$\langle \hat{s}_n^z \rangle$'], loc='best')
            print('mx = ')
            print(str(ob['mx'].T))
            print('mz = ')
            print(str(ob['mz'].T))
            if para['lattice'] == 'square':
                print('Check the numbers of sites in .\\fig_dmrg\\' + para['lattice']
                      + '(%d,%d).png' % (para['square_width'], para['square_height']))
        elif x == 3:
            mp.plot(range(1, A.length), A.ent, '--or')
            mp.xlabel('lattice bond')
            mp.ylabel('entanglement entropy')
            print('entanglement entropy = ')
            print(str(A.ent.T))
        mp.show()
print_sep('Thanks again for using EasyStartDMRG!')

