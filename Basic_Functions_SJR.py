# Notes (about some frequently used BIF):
# np.hstack((m1,m2)): 合并两个矩阵，成（d, d1+d2）
# np.vstack((m1,m2))：合并两个矩阵， 成（d1+d2，d）

import os
import pickle
import inspect
import numpy as np
import matplotlib.pyplot as mp
from termcolor import cprint, colored


def info_contact():
    """Return the information of the contact"""
    info = dict()
    info['name'] = 'S.J. Ran'
    info['email'] = 'ranshiju10@mail.s ucas.ac.cn'
    info['affiliation'] = 'ICFO – The Institute of Photonic Sciences'
    return info


def save_pr(path, file, data, names):
    """
    Save the data as a dict in a file located in the path
    :param path: the location of the saved file
    :param file: the name of the file
    :param data: the data to be saved
    :param names: the names of the data
    Notes: 1. Conventionally, use the suffix \'.pr\'. 2. If the folder does not exist, system will
    automatically create one. 3. use \'load_pr\' to load a .pr file
    Example:
    >>> x = 1
    >>> y = 'good'
    >>> save_pr('.\\test', 'ok.pr', [x, y], ['name1', 'name2'])
      You have a file '.\\test\\ok.pr'
    >>> z = load_pr('.\\test\\ok.pr')
      z = {'name1': 1, 'name2': 'good'}
      type(z) is dict
    """
    mkdir(path)
    s = open(os.path.join(path, file), 'wb')
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    pickle.dump(tmp, s)
    s.close()


def load_pr(path_file, names=None):
    if os.path.isfile(path_file):
        s = open(path_file, 'rb')
        if names is None:
            data = pickle.load(s)
            s.close()
            return data
        else:
            tmp = pickle.load(s)
            if type(names) is str:
                data = tmp[names]
            elif type(names) is list or type(names) is tuple:
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
            s.close()
            return tuple(data)
    else:
        return False


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    path_flag = os.path.exists(path)
    if not path_flag:
        os.makedirs(path)
    return path_flag


def sort_list(a, order):
    return [a[i] for i in order]


def empty_list(n, content=None):
    # make a empty list of size n
    return [content for _ in range(0, n)]


def arg_find_array(arg, n=1, which='first'):
    # find the first/last n True's in the arg
    # like the "find" function in Matlab
    # the input must be array or ndarray
    x = np.nonzero(arg)
    length = x[0].size
    num = min(length, n)
    dim = arg.ndim
    if dim > 1 and (not (dim == 2 and arg.shape[1] == 1)):
        y = np.zeros((dim, num), dtype=int)
        if which == 'last':
            for i in range(0, dim):
                y[i, :] = x[i][length-num:length]
        else:
            for i in range(0, dim):
                y[i, :] = x[i][:num]
    else:
        if which == 'last':
            y = x[0][length - num:length]
        else:
            y = x[0][:num]
        if n == 1:
            y = y[0]
    return y


# =========================================
# Print & Check functions
def trace_stack(level0=2):
    # print the line and file name where this function is used
    info = inspect.stack()
    ns = info.__len__()
    for ns in range(level0, ns):
        cprint('in ' + str(info[ns][1]) + ' at line ' + str(info[ns][2]), 'green')


def print_dict(a, keys=None, welcome='', style_sep=': ', color='white', end='\n'):
    express = welcome
    if keys is None:
        for n in a:
            express += n + style_sep + str(a[n]) + end
    else:
        if type(keys) is str:
            express += keys.capitalize() + style_sep + str(a[keys])
        else:
            for n in keys:
                express += n.capitalize() + style_sep + str(a[n])
                if n is not keys[-1]:
                    express += end
    cprint(express, color)
    return express


def print_error(string, if_trace_stack=True):
    cprint(string, 'magenta')
    if if_trace_stack:
        trace_stack(3)


def print_sep(info='', style='=', length=40, color='cyan'):
    if info == '':
        cprint(style * (length * 2), color)
    else:
        l_info = info.__len__()
        l_new = length * 2 - 2 - l_info
        dl = l_new % 2
        l_new = int(l_new/2)
        l_new = max(l_new, 0)
        mes = style*l_new + ' ' + info + ' ' + style*(l_new + dl)
        cprint(mes, color)


def print_options(options, start=1, welcome='', style_sep=': ', end='    ', color='cyan'):
    message = welcome
    length = len(options)
    for i in range(0, length):
        message += colored(str(i + start) + style_sep + options[i], color)
        if i < length-1:
            message += end
    print(message)


def input_and_check_type(right_type, name, print_result=True, dict_name='para'):
    # right_type should be a tuple
    ok = False
    some_error = True
    while some_error:
        try:
            input_bad = True
            while input_bad:
                value = input('Please input the value of ' + name + ': ')
                if value.__len__() > 0:
                    value = eval(value)
                    input_bad = False
            while not ok:
                if isinstance(value, right_type):
                    ok = True
                else:
                    value = eval(input(name + ' should be ' + str(right_type) + ', please input again: '))
            some_error = False
        except NameError or ValueError or SyntaxError:
            cprint('The input is illegal, please input again ...', 'magenta')
    if print_result:
        print('You have set ' + colored(dict_name + '[\'' + name + '\'] = ' + str(value), 'cyan'))
    return value


def input_and_check_value(right_value, values_str=(), names='', dict_name='', start_ind=-1):
    # right_value should be an array
    ok = False
    some_error = True
    while some_error:
        try:
            input_bad = True
            while input_bad:
                value = input('Please input your choose: ')
                if value.__len__() > 0:
                    value = eval(value)
                    input_bad = False
            right_value = np.array(right_value)
            while not ok:
                if np.any(value == right_value):
                    ok = True
                else:
                    value = eval(input('Input should be ' + colored(str(right_value), 'cyan')
                                       + ', please input again: '))
            some_error = False
        except NameError or ValueError or SyntaxError:
            cprint('The input is illegal, please input again ...', 'magenta')
    if values_str.__len__() > 0:
        if start_ind < 0:
            start_ind = right_value[0]
        print('You have set ' + colored(dict_name + '[\'' + names + '\'] = \'' +
                                        str(values_str[value - start_ind]) + '\'', 'cyan'))
    return value


def check_condition(x, cond):
    from inspect import isfunction
    if not isfunction(cond):
        return False
    try:
        return cond(x)
    except TypeError or IndexError or ValueError:
        cprint('Wrong input in check_condition')
        return False


def input_and_check_type_multiple_items(right_type0, cond=None, name='your terms', max_len=100,
                                        key_stop=-1, is_print=False):
    # cond(inout) is True or False, a function to judge if the input is satisfictary
    output = set()
    # add the type of the key_stop in the tuple of the right types
    if type(right_type0) is tuple:
        right_type = right_type0 + (type(key_stop), )
    elif type(right_type0) is type:
        right_type = (right_type0, type(key_stop))
    not_stop = True
    while not_stop:
        new = input_and_check_type(right_type, name, False)
        if new == key_stop:
            cprint('You input the key to stop. Input completed.', 'cyan')
            not_stop = False
        elif not check_condition(new, cond):
            cprint('The input is invalid since it does not satisfy the condition', 'magenta')
        elif not isinstance(new, right_type0):
            if is_print:
                cprint('This input is invalid since its type is incorrect (should be %s or stop key)'
                       % str(right_type0), 'magenta')
        elif new in output:
            if is_print:
                cprint('This input is invalid since it already exists', 'magenta')
        else:
            output.add(new)
            if output.__len__() > max_len:
                cprint('Number if items exceeds the maximum. Stop the input', 'magenta')
                not_stop = False
    return output


# =========================================
# Plot functions
def plot_square_lattice(width, height, numbered=False, title='', save_path=None):
    from Hamiltonian_Module import positions_nearest_neighbor_square
    pos_1d = np.arange(0, width*height, dtype=int).reshape(height, width)
    index = positions_nearest_neighbor_square(width, height)
    for n in range(0, index.shape[0]):
        pos1 = arg_find_array(pos_1d == index[n, 0])
        pos2 = arg_find_array(pos_1d == index[n, 1])
        mp.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], '-ob', markersize=8)
    mp.axis('equal')
    if numbered:
        for w in range(0, width):
            for h in range(0, height):
                mp.text(h+0.06, w-0.06, str(pos_1d[h, w]), horizontalalignment='left',
                        verticalalignment='top', fontsize=15)
    mp.axis('off')
    mp.title(title)
    if save_path is not None:
        mkdir(save_path)
        mp.savefig(os.path.join(save_path, 'square(%d,%d).png' % (width, height)))
    mp.show()


def plot_connections_polar(positions, numbered=False, title='', save_path=None):
    nb = positions.shape[0]
    ax = mp.subplot(111, projection='polar')
    n_site = np.max(positions) + 1
    theta = np.linspace(0, 2*np.pi, n_site+1)
    x1 = np.zeros((nb, 1))
    x2 = np.zeros((nb, 1))
    for n in range(0, nb):
        x1[n] = theta[positions[n, 0]]
        x2[n] = theta[positions[n, 1]]
        ax.plot([x1[n], x2[n]], [1, 1], '-ob')
    if numbered:
        for n in range(0, n_site):
            mp.text(theta[n]+0.05, 1.1, str(n), horizontalalignment='center',
                    verticalalignment='top', fontsize=15)
    mp.axis('off')
    mp.title(title)
    if type(save_path) is str:
        mkdir(save_path)
        mp.savefig(os.path.join(save_path, 'arbitrary.png'))
    mp.show()
