import os, sys
import re
import pickle
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from termcolor import cprint, colored
from math import factorial
import struct
from mpl_toolkits.mplot3d import Axes3D
import torch as tc


def info_contact():
    """Return the information of the contact"""
    info = dict()
    info['name'] = 'S.J. Ran'
    info['email'] = 'ranshiju10@mail.s ucas.ac.cn'
    info['affiliation'] = 'ICFO – The Institute of Photonic Sciences'
    return info


def project_path(project='T-Nalg\\'):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    return cur_path[:cur_path.find(project) + len(project)]


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
    # print(os.path.join(path, file))
    s = open(os.path.join(path, file), 'wb')
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    pickle.dump(tmp, s)
    s.close()


def load_pr(path_file, names=None):
    """
    Load the file saved by save_pr as a dict from path
    :param path_file: the path and name of the file
    :param names: the specific names of the data you want to load
    :return  the file you loaded
    Notes: the file you load should be a  \'.pr\' file.
    Example:
        >>> x = 1
        >>> y = 'good'
        >>> z = [1, 2, 3]
        >>> save_pr('.\\test', 'ok.pr', [x, y, z], ['name1', 'name2', 'name3'])
        >>> A = load_pr('.\\test\\ok.pr')
          A = {'name1': 1, 'name2': 'good'}
        >>> y, z = load_pr('\\test\\ok.pr', ['y', 'z'])
          y = 'good'
          z = [1, 2, 3]
    """
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
                s.close()
                return data
            elif (type(names) is list) or (type(names) is tuple):
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                s.close()
                return tuple(data)
    else:
        return False


def mkdir(path):
    """
       Create a folder at your path
       :param path: the path of the folder you wish to create
       :return: the path of folder being created
       Notes: if the folder already exist, it will not create a new one.
    """
    path = path.strip()
    path = path.rstrip("\\")
    path_flag = os.path.exists(path)
    if not path_flag:
        os.makedirs(path)
    return path_flag


def search_file(path, exp):
    content = os.listdir(path)
    exp = re.compile(exp)
    result = list()
    for x in content:
        if re.match(exp, x):
            result.append(os.path.join(path, x))
    return result


def search_all_py(path, files_list=None):
    # Modified from: https://blog.csdn.net/CAIJINZHI/article/details/80521200
    if files_list is None:
        files_list = list()
    tmp_list = os.listdir(path)  # 获取path目录下所有文件
    for filename in tmp_list:
        tmp_path = os.path.join(path,filename)
        if os.path.isdir(tmp_path):
            search_all_py(tmp_path, files_list)
        elif filename[-3:].upper() == '.PY':
            files_list.append(tmp_path)
    return files_list


def print_vars_and_memories():
    # Do not directly use this function; copy and paste the code instead
    # Need to import get_size from this module
    var = dir()
    memo = np.zeros((var.__len__(),))
    for n in range(var.__len__()):
        memo[n] = eval('get_size(' + var[n] + ')')
    order = np.argsort(memo)[::-1]
    memo = memo[order]
    var = [var[n] for n in order]
    for n in range(var.__len__()):
        print(var[n] + ': ' + str(memo[n]))


def get_size(obj, seen=None):
    # From: https://www.cnblogs.com/blackprience/p/10692391.html
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def output_txt(x, filename='data'):
    np.savetxt(filename + '.txt', x)


def sort_list(a, order):
    """
    Return the elements sorted in the given order
    :param a: an iterable object
    :param order: the order of elements you want to sort
    :return: the new list contains only elements in the order
    Example:
        >>> a = [1, 2, 'a', 'b']
        >>> order = [1,3]
        >>> z = sort_list(a, order)
          z = [2, 'b']
    """
    return [a[i] for i in order]


def empty_list(n, content=None):
    """
    Create a list of size n with elements as None or content
    :param n: the size of list
    :param content: the content of all elements
    :return: a size n list with all elements are content
    Example:
        >>> z = empty_list(3)
          z = [None, None, None]
        >>> z = empty_list(4, 'a')
          z = ['a', 'a', 'a', 'a']
    """
    # make a empty list of size n
    return [content for _ in range(0, n)]


def remove_element_from_list(x, element):
    """
    Remove an element from a list
    :param x: a list
    :param element: an element to be removed
    :return: a list without 'element'
    Example:
       >>>x = [1, 2, 3]
       >>>print(arg_find_list(x, 3))
         [1, 2]
    """
    return list(filter(lambda a: a != element, x))


def arg_find_array(arg, n=1, which='first'):
    """
    Find the position of positions of elements required
    :param arg:  requirement of the elements needed to fulfil
    :param n:  number of how many elements you need
    :param which:  the first n elements or last elements
    :return:  the position of elements you need
    Notes: 1.arg should be boolean type, 2. if can't find n elements to suit your need, it will return all it can find
    Example:
        >>> x = np.array([-1, 2, -3])
        >>> z = arg_find_array(x < 0, 1, 'last')
          z = 2
    """
    x = np.nonzero(arg)
    length = x[0].size
    if length == 0:
        y = np.zeros(0)
    else:
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


def arg_find_list(x, target, n=1, which='first'):
    """
    Find the position of target elements in list
    :param x: a list
    :param target: target element
    :param n: how much elements needed to be find
    :param which: first or last
    :return: position
    Example:
       >>>x = [1, 2, 1, 3]
       >>>print(arg_find_list(x, 3, which='last'))
         [2]
    """
    # x should be a list or tuple (of course '1D')
    # for array or ndarray, please use arg_find_array
    n_found = 0
    n_start = 0
    ind = list()
    if which is 'last':
        x = x[::-1]
    for i in range(0, n):
        try:
            new_ind = x.index(target, n_start)
        except ValueError:
            break
        else:
            ind.append(new_ind)
            n_found += 1
            n_start = new_ind+1
    if which is 'last':
        length = x.__len__()
        ind = [length - tmp - 1 for tmp in ind]
    return ind


def sort_vecs(mat, order, axis):
    s = mat.shape
    mat1 = np.zeros(s)
    if axis == 0:  # sort as row vectors
        for n in range(0, s[0]):
            mat1[n, :] = mat[order[n], :]
    else:
        for n in range(0, s[1]):
            mat1[:, n] = mat[:, order[n]]
    return mat1


def arrangement(n, m):
    return factorial(n) / factorial(n-m)


def combination(n, m):
    return int(arrangement(n, m) / factorial(m))


def generate_indexes(ndim):
    key0 = {'0', '1'}
    key = set()
    if ndim == 1:
        return key0
    else:
        for x1 in key0:
            for x2 in generate_indexes(ndim - 1):
                key.add(x1 + x2)
        return key


def get_z2_indexes(ndim, parity=0):
    indexes_z2 = set()
    indexes = generate_indexes(ndim)
    for x in indexes:
        x1 = [int(m) for m in x]
        if sum(x1) % 2 == parity:
            indexes_z2.add(x)
    return indexes_z2


def gaussian_fun(x, sigma=1, mu=0):
    return 1 / np.sqrt(np.pi / 2) / sigma * np.exp(
        -((x-mu)**2).sum() / 2 / (sigma ** 2))


def two_dimensional_gaussian_map(sigma=1.0, mu=None, x_range=None,
                                 y_range=None, if_plot=False):
    if mu is None:
        mu = [0, 0]
    if x_range is None:
        x_range = [-1, 1, 0.02]
    if y_range is None:
        y_range = [-1, 1, 0.02]
    x, y = np.mgrid[x_range[0]:x_range[1]:x_range[2], y_range[0]:y_range[1]:y_range[2]]
    z = 1 / np.sqrt(np.pi / 2) / sigma * np.exp(-((x-mu[0])**2+(y-mu[1])**2)/2/(sigma**2))
    if if_plot:
        ax = plt.subplot(111, projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)
        plt.show()
    return x, y, z


def two_dimensional_gaussian_hills(x_range=None, y_range=None, if_plot=False):
    if x_range is None:
        x_range = [-1, 1, 0.02]
    if y_range is None:
        y_range = [-1, 1, 0.02]
    x, y = np.mgrid[x_range[0]:x_range[1]:x_range[2], y_range[0]:y_range[1]:y_range[2]]
    value_mu = np.sin(np.arange(x_range[0], x_range[1], x_range[2]) * np.pi / 1.4) / 2.5
    # value_sigma = 0.05 + 0.1 * np.exp(np.arange(x_range[0], x_range[1], x_range[2]) ** 2)
    value_sigma = 0.1 + 0.08 * np.cos(np.pi * 0.5 + np.arange(x_range[0], x_range[1], x_range[2]) * np.pi * 5)
    value_sigma = value_sigma * (np.arange(x_range[0], x_range[1], x_range[2]))**2 + 0.2
    z = list()
    for n in range(value_mu.size):
        z.append(1 / np.sqrt(np.pi / 2) / value_sigma[n] *
                 np.exp(-((y[n, :]-value_mu[n])**2)/2/(value_sigma[n]**2)))
    z = np.vstack(z)
    if if_plot:
        ax = plt.subplot(111, projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)
        plt.show()
    return x, y, z


# ========================================
# MNIST functions
def decode_idx3_ubyte(idx3_ubyte_file, if_t=True):
    """
    Downloaded from: https://blog.csdn.net/jiede1/article/details/77099326
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'   #'>IIII'是说使用大端法读取4个unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    # print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'   # '>784B'的意思就是用大端法读取784个unsigned byte
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
        #     print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    if if_t:
        images = images.T
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    Downloaded from: https://blog.csdn.net/jiede1/article/details/77099326
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
            # print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def generate_zigzag_order(lx, ly):
    order = (np.zeros((lx * ly, ), dtype=int), np.zeros((lx * ly, ), dtype=int))
    c = 0
    for n in range(lx + ly - 1):
        for nn in range(n + 1):
            if (n % 2) == 0:
                if (n - nn) < lx and (nn < ly):
                    order[0][c] = n - nn
                    order[1][c] = nn
                    c += 1
            else:
                if (n - nn) < ly and (nn < lx):
                    order[0][c] = nn
                    order[1][c] = n - nn
                    c += 1
    return order


def rescale_image(image0, max_, min_, ave_, rescale_way=2):
    if rescale_way == 1:  # consider the range (maximum and minimum)
        image0 -= image0.min()
        image0 /= image0.max()  # [0, 1]
        image0 *= (max_ - min_)
    elif rescale_way == 2:  # consider the average
        image0 += (ave_ - np.average(image0))
        image0[image0 > 1] = 1
        image0[image0 < 0] = 0
    return image0


def psnr(img1, img2, max_pixel=1):
    x = np.sqrt(np.mean((img1.reshape(-1, ) - img2.reshape(-1, )) ** 2))
    x = 20 * np.log10(max_pixel / x)
    return x


def mean_squared_error(img1, img2):
    return np.mean((img1.reshape(-1, ) - img2.reshape(-1, )) ** 2)


# def ssim(img1, img2, k1=0.01, k2=0.03):
#     mu1 = np.average(img1)
#     mu2 = np.average(img2)
#     dxy = np.cov(img1, img2)
#     return


# =========================================
# Print or Check functions
def trace_stack(level0=2):
    """
    Print the line and file name where this function is used
    :param level0:  previous level0 level in files
    :return: previous level0 line and file name
    Example
        in fileA.py
        >>> def fucntion1():
        >>>    print(trace_stack(2))
        in fileB.py
        if import fileA
        >>> def function2():
        >>>    fileA.function1()
        in fileC.py
        if import fileB
        >>> def function3():
        >>>    fileB.function2()
        >>>function3()
          in file_path\fileC.py at line 2
    """
    # print the line and file name where this function is used
    info = inspect.stack()
    ns = info.__len__()
    for ns in range(level0, ns):
        cprint('in ' + str(info[ns][1]) + ' at line ' + str(info[ns][2]), 'green')


def print_dict(a, keys=None, welcome='', style_sep=': ', color='white', end='\n'):
    """
    Print dictionary
    :param a: dictionary
    :param keys: names in dictionary
    :param welcome:  front words of dictionary
    :param style_sep:  separator
    :param color: print in what color
    :param end: how to end each line
    :return: what need to be print
    Example:
        >>>A = {'name1': 1, 'name2': 'a'}
        >>>print_dict(A, 'this is an example', '-')
          this is an example
          name1-1
          name2-2
    """
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
    """
    Print an error
    :param string: error information
    :param if_trace_stack: if need to print file name and line
    Example:
        >>>print_error('error: this is an example', 0)
          error: this is an example
    """
    cprint(string, 'magenta')
    if if_trace_stack:
        trace_stack(3)


def print_sep(info='', style='=', length=40, color='cyan'):
    """
    Print a separator
    :param info:  information
    :param style:  separator type
    :param length:  total length
    :param color:  color
    Example:
        >>>print_sep('This is an example', '@', '20')
          @@@@@@@@@@ This is an example @@@@@@@@@@
    """
    if info == '':
        cprint(style * (length * 2), color)
    else:
        l_info = info.__len__()
        l_new = length * 2 - 2 - l_info
        dl = l_new % 2
        l_new = int(l_new/2)
        l_new = max(l_new, 0)
        mes = style*max(l_new, 0) + ' ' + info + ' ' + style*((l_new + dl)*(l_new > 0))
        cprint(mes, color)


def print_options(options, start=None, welcome='', style_sep=': ', end='    ', color='cyan', quote=None):
    """
    Print the options
    :param options: possible options
    :param start: options count start with
    :param welcome: explaining of options
    :param style_sep:  separator between counts and options
    :param end: end
    :param color: color
    Example:
        >>>a = ['left', 'right']
        >>>print_options(a, [1, 2], 'Where to go:')
          Where to go:1: left    2: right
    """
    message = welcome
    length = len(options)
    if start is None:
        start = list(range(0, options.__len__()))
    for i in range(0, length):
        if quote is None:
            message += colored(str(start[i]) + style_sep + options[i], color)
        elif type(quote) is str:
            message += colored(str(start[i]) + style_sep + quote + options[i] + quote, color)
        if i < length-1:
            message += end
    print(message)


def input_and_check_type(right_type, name, print_result=True, dict_name='para'):
    """
    Input and check input type
    :param right_type: allowed types for input
    :param name: name of input
    :param print_result: if print out the input
    :param dict_name: dictionary of input belongs to
    :return: input
    Example:
        >>>input_and_check_type(int, 'number',True, 'input')
          Please input the value of number:
        >>> a
          number should be int, please input again:
        >>> 2
          You have set input 'number' = 2
    """
    # right_type should be a tuple
    ok = False
    some_error = True
    while some_error:
        try:
            while not ok:
                value = eval(input('Please input the value of ' + name + ': '))
                if isinstance(value, right_type):
                    ok = True
                else:
                    print(name + ' should be ' + str(right_type) + ', please input again.')
            some_error = False
        except (NameError, ValueError, SyntaxError):
            cprint('The input is illegal, please input again ...', 'magenta')
    if print_result:
        print('You have set ' + colored(dict_name + '[\'' + name + '\'] = ' + str(value), 'cyan'))
    return value


def input_and_check_value(right_value, values_str, names='', dict_name=''):
    """
    Input and check the value of input
    :param right_value:  allowed values of input
    :param values_str: describe of input
    :param names:  name of input
    :param dict_name: dictionary name of input
    :param start_ind: start with 1
    :return: input
    Example:
        >>>input_and_check_value([1, 2, 3], ('one', 'two', 'three'), names='Example', dict_name='Only an')
          Please input your choice:
        >>> 2
          You have set Only an['Example'] = 'two'
    """
    # right_value should be an array
    ok = False
    some_error = True
    while some_error:
        try:
            while not ok:
                value = eval(input('Please input your choice: '))
                if value in right_value:
                    ok = True
                else:
                    print('Input should be ' + colored(str(right_value), 'cyan') + ', please input again: ')
            some_error = False
        except (NameError, ValueError, SyntaxError):
            cprint('The input is illegal, please input again ...', 'magenta')
    ind = right_value.index(value)
    print('You have set ' + colored(dict_name + '[\'' + names + '\'] = \'' + str(values_str[ind]) + '\'', 'cyan'))
    return value


def check_condition(x, cond):
    """
    check if x satisfied condition
    :param x: a variable
    :param cond: a function that return boolean variable
    :return: true or false
    Example:
        >>>y = check_condition(3, lambda x: x > 0)
        >>> print(y)
          True
    """
    from inspect import isfunction
    if not isfunction(cond):
        return False
    try:
        return cond(x)
    except (TypeError, IndexError, ValueError):
        cprint('Wrong input in check_condition')
        return False


def input_and_check_type_multiple_items(right_type0, cond=None, name='your terms', max_len=100,
                                        key_stop=-1, key_clear=-3, is_print=False):
    """
    Input multiple items and check type
    :param right_type0:  allowed input type
    :param cond:  condition of input
    :param name:  name of input
    :param max_len:  maximal number inputs
    :param key_stop:  keyword to end inputs
    :param key_clear:  keyword to clean all inputs
    :param is_print:  if print your inputs
    :return:  all inputs
    Example:
        >>>y = input_and_check_type_multiple_items(int, lambda x: x > 0, 'int',key_stop='stop', key_clear='clean')
          Please input the value of int:
        >>> 2
          Please input the value of int:
        >>> -1
          The input is invalid since it does not satisfy the condition
          Please input the value of int:
        >>> 3
          Please input the value of int:
        >>>'stop'
          You input the key to stop. Input completed.
        >>> print(y)
          {2, 3}
    """
    # cond(inout) is True or False, a function to judge if the input is satisfactory
    if is_print:
        cprint('To finish inputting, input -1', 'cyan')
        cprint('To clear all the inputs and start over, input -3', 'cyan')
    output = set()
    # add the type of the key_stop in the tuple of the right types
    if type(right_type0) is type:
        right_type = {right_type0, type(key_stop), type(key_clear)}
    else:
        if type(right_type0) is tuple:
            right_type0 = set(right_type0)
        right_type = right_type0 | {type(key_stop), type(key_clear)}
    right_type = tuple(right_type)
    not_stop = True
    while not_stop:
        new = input_and_check_type(right_type, name, False)
        if new == key_stop:
            cprint('You input the key to stop. Input completed.', 'cyan')
            not_stop = False
        elif new == key_clear:
            output.clear()
            cprint('You have cleared all the inputs.', 'cyan')
        elif (cond is not None) and (not check_condition(new, cond)):
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


def check_existence(x, y=np.nan):
    flag = False
    if type(x) == np.ndarray:
        x = x.reshape(-1, )
    for n in x:
        if n == y:
            flag = True
            break
    return flag


def check_irregular_value(x):
    flag = False
    for n in x:
        if not abs(n) > 1e-30:
            flag = True
            return flag


# =========================================
# Plot functions
def plot(x, *y, marker='s'):
    if type(x) is tc.Tensor:
        if x.device != 'cpu':
            x = x.cpu()
        x = x.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if len(y) > 0.5:
        for y0 in y:
            if type(y0) is tc.Tensor:
                if y0.device != 'cpu':
                    y0 = y0.cpu()
                y0 = y0.numpy()
            ax.plot(x, y0, marker=marker)
    else:
        ax.plot(x, marker=marker)
    plt.show()


def plot_v1(x, *y, options=None, save=None):
    if options is None:
        options = dict()
    num_curves = max(1, y.__len__())
    # Default font
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 40}
    # Default values
    label_names = list()
    for n in range(num_curves):
        label_names.append('curve-' + str(n))
    default_ops = ['labelsize', 'axfontname', 'labelfont', 'axnames',
                   'legendfont', 'markers', 'labelnames']
    default_val = [32, 'Times New Roman', font1, ['', ''], font1,
                   ['o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+'],
                   label_names]

    save_opts = dict()
    save_opts['name'] = 'img.png'
    if type(save) is dict:
        for s in save:
            save_opts[s] = save[s]

    opts = dict()
    for n in range(default_ops.__len__()):
        if default_ops[n] in options:
            opts[default_ops[n]] = options[default_ops[n]]
        else:
            opts[default_ops[n]] = default_val[n]
    while num_curves > opts['markers'].__len__():
        opts['markers'] = opts['markers'] * 2
    opts['markers'] = opts['markers'][:num_curves]

    # start plotting
    if type(x) is tc.Tensor:
        if x.device != 'cpu':
            x = x.cpu()
        x = x.numpy()
    x = np.array(x).reshape(-1,)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1)
    if len(y) > 0.5:
        n = 0
        for y0 in y:
            if type(y0) is tc.Tensor:
                if y0.device != 'cpu':
                    y0 = y0.cpu()
                y0 = y0.numpy()
            ax.plot(x, np.array(y0).reshape(-1,), marker=opts['markers'][n], markerfacecolor='w', markersize=12,
                    markeredgewidth=2, label=opts['labelnames'][n])
            n += 1
    else:
        ax.plot(x, marker=opts['markers'][0], markerfacecolor='w', markersize=12,
                markeredgewidth=2, label=opts['labelnames'][0])

    plt.tick_params(labelsize=opts['labelsize'])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(opts['axfontname']) for label in labels]

    plt.xlabel(opts['axnames'][0], opts['labelfont'])
    plt.ylabel(opts['axnames'][1], opts['labelfont'])

    plt.legend(prop=opts['legendfont'])
    if type(save) is dict:
        # plt.subplots_adjust(left=0.09, right=1, wspace=0.25, hspace=0.25, bottom=0.13, top=0.91)
        plt.savefig(save_opts['name'])
    plt.show()


def plot_square_lattice(width, height, numbered=False, title='', save_path=None):
    """
    Plot a figure of square lattice
    :param width: width of the square lattice
    :param height:  height of the square lattice
    :param numbered:  if show each each lattice dot a number
    :param title:  title of the figure
    :param save_path:  if save the figure
    Example:
        >>>plot_square_lattice(2, 2)
          show a figure of a 2x2 square lattice
    """
    from library.HamiltonianModule import positions_nearest_neighbor_square
    pos_1d = np.arange(0, width*height, dtype=int).reshape(height, width)
    index = positions_nearest_neighbor_square(width, height)
    for n in range(0, index.shape[0]):
        pos1 = arg_find_array(pos_1d == index[n, 0])
        pos2 = arg_find_array(pos_1d == index[n, 1])
        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], '-ob', markersize=8)
    plt.axis('equal')
    if numbered:
        for w in range(0, width):
            for h in range(0, height):
                plt.text(h + 0.06, w - 0.06, str(pos_1d[h, w]), horizontalalignment='left',
                         verticalalignment='top', fontsize=15)
    plt.axis('off')
    plt.title(title)
    if save_path is not None:
        mkdir(save_path)
        plt.savefig(os.path.join(save_path, 'square(%d,%d).png' % (width, height)))
    plt.show()


def plot_connections_polar(positions, numbered=False, title='', save_path=None):
    """
    Plot a figure of points on polar coordinate with connections
    :param positions: position of points
    :param numbered: if show each each lattice dot a number
    :param title: title of the figure
    :param save_path: if save the figure
    Example:
        >>>x = np.array([[1, 3], [1, 4], [2, 4]])
        >>>plot_connections_polar(x, True)
          plot a figure with [1, 3] are connected, [1, 4] are connected, [2, 4] are connected
    """
    nb = positions.shape[0]
    ax = plt.subplot(111, projection='polar')
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
            plt.text(theta[n] + 0.05, 1.1, str(n), horizontalalignment='center',
                     verticalalignment='top', fontsize=15)
    plt.axis('off')
    plt.title(title)
    if type(save_path) is str:
        mkdir(save_path)
        plt.savefig(os.path.join(save_path, 'arbitrary.png'))
    plt.show()


def plot_surf(x=None, y=None, z=None, xlabel='x label', ylabel='y label', zlabel='z label', title=''):
    fig = plt.figure()
    ax = Axes3D(fig)
    if x is None:
        x = np.array(range(z.shape[1]))
    if y is None:
        y = np.array(range(z.shape[0]))
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    ax.set_xlabel(xlabel, color='r')
    ax.set_ylabel(ylabel, color='r')
    ax.set_zlabel(zlabel)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.suptitle(title)
    plt.show()


def scatter3d(x, y, z):
    ax1 = plt.figure().add_subplot(111, projection='3d')
    ax1.set_title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.scatter(x, y, z, c='r', marker='o')
    plt.legend('x1')
    plt.show()


def show_multiple_images_v0(imgs, lx=10, ly=10, gap=None, back_color=0):
    ni = len(imgs)
    shapes = [[], []]
    for x in imgs:
        shapes[0].append(x.shape[0])
        shapes[1].append(x.shape[1])
    shape_max = [max(x) for x in shapes]
    if gap is None:
        gap = int(max(shape_max) / 6)
    img = np.ones(((shape_max[0] + gap) * lx - gap,
                   (shape_max[1] + gap) * ly - gap)) * back_color
    px_now = 0
    py_now = 0
    shapes_x = list()
    shapes_y = list()
    num_each_column = [0]
    for n in range(ni):
        if px_now + imgs[n].shape[0] > img.shape[0]:
            shapes_x.append(px_now)
            px_now = 0
            py_now += max(shapes_y) + gap
            shapes_y = list()
            num_each_column.append(0)
        img[px_now:px_now + imgs[n].shape[0], py_now:py_now + imgs[n].shape[1]] = imgs[n]
        num_each_column[-1] += 1
        if n != (ni - 1):
            px_now += imgs[n].shape[0] + gap
        shapes_y.append(imgs[n].shape[1])
    shapes_x.append(px_now + imgs[ni - 1].shape[0])
    plt.imshow(img[: max(shapes_x), :py_now + max(shapes_y)], cmap=plt.cm.gray)
    plt.show()
    # viewer = ImageViewer(img[: max(shapes_x), :py_now + max(shapes_y)])
    # viewer.show()
    return


def show_multiple_images_v1(imgs, lxy=None, titles=None, save_name=None, cmap=None):
    if cmap is None:
        cmap = plt.cm.gray
    ni = len(imgs)
    if lxy is None:
        lx = int(np.sqrt(ni)) + 1
        ly = int(ni / lx) + 1
    else:
        lx, ly = tuple(lxy)
    plt.figure()
    for n in range(ni):
        plt.subplot(lx, ly, n + 1)
        if imgs[n].ndim == 2:
            plt.imshow(imgs[n], cmap=cmap)
        else:
            plt.imshow(imgs[n])
        if titles is not None:
            plt.title(str(titles[n]))
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    if type(save_name) is str:
        plt.savefig(save_name)
    plt.show()


def save_one_image(img, name, im_type=None, if_show=False):
    if img.ndim == 2:
        plt.imshow(img, cmap=plt.cm.gray)
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    if type(im_type) is str:
        name += im_type
    plt.savefig(name)
    if if_show:
        plt.show()


def save_imgs_respectively(images, names, path='.\\imgs_tmp\\', im_type='.jpg'):
    for n in images.__len__():
        plt.figure()
        if images[n].ndim == 2:
            plt.imshow(images[n], cmap=plt.cm.gray)
        else:
            plt.imshow(images[n])
        plt.savefig(os.path.join(path, names[n]+im_type))


def join_imgs_in_one_row(imgs, spacing=None, back_color=None):
    # All images must be in the same shape (size)
    num_imgs = imgs.__len__()
    shape = imgs[0].shape
    num_chan = (imgs[0].ndim == 3) * 3 + (imgs[0].ndim == 2)
    if spacing is None:
        spacing = int(shape[1] / 10)
    if back_color is None:
        if imgs[0].max() > 1.001:
            back_color = 255
        else:
            back_color = 1
    img = back_color * np.ones((shape[0] + spacing, (shape[1] + spacing) * num_imgs, num_chan))
    spacing_h = int(spacing / 2)
    for n in range(num_imgs):
        img[spacing_h:spacing_h+shape[0], spacing_h+n*(
                spacing + shape[1]):spacing_h+n*(spacing + shape[1])+shape[1], :] = \
            imgs[n].reshape(imgs[n].shape[:2]+(-1,))
    return img.squeeze()


def join_imgs_in_one_column(imgs, spacing=None, back_color=None):
    # All images must be in the same shape (size)
    num_imgs = imgs.__len__()
    shape = imgs[0].shape
    num_chan = (imgs[0].ndim == 3) * 3 + (imgs[0].ndim == 2)
    if spacing is None:
        spacing = int(shape[0] / 10)
    if back_color is None:
        if imgs[0].max() > 1.001:
            back_color = 255
        else:
            back_color = 1
    img = back_color * np.ones(((shape[0] + spacing) * num_imgs, shape[1] + spacing, num_chan))
    spacing_h = int(spacing / 2)
    for n in range(num_imgs):
        img[spacing_h+n*(spacing + shape[0]):spacing_h+n*(
                spacing + shape[0])+shape[0], spacing_h:spacing_h+shape[1], :] = \
            imgs[n].reshape(imgs[n].shape[:2]+(-1,))
    return img.squeeze()

