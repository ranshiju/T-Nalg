import numpy as np
import matplotlib.pyplot as mpy


def plot_square_map(width, height, _is_show=False):
    figure = mpy.plot()
    for i in range(0, height):
        pos0 = [i, i]
        pos1 = [0, width - 1]
        figure = mpy.plot(pos0, pos1, 'black')
    for j in range(0, width):
        pos0 = [0, height - 1]
        pos1 = [j, j]
        mpy.plot(pos0, pos1, 'black')
    if _is_show:
        mpy.show()
    return figure


def list_color():
    color_list = ('c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9')
    return color_list


def list_marker():
    marker_list = ('o', 's', 'p', '*', 'h', 'x', 'D')
    return marker_list


def plot_spotty(spotty, _is_show=False):
    try:
        nation = spotty.nation
    except KeyError:
        nation = 0
    color = list_color()[nation]
    marker = list_marker()[spotty.tribe - 1]
    figure = mpy.plot(spotty.position, color, marker)
    if _is_show:
        mpy.show(figure)
    return figure

