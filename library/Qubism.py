import numpy as np
from PIL import Image


def state2image(state, d, is_rescale=False):
    num = int(round(np.log(state.size) / np.log(d)))
    if state.ndim != num:
        state = state.reshape(np.ones((num, ), dtype=int) * d)
    num_h = int(round(num/2))
    image = np.zeros((d**num_h, d**num_h))
    config = [0] * (num + 1)
    while config[0] == 0:
        x = list()
        for nx in range(0, num+1, 2):
            x.append(config[nx])
        y = list()
        for ny in range(1, num+1, 2):
            y.append(config[ny])
        x = list2num(x, d)
        y = list2num(y, d)
        ind = list2index(config[1:])
        image[x, y] = eval('state[' + ind + ']')
        config[-1] += 1
        for n in range(num, 0, -1):
            if config[n] == d:
                config[n] = 0
                config[n-1] += 1
            else:
                break
    if is_rescale:
        image = image / max(abs(image.reshape(-1, ))) * 255
    return image


def list2num(x, d):
    num = 0
    length = x.__len__()
    for n in range(0, length):
        num += x[n] * d**(length - n - 1)
    return num


def list2index(x):
    ind = ''
    for n in range(0, x.__len__()):
        ind = ind + str(x[n]) + ','
    return ind[:-1]


def image2rgb(image, if_rescale_1=False):
    # Positive: red; negative: blue
    shape = image.shape
    im = Image.new("RGB", shape)
    for nx in range(0, shape[0]):
        for ny in range(0, shape[1]):
            if if_rescale_1:
                if image[nx, ny] > 0:
                    im.putpixel((nx, ny), (0, 255, 255))
                else:
                    im.putpixel((nx, ny), (255, 0, 255))
            else:
                if image[nx, ny] > 0:
                    im.putpixel((nx, ny), (255 - int(image[nx, ny]), 255, 255))
                else:
                    im.putpixel((nx, ny), (255, 255 + int(image[nx, ny]), 255))
    return im





