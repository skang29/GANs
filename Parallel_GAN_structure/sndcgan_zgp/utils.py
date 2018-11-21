"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import scipy.misc
import numpy as np


import tensorflow as tf
import tensorflow.contrib.slim as slim

from contextlib import contextmanager

VERSION_INFO = \
"""----------------------------
| UTILS
| Version: 1.0.1
| Change log
|   * Version established.
|   * print_info added.
| Modified date: 2018.09.05.
----------------------------"""
print(VERSION_INFO)


def get_stddev(x, k_h, k_w):
    return 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def get_patch(data, index, div=3):
    """
    get_patch
    :param data: [np.array] data
    :param index: [int] index of patch.
                    0   1   2
                    3   4   5
                    6   7   8
    :param size: [int] width and height.
    :param div: [int] patch division.
    :return: [np.array] sliced array.
    """
    if type(data) == np.ndarray:
        size = data.shape[1]
    else:
        size = data.get_shape().as_list()[1]

    base_size = size // div
    h_index = index // div
    w_index = index % div

    return data[:, base_size * h_index:base_size * (h_index + 1), base_size * w_index:base_size * (w_index + 1), :]


def get_quadrant(data, index):
    """
    get_quadrant
    :param data: [np.array] data
    :param index: [int] index of quadrant.
                    0   1
                    2   3
    :param size: [int] width and height.
    :return: [np.array] sliced array.
    """
    if type(data) == np.ndarray:
        size = data.shape[1]
    else:
        size = data.get_shape().as_list()[1]
    base_size = size // 3
    h_index = index // 2
    w_index = index % 2

    return data[:, base_size * h_index:base_size * (h_index + 2), base_size * w_index:base_size * (w_index + 2), :]


def glue_patch_to_quadrant(patch, quadrant, index):
    q0 = patch if index == 0 else get_patch(quadrant, 0, div=2)
    q1 = patch if index == 1 else get_patch(quadrant, 1, div=2)
    q2 = patch if index == 2 else get_patch(quadrant, 2, div=2)
    q3 = patch if index == 3 else get_patch(quadrant, 3, div=2)

    h0 = tf.concat([q0, q1], axis=2)
    h1 = tf.concat([q2, q3], axis=2)

    full = tf.concat([h0, h1], axis=1)

    return full


def glue_patch_to_full(patch, full, index):
    p = dict()
    for i in range(9):
        p[i] = patch if index == i else get_patch(full, i)

    r0 = tf.concat([p[0], p[1], p[2]], axis=2)
    r1 = tf.concat([p[3], p[4], p[5]], axis=2)
    r2 = tf.concat([p[6], p[7], p[8]], axis=2)

    f = tf.concat([r0, r1, r2], axis=1)

    return f


def crop_center(data):
    if type(data) == np.ndarray:
        size = data.shape[1]
    else:
        size = data.get_shape().as_list()[1]

    return data[:, size//4:size//4 * 3, size//4:size//4 * 3, :]


def glue_center(patch, data):
    if type(data) == np.ndarray:
        size = data.shape[1]
    else:
        size = data.get_shape().as_list()[1]

    u_ = data[:, :size//4, :, :]
    d_ = data[:, size//4 * 3:, :, :]

    l_ = data[:, size//4:size//4 * 3, :size//4, :]
    r_ = data[:, size//4:size//4 * 3, size//4 * 3:, :]

    if type(data) == np.ndarray:
        c_ = np.concatenate([l_, patch, r_], axis=2)
        f = np.concatenate([u_, c_, d_], axis=1)

    else:
        c_ = tf.concat([l_, patch, r_], axis=2)
        f = tf.concat([u_, c_, d_], axis=1)

    return f


def empty_center(data, value):
    if type(data) == np.ndarray:
        shape = data.shape
        patch = np.ones(shape=(shape[0], shape[1] // 2, shape[2] // 2, shape[3])) * value
    else:
        shape = data.get_shape().as_list()
        patch = tf.constant(value, dtype=tf.float32, shape=(shape[0], shape[1] // 2, shape[2] // 2, shape[3]))

    f = glue_center(patch, data)
    return f


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def save_images(images, size, image_path):
    if type(images) == list:
        images = np.concatenate(images, axis=0)
    return imsave(inverse_scale(images), size, image_path)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def print_info(name, tensor):
    shape_list = [str(x) for x in tensor.get_shape().as_list()]
    tensor_name = tensor.name

    shape_string = "[{}]".format(" x ".join(shape_list))

    print("{:15s}: {:30s}   {}".format(name, shape_string, tensor_name))


from collections import defaultdict

nested_dict = lambda: defaultdict(nested_dict)


def rescale(data, func=lambda x: x / 127.5 - 1):
    return func(data)


def inverse_scale(data, func=lambda x: (x + 1. ) * 127.5):
    return func(data)


