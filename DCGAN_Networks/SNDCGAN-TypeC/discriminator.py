import tensorflow as tf
import numpy as np

from .ops import *
from .utils import print_info


def discriminator(image_, batch_size, df_dim, name="discriminator", sn=True, reuse=False, debug=True):
    with tf.variable_scope("type__discriminator"):
        with tf.variable_scope(name, reuse=reuse):
            minimum_image_size = 16

            if debug: print_info("Input", image_)
            im_shape = image_.get_shape().as_list()
            image_.set_shape([batch_size] + im_shape[1:])
            nets = init_stage(image_, batch_size, df_dim, sn=sn)
            if debug: print_info("Init", nets)
            # nets: [batch_size, size, size, df_dim]

            size = nets.get_shape().as_list()[1]
            num_layers = int(np.log2(size / minimum_image_size))

            for i in range(num_layers):
                conv_dims = df_dim * int(2 ** (i + 1))
                nets = repeat_stage(nets, batch_size, conv_dims, name="repeat_stage_{}".format(i), sn=sn)
                if debug: print_info("repeat_{}".format(i), nets)

            # nets: [batch_size, 4, 4, df_dim * multiplier]

            nets = final_stage(nets, batch_size, df_dim * int(2 ** num_layers), sn=sn)
            if debug: print_info("Final", nets)
            # nets: [batch_size, 4 * 4 * df_dim * multiplier]

            nets = linear(nets, 1, sn=sn)
            if debug: print_info("Logits", nets)

            return tf.nn.sigmoid(nets), nets


def init_stage(image_, batch_size, conv_dims, name="init_stage", sn=False):
    with tf.variable_scope(name):
        nets = conv2d(image_, conv_dims, kernel=(3, 3), strides=(1, 1), sn=sn)
        nets = lrelu(nets)

    return nets


def repeat_stage(nets, batch_size, conv_dims, name="repeat_stage", sn=False):
    with tf.variable_scope(name):
        nets = conv2d(nets, conv_dims, kernel=(4, 4), strides=(2, 2), sn=sn, name="conv0")
        nets = lrelu(nets)
        nets = conv2d(nets, conv_dims, kernel=(3, 3), strides=(1, 1), sn=sn, name="conv1")
        nets = lrelu(nets)

        return nets


def final_stage(nets, batch_size, conv_dims, name="final_stage", sn=False):
    with tf.variable_scope(name):
        nets = tf.reshape(nets, [batch_size, conv_dims * 16 * 16])
        return nets
