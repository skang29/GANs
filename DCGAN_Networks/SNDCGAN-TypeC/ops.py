import functools
import tensorflow as tf
import numpy as np
import inspect
import os

VERSION_INFO = \
"""----------------------------
| OPS
| Version: 1.0.2
| Change log
|   * Version established.
|   * New residual block.
|   * conv2d Unknown shape bug fix.
|   * loss_iterator logits update.
| Modified date: 2018.08.31.
----------------------------"""
print(VERSION_INFO)

# Summary version compatibility
try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter


if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


def batch_norm(x, is_training=True, name="batch_norm", scope=None):
    if scope is not None:
        DeprecationWarning("Argument 'scope' is deprecated. Used 'name' instead.")
        name = scope

    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=name)


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def group_norm(x, G=32, eps=1e-5, name='group_norm'):
    with tf.variable_scope(name):
        N, H, W, C = x.get_shape().as_list()
        G = min(G, C)

        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u",
                        [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def instance_norm(x, name="instance_norm"):
    return tf.contrib.layers.instance_norm(x)


def layer_norm(x, name="batch_norm"):
    return tf.contrib.layers.layer_norm(x)


def conv2d(input_, output_dim, kernel=(5, 5), strides=(2, 2), stddev=0.01, sn=False, name="conv2d", scope=None):
    if scope is not None:
        DeprecationWarning("Argument 'scope' is deprecated. Used 'name' instead.")
        name = scope

    with tf.variable_scope(name):
        w = tf.get_variable(name="w",
                            shape=[kernel[0], kernel[1], input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b',
                            shape=[output_dim],
                            initializer=tf.constant_initializer(0.0))

        if sn:
            conv = tf.nn.conv2d(input_, spectral_norm(w),
                                strides=[1, strides[0], strides[1], 1],
                                padding='SAME')

        else:
            conv = tf.nn.conv2d(input_, w,
                                strides=[1, strides[0], strides[1], 1],
                                padding='SAME')

        # conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return tf.nn.bias_add(conv, b)


def conv2d_transpose(input_, output_shape, kernel=(4, 4), strides=(2, 2), sn=False,
                     stddev=0.01, name="conv2d_transpose", scope=None, with_w=False):
    if scope is not None:
        DeprecationWarning("Argument 'scope' is deprecated. Used 'name' instead.")
        name = scope

    with tf.variable_scope(name):
        # filter : (height, width, output_channels, in_channels)
        w = tf.get_variable(name="w",
                            shape=[kernel[0], kernel[1], output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name="b",
                            shape=[output_shape[-1]],
                            initializer=tf.constant_initializer(0.0))

        if sn:
            conv_tp = tf.nn.conv2d_transpose(input_, spectral_norm(w),
                                             output_shape=output_shape,
                                             strides=[1, strides[0], strides[1], 1])

        else:
            conv_tp = tf.nn.conv2d_transpose(input_, w,
                                             output_shape=output_shape,
                                             strides=[1, strides[0], strides[1], 1])

        conv_tp = tf.reshape(tf.nn.bias_add(conv_tp, b), conv_tp.get_shape())

        if with_w:
            return conv_tp, w, b
        else:
            return conv_tp


def lrelu(x, leak=0.2, scope='lrelu'):
    return tf.maximum(x, leak * x)


def relu(x, scope='relu'):
    return tf.maximum(x, 0)


def linear(input_, output_size, scope=None, name='linear', stddev=0.02, bias_init=0.0, sn=False, with_w=False):
    if scope is not None:
        DeprecationWarning("Argument 'scope' is deprecated. Used 'name' instead.")
        name = scope

    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable(name="w",
                            shape=[shape[1], output_size],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name="b",
                             shape=[output_size],
                             initializer=tf.constant_initializer(bias_init))

        if sn:
            y = tf.matmul(input_, spectral_norm(w)) + b

        else:
            y = tf.matmul(input_, w) + b

        if with_w:
            return y, w, b
        else:
            return y


def loss_iterator(logits, fitting):
    loss = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(
            logits,
            fitting(logits))
    )

    return loss


def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


def glu(x, name="glu"):
    input_channel = x.get_shape().as_list()[-1]
    assert input_channel % 2 == 0, "Channel not separable."
    sliced_channel = input_channel // 2

    return x[:, :sliced_channel] * tf.nn.sigmoid(x[:, sliced_channel:])


def glu_4d(x, sn=False, name="gcnn"):
    input_channel = x.get_shape().as_list()[-1]
    assert input_channel % 2 == 0, "Channel not separable."
    sliced_channel = input_channel // 2

    return tf.multiply(x[:, :, :, :sliced_channel],
                       tf.nn.sigmoid(x[:, :, :, sliced_channel:]))


def residual_block(input_layer, output_channel=None, hidden_channel=None, normalization=False, sn=False, debug=False, name="res_block"):
    """
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param hidden_channel: int.
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.

    ### Important ###
    This residual block is from Which Training Method ....
    """

    UserWarning("This residual block is from Which Training Method ... . ")

    with tf.variable_scope(name) as scope:
        input_channel = input_layer.get_shape().as_list()[-1]

        if output_channel is None:
            output_channel = input_channel

        if hidden_channel is None:
            hidden_channel = min(input_channel, output_channel)

        if debug:
            print("="*38)
            print("☆★☆★☆★ DEBUG SCREEN ☆★☆★☆★")
            print("-"*38)
            print("Scope: ", tf.get_variable_scope().name)
            scope_name = str(tf.get_variable_scope().name)
            print("Input channel: {}".format(input_channel))
            print("Hidden channel: {}".format(hidden_channel))
            print("Output channel: {}".format(output_channel))
            print("-"*38)

        x = nets = input_layer
        if debug: print("Input: ", str(nets).replace(scope_name, ""))
        # nets = batch_norm(nets, name="bn0")
        # if debug: print("BN0: ", str(nets).replace(scope_name, ""))
        if normalization: nets = group_norm(nets, name="gn0")
        nets = relu(nets)
        if debug: print("LReLU: ", str(nets).replace(scope_name, ""))
        nets = conv2d(input_=nets,
                      output_dim=hidden_channel,
                      kernel=(3, 3),
                      strides=(1, 1),
                      name="conv_h0",
                      sn=sn)
        if debug: print("Conv 0: ", str(nets).replace(scope_name, ""))

        # nets = batch_norm(nets, name="bn1")
        # if debug: print("BN1: ", str(nets).replace(scope_name, ""))
        if normalization: nets = group_norm(nets, name="gn1")
        nets = relu(nets)
        if debug: print("LReLU: ", str(nets).replace(scope_name, ""))
        dx = conv2d(input_=nets,
                    output_dim=output_channel,
                    kernel=(3, 3),
                    strides=(1, 1),
                    name="conv_h1",
                    sn=sn)
        if debug: print("Conv 1: ", str(dx).replace(scope_name, ""))

        if output_channel != input_channel:
            if debug: print("Creating new identity path.")
            x = conv2d(input_=x,
                       output_dim=output_channel,
                       kernel=(3, 3),
                       strides=(1, 1),
                       name="conv_s",
                       sn=sn)
            if debug: print("Conv S: ", str(x).replace(scope_name, ""))

        ret = x + dx
        if debug: print("Return: ", str(ret).replace(scope_name, ""))
        if debug: print("-" * 38)
        return ret


def down_sampling_block(_nets, conv_dims, do_bn=True, sn=False, scope=None, name="downsampling"):
    if scope is not None:
        DeprecationWarning("Argument 'scope' is deprecated. Used 'name' instead.")
        name = scope

    convParams = dict(kernel=(5, 5), strides=(2, 2), name=name+"conv", sn=sn)
    nets = conv2d(_nets, conv_dims, **convParams)
    if do_bn:
        nets = batch_norm(nets, is_training=True, name=name+"bn")
    nets = lrelu(nets)

    return nets


def gram_matrix(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)

    dim = v.get_shape().as_list()
    v = tf.reshape(v, [dim[0], dim[1] * dim[2], dim[3]])
    if dim[1] * dim[2] < dim[3]:
        return tf.matmul(v, v, transpose_b=True)
    else:
        return tf.matmul(v, v, transpose_a=True)


def l2_norm_cost(v):
    dim = v.get_shape().as_list()
    size = functools.reduce(lambda x, y: x * y, dim)
    return tf.reduce_sum(tf.square(v)) / (size ** 2)


def compute_mean_covariance(img):
    shape = img.get_shape().as_list()
    numBatchs = shape[0]
    numPixels = shape[1] * shape[2]
    cDims = shape[3]

    mu = tf.reduce_mean(img, axis=[1, 2])

    img_hat = img - tf.reshape(mu, shape=[mu.shape[0], 1, 1, mu.shape[1]])
    img_hat = tf.reshape(img_hat, [numBatchs, cDims, numPixels])

    cov = tf.matmul(img_hat, img_hat, transpose_b=True)
    cov = cov / numPixels

    return mu, cov


def gaussian_noise_layer(input_layer, stddev=0.1):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=stddev, dtype=tf.float32)
    return input_layer + noise


VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        # self.relu6 = tf.nn.relu(self.fc6)
        #
        # self.fc7 = self.fc_layer(self.relu6, "fc7")
        # self.relu7 = tf.nn.relu(self.fc7)
        #
        # self.fc8 = self.fc_layer(self.relu7, "fc8")
        #
        # self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")