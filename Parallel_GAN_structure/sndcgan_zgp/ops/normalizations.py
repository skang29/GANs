""""
Normalizations under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf
from tensorflow.contrib.nccl.ops import gen_nccl_ops
NCCL_FLAG = os.environ.get('nccl_multigpu_env')


if NCCL_FLAG == 'true':
    # NCCL multi-gpu environment.
    def spectral_norm(w, iteration=1, tower_config=None):
        """Multi-gpu independent."""

        def l2_norm(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u",
                            [1, w_shape[-1]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
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


    def __group_norm(xxx, G=32, eps=1e-5, name='group_norm', tower_config=None):
        x = xxx
        with tf.variable_scope(name):
            N, H, W, C = x.get_shape().as_list()
            G = min(G, C)

            x = tf.reshape(x, [N, H, W, G, C // G])

            # Go! NCCL!
            with tf.variable_scope("mean"):
                shared_name = tf.get_variable_scope().name. \
                    replace(tower_config.name, tower_config.prefix.format("NCCL"))

                device_mean = tf.reduce_mean(x, axis=[1, 2, 4], keepdims=True)
                mean = gen_nccl_ops.nccl_all_reduce(
                    input=device_mean,
                    reduction="sum",
                    num_devices=tower_config.num_devices,
                    shared_name=shared_name
                ) / (1.0 * tower_config.num_devices)

            with tf.variable_scope("var"):
                shared_name = tf.get_variable_scope().name. \
                    replace(tower_config.name, tower_config.prefix.format("NCCL"))

                device_var = tf.reduce_mean(tf.square(x - mean), axis=[1, 2, 4], keepdims=True)
                var = gen_nccl_ops.nccl_all_reduce(
                    input=device_var,
                    reduction="sum",
                    num_devices=tower_config.num_devices,
                    shared_name=shared_name
                ) / (1.0 * tower_config.num_devices)

            x = (x - mean) / tf.sqrt(var + eps)
            print("### X:", x)

            gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

            x = tf.reshape(x, [N, H, W, C]) * gamma + beta
            print("### X Final:", x)

        return xxx


    def group_norm(x, G=32, eps=1e-5, name='group_norm', tower_config=None):
        with tf.variable_scope(name):
            N, H, W, C = x.get_shape().as_list()
            G = min(G, C)

            x = tf.reshape(x, [N, H, W, G, C // G])

            # Go! NCCL!
            with tf.variable_scope("mean"):
                shared_name = tf.get_variable_scope().name. \
                    replace(tower_config.name, tower_config.prefix.format("NCCL"))
                device_mean = tf.reduce_mean(x, axis=[1, 2, 4], keepdims=True)
                mean = gen_nccl_ops.nccl_all_reduce(
                    input=device_mean,
                    reduction="sum",
                    num_devices=tower_config.num_devices,
                    shared_name=shared_name
                ) / (1.0 * tower_config.num_devices)

            with tf.variable_scope("var"):
                shared_name = tf.get_variable_scope().name. \
                    replace(tower_config.name, tower_config.prefix.format("NCCL"))
                device_var = tf.reduce_mean(tf.square(x - mean), axis=[1, 2, 4], keepdims=True)
                var = gen_nccl_ops.nccl_all_reduce(
                    input=device_var,
                    reduction="sum",
                    num_devices=tower_config.num_devices,
                    shared_name=shared_name
                ) / (1.0 * tower_config.num_devices)

            x = (x - mean) / tf.sqrt(var + eps)

            gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

            x = tf.reshape(x, [N, H, W, C]) * gamma + beta

        return x


    def layer_norm(x, name="layer_norm", tower_config=None):
        raise NotImplementedError("Layer normalization for NCCL environment has not been implemented.")


    def batch_norm(x, is_training=True, name="batch_norm", scope=None, tower_config=None):
        raise NotImplementedError("Batch normalization for NCCL environment has not been implemented.")

else:
    # None NCCL multi-gpu environment.
    def spectral_norm(w, iteration=1):
        """Multi-gpu independent."""

        def l2_norm(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u",
                            [1, w_shape[-1]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
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


    def layer_norm(x, name="layer_norm"):
        return tf.contrib.layers.layer_norm(x)


    def batch_norm(x, is_training=True, name="batch_norm"):
        return tf.contrib.layers.batch_norm(x,
                                            decay=0.9,
                                            updates_collections=None,
                                            epsilon=1e-5,
                                            scale=True,
                                            is_training=is_training,
                                            scope=name)
