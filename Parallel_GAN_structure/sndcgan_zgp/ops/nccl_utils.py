""""
NCCL utils layers under tensorflow environment.
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


if not NCCL_FLAG == 'true':
    raise NotImplementedError("NCCL environment is not set. Please enable NCCL environment flag using $ os.environ['nccl_multigpu_env'] = 'true' in your main.py.")


def nccl_all_sum(network_list, lambda_function):
    return tf.contrib.nccl.all_sum([lambda_function(network) for network in network_list])


def nccl_all_mean(network_list, lambda_function):
    sum_list = nccl_all_sum(network_list, lambda_function)

    data_list = list()
    for element in sum_list:
        with tf.device(element.device):
            data_list.append(element / (1.0 * len(network_list)))

    return data_list


def nccl_device_sum(input_, tower_config):
    nccl_name = "NCCL" if not tower_config.is_test else "NCCL_TEST"

    shared_name = input_.name.replace(tower_config.name, tower_config.prefix.format(nccl_name))

    output_ = gen_nccl_ops.nccl_all_reduce(
        input=input_,
        reduction="sum",
        num_devices=tower_config.num_devices,
        shared_name=shared_name
    )

    return output_


def nccl_device_mean(input_, tower_config):
    return nccl_device_sum(input_, tower_config) / (1.0 * tower_config.num_devices)


def all_reduce_grads(all_grads, average=True):
    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads
    new_all_grads = []  # N x K
    for grads in zip(*all_grads):
        # grads: ((d0 v0 grad, d0 v0), (d1 v0 grad, d1 v0))
        grad_list = [x[0] for x in grads]
        summed = tf.contrib.nccl.all_sum(grad_list)
        device_var_list = [x[1] for x in grads]
        grad_and_vars = zip(summed, device_var_list)

        grads_for_devices = []  # K
        for g in grad_and_vars:
            with tf.device(g[0].device):
                if average:
                    grad_ = tf.multiply(g[0], 1.0 / nr_tower, name="all_reduce_avg")
                else:
                    grad_ = g[0]
                var_ = g[1]
            grads_for_devices.append((grad_, var_))
        new_all_grads.append(grads_for_devices)

    ret = list(zip(*new_all_grads))

    return ret


def optimizer_op(losses, network_list, optimizer_lambda_function, var_name=None):
    tower_grads = list()
    optimizer_list = list()

    var_name_str = "" if var_name is None else "_{}".format(var_name)

    for idx, (network, loss) in enumerate(zip(network_list, losses)):
        optimizer_ = optimizer_lambda_function()
        optimizer_list.append(optimizer_)
        with tf.device(network.tower_config.device_name), tf.variable_scope(network.tower_config.name):
            tvars = [var for var in tf.trainable_variables() if network.tower_config.name in var.name]

            if var_name is None:
                op_vars = tvars
            else:
                op_vars = [var for var in tvars if var_name in var.name]

            tower_grads.append(
                [grad for grad in optimizer_.compute_gradients(loss, var_list=op_vars) if grad[0] is not None])

    grads = all_reduce_grads(tower_grads)

    train_ops = []
    for idx, (grad_and_vars, network, optimizer) in enumerate(zip(grads, network_list, optimizer_list)):
        with tf.name_scope("apply_gradients"), tf.device(network.tower_config.device_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=network.tower_config.name)
            with tf.control_dependencies(update_ops):
                train_ops.append(optimizer.apply_gradients(grad_and_vars,
                                                           name='apply_grad{}_{}'.format(var_name_str,
                                                                                         network.tower_config.idx)))

    optimize_op = tf.group(*train_ops, name="train_op{}".format(var_name_str))

    return optimize_op


def device_sync_op(prefix, main_idx):
    import re
    all_vars = tf.global_variables() + tf.local_variables()
    var_by_name = dict([(v.name, v) for v in all_vars])

    post_init_ops = []
    match_re = r"{}\d+".format(prefix.format(""))
    regex = re.compile(match_re)
    main_tower_name = prefix.format(main_idx)
    for v in all_vars:
        tower_name = regex.search(v.name)
        if tower_name is None:
            continue

        if main_tower_name == tower_name:
            # no need to copy to main tower
            continue

        tower_name = tower_name.group()

        copy_from = var_by_name.get(v.name.replace(tower_name, main_tower_name))
        if v.name == copy_from:
            continue

        if copy_from is not None:
            post_init_ops.append(v.assign(copy_from.read_value()))
        else:
            UserWarning("Cannot find {} in the graph!".format(v.name.replace(tower_name, main_tower_name)))

    return tf.group(*post_init_ops, name="sync_variables_from_main_tower")


class TowerConfig(object):
    def __init__(self, idx, prefix, is_main, num_devices, device_name, is_test):
        self.idx = idx
        self.prefix = prefix
        self.is_main = is_main
        self.name = prefix.format(idx)
        self.num_devices = num_devices
        self.device_name = device_name
        self.is_test = is_test
