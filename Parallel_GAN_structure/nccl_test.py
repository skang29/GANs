import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['nccl_multigpu_env'] = 'true'

import tensorflow as tf
from tensorflow.contrib.nccl.python.ops import nccl_ops
nccl_ops._maybe_load_nccl_ops_so()

from sndcgan_zgp.ops.core.nn import moments
from sndcgan_zgp.ops import TowerConfig

sess = tf.Session()

var_list = list()
gpu_list = [0, 1]
for i in gpu_list:
    tower_config = TowerConfig(idx=i,
                               prefix="Tower_{}",
                               is_main=i == 0,
                               num_devices=2,
                               device_name="/gpu:{}".format(i),
                               is_test=False)
    with tf.device("/gpu:{}".format(i)), tf.variable_scope("Tower_{}".format(i)):
        a = tf.random_normal([64, 128, 128, 1024])
        # b = tf.random_normal([10, 10, 10])
        m, v = moments(a, [0, 1, 2, 3], tower_config, keep_dims=False)
        # for i in range(10):
        #     with tf.variable_scope("sub_nets/sub_{}".format(i)):
        #         m = batch_norm(a, tower_config=tower_config, name="norm{}".format(i))
        var_list.append(v)


nccl_list = tf.contrib.nccl.all_sum(var_list)
hey = sum(var_list)

print("Initialize ... ", end=' ', flush=True)
sess.run(tf.global_variables_initializer())
print("Done !")


while True:
    print("get result:", end=' ', flush=True)
    result, hey2 = sess.run([nccl_list, hey])
    print(result[0])
    print(hey2)
    ttt = input("Press any key to continue.")