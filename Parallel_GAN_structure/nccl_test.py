import tensorflow as tf
from tensorflow.contrib.nccl.python.ops import nccl_ops
nccl_ops._maybe_load_nccl_ops_so()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

sess = tf.Session()


var_list = list()
gpu_list = [0, 1, 2, 3]
for i in gpu_list:
    a = tf.random_normal([1])
    b = tf.random_normal([1])

    with tf.device("/gpu:{}".format(i)):
        c = a * b
    var_list.append(c)


nccl_list = tf.contrib.nccl.all_sum(var_list)


while True:
    print("get result:", end=' ', flush=True)
    result = sess.run(nccl_list)
    print(result)