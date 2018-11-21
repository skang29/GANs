import os
os.environ["nccl_multigpu_env"] = 'true'
os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

import numpy as np
import tensorflow as tf
from parallel_training_model import TrainingModel as model

flags = tf.app.flags

# Network options
flags.DEFINE_integer("final_size", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("z_dim", 128, "Latent space dimensions [100]")
flags.DEFINE_integer("gf_dim", 64, "Latent space dimensions [100]")
flags.DEFINE_integer("df_dim", 64, "Latent space dimensions [100]")

# Model options
flags.DEFINE_string("model_name", "SNDCGAN_TypeB_ZGP_BN", "Epoch to train [25]")
flags.DEFINE_integer("epoch", 400, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 4, "The size of batch images per one GPU. [64]")
flags.DEFINE_integer("sample_num", 64, "The size of sample images [64]")
flags.DEFINE_string("dataset", "LSUN_CHURCH_CENTER_SQUARE_256/images", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("data_dir", "/home/Databases/LSUN", "Root directory of dataset [data]")

flags.DEFINE_integer("discriminator_iteration", 3, "How many times to repeat discriminator optimizer.")
flags.DEFINE_integer("generator_iteration", 2, "How many times to repeat generator optimizer.")

flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

flags.DEFINE_string("checkpoint_dir", None, "Restoring checkpoint dir. [None : No restoring]")
flags.DEFINE_list("restore_list", ["type__"], "Restoring variable list. [None : No restoring]")

flags.DEFINE_integer("sample_interval", 400, "Sample image interval. [50]")
flags.DEFINE_integer("print_interval", 25, "Status print interval. [50]")
flags.DEFINE_integer("ckpt_interval", 100, "Ckpt save interval. [50]")

# System options
gpu_list = [str(x) for x in [0, 1]]
print("Visible devices: ", ", ".join(gpu_list))
flags.DEFINE_list("device_list", ['/gpu:{}'.format(x) for x in range(len(gpu_list))], "GPU to utilize.")

flags.DEFINE_string("base_dir", "./Container", "Root directory of ckpt, results, logs [container]")
flags.DEFINE_float("memory_limit", None, "Per GPU memory fraction. [None: No limit]")
flags.DEFINE_string("visible_devices", ", ".join(gpu_list), "Visible GPU selection for the training. [None: Use all]")

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.visible_devices:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_devices


    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    if FLAGS.memory_limit:
        # Limit memory for each process
        run_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory_limit

    with tf.Session(config=run_config) as sess:
        m = model(
            sess,
            config=FLAGS,
            dataset_name=FLAGS.dataset,
            data_dir=FLAGS.data_dir,
            base_dir=FLAGS.base_dir,
            model_name=FLAGS.model_name,
            device_list=FLAGS.device_list
        )

        if FLAGS.train:
            m.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
