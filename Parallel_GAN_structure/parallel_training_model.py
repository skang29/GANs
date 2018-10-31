import os
import time
import numpy as np
import tensorflow as tf

from base_model import BaseModel

from sndcgan_zgp import Network
from sndcgan_zgp.utils import *
from sndcgan_zgp.ops import nccl_all_mean, network_mean, nccl_all_sum
from sndcgan_zgp.ops import TowerConfig, device_sync_op, optimizer_op

from data_loader import parallel_image_filename_loader


class TrainingModel(BaseModel):
    def __init__(self,
                 sess,
                 config,
                 device_list,
                 model_name="model_name",
                 dataset_name="default",
                 data_dir="./data",
                 base_dir="./"):
        super(TrainingModel, self).__init__(name=model_name, base_dir=base_dir)
        self.connect_paths()

        snapshot_list = list()
        snapshot_list.append(dict(type="file", dir="base_model.py"))
        snapshot_list.append(dict(type="file", dir="main.py"))
        snapshot_list.append(dict(type="file", dir="parallel_training_model.py"))
        snapshot_list.append(dict(type="dir", dir="sndcgan_zgp"))
        self.snapshot(snapshot_list)

        self.sess = sess
        self.config = config
        self.device_list = device_list
        self.num_device = len(device_list)

        self.model_name = model_name

        self.train_data, \
        self.sample_data, \
        self.num_batch, \
        self.file_type = parallel_image_filename_loader(data_dir=data_dir,
                                                        dataset_name=dataset_name,
                                                        sample_num=config.sample_num,
                                                        batch_size=config.batch_size
                                                        )
        self.build_model()

    def build_model(self):
        size = self.config.final_size
        batch_size = self.config.batch_size
        gf_dim = self.config.gf_dim
        df_dim = self.config.df_dim
        z_dim = self.config.z_dim
        c_dim = 3

        tower_prefix = "Tower_{}"
        main_idx = 0

        self.gOptim = lambda: tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                                     beta1=self.config.beta1)
        self.dOptim = lambda: tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                                     beta1=self.config.beta1)

        # Data flow
        # Data parser
        def _parse_function(fname, number):
            image_string = tf.read_file(fname)
            if self.file_type in ['jpg', 'jpeg', 'JPG', 'JPEG']:
                image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            elif self.file_type in ['png', 'PNG']:
                image_decoded = tf.image.decode_png(image_string, channels=3)
            else:
                raise ValueError("Image type should be in 'jpg', 'png'. Got {}.".format(self.file_type))
            image_resized = tf.image.resize_images(image_decoded, (size, size))

            image_resized = image_resized / 127.5 - 1

            return image_resized, number

        def _parse_function_test(fname, z):
            image_string = tf.read_file(fname)
            if self.file_type in ['jpg', 'jpeg', 'JPG', 'JPEG']:
                image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            elif self.file_type in ['png', 'PNG']:
                image_decoded = tf.image.decode_png(image_string, channels=3)
            else:
                raise ValueError("Image type should be in 'jpg', 'png'. Got {}.".format(self.file_type))
            image_resized = tf.image.resize_images(image_decoded, (size, size))

            image_resized = image_resized / 127.5 - 1

            return image_resized, z


        # Start towering
        # Iterator
        self.iterator = nested_dict()

        # Loss container
        # Tower, data config
        print("Tower configuration ... ", end=" ", flush=True)
        tower_config_list = list()
        for idx, device in enumerate(self.device_list):
            # Tower config
            tower_config = TowerConfig(idx=idx,
                                       prefix=tower_prefix,
                                       is_main=idx == main_idx,
                                       num_devices=len(self.device_list),
                                       device_name=device)
            tower_config_list.append(tower_config)

            # Data flow
            # For train
            dataset = tf.data.Dataset.from_tensor_slices(self.train_data)
            dataset = dataset.repeat().shuffle(len(self.train_data) * 2)
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(
                    map_func=_parse_function,
                    batch_size=batch_size,
                    num_parallel_batches=int(batch_size * 1.5)
                )
            )
            self.iterator[device] = dataset.make_initializable_iterator()

            x, self.num = self.iterator[device].get_next()
            self.z = z = tf.random_normal(shape=[batch_size, z_dim], dtype=tf.float32, name="z")

            # For test
            if tower_config.is_main:
                self.sample_z = tf.constant(np.random.normal(0.0, 1.0,
                                                             size=[self.config.sample_num, z_dim])
                                            .astype(dtype=np.float32))

                test_dataset = tf.data.Dataset.from_tensor_slices((self.sample_data, self.sample_z))

                test_dataset = test_dataset.apply(
                    tf.contrib.data.map_and_batch(
                        map_func=_parse_function_test,
                        batch_size=1,
                        num_parallel_batches=batch_size * 4
                    )
                )

                self.iterator['test'] = test_dataset.make_initializable_iterator()

                sample_x, sample_z = self.iterator['test'].get_next()
        print("Done !")

        # Building network
        print("Build Network ...")
        network_list = list()
        dummy_test_network_list = list()
        for idx, (device, tower_config) in enumerate(zip(self.device_list, tower_config_list)):
            with tf.device(device), tf.variable_scope(tower_prefix.format(idx)):
                print("\tCreating gpu tower @ {:d} on device {:s}".format(idx, device))

                # Establish network
                network = Network(name="Network",
                                       batch_size=batch_size,
                                       size=size,
                                       gf_dim=gf_dim,
                                       df_dim=df_dim,
                                       reuse=False,
                                       is_training=True,
                                       tower_config=tower_config)
                network.build_network(x=x, z=z)
                network_list.append(network)

                # Establish dummy test network
                dummy = Network(name="Network",
                                batch_size=1,
                                size=size,
                                gf_dim=gf_dim,
                                df_dim=df_dim,
                                reuse=True,
                                is_training=False,
                                tower_config=tower_config)

                dummy.build_network(z=sample_z, x=sample_x)
                dummy_test_network_list.append(dummy)

                # Establish test network
                if tower_config.is_main:
                    print("\t +- Test network @  tower {:d} on device {:s}".format(idx, device))

                    self.test_network = dummy
                    self.sampler = self.test_network.y
                    self.input_checker = self.test_network.x

        self.test_enforcer = nccl_all_sum(dummy_test_network_list, lambda x: x.y)

        g_loss_list = nccl_all_mean(network_list, lambda x: x.g_loss)
        d_loss_list = nccl_all_mean(network_list, lambda x: x.d_loss)

        self.g_loss = network_mean(network_list, lambda x: x.g_loss)
        self.d_loss = network_mean(network_list, lambda x: x.d_loss)
        self.gp_loss = network_mean(network_list, lambda x: x.gp_loss)

        print(">> Done.")

        # Compute gradients
        print("Compute gradients ... ", end=' ', flush=True)
        self.g_optimize_op = optimizer_op(g_loss_list, network_list, self.gOptim, var_name="type__generator")
        self.d_optimize_op = optimizer_op(d_loss_list, network_list, self.dOptim, var_name="type__discriminator")
        print("Done !")

        self.sync_op = device_sync_op(tower_prefix, main_idx)
        self.network_list = network_list

        # Saver to save only main tower.
        excluding_towers = [tower_config.name for tower_config in tower_config_list if not tower_config.is_main]
        tracking_variables_list = [v for v in tf.global_variables() if not any(tower_name in v.op.name for tower_name in excluding_towers)]
        self.saver = tf.train.Saver(var_list=tracking_variables_list, max_to_keep=None)

    def train(self, config):
        # Initializing
        print("Initializing ... ", end=' ', flush=True)
        self.sess.run(tf.global_variables_initializer())
        print("Done !")

        # Restoring
        if config.checkpoint_dir is None:
            pass
        else:
            print("Restoring ... ", end=' ', flush=True)
            self.restore_checkpoints(config.checkpoint_dir, config.restore_list)
            print("Done !")

        print("Sync towers ... ", end=' ', flush=True)
        if self.num_device == 1:
            print(" >> Using only one device. Skip syncing. ", end=' ', flush=True)
        else:
            self.sess.run(self.sync_op)
        print("Done !")

        counter = 0
        start_time = time.time()

        # Check test input
        print("Checking input pipeline ... ", end=' ', flush=True)
        y_sample, x_sample, _ = self.sample_runner([self.test_network.y, self.test_network.x, self.test_enforcer])
        save_images(y_sample,
                    image_manifold_size(len(y_sample)),
                    os.path.join(self.get_result_dir(), "9_naive_gen_image.jpg"))
        save_images(x_sample,
                    image_manifold_size(len(x_sample)),
                    os.path.join(self.get_result_dir(), "9_input_image.jpg"))

        os.mkdir(os.path.join(self.get_result_dir(), "Result"))
        print("Done !")

        print("Train iterator initializing ... ", end=' ', flush=True)
        for device in self.device_list:
            self.sess.run(self.iterator[device].initializer)
        print("Done !")

        # Batch training
        epoch_time = time.time()

        effective_num_batch = (self.num_batch // len(self.device_list))
        print("CKPT directory: {}".format(self.get_checkpoint_dir()))
        print("Effective batch size: {}".format(self.config.batch_size * len(self.device_list)))
        while counter // effective_num_batch < config.epoch:
            epoch = counter // effective_num_batch
            idx = counter % effective_num_batch
            counter += 1

            # Syncing
            if np.mod(counter, 10000) == 0:
                if not self.num_device == 1:
                    self.sess.run(self.sync_op)
                    print("Sync towers on step {}.".format(counter))

            for i_discriminator in range(config.discriminator_iteration):
                # Update D network
                _, num_d = self.sess.run([self.d_optimize_op, self.num])

            # Update G network
            for i_generator in range(config.generator_iteration):
                _, num_g = self.sess.run([self.g_optimize_op, self.num])

            if np.mod(counter, config.print_interval) == 1:
                elapsed_time = int(time.time() - start_time)
                e_sec = elapsed_time % 60
                e_min = (elapsed_time // 60) % 60
                e_hr = elapsed_time // 3600

                dLoss, gLoss, gpLoss = self.sess.run([self.d_loss, self.g_loss, self.gp_loss])
                print("Epoch: [{epoch:2d}/{config_epoch:2d}] ".format(epoch=epoch, config_epoch=config.epoch) + \
                      "[{idx:4d}/{batch_idxs:2d}] ".format(idx=idx, batch_idxs=effective_num_batch) + \
                      "time: {e_hr:02d}:{e_min:02d}:{e_sec:02d} ".format(e_hr=e_hr, e_min=e_min, e_sec=e_sec) + \
                      "d_loss: {dLoss:.4f} ".format(dLoss=dLoss) + \
                      "penalty: {gp:.4f} ".format(gp=gpLoss) + \
                      "g_loss: {gLoss:.4f} ".format(gLoss=gLoss)
                      )

            if np.mod(counter, config.sample_interval) == 0:
                y_sample, _ = self.sample_runner([self.test_network.y, self.test_enforcer])
                save_images(y_sample,
                            image_manifold_size(len(y_sample)),
                            os.path.join(self.get_result_dir(),
                                         "Result",
                                         '{epoch:05d}_{idx:04d}.jpg'.format(epoch=epoch, idx=idx)))

                step_time = int((time.time() - epoch_time) / config.sample_interval * effective_num_batch) # time / epoch
                epoch_time = time.time()

                s_sec = step_time % 60
                s_min = (step_time // 60) % 60
                s_hr = step_time // 3600

                epoch_speed = 1 / step_time * 3600
                print("[Sample] " + \
                      "Time per epoch: {e_hr:02d}:{e_min:02d}:{e_sec:02d}\t".
                      format(e_hr=s_hr, e_min=s_min, e_sec=s_sec) + \
                      "Epoch speed: {:.3f} eph".format(epoch_speed))

            if idx == effective_num_batch // 2:
                self.saver.save(self.sess, os.path.join(self.get_checkpoint_dir(), "Ckpt"), global_step=epoch * 10)
            if idx == effective_num_batch - 1:
                self.saver.save(self.sess, os.path.join(self.get_checkpoint_dir(), "Ckpt"), global_step=epoch * 10 + 5)

    def sample_runner(self, elements):
        self.sess.run(self.iterator['test'].initializer)

        retList = [[] for x in elements]
        for i in range(len(self.sample_data)):
            results = list(self.sess.run(elements))
            for idx, result in enumerate(results):
                retList[idx].append(result)

        # Check actual end.
        try:
            self.sess.run(elements)
            raise ValueError("Iterator doesn't end. Something wrong!")
        except tf.errors.OutOfRangeError:
            pass
        except Exception as ex:
            raise ex

        return retList

    def restore_checkpoints(self, dir_, restore_list=None):
        if dir_ is None:
            return
        else:
            def checker(var, restore_list):
                for p in restore_list:
                    if p in var.name:
                        return True
                return False

            restoreVars = [v for v in tf.trainable_variables() if checker(v, restore_list)]
            for v in tf.trainable_variables():
                if v in restoreVars:
                    prefix = "O"
                else:
                    prefix = "X"
                print(prefix, v.name)
            saver = tf.train.Saver(var_list=restoreVars)
            saver.restore(self.sess, dir_)
