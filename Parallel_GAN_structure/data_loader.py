import os
import numpy as np
import tensorflow as tf


def image_filename_loader(data_dir,
                          dataset_name,
                          sample_num,
                          batch_size,
                          repeat=1,
                          remainder_handling_policy="drop",
                          preload_filename="data_list.txt",
                          image_type='jpg'):
    """
    Image Filename Loader

    Version: 1.0.2
    Change log
        * Applying multiple iteration for one step.
        * Add 'FULL' Policy.
    Modified date: 2018.09.06.
    """
    print("\n:::: Data Loader ::::")
    assert remainder_handling_policy in ['drop', 'fill', 'random_fill', 'full'], \
        "Unsupport policy {}. Policy should be in [drop, fill, random_fill, full].".format(remainder_handling_policy)

    image_type_list = ['jpg', 'png']
    image_type_list.append(image_type)

    dataset_dir = os.path.join(data_dir, dataset_name)

    data_list = list()
    OVERRIDE = True
    if not OVERRIDE and os.path.exists(os.path.join(dataset_dir, preload_filename)):
        print("Load data list from existing list.")
        with open(os.path.join(dataset_dir, preload_filename), 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_list.append(line.split("\n")[0])

    else:
        print("Creating file list.")
        for path, _, files in os.walk(dataset_dir):
            for filename in files:
                if filename.split(".")[-1] in image_type_list:
                    data_list.append(os.path.join(os.path.abspath(path), filename))

        data_list = sorted(data_list)

        with open(os.path.join(dataset_dir, preload_filename), 'w') as f:
            f.writelines([str(data) + "\n" for data in data_list])

    assert sample_num + batch_size <= len(data_list), \
        "The number of data is not enough to make at least one batch."

    sample_data = data_list[:sample_num]
    data_list = list(data_list[sample_num:])


    if remainder_handling_policy == 'full':
        file_type = data_list[0].split("\n")[0].split(".")[-1]
        num_element = len(data_list)
        step_per_epoch = num_element / batch_size
        print("Remainder policy: FULL")
        print("Data list loaded.")
        print("File type: {}".format(file_type))
        print("Sample: {}".format(len(sample_data)))
        print("Data: {:,}".format(num_element))
        print("Batch: {:,}(batch) x {:,.2f}(step/epoch)".format(batch_size, step_per_epoch))
        print(":::::::::::::::::::::\n")

        return (data_list, tf.constant(np.arange(num_element, dtype=np.int32))), \
               sample_data, \
               int(num_element / batch_size) + 1, \
               file_type

    else:
        remainder = len(data_list) % batch_size

        if remainder_handling_policy == 'drop':
            print("Remainder policy: DROP")
            print("Drop {} items.".format(remainder))
            data_list = list(data_list[:len(data_list) - remainder])

        elif remainder_handling_policy == 'fill':
            print("Remainder policy: FILL")
            if remainder == 0:
                pass
            else:
                num_to_fill = batch_size - remainder
                fill_list = data_list[len(data_list) // 2:len(data_list) // 2 + num_to_fill]
                data_list = list(data_list + fill_list)

        elif remainder_handling_policy == 'random_fill':
            print("Remainder policy: RANDOM_FILL")
            if remainder == 0:
                pass
            else:
                num_to_fill = batch_size- remainder
                import random
                fill_list = random.sample(data_list, num_to_fill)
                data_list = list(data_list + fill_list)


        else:
            raise ValueError("Unsupport policy {}. Policy should be in [drop, fill, random_fill".
                             format(remainder_handling_policy))

        num_element = len(data_list)

        DEL_data_index = list(range(len(data_list)))

        DEL_zip = list(zip(data_list, DEL_data_index))

        # Repeat procedure
        step_per_epoch = num_element // batch_size
        assert num_element % batch_size == 0, "num_element has remainder!"

        repeat_data_list = list()
        DEL_repeat_number_list = list()
        for batch_index in range(step_per_epoch):
            for _ in range(repeat):
                for data, data_index in DEL_zip[batch_size * batch_index:batch_size * (batch_index + 1)]:
                    repeat_data_list.append(data)
                    DEL_repeat_number_list.append(data_index)

        data_list = repeat_data_list
        file_type = data_list[0].split("\n")[0].split(".")[-1]

        assert repeat * num_element == len(repeat_data_list), "Something went wrong in repeat procedure."

        print("Data list loaded.")
        print("File type: {}".format(file_type))
        print("Sample: {}".format(len(sample_data)))
        print("Data: {:,}".format(num_element))
        print("Batch: {:,}(batch) x {:,}(iteration) x {:,}(step/epoch)".format(batch_size,
                                                                               repeat,
                                                                               step_per_epoch))
        print(":::::::::::::::::::::\n")

        return (data_list, tf.constant(np.asarray(DEL_repeat_number_list))), \
               sample_data, \
               int(num_element/batch_size), \
               file_type


def parallel_image_filename_loader(data_dir,
                                   dataset_name,
                                   sample_num,
                                   batch_size,
                                   remainder_handling_policy="full",
                                   preload_filename="data_list.txt",
                                   image_type='jpg'):
    """
    Image Filename Loader

    Version: 1.0.2
    Change log
        * Applying multiple iteration for one step.
        * Add 'FULL' Policy.
    Modified date: 2018.09.06.
    """
    print("\n:::: Data Loader ::::")
    assert remainder_handling_policy in ['full'], \
        "Unsupport policy {}. Policy should be in [full].".format(remainder_handling_policy)

    image_type_list = ['jpg', 'png']
    image_type_list.append(image_type)

    dataset_dir = os.path.join(data_dir, dataset_name)

    data_list = list()
    OVERRIDE = False
    if not OVERRIDE and os.path.exists(os.path.join(dataset_dir, preload_filename)):
        print("Load data list from existing list.")
        with open(os.path.join(dataset_dir, preload_filename), 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_list.append(line.split("\n")[0])

    else:
        print("Creating file list.")
        for path, _, files in os.walk(dataset_dir):
            for filename in files:
                if filename.split(".")[-1] in image_type_list:
                    data_list.append(os.path.join(os.path.abspath(path), filename))

        data_list = sorted(data_list)

        with open(os.path.join(dataset_dir, preload_filename), 'w') as f:
            f.writelines([str(data) + "\n" for data in data_list])

    assert sample_num + batch_size <= len(data_list), \
        "The number of data is not enough to make at least one batch."

    sample_data = data_list[:sample_num]
    data_list = list(data_list[sample_num:])

    num_element = len(data_list)
    file_type = data_list[0].split("\n")[0].split(".")[-1]

    return (data_list, tf.constant(np.arange(num_element), dtype=np.int32)), \
               sample_data, \
               int(num_element/batch_size) + 1, \
               file_type