"""
N x N Parser.
"""
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

    return image_resized


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


"""
N x N Parser.
CelebA cropping.
"""
def _parse_function(fname, number):
    image_string = tf.read_file(fname)
    if self.file_type in ['jpg', 'jpeg', 'JPG', 'JPEG']:
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    elif self.file_type in ['png', 'PNG']:
        image_decoded = tf.image.decode_png(image_string, channels=3)
    else:
        raise ValueError("Image type should be in 'jpg', 'png'. Got {}.".format(self.file_type))
    image_cropped = tf.image.crop_to_bounding_box(image_decoded, 41, 21, 136, 136)
    image_resized = tf.image.resize_images(image_cropped, (size, size))
    image_resized = image_resized / 127.5 - 1

    return image_resized


def _parse_function_test(fname, z):
    image_string = tf.read_file(fname)
    if self.file_type in ['jpg', 'jpeg', 'JPG', 'JPEG']:
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    elif self.file_type in ['png', 'PNG']:
        image_decoded = tf.image.decode_png(image_string, channels=3)
    else:
        raise ValueError("Image type should be in 'jpg', 'png'. Got {}.".format(self.file_type))
    image_cropped = tf.image.crop_to_bounding_box(image_decoded, 41, 21, 136, 136)
    image_resized = tf.image.resize_images(image_cropped, (size, size))
    image_resized = image_resized / 127.5 - 1

    return image_resized, z
