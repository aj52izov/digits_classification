import pdb

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def apply(inputs, batch_size=500, kernel_shape=3, padding: str = 'CONSTANT', constant_values: tfa.types.TensorLike = 0):
    '''
            apply the median filter and normalize the data between 0 and 1
            :param batch_size: the batch_size for processing -> to avoid out of memory situation
            :param inputs: the inputs images
            :param kernel_shape: the filter shape
            :param padding: the padding type ("CONSTANT" , "REFLECT" or "SYMMETRIC")
            :param constant_values: the padding value in constant padding case
            :return: the filtered and normalized images
    '''
    inputs = tf.cast(inputs, dtype=tf.float32).numpy()
    for idx in range(0, len(inputs), batch_size):
        inputs[idx:batch_size] = tfa.image.median_filter2d(inputs[idx:batch_size], kernel_shape, padding, constant_values) / 255.0
    return tf.convert_to_tensor(inputs)
