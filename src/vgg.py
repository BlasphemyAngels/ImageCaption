#########################################################################
# File Name: vgg.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-11-26 17:52:49
# Last modified: 2017-11-26 17:52:49
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_

"""
"""

import tensorflow as tf
from tensorflow.contrib.slim import nets
import tensorflow.contrib.slim as slim
from tensorflow.contrib import layers
from hparams import vgg16_hparams
from tensorflow.contrib.slim import assign_from_checkpoint_fn
from tensorflow.contrib.layers.python.layers import layers as layers_lib


def vgg16(images, hparams):
    vgg = nets.vgg

    print(hparams)
    # Create the model
    with slim.arg_scope(
            [layers.conv2d, layers_lib.fully_connected],
            trainable=hparams.trainable):
        is_training = hparams.is_training
        fc7 = vgg.vgg_16(images, is_training=is_training)[1][hparams.vgg_layer]

    # Restore only the convolutional layers:
    variables_to_restore = slim.get_variables_to_restore(exclude=['fc8'])
    init_fn = assign_from_checkpoint_fn(
        hparams.vgg16_model_file, variables_to_restore)
    return fc7, init_fn
