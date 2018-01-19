#########################################################################
# File Name: hparams.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-11-26 11:07:08
# Last modified: 2017-11-26 11:07:08
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_

"""
vgg_16/conv1/conv1_1
vgg_16/conv1/conv1_2
vgg_16/pool1
vgg_16/conv2/conv 2_1
vgg_16/conv2/conv2_2
vgg_16/pool2
vgg_16/conv3/conv3_1
vgg_16/con v3/conv3_2
vgg_16/conv3/conv3_3
vgg_16/pool3
vgg_16/conv4/conv4_1
vgg_16/conv4/conv4_2
vgg_16/conv4/conv4_3
vgg_16/pool4
vgg_16/conv5/conv5_1
vgg_16/conv5/conv5_2
vgg_16/conv5/conv5_3
vgg_16/pool5
vgg_16/fc6
vgg_16/fc7
vgg_16/fc8
"""

import tensorflow as tf


def lstm_image_caption_hparams():
    return tf.contrib.training.HParams(
        images_dir="images",
        captions_file="captions",
        vocab_file="vocab.txt",
        min_word_count=1,
        skip_window=5,
        word_vec_size=1024,
        word2vec_model_file="w2v.model",
        word2vec_model_vector_file="w2v.model.vector",
        image_height=224,
        image_wight=224,
        is_delete=False,
        is_reproduce=False,
        config_path="config",
        get_images_name_fn="",
        get_captions_fn=""
    )


def vgg16_hparams():
    return tf.contrib.training.HParams(
        is_training=True,
        trainable=False,
        vgg16_model_file="./checkpoint/vgg_16.ckpt",
        vgg_layer="vgg_16/fc7"
    )


def vgg16_hparams_infer():
    return tf.contrib.training.HParams(
        is_training=False,
        trainable=False,
        vgg16_model_file="./checkpoint/vgg_16.ckpt",
        vgg_layer="vgg_16/fc7"
    )


def data_generate_hparams():
    return tf.contrib.training.HParams(

    )
