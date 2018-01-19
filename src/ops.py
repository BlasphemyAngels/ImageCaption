#########################################################################
# File Name: ops.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-11-12 11:39:20
# Last modified: 2017-11-12 11:39:20
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_
import os
import sys
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from utils import distort_image
from tensorlayer.layers import *
from vgg import vgg16
from hparams import vgg16_hparams


def read_and_decode(data_path, height, width, is_training):
    f_queue = tf.train.string_input_producer([data_path])
    reader = tf.TFRecordReader()
    _, serizlized_example = reader.read(f_queue)
    features, sequence_features = tf.parse_single_sequence_example(
        serizlized_example,
        context_features={
            "image/image_raw": tf.FixedLenFeature([], dtype=tf.string),
        },
        sequence_features={
            "image/caption": tf.FixedLenSequenceFeature([], dtype=tf.string),
            "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
    )
    img = tf.decode_raw(features["image/image_raw"], tf.uint8)
    img = tf.reshape(img, [height, width, 3])
    #  img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    if is_training:
        img = distort_image(img, 0)
    #  img = tf.image.convert_image_dtype(img, dtype=tf.int64)
    #  img = tf.clip_by_value(img, 0.0, 1.0)
    #  img = tf.subtract(img, 0.5)
    #  img = tf.multiply(img, 2)
    caption = sequence_features["image/caption_ids"]
    return img, caption


def get_one_batch(image_and_captions, batch_size, queue_capacity):
    enqueue_list = []
    for image, caption in image_and_captions:
        caption_length = tf.shape(caption)[0]
        input_length = tf.expand_dims(tf.subtract(caption_length, 1), axis=0)
        input_seq = tf.slice(caption, [0], input_length)
        target_seq = tf.slice(caption, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.float32)
        enqueue_list.append([image, input_seq, target_seq, indicator])
        images, input_seqs, target_seqs, mask = tf.train.batch_join(
            enqueue_list,
            batch_size=batch_size,
            capacity=queue_capacity,
            dynamic_pad=True,
            name="batch_and_pad")
    return images, input_seqs, target_seqs, mask


def seq_embedding(input_seqs, vocabulary_size, embedding_size, initializer):
    network = tl.layers.EmbeddingInputlayer(
        input_seqs,
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        E_init=initializer,
        name="seq_embedding")
    return network


def image_embedding(images, hparams):
    fc7, init_fn = vgg16(images, hparams)
    return fc7, init_fn
