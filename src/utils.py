#########################################################################
# File Name: util.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-11-04 20:57:10
# Last modified: 2017-11-04 20:57:10
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_
import tensorflow as tf


def distort_image(image, thread_id):
    with tf.variable_scope("flip_horizontal"):
        image = tf.image.random_flip_left_right(image)
    color_ordering = thread_id % 2
    with tf.variable_scope("distort_color"):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        else:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)
    return image


def image_int_to_float(img, sess):
    with tf.name_scope("image_int_to_float"):
        img = tf.image.convert_image_dtype(
            img, dtype=tf.float32, name="convert_to_float32")
        img = tf.clip_by_value(img, 0.0, 1.0)
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2)
    img = sess.run(img)
    return img
