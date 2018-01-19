#########################################################################
# File Name: TFC/decoder.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-11-27 15:35:06
# Last modified: 2017-11-27 15:35:06
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_

"""Generate captions for images by a given model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from ops import get_one_batch
from ops import seq_embedding
from model import image_caption
import tensorflow as tf
import tensorlayer as tl
from hparams import vgg16_hparams_infer
from ops import image_embedding
from PIL import Image
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buckets", type=str, default='', dest="buckets",
        help="input data path")
    parser.add_argument(
        "-d, --data_path", action="store", type=str, default="data",
        dest="data_path", help="the path of the data")
    parser.add_argument(
        "-m,--model_dir", action="store", type=str, default="train",
        dest="model_dir", help="the directory of the model")
    parser.add_argument(
        "-i,--initializer_scale", action="store", type=float, default=0.08,
        dest="initializer_scale", help="the initializer scale")
    parser.add_argument(
        "-p,--lstm_dropout_keep_prob", action="store", type=float, default=0.7,
        dest="lstm_dropout_keep_prob", help="the dropout keep probability")
    args, _ = parser.parse_known_args()
    images_dir = os.path.join(args.buckets, "images/")
    print(images_dir)
    max_caption_length = 20
    top_k = 4
    batch_size = 1
    print("top k:%d" % top_k)
    n_captions = 50
    mode = 'inference'
    model_dir = os.path.join(args.buckets, args.model_dir)
    vocab_path = os.path.join(args.buckets, args.data_path, "vocab.txt")
    vocab = tl.nlp.Vocabulary(vocab_path, start_word="<S>",
                              end_word="</S>", unk_word="<UNK>")
    print(len(vocab.vocab))
    vocabulary_size = len(vocab.vocab)
    embedding_size = 1024
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        initializer = tf.random_uniform_initializer(
            maxval=args.initializer_scale, minval=-args.initializer_scale)
        # In inference mode, images and inputs are fed via placeholders.
        image_feed = tf.placeholder(
            dtype=tf.string, shape=[], name="image_feed")
        image_feed = tf.image.decode_jpeg(image_feed, channels=3)
        input_feed = tf.placeholder(dtype=tf.int64,
                                    shape=[None],  # 1 word id
                                    name="input_feed")
        # Process image and insert batch dimensions.
        images = tf.expand_dims(image_feed, 0)
        print(images)
        input_seqs = tf.expand_dims(input_feed, 1)
        # No target sequences or input mask in inference mode.
        target_seqs = None
        input_mask = None
        images = tf.cast(images, tf.float32)
        hparams = vgg16_hparams_infer()
        hparams.vgg16_model_file = os.path.join(
            args.buckets, "checkpoint/vgg_16.ckpt")
        print(hparams.vgg16_model_file)
        img_embed, init_fn = image_embedding(images, hparams)
        img_embed = tl.layers.InputLayer(img_embed, "input_images")
        img_embed = tl.layers.ReshapeLayer(
            img_embed, shape=[batch_size, 4096], name="input_reshape")
        seq_embed = seq_embedding(
            input_seqs, vocabulary_size, embedding_size, initializer)
        softmax, net_img_rnn, net_seq_rnn, state_feed = image_caption(
            "inference", img_embed, seq_embed, target_seqs, input_mask,
            batch_size, embedding_size, vocabulary_size, initializer,
            args.lstm_dropout_keep_prob)

        if tf.gfile.IsDirectory(model_dir):
            model_dir = tf.train.latest_checkpoint(model_dir)
        if not model_dir:
            raise ValueError("No checkpoint file found in: %s" % model_dir)
        saver = tf.train.Saver()

        def _restore_fn(sess):
            tf.logging.info(
                "Loading model from checkpoint: %s", model_dir)
            saver.restore(sess, model_dir)
            tf.logging.info("Successfully loaded checkpoint: %s",
                            os.path.basename(model_dir))

        restore_fn = _restore_fn
        g.finalize()

        files = tf.gfile.Glob(os.path.join(images_dir, "*.jpg"))

    # Generate captions
    with tf.Session(graph=g) as sess:
        init_fn(sess)
        # Load the model from checkpoint.
        restore_fn(sess)
        for filename in files:
            with tf.gfile.GFile(filename, "rb") as f:
                encoded_image = f.read()    # it is string, haven't decode !

            print(filename)
            init_state = sess.run(net_img_rnn.final_state, feed_dict={
                                  "image_feed:0": encoded_image})
            for _ in range(n_captions):
                state = np.hstack((init_state.c, init_state.h))  # (1, 1024)
            a_id = vocab.start_id
            sentence = ''
            for _ in range(max_caption_length - 1):
                softmax_output, state = sess.run(
                    [softmax, net_seq_rnn.final_state],
                    feed_dict={
                        input_feed: [a_id],
                        state_feed: state,
                    })
                state = np.hstack((state.c, state.h))
                a_id = tl.nlp.sample_top(softmax_output[0], top_k=top_k)
                word = vocab.id_to_word(a_id)
                if a_id == vocab.end_id:
                    break
                sentence += word + ' '

            print('  %s' % sentence)
