#########################################################################
# File Name: train.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-11-13 20:34:28
# Last modified: 2017-11-13 20:34:28
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_

import os
import sys
import logging
import tensorflow as tf
import tensorlayer as tl
import argparse
import numpy as np
from model import image_caption
from ops import get_one_batch
from ops import image_embedding
from ops import seq_embedding
from ops import read_and_decode
from hparams import vgg16_hparams

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    logger.setLevel(level=logging.INFO)
    logger.info("run %s" % ' '.join(sys.argv))

    parser = argparse.ArgumentParser(description="train model")

    parser.add_argument(
        "--buckets", type=str, default='', dest="buckets",
        help="input data path")
    parser.add_argument(
        "-d, --data_path", action="store", type=str, default="data",
        dest="data_path", help="the path of the data")
    parser.add_argument(
        "-s,--step", action="store", type=int, default=30000,
        dest="step", help="the num of training steps")
    parser.add_argument(
        "-b,--batch_size", action="store", type=int, default=32,
        dest="batch_size", help="the trianing bath size")
    parser.add_argument(
        "-i,--initializer_scale", action="store", type=float, default=0.08,
        dest="initializer_scale", help="the initializer scale")
    parser.add_argument(
        "-h,--img_h", action="store", type=int, default=224,
        dest="img_h", help="image height")
    parser.add_argument(
        "-w,--img_w", action="store", type=int, default=224,
        dest="img_w", help="image width")
    parser.add_argument(
        "-q,--queue_capacity", action="store", type=int, default=1000,
        dest="queue_capacity", help="queue capacity")
    parser.add_argument(
        "-c,--c_dims", action="store", type=int, default=3,
        dest="c_dims", help="the num of image channel")

    parser.add_argument(
        "-o,--embedding_size", type=int, default=1024,
        dest="embedding_size", help="the dimension of word embedding")
    parser.add_argument(
        "-p,--lstm_dropout_keep_prob", action="store", type=float, default=0.7,
        dest="lstm_dropout_keep_prob", help="the dropout keep probability")
    parser.add_argument(
        "-t,--train_vgg16", action="store_true", default=False,
        dest="train_vgg16", help="wether or not tra2ning the vgg16")
    parser.add_argument(
        "-l,--train_vgg16_learning_rate", action="store", type=float,
        default=0.001, dest="train_vgg16_learning_rate",
        help="the learning rate for training the vgg16")
    parser.add_argument(
        "-r,--initial_learning_rate", action="store", type=float,
        default=2.0,
        dest="initial_learning_rate", help="iniital learning rate")
    parser.add_argument(
        "-g,--clip_gradients", action="store", type=float, default=5.0,
        dest="clip_gradients", help="the clip gradient")
    parser.add_argument(
        "-m,--model_dir", action="store", type=str, default="train",
        dest="model_dir", help="the directory of the model")
    parser.add_argument(
        "-k,--max_ckpt_to_keep", action="store", type=int, default=30,
        dest="max_ckpt_to_keep", help="maximum number of save the model")
    parser.add_argument(
        "-f,--learning_rate_decay_factor", action="store",
        type=float, default=0.5,
        dest="learning_rate_decay_factor",
        help="the decay factor of the learning rate")
    num_examples_per_epoch = 320
    num_epochs_per_decay = 8.0
    args, _ = parser.parse_known_args()

    model_path = os.path.join(args.buckets, args.model_dir)
    if not tf.gfile.IsDirectory(model_path):
        logger.info("create model dir: %s" % args.model_dir)
        tf.gfile.MakeDirs(model_path)

    data_path = os.path.join(args.buckets, args.data_path, "data1.tfrecord")

    vocab_path = os.path.join(args.buckets, args.data_path, "vocab.txt")

    vocab = tl.nlp.Vocabulary(vocab_path, start_word="<S>",
                              end_word="</S>", unk_word="<UNK>")
    vocabulary_size = len(vocab.vocab)

    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()
        initializer = tf.random_uniform_initializer(
            maxval=args.initializer_scale, minval=-args.initializer_scale)
        image, caption = read_and_decode(
            data_path, args.img_h, args.img_w, is_training=True)
        images, input_seqs, target_seqs, masks = get_one_batch(
            [[image, caption]],
            args.batch_size,
            args.queue_capacity)
        images = tf.cast(images, tf.float32)
        hparams = vgg16_hparams()
        hparams.vgg16_model_file = os.path.join(
            args.buckets, "checkpoint/vgg_16.ckpt")
        img_embed, init_fn = image_embedding(images, hparams)
        init_fn(sess)
        img_embed = tl.layers.InputLayer(img_embed, "input_images")
        img_embed = tl.layers.ReshapeLayer(
            img_embed, shape=[args.batch_size, 4096], name="input_reshape")
        seq_embed = seq_embedding(
            input_seqs, vocabulary_size, args.embedding_size, initializer)
        total_loss, _, _, network = image_caption("train",
                                                  img_embed, seq_embed,
                                                  target_seqs,
                                                  masks,
                                                  args.batch_size,
                                                  args.embedding_size,
                                                  vocabulary_size,
                                                  initializer,
                                                  args.lstm_dropout_keep_prob)
        #  network.print_layers()
        global_step = tf.Variable(
            initial_value=0,
            dtype=tf.int32,
            name="global_step",
            trainable=False,
            collections=[
                tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES]
        )

        learning_rate_decay_fn = None
        if args.train_vgg16:
            learning_rate = tf.constant(args.train_vgg16_learning_rate)
        else:
            learning_rate = tf.constant(args.initial_learning_rate)
            if args.learning_rate_decay_factor > 0:
                num_batches_per_epoch = (
                    num_examples_per_epoch / args.batch_size)
                decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=args.learning_rate_decay_factor,
                    staircase=True)
            learning_rate_decay_fn = _learning_rate_decay_fn

        train_ops = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer="SGD",
            clip_gradients=args.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn
        )
        sess.run(tf.global_variables_initializer())
        logger.info("restore model from %s" % args.model_dir)
        try:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        except Exception:
            logger.info("no ckpt model file")
        saver = tf.train.Saver(max_to_keep=args.max_ckpt_to_keep)

        logger.info("start training")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(sess.run(global_step), args.step):
            loss, _ = sess.run([total_loss, train_ops])
            logger.info("step %d: loss %.4f" % (step, loss))
            if step % 100 == 0:
                save_path = saver.save(sess, os.path.join(
                    model_path, "model.ckpt"), global_step=step)

        save_path = saver.save(sess, os.path.join(
            model_path, "model.ckpt"), global_step=args.step)
        coord.request_stop()
        coord.join(threads)
        sess.close()
