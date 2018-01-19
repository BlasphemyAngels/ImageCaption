#########################################################################
# File Name: build_model.py
# Author: caochenglong
# mail: caochenglong@163.com
# Created Time: 2017-11-13 17:47:22
# Last modified: 2017-11-13 17:47:22
#########################################################################
# !/usr/bin/python3
# _*_coding: utf-8_*_
import tensorflow as tf
import tensorlayer as tl


def image_caption(mode, image_embeddings, seq_embeddings, target_seqs,
                  input_mask, batch_size, embedding_size, vocabulary_size,
                  initializer, lstm_dropout_keep_prob):
    if mode == "inference":
        with tf.variable_scope("lstm", initializer) as lstm_scope:
            tl.layers.set_name_reuse(True)
            image_embeddings = tl.layers.DenseLayer(
                image_embeddings, n_units=embedding_size,
                act=tf.nn.relu, name="img_reshape")
            image_embeddings = tl.layers.ReshapeLayer(
                image_embeddings, shape=(-1, 1, embedding_size))
            print(image_embeddings.outputs)
            img_rnn = tl.layers.DynamicRNNLayer(
                image_embeddings,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=embedding_size,
                dropout=None,
                initial_state=None,
                sequence_length=tf.ones([batch_size]),
                return_seq_2d=True,
                name="rnn")
            lstm_scope.reuse_variables()
            state_feed = tf.placeholder(
                dtype=tf.float32,
                shape=[None, sum(img_rnn.cell.state_size)],
                name="state_feed")
            state_tuple = tf.split(state_feed, 2, 1)
            state_tuple = tf.nn.rnn_cell.LSTMStateTuple(
                state_tuple[0], state_tuple[1])
            seq_rnn = tl.layers.DynamicRNNLayer(
                seq_embeddings,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=embedding_size,
                dropout=None,
                initial_state=state_tuple,
                sequence_length=tf.ones([batch_size]),
                return_seq_2d=True,
                name="rnn"
            )
    else:
        with tf.variable_scope("lstm", initializer) as lstm_scope:
            if mode == "train":
                dropout = lstm_dropout_keep_prob
            else:
                dropout = None
            image_embeddings = tl.layers.DenseLayer(
                image_embeddings, n_units=embedding_size,
                act=tf.nn.relu, name="img_reshape")
            image_embeddings = tl.layers.ReshapeLayer(
                image_embeddings, shape=(batch_size, 1,
                                         embedding_size))
            img_rnn = tl.layers.DynamicRNNLayer(
                image_embeddings,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=embedding_size,
                dropout=dropout,
                initial_state=None,
                sequence_length=tf.ones([batch_size]),
                return_seq_2d=True,
                name="rnn")
            lstm_scope.reuse_variables()
            tl.layers.set_name_reuse(True)
            seq_rnn = tl.layers.DynamicRNNLayer(
                seq_embeddings,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=embedding_size,
                dropout=dropout,
                initializer=initializer,
                initial_state=img_rnn.final_state,
                sequence_length=tf.reduce_sum(input_mask, 1),
                return_seq_2d=True,
                name="rnn",
            )

    network = seq_rnn
    network = tl.layers.DenseLayer(
        network, n_units=vocabulary_size, act=tf.identity, name="logits")
    logits = network.outputs
    network.all_layers = image_embeddings.all_layers \
        + network.all_layers
    network.all_params = image_embeddings.all_params \
        + network.all_params
    if mode == "inference":
        softmax = tf.nn.softmax(logits, name="softmax")
        return softmax, img_rnn, seq_rnn, state_feed
    batch_loss, losses, weights, _ = tl.cost.cross_entropy_seq_with_mask(
        logits, target_seqs, input_mask, return_details=True)
    tf.losses.add_loss(batch_loss)
    total_loss = tf.losses.get_total_loss()
    target_cross_entropy_losses = losses
    target_cross_entropy_loss_weights = weights
    return total_loss, target_cross_entropy_losses, \
        target_cross_entropy_loss_weights, network
