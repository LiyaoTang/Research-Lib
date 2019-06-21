#!/usr/bin/env python
# coding: utf-8
'''
module: some tf modeuls from classic networks
'''

import os
import numpy as np
import tensorflow as tf
from .TF_Ops import *

def alexnet_conv_layers(input, auxilary_input=None, prelu_initializer=tf.constant_initializer(0.25), use_spp=False):
    '''
    input: images, expected to be of [batch, width, height, channel]
    '''
    def flatten(feat_map):
        feat_map = tf.transpose(feat_map, perm=[0, 3, 1, 2])
        feat_map = remove_axis(feat_map, [2, 3])
        return feat_map

    with tf.variable_scope('conv1'):
        if auxilary_input:
            conv1 = conv_layer(input, 96, filter_size=11, stride=4, padding='VALID', activation=None)
            conv1 = conv1 + auxilary_input  # join before activation
            conv1 = tf.nn.relu(conv1)
        else:
            conv1 = conv_layer(input, 96, filter_size=11, stride=4, padding='VALID')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0, name='norm1')

    with tf.variable_scope('conv1_skip'):
        conv1_skip = conv_layer(lrn1, 16, filter_size=1, activation=None)
        conv1_skip = prelu(conv1_skip, initializer=prelu_initializer)
        # each img flatten into 1-D
        if not use_spp:
            conv1_skip_flat = flatten(conv1_skip)

    with tf.variable_scope('conv2'):
        # 2-branch by num_groups=2
        conv2 = conv_layer(lrn1, 256, filter_size=5, num_groups=2, padding='SAME')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0, name='norm2')

    with tf.variable_scope('conv2_skip'):
        conv2_skip = conv_layer(lrn2, 32, filter_size=1, activation=None)
        conv2_skip = prelu(conv2_skip, initializer=prelu_initializer)
        if not use_spp:
            conv2_skip_flat = flatten(conv2_skip)

    with tf.variable_scope('conv3'):
        conv3 = conv_layer(lrn2, 384, filter_size=3, padding='SAME')

    with tf.variable_scope('conv4'):
        conv4 = conv_layer(conv3, 384, filter_size=3, num_groups=2, padding='SAME')

    with tf.variable_scope('conv5'):
        conv5 = conv_layer(conv4, 256, filter_size=3, num_groups=2, padding='SAME')
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        if not use_spp:
            pool5_flat = flatten(pool5)

    with tf.variable_scope('conv5_skip'):
        conv5_skip = conv_layer(conv5, 64, filter_size=1, activation=None)
        conv5_skip = prelu(conv5_skip)
        if not use_spp:
            conv5_skip_flat = flatten(conv5_skip)

    with tf.variable_scope('big_concat'):
        # concat all skip layers
        if use_spp:
            feat = [conv1_skip, conv2_skip, conv5_skip, pool5]
            spp_bin_list = [[27], [13], [13], [6]]  # pool size on pixel = [[4x2, 4x2, 4x2, 5x3]]
            for bins, i in zip(spp_bin_list, feat):  # a spp for each layer
                feat[i] = spatial_pyramid_pooling(feat[i], bins)
        else:
            feat = [conv1_skip_flat, conv2_skip_flat, conv5_skip_flat, pool5_flat]
        feat_concat = tf.concat(feat, 1)

    return feat_concat

def re3_lstm_tracker(input, num_unrolls, lstm_size=512, prev_state=None, rnn_type='lstm', log=True):
    '''
    input: object features in time sequence, expected to be [batch, time, feat], with time = num_unrolls
    TODO: migrate to TF 2.0: 
        contrib.rnn.LSTMCell -> keras.layers.LSTMCell
        contrib.rnn.LSTMStateTuple -> get_initial_tuple
        dynamic_rnn -> keras.layers.RNN
    '''
    with tf.variable_scope('fc6'):
        feat_len = input.shape.as_list()[-1]
        flatten_input = tf.reshape(input, [-1, feat_len]) # flatten as [batch x time, feat]
        fc6_out = dense_layer(flatten_input, 1024, name='fc')
        fc6_out = tf.reshape(fc6_out, [-1, num_unrolls, feat_len])  # reshaped back to [batch, time, feat]

    # late fusion: concat feature from t, t-1
    first_frame = fc6_out[:, 0:1, ...]
    other_frame = fc6_out[:, :-1, ...]
    past_frames = tf.concat([first_frame, other_frame], axis=1)  # [B,T,feat]
    fc6_out = tf.concat([fc6_out, past_frames], axis=-1)  # [B,T,feat_t-feat_t-1]
        
    swap_memory = num_unrolls > 1
    assert rnn_type in ['lstm']
    with tf.variable_scope('lstm1'):
        lstm1 = tf.contrib.rnn.LSTMCell(lstm_size)

        # cell state
        if prev_state is not None: # if traker already running
            state1 = tf.contrib.rnn.LSTMStateTuple(prev_state[0], prev_state[1])
        else:
            state1 = lstm1.zero_state(dtype=tf.float32)

        # unroll
        lstm1_outputs, state1 = tf.nn.dynamic_rnn(lstm1, fc6_out, initial_state=state1, swap_memory=swap_memory)
        if log:
            lstmVars = [var for var in tf.trainable_variables() if 'lstm1' in var.name]
            for var in lstmVars:
                variable_summaries(var, var.name[:-2])

    with tf.variable_scope('lstm2'):
        lstm2 = tf.contrib.rnn.LSTMCell(lstm_size)

        # cell state
        if prev_state is not None: # if still one video (traker already running)
            state2 = tf.contrib.rnn.LSTMStateTuple(prev_state[2], prev_state[3])
        else:
            state2 = lstm2.zero_state(dtype=tf.float32)

        # unroll
        lstm2_inputs = tf.concat([fc6_out, lstm1_outputs], 2)
        lstm2_outputs, state2 = tf.nn.dynamic_rnn(lstm2, lstm2_inputs, initial_state=state2, swap_memory=swap_memory)

        if log:
            lstmVars = [var for var in tf.trainable_variables() if 'lstm2' in var.name]
            for var in lstmVars:
                variable_summaries(var, var.name[:-2])

        # [batch, time, feat] ->  [batchxtime, feat]
        flatten_out = tf.reshape(input, [-1, feat_len])  # flatten as [batchxtime, feat]

    # final dense layer.
    with tf.variable_scope('fc_output'):
        fc_output = dense_layer(flatten_out, 4, activation=None)  # [batchxtime, 4]
        fc_output = tf.reshape(fc_output, [-1, num_unrolls, feat_len])  #  [batch, time, 4]
    return fc_output, state1, state2