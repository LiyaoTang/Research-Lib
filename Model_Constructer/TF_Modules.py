#!/usr/bin/env python
# coding: utf-8
'''
module: some tf modeuls from classic networks
'''

import os
import numpy as np
import tensorflow as tf
from .TF_Ops import *

def alexnet_conv_layers(input, auxilary_input=None, prelu_initializer=tf.constant_initializer(0.25), spp_bins=[]):
    '''
    input: images, expected to be of [batch, width, height, channel]
    '''
    with tf.variable_scope('conv1'):
        if auxilary_input:
            conv1 = conv_layer(input, 96, 11, 4, padding='VALID', activation=None)
            conv1 = conv1 + auxilary_input  # join before activation
            conv1 = tf.nn.relu(conv1)
        else:
            conv1 = conv_layer(input, 96, 11, 4, padding='VALID')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0, name='norm1')

    with tf.variable_scope('conv1_skip'):
        conv1_skip = conv_layer(lrn1, 16, 1, activation=None)
        conv1_skip = prelu(conv1_skip, initializer=prelu_initializer)
        # each img flatten into 1-D
        # conv1_skip_flat = tf.transpose(conv1_skip, perm=[0, 3, 1, 2])
        # conv1_skip_flat = remove_axis(conv1_skip, [2, 3])

    with tf.variable_scope('conv2'):
        # 2-branch by num_groups=2
        conv2 = conv_layer(lrn1, 256, 5, num_groups=2, padding='SAME')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0, name='norm2')

    with tf.variable_scope('conv2_skip'):
        conv2_skip = conv_layer(lrn2, 32, 1, activation=None)
        conv2_skip = prelu(conv2_skip, initializer=prelu_initializer)
        # conv2_skip_flat = tf.transpose(conv2_skip, perm=[0, 3, 1, 2])
        # conv2_skip_flat = remove_axis(conv2_skip, [2, 3])

    with tf.variable_scope('conv3'):
        conv3 = conv_layer(lrn2, 384, 3, padding='SAME')

    with tf.variable_scope('conv4'):
        conv4 = conv_layer(conv3, 384, 3, num_groups=2, padding='SAME')

    with tf.variable_scope('conv5'):
        conv5 = conv_layer(conv4, 256, 3, num_groups=2, padding='SAME')
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        # pool5_flat = tf.transpose(pool5, perm=[0, 3, 1, 2])
        # pool5_flat = remove_axis(pool5, [2, 3])

    with tf.variable_scope('conv5_skip'):
        conv5_skip = conv_layer(conv5, 64, 1, activation=None)
        conv5_skip = prelu(conv5_skip)
        # conv5_skip_flat = tf.transpose(conv5_skip, perm=[0, 3, 1, 2])
        # conv5_skip_flat = remove_axis(conv5_skip, [2, 3])

    with tf.variable_scope('big_concat'):
        # Concat all skip layers.
        def flatten(feat_map):
            feat_map = tf.transpose(feat_map, perm=[0, 3, 1, 2])
            feat_map = remove_axis(feat_map, [2, 3])
            return feat_map
        flatten_func = lambda x: spatial_pyramid_pooling(x, spp_bins) if spp_bins else lambda x: flatten(x)
        feat = [conv5_skip, conv2_skip, conv5_skip, pool5]
        feat = [flatten_func(x) for x in feat]
        feat_concat = tf.concat(feat, 1)

    return feat_concat

def re3_lstm_tracker(input, num_unrolls, lstm_size=512, prev_state=None):
    '''
    input: object features in time sequence, expected to be [batch, time, feat]
    '''
    swap_memory = num_unrolls > 1
    with tf.variable_scope('lstm1'):
        lstm1 = tf.keras.layers.LSTMCell(lstm_size, reuse=reuse)

        # cell state
        if prev_state is not None: # if still one video (traker already running)
            state1 = tf.contrib.rnn.LSTMStateTuple(prev_state[0], prev_state[1])
        else:
            state1 = lstm1.zero_state(batch_size, dtype=tf.float32)

        # unroll
        lstm1_outputs, state1 = tf.nn.dynamic_rnn(lstm1, fc6_reshape, initial_state=state1, swap_memory=swap_memory)
        if train: # log
            lstmVars = [var for var in tf.trainable_variables() if 'lstm1' in var.name]
            for var in lstmVars:
                tf_util.variable_summaries(var, var.name[:-2])

    with tf.variable_scope('lstm2'):
        # lstm2 = CaffeLSTMCell(lstm_size, initializer=initializer)
        lstm2 = tf.contrib.rnn.LSTMCell(lstm_size, use_peepholes=True, initializer=initializer, reuse=reuse)

        # cell state
        if prev_state is not None: # if still one video (traker already running)
            state2 = tf.contrib.rnn.LSTMStateTuple(prev_state[2], prev_state[3])
        else:
            state2 = lstm2.zero_state(batch_size, dtype=tf.float32)

        # unroll
        lstm2_inputs = tf.concat([fc6_reshape, lstm1_outputs], 2)
        lstm2_outputs, state2 = tf.nn.dynamic_rnn(lstm2, lstm2_inputs, initial_state=state2, swap_memory=swap_memory)
        if train: # log
            lstmVars = [var for var in tf.trainable_variables() if 'lstm2' in var.name]
            for var in lstmVars:
                tf_util.variable_summaries(var, var.name[:-2])

        # [B,T,C] ->  [BxT,C]
        outputs_reshape = tf_util.remove_axis(lstm2_outputs, 1)

    # Final FC layer.
    with tf.variable_scope('fc_output'):
        fc_output_out = tf_util.fc_layer(outputs_reshape, 4, activation=None) # [BxT,4]
