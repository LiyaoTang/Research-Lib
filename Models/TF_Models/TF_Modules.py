#!/usr/bin/env python
# coding: utf-8
'''
module: some tf modeuls from classic networks
'''

import os
import numpy as np
import tensorflow as tf
from . import TF_Ops as tfops


def fcn_pipe(input, conv_struct, use_batchnorm=False, is_training=None, scope='pipe'):
    '''
    parse conv_struct: e.g. 3-16;5-8;1-32 | 3-8;1-16 | 1
    => concat[ 3x3 out_channel=16, 5x5 out_channel=8, 1x1 out_channel=32]
    => followed by inception concat [3x3 out_channel=8, 1x1 out_channel=16] and so on ...
    => output with a 1x1 conv
    '''
    with tf.variable_scope(scope):
        net = input
        if len(conv_struct) > 1: # if any hidden layer
            for layer_cnt in range(len(conv_struct) - 1):
                layer_cfg = conv_struct[layer_cnt]

                with tf.variable_scope('incep_%d' % layer_cnt):
                    # kernel/bias initializer: default to xavier/zeros
                    if len(layer_cfg) > 1:
                        net = tf.concat([tfops.conv_layer(net, out_channels=cfg[1], filter_size=cfg[0], padding='SAME',
                                         scope='conv%d-%d' % (cfg[0], cfg[1]))
                                         for cfg in layer_cfg], axis=-1)
                    else:
                        cfg = layer_cfg[0]
                        net = tfops.conv_layer(net, out_channels=cfg[1], filter_size=cfg[0], padding='SAME',
                                               scope='conv%d-%d' % (cfg[0], cfg[1]))

                    # it seems BN after ReLU does generally better than BN before ReLU
                    if use_batchnorm:
                        bn_layer = tf.keras.layers.BatchNormalization(name='bn')
                        assert is_training is not None
                        net = bn_layer(net, training=is_training)
    return net

def alexnet_conv_layers(input, auxilary_input=None, prelu_initializer=tf.constant_initializer(0.25), fuse_type='flat'):
    '''
    input: images, expected to be of [batch, width, height, channel]
    '''
    def flatten(feat_map):
        feat_map = tf.transpose(feat_map, perm=[0, 3, 1, 2])
        feat_map = tfops.remove_axis(feat_map, [2, 3])
        return feat_map
    assert fuse_type in ['flat', 'spp', 'resize']

    with tf.variable_scope('conv1'):
        if auxilary_input is not None:
            conv1 = tfops.conv_layer(input, 96, filter_size=11, stride=4, padding='VALID', activation=None)
            conv1 = conv1 + auxilary_input  # join before activation
            conv1 = tf.nn.relu(conv1)
        else:
            conv1 = tfops.conv_layer(input, 96, filter_size=11, stride=4, padding='VALID')
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0, name='norm1')

    with tf.variable_scope('conv1_skip'):
        conv1_skip = tfops.conv_layer(lrn1, 16, filter_size=1, activation=None)
        conv1_skip = tfops.prelu(conv1_skip, initializer=prelu_initializer)
        if fuse_type == 'flat':  # each img flatten into 1-D
            conv1_skip_flat = flatten(conv1_skip)

    with tf.variable_scope('conv2'):
        # 2-branch by num_groups=2
        conv2 = tfops.conv_layer(lrn1, 256, filter_size=5, num_groups=2, padding='SAME')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0, name='norm2')

    with tf.variable_scope('conv2_skip'):
        conv2_skip = tfops.conv_layer(lrn2, 32, filter_size=1, activation=None)
        conv2_skip = tfops.prelu(conv2_skip, initializer=prelu_initializer)
        if fuse_type == 'flat':
            conv2_skip_flat = flatten(conv2_skip)

    with tf.variable_scope('conv3'):
        conv3 = tfops.conv_layer(lrn2, 384, filter_size=3, padding='SAME')

    with tf.variable_scope('conv4'):
        conv4 = tfops.conv_layer(conv3, 384, filter_size=3, num_groups=2, padding='SAME')

    with tf.variable_scope('conv5'):
        conv5 = tfops.conv_layer(conv4, 256, filter_size=3, num_groups=2, padding='SAME')
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        if fuse_type == 'flat':
            pool5_flat = flatten(pool5)

    with tf.variable_scope('conv5_skip'):
        conv5_skip = tfops.conv_layer(conv5, 64, filter_size=1, activation=None)
        conv5_skip = tfops.prelu(conv5_skip)
        if fuse_type == 'flat':
            conv5_skip_flat = flatten(conv5_skip)

    with tf.variable_scope('big_concat'):
        # concat all skip layers
        if fuse_type == 'flat':
            feat = [conv1_skip_flat, conv2_skip_flat, conv5_skip_flat, pool5_flat]
        elif fuse_type == 'spp':
            feat = [conv1_skip, conv2_skip, conv5_skip, pool5]
            spp_bin_list = [[27], [13], [13], [6]]  # pool size on pixel = [[4x2, 4x2, 4x2, 5x3]]
            for i, bins in enumerate(spp_bin_list):  # a spp for each layer
                feat[i] = tfops.spatial_pyramid_pooling(feat[i], bins)
        else:  # resize as image
            feat = [conv1_skip, conv2_skip, conv5_skip, pool5]
            size_list = [27, 13, 13, 6]
            for i, sz in enumerate(size_list):
                feat[i] = flatten(tf.image.resize(feat[i], (sz, sz)))
        feat_concat = tf.concat(feat, 1)

    return feat_concat

def re3_lstm_tracker(input, num_unrolls, batch_size, prev_state=None, lstm_size=512, rnn_type='lstm'):
    '''
    input: object features in time sequence, expected to be [batch, time, feat_t + feat_t-1], with time = num_unrolls
    prev_state: the initial state for RNN cell, set to placeholder to enable single-step inference
    TODO: migrate to TF 2.0: 
        contrib.rnn.LSTMCell -> keras.layers.LSTMCell
        contrib.rnn.LSTMStateTuple -> get_initial_tuple
        dynamic_rnn -> keras.layers.RNN
    '''
    assert rnn_type in ['lstm']
    with tf.variable_scope('lstm1'):
        lstm1 = tf.contrib.rnn.LSTMCell(lstm_size)

        # cell state
        if prev_state is not None: # if traker already running
            state1 = tf.contrib.rnn.LSTMStateTuple(prev_state[0], prev_state[1])
        else:
            state1 = lstm1.zero_state(batch_size, dtype=tf.float32)

        # unroll
        lstm1_outputs, state1 = tf.nn.dynamic_rnn(lstm1, input, initial_state=state1, swap_memory=True)

    with tf.variable_scope('lstm2'):
        lstm2 = tf.contrib.rnn.LSTMCell(lstm_size)

        # cell state
        if prev_state is not None: # if still one video (traker already running)
            state2 = tf.contrib.rnn.LSTMStateTuple(prev_state[2], prev_state[3])
        else:
            state2 = lstm2.zero_state(batch_size, dtype=tf.float32)

        # unroll
        lstm2_inputs = tf.concat([input, lstm1_outputs], -1)
        lstm2_outputs, state2 = tf.nn.dynamic_rnn(lstm2, lstm2_inputs, initial_state=state2, swap_memory=True)

        flatten_out = tf.reshape(lstm2_outputs, [-1, lstm2_outputs.get_shape().as_list()[-1]])  # flatten as [batch x time, feat]

    # final dense layer.
    with tf.variable_scope('fc_output'):
        fc_output = tfops.dense_layer(flatten_out, 4, activation=None, weight_name='W_fc', bias_name='b_fc')  # [batch x time, 4]
        fc_output = tf.reshape(fc_output, [-1, num_unrolls, 4])  #  [batch, time, 4]

    return fc_output, (state1, state2)