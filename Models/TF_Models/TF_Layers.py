#!/usr/bin/env python
# coding: utf-8
"""
module: some self-constructed tf layers as classes, consistent with TF 2.0
"""

import numpy as np
import tensorflow as tf
from . import TF_Ops as tfops

class Conv_Layer(object):
    """
    callable layer, with options stored as attributes
    """
    def __init__(self, out_channels, kernel_size, stride=1, num_groups=1, padding='VALID', scope=None,
                 activation=tf.nn.relu, weights_initializer=None, bias_initializer=None, return_vars=False,
                 kernel_name='W_conv', bias_name='b_conv'):
        if type(kernel_size) == int:
            kernel_width, self.kernel_height = kernel_size, kernel_size
        elif type(kernel_size) == tuple:
            kernel_width, self.kernel_height = kernel_size
        else:
            raise Exception('kernel_size is not int or tuple')

        if type(stride) == int:
            self.stride_width, self.stride_height = stride, stride
        elif type(stride) == tuple:
            self.stride_width, self.stride_height = stride
        else:
            raise Exception('stride is not int or tuple')
        self.padding = padding
        self.num_groups = num_groups
        self.activation = activation
        self.tf_scope = scope

        if weights_initializer is None:
            # equivalent to TF 1.0+ tf.contrib.layers.xavier_initializer()
            weights_initializer = tf.glorot_uniform_initializer()
        if bias_initializer is None:
            bias_initializer = tf.zeros_initializer()    

        shape = [kernel_width, kernel_height, input.get_shape().as_list()[3] / num_groups, out_channels]
        with tfops.var_scope(self.tf_scope):
            self.kernel = tf.get_variable(kernel_name, shape, dtype=tf.float32, initializer=weights_initializer)
            self.bias = tf.get_variable(bias_name, shape, dtype=tf.float32, initializer=bias_initializer)

    def __call__(self, input):
        with tfops.var_scope(self.tf_scope):
            conv_out = tfops.conv(input, self.kernel, self.bias, self.stride_width, self.stride_height, self.padding, self.num_groups)
            if self.activation is not None:
                conv_out = self.activation(conv_out)
        return conv_out