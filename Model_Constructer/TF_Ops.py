#!/usr/bin/env python
# coding: utf-8
'''
module: some self-constructed tf ops
'''

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops

def unpooling(inputs, before_pool, padding, ksize=[1,2,2,1], strides=[1,2,2,1], name=None, data_format="NHWC"):
    '''
    apply unpooling given the corresponding pooling op
    by using the gradient of pooling op
    '''
    raise PermissionError('not permitted to use: not tested yet')
    unpool = gen_nn_ops._max_pool_grad(orig_input=before_pool,
                                       orig_output=inputs,
                                       grad=inputs,
                                       ksize=ksize,
                                       strides=strides,
                                       padding=padding,
                                       data_format=data_format,
                                       name=name)
    return unpool

def max_unpooling(inputs, factor, scope='max_unpooling'):
    '''
    N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    inputs: A Tensor of shape [batch, d0, d1...dn, channel]
    return: A Tensor of shape [batch, facor*d0, facor*d1...facor*dn, channel]
    '''
    with tf.name_scope(scope) as sc:
        shape = inputs.get_shape().as_list()
        dim = len(shape[1:-1])
        out = (tf.reshape(inputs, [-1] + shape[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out] + (factor - 1) * [tf.zeros_like(out)], i)
        out_size = [-1] + [s * factor for s in shape[1:-1]] + [shape[-1]]
        out = tf.reshape(out, out_size, name=sc)
    return out

def average_unpooling(inputs, factor, scope='average_unpooling'):
    '''
    N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    inputs: A Tensor of shape [batch, d0, d1...dn, channel]
    return: A Tensor of shape [batch, facor*d0, facor*d1...facor*dn, channel]
    '''
    with tf.name_scope(scope) as sc:
        shape = inputs.get_shape().as_list()
        dim = len(shape[1:-1])
        out = (tf.reshape(inputs, [-1] + shape[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat(factor * [out], i)
        out_size = [-1] + [s * factor for s in shape[1:-1]] + [shape[-1]]
        out = tf.reshape(out, out_size, name=sc)
    return out