#!/usr/bin/env python
# coding: utf-8
'''
module: some self-constructed tf ops
'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops

# reshaping
def remove_axis_get_shape(curr_shape, axis):
    assert axis > 0, 'Axis must be greater than 0'
    axis_shape = curr_shape.pop(axis)
    curr_shape[axis - 1] *= axis_shape
    return curr_shape

def remove_axis(input, axis):
    tensor_shape = tf.shape(input)
    curr_shape = input.get_shape().as_list()
    curr_shape = [ss if ss is not None else tensor_shape[ii] for ii,ss in enumerate(curr_shape)]
    if type(axis) == int:
        new_shape = remove_axis_get_shape(curr_shape, axis)
    else:
        for ax in sorted(axis, reverse=True):
            new_shape = remove_axis_get_shape(curr_shape, ax)
    return tf.reshape(input, tf.stack(new_shape))

# activation
def leaky_relu(input, slope=0.01, name='lrelu'):
    with tf.variable_scope(name):
        return tf.nn.relu(input) - slope * tf.nn.relu(-input)

def prelu(input, weights, name='prelu'):
    with tf.variable_scope(name):
        return tf.nn.relu(input) - weights * tf.nn.relu(-input)

# pooling
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

# operation
def conv(input, kernel, biases, stride_w, stride_h, padding, num_groups=1):
    # Creates convolutional layers supporting the "group" parameter
    '''
    From https://github.com/ethereon/caffe-tensorflow
    '''
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, stride_h, stride_w, 1], padding=padding)
    if num_groups == 1:
        conv = convolve(input, kernel)
    else:
        #group means we split the input  into 'num_groups' groups along the third dimension
        input_groups = tf.split(input, num_groups, 3)
        kernel_groups = tf.split(kernel, num_groups, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return tf.nn.bias_add(conv, biases)

def dense_layer(input, num_channels, activation=tf.nn.relu,
                weights_initializer=None, bias_initializer=None, return_vars=False, summary=True):
    if weights_initializer is None:
        # TF 2.0: tf.initializers.GlorotUniform()
        weights_initializer = tf.contrib.layers.xavier_initializer()
    if bias_initializer is None:
        bias_initializer = tf.zeros_initializer()
    input_shape = input.get_shape().as_list()
    if len(input_shape) > 2:
        input = tf.reshape(input, [-1, np.prod(input_shape[1:])])
        input_shape = input.get_shape().as_list()
    input_channels = input.get_shape().as_list()[1]
    W_dense = get_variable('W_dense', [input_channels, num_channels], initializer=weights_initializer, summary=summary)
    b_dense = get_variable('b_dense', [num_channels], initializer=bias_initializer, summary=summary)
    dense_out = tf.matmul(input, W_dense) + b_dense
    if activation is not None:
        dense_out = activation(dense_out)
    if return_vars:
        return dense_out, W_dense, b_dense
    else:
        return dense_out

def conv_layer(input, num_filters, filter_size, stride=1, num_groups=1, padding='VALID', scope=None,
        activation=tf.nn.relu, weights_initializer=None, bias_initializer=None, return_vars=False, summary=True):
    if type(filter_size) == int:
        filter_width = filter_size
        filter_height = filter_size
    elif type(filter_size) == tuple:
        filter_width, filter_height = filter_size
    else:
        raise Exception('filter_size is not int or tuple')
    if type(stride) == int:
        stride_width = stride
        stride_height = stride
    elif type(stride) == tuple:
        stride_width, stride_height = stride
    else:
        raise Exception('stride is not int or tuple')
    if weights_initializer is None:
        weights_initializer = tf.contrib.layers.xavier_initializer()
    if bias_initializer is None:
        bias_initializer = tf.zeros_initializer()
    shape = [filter_width, filter_height, input.get_shape().as_list()[3] / num_groups, num_filters]
    with cond_scope(scope):
        W_conv = get_variable('W_conv', shape, initializer=weights_initializer, summary=summary)
        b_conv = get_variable('b_conv', [num_filters], initializer=bias_initializer, summary=summary)
        if summary:
            conv_variable_summaries(W_conv)
        conv_out = conv(input, W_conv, b_conv, stride_width, stride_height, padding, num_groups)
        if activation is not None:
            conv_out = activation(conv_out)
        if return_vars:
            return conv_out, W_conv, b_conv
        else:
            return conv_out

# save & restore
def restore(session, save_file, raise_if_not_found=False):
    if not os.path.exists(save_file) and raise_if_not_found:
        raise Exception('File %s not found' % save_file)
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()  # dict {op name : shape in list}
    var_list = sorted([(var.name, var.name.split(':')[0], var) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes]) # [(tensor name, op name, tensor)...]
    restore_vars = []
    restored_var_names = set()
    # restored_var_new_shape = []
    print('Restoring:')
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for var_name, op_name, cur_var in var_list:
            if 'global_step' in var_name:
                restored_var_names.add(op_name)
                continue
            var_shape = cur_var.get_shape().as_list()
            if var_shape == saved_shapes[op_name]:
                restore_vars.append(cur_var)
                print(str(op_name) + ' -> \t' + str(var_shape) + ' = ' +
                      str(int(np.prod(var_shape) * 4 / 10**6)) + 'MB')
                restored_var_names.add(op_name)
            else:
                print('Shape mismatch for var', op_name, 'expected', var_shape,
                      'got', saved_shapes[op_name])
                # restored_var_new_shape.append((saved_var_name, cur_var, reader.get_tensor(saved_var_name)))
                # print('bad things')
    ignored_var_names = sorted(list(set(saved_shapes.keys()) - restored_var_names))
    print('\n')
    if len(ignored_var_names) == 0:
        print('Restored all variables')
    else:
        print('Did not restore:\n' + '\n\t'.join(ignored_var_names))

    if len(restore_vars) > 0:
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    '''
    if len(restored_var_new_shape) > 0:
        print('trying to restore misshapen variables')
        assign_ops = []
        for name, kk, vv in restored_var_new_shape:
            copy_sizes = np.minimum(kk.get_shape().as_list(), vv.shape)
            slices = [slice(0,cs) for cs in copy_sizes]
            print('copy shape', name, kk.get_shape().as_list(), '->', copy_sizes.tolist())
            new_arr = session.run(kk)
            new_arr[slices] = vv[slices]
            assign_ops.append(tf.assign(kk, new_arr))
        session.run(assign_ops)
        print('Copying unmatched weights done')
    '''
    print('Restored %s' % save_file)
    try:
        start_iter = int(save_file.split('-')[-1]) # get global_step
    except ValueError:
        print('Could not parse start iter, assuming 0')
        start_iter = 0
    return start_iter


def restore_from_dir(sess, folder_path, raise_if_not_found=False):
    start_iter = 0
    ckpt = tf.train.get_checkpoint_state(folder_path) # get all checkpoints into "CheckpointState" object
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring')
        start_iter = restore(sess, ckpt.model_checkpoint_path) # use the latest one
    else:
        if raise_if_not_found:
            raise Exception('No checkpoint to restore in %s' % folder_path)
        else:
            print('No checkpoint to restore in %s' % folder_path)
    return start_iter

# session
def Session():
    return tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))

# summary
def kernel_to_image(data, padsize=1, padval=0):
    # Turns a convolutional kernel into an image of nicely tiled filters.
    # Useful for viewing purposes.
    if len(data.get_shape().as_list()) > 4:
        data = tf.squeeze(data)
    data = tf.transpose(data, (3,0,1,2))
    dataShape = tuple(data.get_shape().as_list())
    min = tf.reduce_min(tf.reshape(data, (dataShape[0], -1)), reduction_indices=1)
    data = tf.transpose((tf.transpose(data, (1,2,3,0)) - min), (3,0,1,2))
    max = tf.reduce_max(tf.reshape(data, (dataShape[0], -1)), reduction_indices=1)
    data = tf.transpose((tf.transpose(data, (1,2,3,0)) / max), (3,0,1,2))

    n = int(np.ceil(np.sqrt(dataShape[0])))
    ndim = data.get_shape().ndims
    padding = ((0, n ** 2 - dataShape[0]), (0, padsize),
            (0, padsize)) + ((0, 0),) * (ndim - 3)
    data = tf.pad(data, padding, mode='constant')
    # tile the filters into an image
    dataShape = tuple(data.get_shape().as_list())
    data = tf.transpose(tf.reshape(data, ((n, n) + dataShape[1:])), ((0, 2, 1, 3)
            + tuple(range(4, ndim + 1))))
    dataShape = tuple(data.get_shape().as_list())
    data = tf.reshape(data, ((n * dataShape[1], n * dataShape[3]) + dataShape[4:]))
    return tf.image.convert_image_dtype(data, dtype=tf.uint8)

class empty_scope():
     def __init__(self):
         pass
     def __enter__(self):
         pass
     def __exit__(self, type, value, traceback):
         pass

def cond_scope(scope):
    return empty_scope() if scope is None else tf.variable_scope(scope)

def variable_summaries(var, scope=''):
    # Some useful stats for variables.
    if len(scope) > 0:
        scope = '/' + scope
    with tf.name_scope('summaries' + scope):
        mean = tf.reduce_mean(var)
        with tf.device('/cpu:0'):
            tf.summary.scalar('mean', mean)
            #tf.summary.histogram('histogram', var)

def conv_variable_summaries(var, scope=''):
    # Useful stats for variables and the kernel images.
    variable_summaries(var, scope)
    if len(scope) > 0:
        scope = '/' + scope
    with tf.name_scope('conv_summaries' + scope):
        varShape = var.get_shape().as_list()
        if not(varShape[0] == 1 and varShape[1] == 1):
            if varShape[2] < 3:
                var = tf.tile(var, [1,1,3,1])
                varShape = var.get_shape().as_list()
            summary_image = tf.expand_dims(
                    kernel_to_image(tf.slice(
                        var, [0,0,0,0], [varShape[0], varShape[1], 3, varShape[3]])),
                    0)
            with tf.device('/cpu:0'):
                tf.summary.image('filters', summary_image)

# variable
def get_variable(name, shape, dtype=tf.float32, initializer=None, summary=True):
    var = tf.get_variable(name, shape, dtype=dtype, initializer=initializer)
    if summary:
        variable_summaries(var, name)
    return var
