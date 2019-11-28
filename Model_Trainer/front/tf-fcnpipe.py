#!/usr/bin/env python
# coding: utf-8
"""
construct CNN given setting
"""

import sys
root_dir = '../../'
sys.path.append(root_dir + 'Data_Feeder/')
sys.path.append(root_dir + 'Metric_Recorder/')

import os
import psutil
import matplotlib
# matplotlib.use('agg') # so that plt works in command line
import matplotlib.pyplot as plt
import tensorflow as tf
import Data_Feeder as feeder
import Model_Analyzer as analyzer
from optparse import OptionParser

""" parsing args """

parser = OptionParser()
parser.add_option('--name', dest='model_name')
parser.add_option('--save', dest='save_path', default='./Model/')
parser.add_option('--analysis_dir', dest='analysis_dir', default='./Model_Analysis/')
parser.add_option('--log_dir', dest='log_dir', default='./Log/')
parser.add_option('--record_summary', action='store_true', default=False, dest='record_summary')

# all dataset assumed able to be read by tf.train.Data
parser.add_option('--train', dest='path_train_set', default=root_dir + 'Data/front/train/')
parser.add_option('--val', dest='path_val_set', default=root_dir + 'Data/front/val/')
parser.add_option('--test', dest='path_test_set', default=root_dir + 'Data/front/test/')
parser.add_option('--meta_postfix', dest='meta_postfix', default='_meta')
parser.add_option('--use_cv', dest='use_cv', action='store_true', default=False)
parser.add_option('--height', dest='height', type=int, default=1024)
parser.add_option('--width', dest='width', type=int, default=448)
parser.add_option('--band', dest='band', type=int, default=8)
parser.add_option('--class_num', dest='class_num', type=int, default=3)

parser.add_option('--conv_struct', default='3-16|3', dest='conv_struct')

parser.add_option('-e', '--epoch', dest='epoch', default=20, type='int')
parser.add_option('-b', '--batch', dest='batch', default=1, type='int')
parser.add_option('--norm', dest='norm_type', default='')
parser.add_option('--learning_rate', dest='learning_rate', default=1e-5, type='float')
parser.add_option('--reg_type', dest='regularizer_type', default='')
parser.add_option('--reg_scale', dest='regularizer_scale', default=0.1, type='float')
parser.add_option('--use_batch_norm', dest='use_batch_norm', action='store_true', default=False)
parser.add_option('--weighted_loss', dest='weighted_loss', default='')

parser.add_option("--gpu", default="", dest="gpu")
parser.add_option("--gpu_max_mem", type="float", default=0.99, dest="gpu_max_mem")
(options, args) = parser.parse_args()

model_name = options.model_name
save_path = options.save_path.rstrip('/') + '/'
analysis_dir = options.analysis_dir.rstrip('/') + '/'
log_dir = options.log_dir.rstrip('/') + '/'
summary_dir = options.summary_dir.rstrip('/') + '/'

path_train_set = options.path_train_set
path_val_set = options.path_val_set
path_test_set = options.path_test_set
meta_postfix = options.meta_postfix
use_cv = options.use_cv
height = options.height
width = options.width
band = options.band
class_num = options.class_num

conv_struct = options.conv_struct

epoch = options.epoch
batch = options.batch
learning_rate = options.learning_rate
norm_type = options.norm_type
regularizer_type = options.regularizer_type
regularizer_scale = options.regularizer_scale
use_batch_norm = options.use_batch_norm
weighted_loss = options.weighted_loss
record_summary = options.record_summary

gpu = options.gpu
gpu_max_mem = options.gpu_max_mem

# assert options to be valid
assert norm_type in ('m', 's')  # m: mean-centered, s: standardized
if norm_type == 'm': norm_type = 'mean'
else: norm_type = 'std'

assert regularizer_type in ('', 'L1', 'L2') # e.g. 'L2-0.1' => use L2 norm with scale of 0.1
assert weighted_loss in ('', 'bal') # b: balanced

# parse conv_struct: e.g. 3-16;5-8;1-32 | 3-8;1-16 | 1
# => concat[ 3x3 out_channel=16, 5x5 out_channel=8, 1x1 out_channel=32]
# => followed by inception concat [3x3 out_channel=8, 1x1 out_channel=16] and so on ...
# => output with a 1x1 conv
# note: size must be specified for the kernel at output (logits) layer 
if not conv_struct:
    print("must provide structure for conv")
    sys.exit()
else:
    conv_struct = [[[int(x) for x in config.split('-')] for config in layer.split(';')] for layer in conv_struct.split('|')]
    assert len(conv_struct[-1]) == 1 and len(conv_struct[-1][0]) == 1  # assert the kernel at output layer is given
    conv_struct[-1][0].append(class_num) # output vector with dimension of class_num
    print("conv_struct = ", conv_struct)

# naming
if not model_name:
    model_name = 'FCN-pipe_'
    model_name += options.conv_struct + '_'
    model_name += norm_type[0] + '_'
    if use_batch_norm: model_name += 'bn_'
    if regularizer_type: model_name += regularizer_type + '-' + str(regularizer_scale) + '_'
    if weighted_loss: model_name += weighted_loss +'_'
    model_name += 'e' + str(epoch) + '_'
    model_name += 'b' + str(batch)

# mkdir if not existed, deafult to folder with same name
save_path = save_path + model_name + '/'
analysis_dir = analysis_dir + model_name + '/'
if not record_summary:
    summary_dir = summary_dir + model_name + '/'
os.makedirs(save_path, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)

# change std out if log dir originally given
if options.log_dir:
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = log_dir + model_name
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file

# reset tf graph environment before everything
tf.reset_default_graph()

""" constructing dataset """

train_set = feeder.Img_from_Record_Feeder(path_train_set, meta_data_file=path_train_set + meta_postfix,norm_type=norm_type)
val_set = feeder.Img_from_Record_Feeder(path_val_set, meta_data_file=path_val_set + meta_postfix, norm_type=norm_type)
test_set = feeder.Img_from_Record_Feeder(path_test_set, meta_data_file=path_test_set + meta_postfix, norm_type=norm_type)

# general iterator => able to switch dataset
general_itr = train_set.get_general_iterator()

train_init_op = general_itr.make_initializer(train_set.config_dataset(epoch=1, batch=batch)) # controlling epoch at outter loop
val_init_op = general_itr.make_initializer(val_set.config_dataset(shuffle=False))
test_init_op = general_itr.make_initializer(test_set.config_dataset(shuffle=False))

""" constructing model """

# general layer configuration
if regularizer_type == 'L2':
    regularizer = tf.contrib.layers.l2_regularizer(scale=regularizer_scale)
elif regularizer_type == 'L1':
    regularizer = tf.contrib.layers.l1_regularizer(scale=regularizer_scale)
else:
    regularizer = None

with tf.variable_scope('input'):
    input_img, img_mask = general_itr.get_next()
    # input_img   = tf.placeholder(tf.float32, shape=[None, height, width, band], name='input_img')
    # img_mask    = tf.placeholder(tf.float32, shape=[None, height, width, class_num], name='img_mask')
    is_training = tf.placeholder(tf.bool, name='is_training')  # batch norm
    
    train_weight = tf.constant(1 - train_set.class_ratio)
    val_weight = tf.constant(1 - val_set.class_ratio)
    test_weight = tf.constant(1 - test_set.class_ratio)



""" training & monitoring """

# saver to save model
saver = tf.train.Saver()

# sess config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# initialization
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

# visualize graph into tensorboard
if record_summary:
    train_writer = tf.summary.FileWriter(summary_dir + model_name, sess.graph)

val_metric = analyzer.Metric_Record(tf_loss=val_xen, tf_pred=prob_out, tf_label=img_mask, tf_itr_init_op=val_init_op,
                                    class_num=class_num, sess=sess, tf_feed_dict={is_training:False})
for epoch_num in range(epoch):
    sess.run(train_init_op)

    try:
        while True:
            train_step.run(feed_dict={is_training: True})
    # epoch finished
    except tf.errors.OutOfRangeError:
        pass

    # save model
    saver.save(sess, save_path, epoch_num)

    # tensor board
    if record_summary:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary = sess.run(merged_summary, feed_dict={is_training: False}, options=run_options, run_metadata=run_metadata)

        train_writer.add_run_metadata(run_metadata, 'epoch_%03d' % (epoch_num+1))
        train_writer.add_summary(summary, epoch_num + 1)
            
    # snap shot on CV set
    val_metric.evaluate_model()
    val_metric.print_result()
    if epoch_num < epoch:
        val_metric.clear_cur_epoch()

val_metric.plot_cur_epoch_curve(save_path=analysis_dir, model_name=model_name)
val_metric.plot_evolve_curve(save_path=analysis_dir, model_name=model_name)

print("finish")
if options.log_dir:
    log_file.close()
sys.exit()
