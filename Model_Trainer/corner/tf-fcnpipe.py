#!/usr/bin/env python
# coding: utf-8
"""
construct CNN given setting
"""

import sys
root_dir = '../../'
sys.path.append(root_dir)

import os
import time
import dill
import random
import psutil
import matplotlib
# matplotlib.use('agg') # so that plt works in command line

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import Data_Feeder as feeder
import Model_Analyzer as analyzer
import Model_Constructer as constructer

from collections import defaultdict

""" parsing args """

model_group_name = 'tf-fcnpipe'
parser = constructer.Default_Optparser(model_group_name, root_dir, class_num=2)
parser.add_option('--conv_struct', dest='conv_struct', default='3-16|3')
parser.add_option('--focus_size', dest='focus_size', default='15-20')
parser.add_option('--resolution', dest='resolution', default=0.5, type=float)

parser.add_option('--use_tfdeprecate', dest='use_tfdeprecate', action='store_true', default=False)
parser.add_option('--use_tfdataset', dest='use_tfdataset', action='store_true', default=False)
parser.add_option('--select_cols', dest='select_cols', default='0-1-2-3-4-5-6-7-8-9')
parser.add_option('--all_radar', dest='all_radar', action='store_true', default=False)
parser.add_option('--split', dest='split', default='4-1')

parser.add_option('--gpu', default='', dest='gpu')
parser.add_option('--gpu_max_mem', type='float', default=0.99, dest='gpu_max_mem')

(options, args) = parser.parse_args()

model_name = options.model_name
save_path = options.save_path
analysis_dir = options.analysis_dir
log_dir = options.log_dir
summary_dir = options.summary_dir
groupby_model = options.groupby_model

path_train_set = options.path_train_set
path_val_set = options.path_val_set
path_test_set = options.path_test_set
class_num = options.class_num
class_name = options.class_name
use_onehot = options.use_onehot
cv_fold = options.cv_fold
all_radar = options.all_radar

# parse conv_struct: e.g. 3-16;5-8;1-32 | 3-8;1-16 | 1
# => concat[ 3x3 out_channel=16, 5x5 out_channel=8, 1x1 out_channel=32]
# => followed by inception concat [3x3 out_channel=8, 1x1 out_channel=16] and so on ...
# => output with a 1x1 conv
# note: size must be specified for the kernel at output (logits) layer
select_cols = [int(num) for num in options.select_cols.split('-')]  # parse select_cols: str to int
assert any((0 < n and n < 10) for n in select_cols)  # chk valid
focus_size = [int(i) for i in options.focus_size.split('-')]
resolution = options.resolution

# naming
if not model_name:
    model_name = '%s_%s_%s_' % (model_group_name, options.conv_struct, options.select_cols)
    if options.norm_type: model_name += '%s_' % options.norm_type
    if options.batchnorm: model_name += 'bn_'
    if options.regularizer_type: model_name += '%s%.3f_' % (options.regularizer_type, options.regularizer_scale)
    if options.weighted_loss: model_name += options.weighted_loss + '_'
    if options.loss_type != 'xen': model_name += options.loss_type + '_'
    if options.rand_seed is not None: model_name += 'r%d_' % options.rand_seed
    if options.add_noise: model_name += 'n_'
    model_name += 'e%db%d' % (options.epoch, options.batch)
print('\n', model_name)
# mkdir if not existed
if groupby_model:
    analysis_dir = os.path.join(analysis_dir, model_name) + '/'  # group analysis under folder with same name
os.makedirs(analysis_dir, exist_ok=True)
if options.save_path:
    os.makedirs(options.save_path, exist_ok=True)

# change std out if log dir originally given
if options.log_dir:
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, model_name)
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    print(model_name)
else:
    log_file = ''

if options.rand_seed is not None:
    tf.set_random_seed(options.rand_seed) # graph-level seed
    np.random.seed(options.rand_seed)

# re expr for selecting lines
if all_radar:
    line_re = '.*'
else:
    line_re = '\t (?!3).*'

# reset tf graph environment before everything
tf.reset_default_graph()

""" constructing dataset """

if options.use_tfdataset:  # tf dataset
    raise NotImplementedError
    if options.use_tfdeprecate:
        train_set = feeder.Corner_Radar_Points_Gen_Feeder(path_train_set, class_num=class_num,
                                                        use_onehot=use_onehot, line_re=line_re, select_cols=select_cols)
        if cv_fold <= 0:
            val_set = feeder.Corner_Radar_Points_Gen_Feeder(path_val_set, class_num=class_num,
                                                            use_onehot=use_onehot, line_re=line_re, select_cols=select_cols)
            test_set = feeder.Corner_Radar_Points_Gen_Feeder(path_test_set, class_num=class_num,
                                                            use_onehot=use_onehot, line_re=line_re, select_cols=select_cols)

        # get all data points
        train_input, train_label = train_set.get_all_data()
        train_input = np.vstack(train_input)
        train_label = np.concatenate(train_label)

        if add_noise:
            train_input = np.concatenate([train_input, np.transpose([np.random.rand(train_input.shape[0])])], axis=-1)
            train_set.feature_names.append('noise')
    else:
        raise NotImplementedError
        # general iterator => able to switch dataset
        general_itr = train_set.get_general_iterator()

        train_init_op = general_itr.make_initializer(train_set.config_dataset(epoch=1, batch=batch)) # controlling epoch at outter loop
        val_init_op = general_itr.make_initializer(val_set.config_dataset(shuffle=False))
        test_init_op = general_itr.make_initializer(test_set.config_dataset(shuffle=False))
else:  # general feeder
    dataset = feeder.Corner_Radar_Boxcenter_Gen_Feeder(data_dir=path_train_set,
                                                       select_cols=select_cols,
                                                       focus_size=focus_size,
                                                       resolution=resolution,
                                                       class_num=class_num,
                                                       split=options.split,
                                                       weight_type=options.weighted_loss,
                                                       norm_type=options.norm_type,
                                                       header=True)
process = psutil.Process(os.getpid())
print('mem usage after data loaded:', process.memory_info().rss / 1024 / 1024, 'MB')

""" constructing model """

with tf.variable_scope('input'):
    if options.use_tfdataset:
        input_img, img_mask = general_itr.get_next()
    else:
        tf_input = tf.placeholder(tf.float32, shape=[None] + dataset.input_shape, name='X')
        tf_label = tf.placeholder(tf.int32, shape=[None] + dataset.label_shape, name='Y')
        tf_phase = tf.placeholder(tf.string, name='phase')

options.model_config['norm_params'] = dict(zip(['train', 'val', 'test'], dataset.norm_params))
options.model_config['weight'] = defaultdict(lambda: [0, 0], zip(['train', 'val', 'test'], dataset.weight))

print('model config:')
for k in options.model_config:
    print(k, options.model_config[k])

pipe = constructer.FCN_Pipe_Constructer(conv_struct=options.conv_struct,
                                        class_num=class_num,
                                        tf_input=tf_input,
                                        tf_label=tf_label,
                                        tf_phase = tf_phase,
                                        config_dict=options.model_config)
process = psutil.Process(os.getpid())
print('mem usage after model constructed:', process.memory_info().rss / 1024 / 1024, 'MB')

""" training & monitoring """

# saver to save model
saver = tf.train.Saver(max_to_keep=None)

# sess config
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

# initialization
sess = tf.InteractiveSession(config=sess_config)
sess.run(tf.global_variables_initializer())

# visualize graph into tensorboard
if options.record_summary:
    train_writer = tf.summary.FileWriter(os.path.join(summary_dir, model_name), sess.graph)
sess.graph.finalize()

if options.use_tfdataset:
    val_metric = analyzer.TF_Metric_Record(tf_loss=val_xen, tf_pred=prob_out, tf_label=img_mask, tf_itr_init_op=val_init_op,
                                        class_num=class_num, sess=sess, tf_feed_dict={is_training:False})
    for epoch_num in range(epoch):
        print('epoch:',epoch_num)

        try:
            while True:
                train_step.run(feed_dict={is_training: True})
        # epoch finished
        except tf.errors.OutOfRangeError:
            pass

        # save model
        saver.save(sess, save_path, epoch_num)

        # tensor board
        if options.record_summary:
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
else:
    train_metric = analyzer.General_Classif_Record(class_num=options.class_num, class_name=options.class_name, title='\ntrain eval')
    val_metric = analyzer.General_Classif_Record(class_num=options.class_num, class_name=options.class_name, title='\nval eval')
    
    dataset.switch_split(0)
    for epoch_num in range(options.epoch):
        print('\nepoch =====>>>>>>', epoch_num)
        start = time.time()
        for img_input, img_mask in dataset.iterate_data():
            pipe.train_step.run(feed_dict={pipe.tf_input: [img_input], pipe.tf_label: [img_mask], pipe.tf_phase: 'train'})
        print('train:', time.time() - start)

        # save model
        if options.save_path:
            saver.save(sess, os.path.join(options.save_path, model_name), epoch_num)

        # tensor board
        if options.record_summary:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary = sess.run(pipe.merged_summary,
                               feed_dict={pipe.tf_input: [img_input], pipe.tf_label: [img_mask], pipe.tf_phase: 'train-eval'},
                               options=run_options, run_metadata=run_metadata)

            train_writer.add_run_metadata(run_metadata, 'epoch_%03d' % epoch_num)
            train_writer.add_summary(summary, epoch_num)

        cur_model_cfg = {'dataset_name': 'corner', 'name': model_name, 'epoch': epoch_num, 'root_dir': root_dir}

        # snap shot on train set
        train_metric.evaluate_model(model_func=lambda x, y: sess.run([pipe.prob_out, pipe.loss['train']],
                                                                     feed_dict={pipe.tf_phase: 'train-eval',
                                                                                pipe.tf_input: [x],
                                                                                pipe.tf_label: [y]})[0],
                                    input_label_itr=dataset.iterate_data)
        train_metric.print_result()
        train_metric.clear_cur_epoch()

        # snap shot on val set
        dataset.switch_split(1)
        val_metric.evaluate_model(model_func=lambda x, y: sess.run([pipe.prob_out, pipe.loss['val']],
                                                                   feed_dict={pipe.tf_phase: 'val',
                                                                              pipe.tf_input: [x],
                                                                              pipe.tf_label: [y]})[0],
                                  input_label_itr=dataset.iterate_data)
        val_metric.print_result()
        val_metric.clear_cur_epoch()
        dataset.switch_split(0, shuffle=True)  # switch back to train set

print("finish")
if options.log_dir:
    log_file.close()
sys.exit()