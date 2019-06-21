#!/usr/bin/env python
# coding: utf-8
'''
script: train re3 tracker
'''

import sys
root_dir = '../../'
sys.path.append(root_dir)

import os
import time
import random
import psutil
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import Data_Feeder as feeder
import Model_Analyzer as analyzer
import Model_Constructer as constructer
from constructer import tfops

DEBUG = True

''' parsing args '''


parser = constructer.ArgumentParser(description='training re3')
parser.add_argument('--class_num', type=int, dest='class_num')
parser.add_argument('--num_unrolls', default=2, type=int, dest='num_unrolls')
parser.add_argument('--img_size', default=None, type=int, dest='img_size')
parser.add_argument('--lstm_size', default=512, type=int, dest='lstm_size')

parser.add_argument('--unroll_type', default='dynamic', type=str, dest='unroll_type')
parser.add_argument('--bbox_encoding', default='mask', type=str, dest='bbox_encoding')
parser.add_argument('--restore', default=True, type=bool, dest='restore')

parser.add_argument('--buffer_size', default=5, type=int, dest='buffer_size')
parser.add_argument('--worker_num', default=1, type=int, dest='worker_num')
parser.add_argument('--epoch', default=20, type=int, dest='epoch')

parser.add_argument('--tf_dataset', default=True, type=bool, dest='tf_dataset')
parser.add_argument('--rand_seed', default=None, type=int, dest='rand_seed')
parser.add_argument('--log_dir', default='./Log', type=str, dest='log_dir')
parser.add_argument('--summary_dir', default='./Summary', type=str, dest='summary_dir')
parser.add_argument('--restore_dir', default='', type=str, dest='restore_dir')

args = parser.parse_args()

args.batch_size = int(max(64 / args.num_unrolls, 2))
args.channel_size = 4 if args.bbox_encoding == 'mask' else 3
assert args.img_size in [None, 227]
if args.rand_seed is not None:
    np.random.seed(args.rand_seed)
    tf.random.set_random_seed(args.rand_seed)
if not args.restore_dir:
    args.restore_dir = args.log_dir
    
# model name
tt = time.localtime()
time_str = ('%04d_%02d_%02d_%02d_%02d_%02d' % (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
args.model_name = 're3-%s_un%d_img%d_lstm%d_b%d_e%d' % \
                  (args.bbox_encoding, args.num_unrolls, args.img_size, args.lstm_size, args.batch_size, args.epoch)
args.model_name += '' if args.rand_seed is None else '_r%d' % args.rand_seed
args.model_name += time_str

# make dirs
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.summary_dir, exist_ok=True)
os.makedirs(args.restore_dir, exist_ok=True)


''' construct feeder '''


data_ref_dir = os.path.join(root_dir, 'Data/ILSVRC2015')
train_ref = os.path.join(data_ref_dir, 'train_label.npy')
train_feeder = feeder.Imagenet_VID_Feeder(train_ref, class_num=30, num_unrolls=args.num_unrolls)

if args.tf_dataset:
    raise NotImplementedError
    # para_train_feeder = feeder.Parallel_Feeder(train_feeder, batch_size=batch_size,
    #                                            buffer_size=args.buffer_size, worker_num=args.worker_num)
    # feeder_gen = para_train_feeder.iterate_batch()

else:
    tf_img = tf.placeholder(tf.uint8, shape=[args.batch_size, args.num_unrolls, args.img_size, args.img_size, args.channel_size])
    tf_label = tf.placeholder(tf.float32, shape=[args.batch_size, args.num_unrolls, 4])


''' construct model '''


tracker = constructer.Re3_Tracker(tf_img, tf_label,
                                  num_unrolls=args.num_unrolls,
                                  img_size=args.img_size,
                                  lstm_size=args.lstm_size,
                                  unroll_type=args.unroll_type,
                                  bbox_encoding=args.bbox_encoding)
train_step = tracker.get_train_step()


''' training '''


def train(tf_img):
    pass

def train_tfdataset(tf_img):
    raise NotImplementedError
    pass

train_func = train_tfdataset if args.tf_dataset else train

sess = constructer.tfops.Session()
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
saver = tf.train.Saver()
longSaver = tf.train.Saver()

# initialize/restore
global_step = 0
sess.run(tf.global_variables_initializer())
if args.restore:
    global_step = tfops.restore_from_dir(sess, args.restore_dir)

# create summary
conv_var_list = [v for v in tf.trainable_variables() if 'conv' in v.name and 'weight' in v.name and
                 (v.get_shape().as_list()[0] != 1 or v.get_shape().as_list()[1] != 1)]
for var in conv_var_list:
    tfops.conv_variable_summaries(var, scope=var.name.replace('/', '_')[:-2])
summary_with_images = tf.summary.merge_all()

# 
sess.graph.finalize()
summary_writer = tf.summary.FileWriter(os.path.join(args.summary_dir, args.model_name), sess.graph)
summary_full = tf.summary.merge_all()

# start training
start_time = time.time()
try:
    for ep_cnt in range(args.epoch):
        batch_cnt = 0
        for input_batch, label_batch in train_feeder.iterate_batch(args.batch_size):
            feed_dict = {tracker.tf_img: input_batch,
                        tracker.tf_label: label_batch}
            sess.run([tracker.train_step], feed_dict=feed_dict)

except:  # save on unexpected termination
    if not DEBUG:
        print('saving...')
        checkpoint_file = os.path.join(args.log_dir, 'checkpoints', 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=global_step)
    raise