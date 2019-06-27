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
import matplotlib as mlt
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

parser.add_argument('--lrn_rate', default=None, type=int, dest='lrn_rate')
parser.add_argument('--buffer_size', default=5, type=int, dest='buffer_size')
parser.add_argument('--max_step', default=1e6, type=int, dest='max_step')

parser.add_argument('--run_val', default=True, type=bool, dest='run_val')
parser.add_argument('--rand_seed', default=None, type=int, dest='rand_seed')
parser.add_argument('--worker_num', default=1, type=int, dest='worker_num')
parser.add_argument('--tf_dataset', default=True, type=bool, dest='tf_dataset')

parser.add_argument('--log_dir', default='./Log', type=str, dest='log_dir')
parser.add_argument('--model_dir', default='./Model', type=str, dest='model_dir')
parser.add_argument('--summary_dir', default='./Summary', type=str, dest='summary_dir')
parser.add_argument('--restore_dir', default='', type=str, dest='restore_dir')

args = parser.parse_args()

args.batch_size = int(max(64 / args.num_unrolls, 2))
args.channel_size = 4 if args.bbox_encoding in ['mask', 'mesh'] else 3
assert args.img_size in [None, 227]
if args.rand_seed is not None:
    np.random.seed(args.rand_seed)
    tf.random.set_random_seed(args.rand_seed)
if not args.restore_dir:
    args.restore_dir = args.model_dir
    
# model name
tt = time.localtime()
time_str = ('%04d_%02d_%02d_%02d_%02d_%02d' % (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
args.model_name = 're3-%s_un%d_img%d_lstm%d_b%d_e%d' % \
                  (args.bbox_encoding, args.num_unrolls, args.img_size, args.lstm_size, args.batch_size, args.epoch)
args.model_name += '' if args.rand_seed is None else '_r%d' % args.rand_seed
args.model_name += time_str

# make dirs
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.summary_dir, exist_ok=True)


''' construct feeder '''


feeder_cfg = {
    'label_type': args.label_type,
    'bbox_encoding':'crop',
    'use_inference_prob':-1,
    'data_split': 'train',
}
data_ref_dir = os.path.join(root_dir, 'Data/ILSVRC2015')
train_ref = os.path.join(data_ref_dir, 'train_label.npy')
train_feeder = feeder.Imagenet_VID_Feeder(train_ref, class_num=30, num_unrolls=args.num_unrolls)

if args.tf_dataset:
    raise NotImplementedError
    # para_train_feeder = feeder.Parallel_Feeder(train_feeder, batch_size=batch_size,
    #                                            buffer_size=args.buffer_size, worker_num=args.worker_num)
    # feeder_gen = para_train_feeder.iterate_batch()
    # tf_dataset = tf.data.Dataset.from_generator(feeder_gen, (tf.uint8, tf.float32))
    # tf_dataset_iterator = feeder_gen

else:
    tf_input = tf.placeholder(tf.uint8, shape=[None, None, None, None, args.channel_size])
    tf_label = tf.placeholder(tf.float32, shape=[args.batch_size, None, 4])
    tf_unroll = tf.placeholder(tf.int32)

def display_img_pred_label(track_img, label_box, track_pred):
    fig, ax = plt.subplots(1, figsize=(10,10))
    for img, label, pred in zip(track_img, label_box, track_pred):  # assume xyxy box
        pred_rect = mlt.patches.Rectangle((pred[[0, 1]]), pred[2] - pred[0], pred[3] - pred[1], color='g', fill=False)
        label_rect = mlt.patches.Rectangle((label[[0, 1]]), label[2] - label[0], label[3] - label[1], color='r', fill=False)

        ax.imshow(img)
        ax.add_patches(label_rect)
        ax.add_patches(pred_rect)
        fig.canvas.draw()
        plt.show()
        plt.waitforbuttonpress()
        pred_rect.remove()
        label_rect.remove
        ax.clear()


''' metric recoder '''


train_recorder = analyzer.Tracking_SOT_Record()
val_recorder = analyzer.Tracking_SOT_Record()


''' construct model '''


sess = constructer.tfops.Session()
saver = tf.train.Saver()
longSaver = tf.train.Saver()

tracker = constructer.Re3_Tracker(tf_input, tf_label, num_unrolls=tf_unroll,
                                  img_size=args.img_size,
                                  lstm_size=args.lstm_size,
                                  unroll_type=args.unroll_type,
                                  bbox_encoding=args.bbox_encoding)
learning_rate = tf.placeholder(tf.float32) if args.lrn_rate is None else args.lrn_rate
train_step = tracker.get_train_step(learning_rate)
summary_op = tracker.summary['all']

# logging validation
val_scope = 'val'
with tf.variable_scope(val_scope):
    robustness_ph = tf.placeholder(tf.float32, shape=[])
    lost_targets_ph = tf.placeholder(tf.float32, shape=[])
    mean_iou_ph = tf.placeholder(tf.float32, shape=[])
    avg_ph = tf.placeholder(tf.float32, shape=[])
    val_tracker = constructer.Re3_Tracker(tf_img, tf_label,
                                        num_unrolls=args.num_unrolls,
                                        img_size=args.img_size,
                                        lstm_size=args.lstm_size,
                                        unroll_type=args.unroll_type,
                                        bbox_encoding=args.bbox_encoding)
    val_vars = list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=val_scope))
    restore_dict = dict([v.name.strip(val_scope + '/').split(':')[0] for v in val_vars], val_vars)
    val_model = constructer.Val_Model(sess, val_tracker, feeder, val_recorder, var_dict=restore_dict)


''' training '''


# initialize/restore
global_step = 0
sess.run(tf.global_variables_initializer())
if args.restore:
    global_step = tfops.restore_from_dir(sess, args.restore_dir)

# writer for tf.summary io
sess.graph.finalize()
summary_writer = tf.summary.FileWriter(os.path.join(args.summary_dir, args.model_name), sess.graph)

# routine to update model
def run_train_step_feeddict(input_batch, label_batch, display=False):
    feed_dict = {
        tracker.tf_input: input_batch,
        tracker.tf_label: label_batch,
        tracker.num_unrolls: num_unrolls,
    }
    op_to_run = [tracker.train_step]

    # record summary
    if global_step % 1000 == 0:  # conv, lstm, loss => all summary
        op_to_run += [tracker.summary['all']]
    elif global_step % 100 == 0:  # lstm, loss
        op_to_run += [tracker.summary['lstm'], tracker.summary['loss']]
    elif global_step % 10 == 0:  # loss
        op_to_run += [tracker.summary['loss']]

    # get pred bbox for display
    if display:
        op_to_run += [tracker.pred]

    # run ops
    op_output = sess.run([op_to_run], feed_dict=feed_dict)

    # display
    if display:
        track_pred = op_to_run[-1][0]
        track_img = input_batch[0]
        label_box = [train_feeder.revert_label_type(l) for l in label_batch[0]]
        pred_box = [train_feeder.revert_label_type(p) for p in track_pred]

        display_img_pred_label(track_img, label_box, pred_box)
        op_to_run = op_to_run[:-1] # get rid of pred

    # write summary
    cur_summary = op_output[1:]
    if cur_summary:
        for s in cur_summary:
            summary_writer.add_summary(s, global_step=global_step)

    # save new ckpt (over-write old one)
    if global_step % 500 == 0:
        ckpt_path = os.path.join(args.model_dir, args.model_name)
        saver.save(sess, ckpt_path, global_step=global_step)

        if args.run_val:
            val_model.record_val(ckpt_path)

def run_train_step_tfdataset(tf_img):
    raise NotImplementedError
    pass

run_train_step = run_train_step_tfdataset if args.tf_dataset else run_train_step_feeddict

# start training
start_time = time.time()
try:
    if args.bbox_encoding in ['mask', 'mesh']:  # use mask => varying input img size
    # => training strategy: initial unrolls=2, then unrolls*=2 till unroll=32; batch=1 to avoid searching for same-size img
        num_unrolls = 2
        batch_size = 1
        for input_batch, label_batch in train_feeder.iterate_batch(batch_size):
            run_train_step(input_batch, label_batch)
            global_step += 1

    else:  # use crop => fixed input img size
    # => training strategy: initial unrolls=2, batch=64; then, unrolls*=2, batch/=2, till unroll=32 (batch=4)
        num_unrolls = 2
        batch_size = 64
        for ep_cnt in range(args.epoch):
            batch_cnt = 0
            for input_batch, label_batch in train_feeder.iterate_batch(batch_size):
                feed_dict = {tracker.tf_img: input_batch,
                            tracker.tf_label: label_batch}
                sess.run([tracker.train_step], feed_dict=feed_dict)

except:  # save on unexpected termination
    if not DEBUG:
        print('saving...')
        checkpoint_file = os.path.join(args.log_dir, 'checkpoints', 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=global_step)
    raise