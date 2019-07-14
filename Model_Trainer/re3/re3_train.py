#!/usr/bin/env python
# coding: utf-8
'''
script: train re3 tracker (directly)
'''

import sys
root_dir = '../../'
sys.path.append(root_dir)

import os
import time
import random
import psutil
import argparse

import Re3_Trainer as trainer

DEBUG = True

''' parsing args '''

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected, given ', type(v))
parser = argparse.ArgumentParser(description='training re3')

parser.add_argument('--attention', type=str, dest='attention')
parser.add_argument('--num_unrolls', default=2, type=int, dest='num_unrolls')
parser.add_argument('--batch_size', default=64, type=int, dest='num_unrolls')
parser.add_argument('--lstm_size', default=512, type=int, dest='lstm_size')
parser.add_argument('--lrn_rate', default=None, type=float, dest='lrn_rate')

parser.add_argument('--fuse_type', default='spp', type=str, dest='fuse_type')
parser.add_argument('--label_type', default='center', type=str, dest='label_type')
parser.add_argument('--label_norm', default='dynamic', type=str, dest='label_norm')
parser.add_argument('--unroll_type', default='dynamic', type=str, dest='unroll_type')
parser.add_argument('--bbox_encoding', default='mask', type=str, dest='bbox_encoding')
parser.add_argument('--use_inference_prob', default=-1, type=float, dest='use_inference_prob')

parser.add_argument('--max_step', default=1e6, type=float, dest='max_step')
parser.add_argument('--rand_seed', default=None, type=int, dest='rand_seed')

parser.add_argument('--run_val', default=True, type=str2bool, dest='run_val')
parser.add_argument('--worker_num', default=1, type=int, dest='worker_num')
parser.add_argument('--buffer_size', default=5, type=int, dest='buffer_size')
parser.add_argument('--use_parallel', default=True, type=str2bool, dest='use_parallel')
parser.add_argument('--use_tfdataset', default=False, type=str2bool, dest='use_tfdataset')

parser.add_argument('--model_name', type=str, dest='model_name')
parser.add_argument('--root_dir', default='../../', type=str, dest='root_dir')
parser.add_argument('--log_dir', default='./Log', type=str, dest='log_dir')
parser.add_argument('--model_dir', default='./Model', type=str, dest='model_dir')
parser.add_argument('--summary_dir', default='./Summary', type=str, dest='summary_dir')
parser.add_argument('--restore', default=True, type=str2bool, dest='restore')
parser.add_argument('--restore_dir', default=None, type=str, dest='restore_dir')

parser.add_argument('--debug', default=True, type=str2bool, dest='debug')
parser.add_argument('--display', default=False, type=str2bool, dest='display')

args = parser.parse_args()

if args.rand_seed is not None:
    np.random.seed(args.rand_seed)
    tf.random.set_random_seed(args.rand_seed)

# model name
if not args.model_name:
    tt = time.localtime()
    time_str = ('_%04d_%02d_%02d_%02d_%02d_%02d' % (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
    args.model_name = 're3-%s_lstm%d_%s_%s_%s_%s' % (args.bbox_encoding, args.lstm_size, args.attention, \
                                                     args.label_type, args.label_norm, args.fuse_type)
    args.model_name += '' if args.rand_seed is None else '_r%d' % args.rand_seed
    args.model_name += time_str

# config dir path
if not args.restore_dir:
    args.restore_dir = os.path.join(args.model_dir, 're3')
args.max_step = int(args.max_step)
args.model_dir = os.path.join(args.model_dir, args.model_name)

# make dirs
os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(args.summary_dir, exist_ok=True)

# change std out if log dir originally given
if args.log_dir:
    os.makedirs(args.log_dir, exist_ok=True)
    log_file_path = os.path.join(args.log_dir, args.model_name)
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    print(args.model_name)
else:
    log_file = ''



''' construct feeder '''


feeder_cfg = {
    'label_type': args.label_type,
    'bbox_encoding': args.bbox_encoding,
    'use_inference_prob': args.use_inference_prob,
    'data_split': 'train',
}
data_ref_dir = os.path.join(root_dir, 'Data/ILSVRC2015')
train_ref = os.path.join(data_ref_dir, 'train_label.npy') 
train_feeder = feeder.Imagenet_VID_Feeder(train_ref, class_num=30, config=feeder_cfg)

feeder_cfg['data_split'] = 'val'
val_ref = os.path.join(data_ref_dir, 'val_label.npy')
val_feeder = feeder.Imagenet_VID_Feeder(val_ref, class_num=30, config=feeder_cfg)

if args.use_parallel:
    cur_feeder = feeder.Parallel_Feeder(train_feeder, buffer_size=args.buffer_size, worker_num=args.worker_num)
    paral_feeder = cur_feeder
else:
    cur_feeder = train_feeder
    paral_feeder = None

if args.use_tfdataset:
    raise NotImplementedError # TODO: not tested yet
    feeder_gen = cur_feeder.iterate_data()
    tf_dataset = tf.data.Dataset.from_generator(feeder_gen, (tf.uint8, tf.float32))
    tf_dataset = tf_dataset.prefetch(2)
    tf_dataset_iter = tf_dataset.make_one_shot_iterator()
    tf_input, tf_label = tf_dataset_iter.get_next()
    cur_feeder = None
else:
    if args.bbox_encoding == 'crop':
        channel_size = 3
    elif args.bbox_encoding == 'mask':
        channel_size = 4
    else:  # mesh
        channel_size = 6
    tf_input = tf.placeholder(tf.float32, shape=[None, None, None, None, channel_size])
    tf_label = tf.placeholder(tf.float32, shape=[None, None, 4])
tf_prev_state = tuple([tf.placeholder(tf.float32, shape=(1, args.lstm_size)) for _ in range(4)])  # hashable tuple
tf_init_state = tuple([np.zeros(shape=(1, args.lstm_size)) for _ in range(4)])


''' construct model '''


self.sess = constructer.tfops.Session()
tracker_cfg = {
    'attention': args.attention,
    'bbox_encoding': args.bbox_encoding,
    'unroll_type': args.unroll_type,
    'label_type': args.label_type,
    'label_norm': args.label_norm,
    'fuse_type': args.fuse_type,
}
self.tracker = constructer.Re3_Tracker(self.tf_input, self.tf_label, self.tf_prev_state,
                                        lstm_size=args.lstm_size, config=tracker_cfg)
train_cfg = {
    'learning_rate': tf.placeholder(tf.float32) if args.lrn_rate is None else args.lrn_rate,
}
self.tracker.build_train_step(config=train_cfg)
self.tracker.build_summary()
self.encode_bbox = train_feeder.encode_bbox_to_img
self.decode_bbox = train_feeder.decode_bbox
train_feeder.model = tracker

# logging validation
if args.run_val:
    val_scope = 'val'
    with tf.variable_scope(val_scope):
        with tf.variable_scope('summary'):
            val_rst_placeholders = {
                'robustness_ph': tf.placeholder(tf.float32, shape=[]),
                'lost_targets_ph': tf.placeholder(tf.float32, shape=[]),
                'mean_iou_ph': tf.placeholder(tf.float32, shape=[]),
                'avg_ph': tf.placeholder(tf.float32, shape=[]),
            }
            summary_op = [tf.summary.scalar(n, v) for n, v in val_rst_placeholders.items()]
            summary_op = tf.summary.merge(summary_op)
        tf_input = self.val_placeholders['input']
        tf_label = self.val_placeholders['label']
        tf_prev_state = self.val_placeholders['prev_state']
        self.val_tracker = constructer.Re3_Tracker(tf_input, tf_label, tf_prev_state,
                                                lstm_size=args.lstm_size, config=tracker_cfg)
        # restore latest ckpt to val model (under scope 'val')
        val_vars = list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=val_scope))
        restore_dict = dict(zip([v.name.strip(val_scope + '/').split(':')[0] for v in val_vars], val_vars))

        val_cfg = {
            'summary': {'placeholder': val_rst_placeholders, 'op': summary_op}
        }
        self.val_model = constructer.Val_Model(self.sess, self.val_tracker, self.val_feeder, self.val_recorder,
                                            var_dict=restore_dict, config=val_cfg)
        self.val_model.inference = lambda x, y: self.tracker.inference(x, y, self.sess)
