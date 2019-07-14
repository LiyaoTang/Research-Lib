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

parser.add_argument('--batch_size', default=64, type=int, dest='num_unrolls')
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

try:
    re3_trainer = trainer.Re3_Trainer(args.model_name, args.root_dir, args)
    re3_trainer.train()
except:
    log_file.close()
    raise
