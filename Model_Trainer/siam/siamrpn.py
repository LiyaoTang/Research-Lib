#!/usr/bin/env python
# coding: utf-8
'''
script: train siamrpn tracker
'''

import sys
root_dir = '../../'
sys.path.append(root_dir)

import os
import time
import random
import psutil
import Utilities as utils


''' parsing args '''

config = utils.Config()
config.merge_yaml('./config.yaml')
parser = config.construct_argparser()
args = parser.parse_args()
config.merge_args(args)

config['DEBUG'] = True
print(config.config)

exit()

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
    sys.stderr = log_file  # redirect runtime warining/error as well
    print(args.model_name)
else:
    log_file = ''

try:
    re3_trainer = trainer.Re3_Trainer(args.model_name, args.root_dir, args)
    re3_trainer.train()
except:
    log_file.close()
    raise
