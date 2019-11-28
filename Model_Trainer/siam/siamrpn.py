#!/usr/bin/env python
# coding: utf-8
"""
script: train siamrpn tracker
"""

import sys
root_dir = '../../'
sys.path.append(root_dir)

import os
import time
import torch
import random
import psutil
import Utilities as utils


""" parsing args """

cfg = utils.Config()
cfg.merge_yaml('./config.yaml')
parser = config.construct_argparser()
args = parser.parse_args()
cfg.merge_args(args)
cfg['DEBUG'] = True
cfg = cfg.freeze()

path_cfg = cfg['path']
train_cfg = cfg['train']
model_cfg = cfg['model']
feeder_cfg = cfg['feeder']

def seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # disable online tunning for specific input size (gpu mem related, e.g. layout)
    # => useful when input size fixed 
    # (otherwise may benchmark different algorithms multiple times and hence worsen performance)
    torch.backends.cudnn.benchmark = False
    # only allow cudnn to choose (believed to be) determinstic algorithm
    # for reproducibility: turn off benchmark (as may use different algorithms depending on host)
    torch.backends.cudnn.deterministic = True

if train_cfg['rand_seed'] is not None:
    seed(train_cfg['rand_seed'])

# model name
if not model_cfg['name']:
    tt = time.localtime()
    time_str = ('_%04d_%02d_%02d_%02d_%02d_%02d' % (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
    model_name = 'siam-%s%s_%s%s_%s' % (
        feeder_cfg['bbox_encoding'],
        '_' + feeder_cfg['attention'] if feeder_cfg['bbox_encoding'] != 'crop' else '',
        model_cfg['backbone']['model'],
        '_neck' if model_cfg['neck'] else '',
        model_cfg['rpn']['model'],
    )

    model_name += '_r%d' % train_cfg['rand_seed'] if train_cfg['rand_seed'] else ''
    model_name += time_str
else:
    model_name = model_cfg['name']

# config dir path
assert path_cfg['restore_dir']
os.makedirs(os.path.join(path_cfg['model_dir'], model_name), exist_ok=True)
os.makedirs(path_cfg['summary_dir'], exist_ok=True)

# change std out if log dir originally given
if path_cfg['log_dir']:
    os.makedirs(path_cfg['log_dir'], exist_ok=True)
    log_file_path = os.path.join(path_cfg['log_dir'], model_name)
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file  # redirect runtime warining/error as well
    print(model_name)
else:
    log_file = ''

# import training after env prepared
import Siam_Trainer as trainer
try:
    siam_trainer = trainer.SiamRPN_Trainer(model_name, cfg)
    siam_trainer.train()
except:
    log_file.close()
    raise
