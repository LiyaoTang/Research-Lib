#!/usr/bin/env python
# coding: utf-8
'''
module: pipeline to train a neural net, with corresponding feeder, model, recorder and etc. carefully connected.
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

class Re3_Trainer(object):
    def __init__(self, model_name, root_dir, config):
        super(Re3_Trainer, self).__init__()
        self.config = config
        self.model_name = model_name
        self.root_dir = root_dir

        # prepare env
        assert 'log_dir' in config
        assert 'model_dir' in config
        assert 'summary_dir' in config
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['summary_dir'], exist_ok=True)
        if 'restore_dir' not in self.config:
            self.config['restore_dir'] = self.config['log_dir']

        # get feeder
        assert 'num_unrolls' in config
        assert config['label_type'] in ['corner', 'center']
        assert config['bbox_encoding'] in ['mask', 'crop', 'mesh']
        assert config['use_inference_prob'] <= 1
        assert config['unroll_type'] in ['manual', 'dynamic']
        assert config['run_val'] in [True, False]
        feeder_cfg = {
            'label_type':config['label_type'],
            'bbox_encoding':config['bbox_encoding'],
            'use_inference_prob': config['use_inference_prob'],
            'data_split': 'train',
        }

        data_ref_dir = os.path.join(root_dir, 'Data/ILSVRC2015')
        train_ref = os.path.join(data_ref_dir, 'train_label.npy')
        self.train_feeder = feeder.Imagenet_VID_Feeder(train_ref, class_num=30, num_unrolls=config['num_unrolls'], config=self.feeder_cfg)

        if config['run_val']:
            feeder_cfg['data_split'] = 'val'
            feeder_cfg['use_inference_prob'] = -1
            val_ref = os.path.join(data_ref_dir, 'val_label.npy')
            self.val_feeder = feeder.Imagenet_VID_Feeder(val_ref, class_num=30, num_unrolls=config['num_unrolls'], config=self.feeder_cfg)
        else:
            self.val_feeder = None

        # get tracker
        assert config['label_norm'] in ['fix', 'dynamic', 'raw']
        assert config['use_tfdataset'] in [True, False]
        assert config['use_parallel'] in [True, False]
        model_cfg = {
            'label_type':config['label_type'],
            'bbox_encoding':config['bbox_encoding'],
            'label_norm': config['label_norm'],
            'unroll_type': config['unroll_type'],
        }

        if config['use_tfdataset']:
            pass
        else:
            tf_input = tf.placeholder(tf.uint8, shape=[None, None, None, None, ])
            tf_label = tf.placeholder(tf.float32, shape=[None, None, 4])
            tf_unroll = tf.placeholder(tf.int32)

        self.tracker = constructer.Re3_Tracker()



        # training config
        assert config['run_val'] in [True, False]
        assert config['run_val'] in [True, False]
        assert config['run_val'] in [True, False]

    def display_img_pred_label(self, track_img, label_box, track_pred):
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

    def run_train_step_feeddict(self, input_batch, label_batch, display=False):
        '''
        routine to run one train step via feed dict
        '''
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