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
plt.ion()

import Data_Feeder as feeder
import Model_Analyzer as analyzer
import Model_Constructer as constructer
import Model_Constructer.TF_Ops as tfops

class Re3_Trainer(object):
    def __init__(self, model_name, root_dir, args):
        super(Re3_Trainer, self).__init__()
        self.model_name = model_name
        self.root_dir = root_dir
        self.args = args

        ''' construct feeder '''
        self.construct_feeder(self.args)

        ''' metric recoder '''
        self.train_recorder = analyzer.Tracking_SOT_Record()
        self.val_recorder = analyzer.Tracking_SOT_Record()

        ''' construct model '''
        # get session
        self.sess = constructer.tfops.Session()
        self.construct_model(self.args, self.sess)

        ''' prepare training '''
        self.prepare_train(self.args)
        self.train = self.train_tfdataset if self.args.use_tfdataset else self.train_feeddict
        self.display_img_pred_label = self.display_img_pred_label_single if args.bbox_encoding != 'crop' else self.display_img_pred_label_double

    def construct_feeder(self, args):
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
            self.paral_feeder = cur_feeder
        else:
            cur_feeder = train_feeder
            self.paral_feeder = None

        if args.use_tfdataset:
            raise NotImplementedError # TODO: consider ways to re-initialize tf_dataset with reconstructed feeder 
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

        self.train_feeder = train_feeder
        self.val_feeder = val_feeder
        self.cur_feeder = cur_feeder
        self.tf_input = tf_input
        self.tf_label = tf_label
        self.tf_prev_state = tf_prev_state
        self.tf_init_state = tf_init_state

        self.val_placeholders = {
            'input': tf.placeholder(tf.float32, shape=[None, None, None, None, channel_size]),
            'label': tf.placeholder(tf.float32, shape=[None, None, 4]),
            'prev_state': tuple([tf.placeholder(tf.float32, shape=(1, args.lstm_size)) for _ in range(4)]),  # hashable tuple
        }

    def reconfig_feeder(self, data_feeder, paral_feeder, num_unrolls, batch_size, args):
        if args.use_parallel:  # shutdown worker
            paral_feeder.shutdown()
        data_feeder.reset(num_unrolls, batch_size)  # reconstruct dataset
        if args.use_parallel: # restart worker
            paral_feeder.refresh(feeder=data_feeder)
        cur_feeder = paral_feeder if args.use_parallel else data_feeder

        if args.use_tfdataset:
            tf.reset_default_graph() # reset all tf ops & rebuild the graph
            feeder_gen = cur_feeder.iterate_data()
            tf_dataset = tf.data.Dataset.from_generator(feeder_gen, (tf.uint8, tf.float32))
            tf_dataset = tf_dataset.prefetch(2)
            tf_dataset_iter = tf_dataset.make_one_shot_iterator()
            tf_input, tf_label = tf_dataset_iter.get_next()
            tf_prev_state = tuple([tf.placeholder(tf.float32, shape=(1, args.lstm_size)) for _ in range(4)])  # hashable tuple
            tf_init_state = tuple([np.zeros(shape=(1, args.lstm_size)) for _ in range(4)])

            self.tf_input = tf_input
            self.tf_label = tf_label
            self.tf_prev_state = tf_prev_state
            self.tf_init_state = tf_init_state

            self.construct_model(self.args, self.sess)
            cur_feeder = None
        self.cur_feeder = cur_feeder

    def display_img_pred_label_double(self, track_img, label_box, track_pred):
        raise

    def display_img_pred_label_single(self, track_img, label_box, track_pred):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        for img, label, pred in zip(track_img, label_box, track_pred):  # assume xyxy box
            pred_rect = mlt.patches.Rectangle((pred[[0, 1]]), pred[2] - pred[0], pred[3] - pred[1], color='g', fill=False)
            label_rect = mlt.patches.Rectangle((label[[0, 1]]), label[2] - label[0], label[3] - label[1], color='r', fill=False)

            ax.imshow(np.array(img, dtype=int)[...,:3])  # in case of mask/mesh
            ax.add_patch(label_rect)
            ax.add_patch(pred_rect)
            fig.canvas.draw()
            plt.show()
            plt.waitforbuttonpress(-1)
            pred_rect.remove()
            label_rect.remove
            ax.clear()
        plt.close()

    def construct_model(self, args, sess):
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

        # link feeder-tracker dependency
        self.tracker.encode_bbox = self.train_feeder.encode_bbox_to_img
        self.tracker.decode_bbox = self.train_feeder.decode_bbox
        self.train_feeder.config['model'] = self.tracker

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
                self.val_tracker.encode_bbox = self.train_feeder.encode_bbox_to_img
                self.val_tracker.decode_bbox = self.train_feeder.decode_bbox
                # restore latest ckpt to val model (under scope 'val')
                val_vars = list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=val_scope))
                restore_dict = dict(zip([v.name.strip(val_scope + '/').split(':')[0] for v in val_vars], val_vars))

                val_cfg = {
                    'summary': {'placeholder': val_rst_placeholders, 'op': summary_op}
                }
                self.val_model = constructer.Val_Model(sess, self.val_tracker, self.val_feeder, self.val_recorder,
                                                    var_dict=restore_dict, config=val_cfg)
                self.val_model.inference = lambda x, y: self.tracker.inference(x, y, sess)

    def prepare_train(self, args):
        # create saver at the last & finalize graph
        model_var = tf.trainable_variables()
        self.saver = tf.train.Saver()
        self.longSaver = tf.train.Saver()

        # initialize
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # restore
        self.global_step = 0
        if args.restore:
            self.global_step = tfops.restore_from_dir(self.sess, args.restore_dir)

        # finalize
        self.sess.graph.finalize()

        # writer for tf.summary io
        self.summary_writer = tf.summary.FileWriter(os.path.join(args.summary_dir, args.model_name), self.sess.graph)

    # routine to update model
    def run_train_step_feeddict(self, input_batch, label_batch, global_step, display=False):
        feed_dict = {
            self.tracker.tf_input: input_batch,
            self.tracker.tf_label: label_batch,
            self.tracker.prev_state: self.tf_init_state
        }
        op_to_run = [self.train_step]

        # record summary
        if global_step % 5000 == 0:  # conv, lstm, loss => all summary
            op_to_run += [self.tracker.summary['all']]
        elif global_step % 1000 == 0:  # lstm, loss
            op_to_run += [self.tracker.summary['lstm'], self.tracker.summary['loss']]
        elif global_step % 100 == 0:  # loss
            op_to_run += [self.tracker.summary['loss']]

        # get pred bbox for display
        if display:
            op_to_run += [self.tracker.pred]

        # run ops
        op_output = self.sess.run(op_to_run, feed_dict=feed_dict)

        # display
        if display:
            track_pred = op_output[-1]
            track_img = input_batch[0]
            label_box = [self.train_feeder.revert_label_type(l) for l in label_batch[0]]
            pred_box = [self.train_feeder.revert_label_type(p) for p in track_pred[0]]

            self.display_img_pred_label(track_img, label_box, pred_box)
            op_output = op_output[:-1] # get rid of pred

        # write summary
        cur_summary = op_output[1:]
        if cur_summary:
            for s in cur_summary:
                self.summary_writer.add_summary(s, global_step=global_step)

        # save new ckpt (over-write old one)
        if global_step % 5000 == 0:
            ckpt_path = os.path.join(self.args.model_dir, self.args.model_name)
            self.saver.save(self.sess, ckpt_path, global_step=global_step)
            self.summary_writer.flush()
            sys.stdout.flush()
            if self.args.run_val:
                self.val_model.record_val(ckpt_path)

    def train_feeddict(self):
        # get tf summary before training
        feed_dict = {self.tracker.tf_input: np.zeros((1, 2, 227, 227, self.tracker.tf_input.get_shape().as_list()[-1])),
                     self.tracker.tf_label: np.zeros((1, 2, self.tracker.tf_label.get_shape().as_list()[-1])),
                     self.tracker.prev_state: self.tf_init_state}
        self.summary_writer.add_summary(self.sess.run(self.tracker.summary['all'], feed_dict=feed_dict))
        self.summary_writer.flush()

        # start training
        args = self.args
        start_time = time.time()
        try:
            if args.bbox_encoding in ['mask', 'mesh']:  # use mask => varying input img size
            # training strategy: initial unrolls=2, then unrolls*=2 till unroll=32; batch=1 to avoid searching for same-size img
            # stablize first few convs, then jointly train
                num_unrolls = 2
                batch_size = 64
                epoch = 2
                # allow only specified vars to train: to stablize (for 2 epoch, no re-config)
                self.train_step = self.tracker.stablize_step['preprocess']
                self.reconfig_feeder(self.train_feeder, self.paral_feeder, num_unrolls, batch_size, args)
                for ep in range(epoch):
                    for input_batch, label_batch in self.cur_feeder.iterate_data():
                        self.run_train_step_feeddict(input_batch, label_batch, self.global_step, args.display)
                        self.global_step += 1

                # num_unrolls = 2
                # batch_size = 1
                # self.train_step = self.tracker.train_step  # join optimization
                # while num_unrolls <= 32 and self.global_step <= args.max_step:
                #     self.reconfig_feeder(self.train_feeder, self.paral_feeder, num_unrolls, batch_size, args)
                #     for input_batch, label_batch in self.cur_feeder.iterate_data():
                #         self.run_train_step_feeddict(input_batch, label_batch, self.global_step, args.display)
                #         self.global_step += 1
                #     num_unrolls *= 2

                # save the lastest model
                ckpt_path = os.path.join(args.model_dir, args.model_name)
                self.saver.save(self.sess, ckpt_path, global_step=self.global_step)
                self.summary_writer.flush()
                sys.stdout.flush()


            else:  # use crop => fixed input img size
            # => training strategy: initial unrolls=2, batch=64; then, unrolls*=2, batch/=2, till unroll=32 (batch=4)
                num_unrolls = 2
                batch_size = 64
                while num_unrolls <= 32 and self.global_step <= args.max_step:
                    self.reconfig_feeder(self.train_feeder, self.paral_feeder, num_unrolls, batch_size, args)
                    for input_batch, label_batch in self.cur_feeder.iterate_data():
                        run_train_step_feeddict(input_batch, label_batch, self.global_step, args.display)
                        self.global_step += 1
                    num_unrolls *= 2
                    batch_size /= 2

        except:  # save on unexpected termination
            if self.paral_feeder is not None:
                self.paral_feeder.shutdown()
            if not args.debug:
                print('saving...')
                checkpoint_file = os.path.join(args.log_dir, 'checkpoints', 'model.ckpt')
                self.saver.save(self.sess, checkpoint_file, global_step=self.global_step)
            raise

    def run_train_step_tfdataset(self, global_step, display=False):
        op_to_run = [self.tracker.train_step]
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
        op_output = self.sess.run([op_to_run], feed_dict=feed_dict)

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
            ckpt_path = os.path.join(self.args.model_dir, self.args.model_name)
            self.saver.save(self.sess, ckpt_path, global_step=global_step)

            if self.args.run_val:
                self.val_model.record_val(ckpt_path)

    def train_tfdataset(self):
        while True:
            try:
                self.run_train_step_feeddict()
            except:
                if self.global_step <= args.max_step:
                    continue
                else:
                    break