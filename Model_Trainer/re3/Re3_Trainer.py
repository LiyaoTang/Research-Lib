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
import Models as models

class Re3_Trainer(object):
    def __init__(self, model_name, root_dir, args):
        super(Re3_Trainer, self).__init__()
        self.args = args
        self.root_dir = root_dir
        self.model_name = model_name
        self.ckpt_path = os.path.join(self.args.model_dir, self.args.model_name)
        self.is_training = False

        ''' feeder '''
        self.train_feeder = None
        self.paral_feeder = None
        self.val_feeder = None
        self.data_feeder = None

        ''' metric recoder '''
        self.train_recorder = analyzer.Tracking_SOT_Record()
        self.val_recorder = analyzer.Tracking_SOT_Record()

        self.train = self.train_tfdataset if self.args.use_tfdataset else self.train_feeddict
        self.display_img_pred_label = self.display_img_pred_label_single if args.bbox_encoding != 'crop' else self.display_img_pred_label_double

    # def _construct_tf_feeder(self, args):
    # def _construct_py_feeder(self, args):

    def construct_feeder(self, num_unrolls, batch_size):
        args = self.args
        feeder_cfg = {
            'label_type': args.label_type,
            'bbox_encoding': args.bbox_encoding,
            'use_inference_prob': args.use_inference_prob,
            'data_split': 'train',
        }
        data_ref_dir = os.path.join(root_dir, 'Data/ILSVRC2015')
        train_ref = os.path.join(data_ref_dir, 'train_label.npy') 
        train_feeder = feeder.Imagenet_VID_Feeder(train_ref, class_num=30, num_unrolls=num_unrolls, batch_size=batch_size, config=feeder_cfg)
        self.train_feeder = train_feeder
        print('total data num = ', len(train_feeder.data_ref))

        if args.run_val:
            feeder_cfg['data_split'] = 'val'
            val_ref = os.path.join(data_ref_dir, 'val_label.npy')
            val_feeder = feeder.Imagenet_VID_Feeder(val_ref, class_num=30, config=feeder_cfg)
            self.val_feeder = val_feeder

        if args.use_parallel:
            data_feeder = feeder.Parallel_Feeder(train_feeder, buffer_size=args.buffer_size, worker_num=args.worker_num, verbose=True)
            self.paral_feeder = data_feeder
        else:
            data_feeder = train_feeder
            self.paral_feeder = None
        self.data_feeder = data_feeder

    def _construct_placeholder(self):
        args = self.args
        if args.bbox_encoding == 'crop':
            channel_size = 3
        elif args.bbox_encoding == 'mask':
            channel_size = 4
        else:  # mesh
            channel_size = 6
        input_shape = [None, None, None, None, channel_size]
        label_shape = [None, None, 4]
        
        if args.use_tfdataset:
            with tf.variable_scope('dataset'):
                data_gen = self.data_feeder.iterate_data
                tf_dataset = tf.data.Dataset.from_generator(data_gen, (tf.float32, tf.float32), (input_shape, label_shape))
                tf_dataset = tf_dataset.prefetch(2)
                tf_dataset_iter = tf_dataset.make_initializable_iterator()
                tf_input, tf_label = tf_dataset_iter.get_next()
        else:
            tf_dataset_iter = None
            tf_input = tf.placeholder(tf.float32, shape=input_shape)
            tf_label = tf.placeholder(tf.float32, shape=label_shape)
        self.tf_input = tf_input
        self.tf_label = tf_label
        self.tf_dataset_iter = tf_dataset_iter

        if args.run_val:
            self.val_placeholders = {
                'input': tf.placeholder(tf.float32, shape=[None, None, None, None, channel_size]),
                'label': tf.placeholder(tf.float32, shape=[None, None, 4]),
                # 'prev_state': tuple([tf.placeholder(tf.float32, shape=(1, args.lstm_size)) for _ in range(4)]),  # hashable tuple
                'prev_state': None,
            }

        # tf_prev_state = tuple([tf.placeholder(tf.float32, shape=(None, args.lstm_size)) for _ in range(4)])  # hashable tuple
        # tf_init_state = lambda batch_size: tuple([np.zeros(shape=(batch_size, args.lstm_size)) for _ in range(4)])
        # self.tf_prev_state = tf_prev_state
        # self.tf_init_state = tf_init_state
        self.tf_prev_state = None

    def construct_model(self):
        args = self.args
        self._construct_placeholder()

        tracker_cfg = {
            'attention': args.attention,
            'bbox_encoding': args.bbox_encoding,
            'unroll_type': args.unroll_type,
            'label_type': args.label_type,
            'label_norm': args.label_norm,
            'fuse_type': args.fuse_type,
        }
        self.tracker = models.Re3_Tracker(self.tf_input, self.tf_label, self.tf_prev_state,
                                               lstm_size=args.lstm_size, config=tracker_cfg)
        print('all trainable variable\n', '\n'.join(sorted([v.name for v in tf.trainable_variables()])))
        train_cfg = {
            'learning_rate': tf.placeholder(tf.float32) if args.lrn_rate is None else args.lrn_rate,
            'var_list': [v for v in tf.trainable_variables() if v.name.startswith(args.weight_prefix)]
        }
        print('train_cfg:\n', train_cfg)
        self.tracker.build_train_step(config=train_cfg)
        summary_cfg = {
            'var_list': [v for v in tf.trainable_variables() if v.name.startswith(args.weight_prefix) or 'conv1' in v.name]
        }
        self.tracker.build_summary(config=summary_cfg)

        # logging validation
        self.val_scope = 'val'
        if args.run_val:
            val_scope = self.val_scope
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
                self.val_tracker = models.Re3_Tracker(tf_input, tf_label, tf_prev_state,
                                                      lstm_size=args.lstm_size, config=tracker_cfg)
                # restore latest ckpt to val model (under scope 'val')
                val_vars = list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=val_scope))
                restore_dict = dict(zip([v.name.strip(val_scope + '/').split(':')[0] for v in val_vars], val_vars))

                val_cfg = {
                    'summary': {'placeholder': val_rst_placeholders, 'op': summary_op}
                }
                self.val_model = models.Val_Model(self.sess, self.val_tracker, self.val_feeder, self.val_recorder,
                                                  var_dict=restore_dict, config=val_cfg)
                self.val_model.inference = lambda x, y: self.tracker.inference(x, y, self.sess)

    def _link_feeder_tracker_dependency(self):
        args = self.args
        self.tracker.encode_bbox = self.train_feeder.encode_bbox_to_img
        self.tracker.decode_bbox = self.train_feeder.decode_bbox
        self.train_feeder.config['model'] = self.tracker
        if args.run_val:
            self.val_tracker.encode_bbox = self.train_feeder.encode_bbox_to_img
            self.val_tracker.decode_bbox = self.train_feeder.decode_bbox

    def _finalize_sess(self):
        # create saver at the last & finalize graph
        args = self.args
        model_var = [v for v in tf.trainable_variables() if not v.name.startswith(self.val_scope + '/')]
        self.saver = tf.train.Saver(model_var)

        # initialize
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # restore
        self.global_step = 0
        if self.is_training:  # already in training
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            if ckpt and ckpt.model_checkpoint_path:  # restore from saved ckpt
                self.global_step = int(ckpt.model_checkpoint_path.split('-')[-1])
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise FileNotFoundError
        elif args.restore:
            self.global_step = models.tfops.restore_from_dir(self.sess, args.restore_dir)
            args.restore = False  # in case restore from init point again

        # finalize
        self.sess.graph.finalize()
        self.summary_writer = tf.summary.FileWriter(os.path.join(args.summary_dir, args.model_name), self.sess.graph)

    def prepare_train(self, num_unrolls, batch_size, dummy_summary=False):
        # prepare training: (re-)construct feeder & model
        args = self.args
        if self.is_training:  # reset
            if args.use_parallel:  # shutdown worker
                self.paral_feeder.shutdown()
            self.train_feeder.reset(num_unrolls, batch_size)  # reset dataset
            if args.use_parallel:  # renew parallel utility
                self.paral_feeder.refresh(feeder=self.train_feeder)
            self.data_feeder = self.paral_feeder if args.use_parallel else self.train_feeder

            if args.use_tfdataset:  # reconstruct tf graph
                self.sess.close()
                tf.reset_default_graph()
                self.sess = models.tfops.Session()  # new sess
                self.construct_model()
                self._link_feeder_tracker_dependency()  # relink with new tracker
                self._finalize_sess()

        else:  # newly construct
            self.sess = models.tfops.Session()
            self.construct_feeder(num_unrolls, batch_size)
            self.construct_model()
            self._link_feeder_tracker_dependency()
            self._finalize_sess()
            self.is_training = True
            if dummy_summary:
                feed_dict = {
                    self.tracker.tf_input: np.zeros((1, 2, 227, 227, self.tracker.tf_input.get_shape().as_list()[-1])),
                    self.tracker.tf_label: np.zeros((1, 2, self.tracker.tf_label.get_shape().as_list()[-1])),
                    # self.tracker.prev_state: self.tf_init_state(1),
                }
                self.summary_writer.add_summary(self.sess.run(self.tracker.summary['all'], feed_dict=feed_dict))
                self.summary_writer.flush()

    @staticmethod
    def display_img_pred_label_double(track_img, label_box, track_pred):
        raise NotImplementedError

    @staticmethod
    def display_img_pred_label_single(track_img, label_box, track_pred):
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

    # routine to update model
    def run_train_step(self, feed_dict, global_step, display=False):
        op_to_run = [self.train_step]

        # record summary
        if global_step % 5000 == 0:  # conv, lstm, loss => all summary
            op_to_run += [self.tracker.summary['all']]
        elif global_step % 1000 == 0:  # lstm, loss
            # op_to_run += [self.tracker.summary['lstm'], self.tracker.summary['loss']]
            pass
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
            self.saver.save(self.sess, self.ckpt_path, global_step=global_step)
            self.summary_writer.flush()
            sys.stdout.flush()
            if self.args.run_val:
                self.val_model.record_val(self.ckpt_path)

    def train_feeddict(self):
        # start training
        args = self.args
        start_time = time.time()
        try:
            # training strategy: initial unrolls=2, batch=64; then, unrolls*=2, batch/=2, till unroll=32 (batch=4)
            # stablize first few convs, then jointly train
            num_unrolls = 2
            batch_size = 2
            epoch = 2
            # allow only specified vars to train: to stablize (for 2 epoch, no re-config)
            self.prepare_train(num_unrolls, batch_size, dummy_summary=True)
            self.train_step = self.tracker.train_step
            print('training prepared')
            for ep_cnt in range(epoch):
                print('starting epoch', ep_cnt)
                for input_batch, label_batch in self.data_feeder.iterate_data():
                    feed_dict = {
                        self.tracker.tf_input: input_batch,
                        self.tracker.tf_label: label_batch,
                        # self.tracker.prev_state: self.tf_init_state(batch_size),
                    }
                    self.run_train_step(feed_dict, self.global_step, args.display)
                    self.global_step += 1

            # num_unrolls = 2
            # batch_size = 64
            # while num_unrolls <= 32:
            #     self.prepare_train(num_unrolls, batch_size)
            #     for input_batch, label_batch in self.data_feeder.iterate_data():
            #         run_train_step(input_batch, label_batch, self.global_step, args.display)
            #         self.global_step += 1
            #     num_unrolls *= 2
            #     batch_size /= 2

            # save the lastest model
            self.saver.save(self.sess, self.ckpt_path, global_step=self.global_step)
            self.summary_writer.flush()
            sys.stdout.flush()

        except:  # save on unexpected termination
            if self.paral_feeder is not None:
                self.paral_feeder.shutdown()
            if not args.debug:
                print('saving...')
                checkpoint_file = os.path.join(args.log_dir, 'checkpoints', 'model.ckpt')
                self.saver.save(self.sess, checkpoint_file, global_step=self.global_step)
            raise

    def train_tfdataset(self):
        # start training
        args = self.args
        start_time = time.time()
        try:
            # training strategy: initial unrolls=2, batch=64; then, unrolls*=2, batch/=2, till unroll=32 (batch=4)
            # stablize first few convs, then jointly train
            num_unrolls = 2
            batch_size = 2
            epoch = 2
            # allow only specified vars to train: to stablize (for 2 epoch, no re-config)
            self.prepare_train(num_unrolls, batch_size, dummy_summary=True)
            self.train_step = self.tracker.train_step
            feed_dict = {}
            print('training prepared')
            for ep_cnt in range(epoch):
                self.sess.run(self.tf_dataset_iter.initializer)
                print('starting epoch', ep_cnt)
                while True:
                    try:
                        self.run_train_step(feed_dict, self.global_step, args.display)
                        self.global_step += 1
                    except:
                        break
                
            # num_unrolls = 2
            # batch_size = 64
            # while num_unrolls <= 32:
            #     self.prepare_train(num_unrolls, batch_size)
            #     for input_batch, label_batch in self.data_feeder.iterate_data():
            #         run_train_step(input_batch, label_batch, self.global_step, args.display)
            #         self.global_step += 1
            #     num_unrolls *= 2
            #     batch_size /= 2

            # save the lastest model
            self.saver.save(self.sess, self.ckpt_path, global_step=self.global_step)
            self.summary_writer.flush()
            print('training done')
            sys.stdout.flush()

        except:  # save on unexpected termination
            if self.paral_feeder is not None:
                self.paral_feeder.shutdown()
            if not args.debug:
                print('saving on unexpected termination...')
                self.saver.save(self.sess, self.ckpt_path, global_step=self.global_step)
            raise