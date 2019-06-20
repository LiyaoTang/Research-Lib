#!/usr/bin/env python
# coding: utf-8
'''
module: base class to construct TF models, using tf.__version__ 1.4.2
'''

import types
import numpy as np
import tensorflow as tf

from . import TF_Ops as tfops
from . import TF_Modules as tfm

class TF_Constructer(object):
    '''
    base class to config TF model with common options
    '''
    def __init__(self, class_num, tf_input, tf_label, tf_phase, config_dict={}):
        super(TF_Constructer, self).__init__()
        
        self.class_num = class_num
        self.tf_input = tf_input
        self.tf_label = tf_label
        self.tf_phase = tf_phase
        self.verbose = 'verbose' in config_dict and config_dict['verbose']

        # assertion
        assert tf_label.get_shape().as_list()[-1] == class_num  # assert to be one-hot
        
        # legal value chk & set
        for argn, legal, default in zip(['regularizer_type', 'weighted_loss', 'batchnorm', 'record_summary', 'loss_type'],
                                        [('', 'L1', 'L2'), ('', 'bal'), (True, False), (True, False), 'xen, crf'],
                                        ['', '', False, True, 'xen']):
            if argn in config_dict:
                cur_val = config_dict[argn]
                if cur_val not in legal:
                    raise ValueError('supported values for args %s are %s, but received %s' % (argn, str(legal), str(cur_val)))
            else:
                config_dict[argn] = default

        # assert leraning rate
        if 'learning_rate' in config_dict:
            assert config_dict['learning_rate'] > 0
        else:
            config_dict['learning_rate'] = 1e-5

        self.config = config_dict
        self._build_preprocess()

    def _build_preprocess(self):
        # build initial net
        # normalization of input
        self.net = self.tf_input  # default to raw input if not defined or all empty
        if 'norm_params' in self.config:
            with tf.variable_scope('Preprocess'):
                norm_params = self.config['norm_params']
                if 'mean' in norm_params:
                    self.net = self.net - norm_params['mean']
                if 'std' in norm_params:
                    self.net = self.net / norm_params['std']

    def print_attributes(self):
        '''
        print all attributes
        '''
        for argn in dir(self):
            arg = getattr(self, argn)
            if not argn.startswith('_') and not isinstance(arg, types.BuiltinFunctionType):
                print('%s = %s' % (argn, str(arg)))

class FCN_Pipe_Constructer(TF_Constructer):
    '''
    construct a FCN-pipe (FCN with no downsampling) model
    '''
    def __init__(self, conv_struct, class_num, tf_input, tf_label, tf_phase, config_dict={}):
        super(FCN_Pipe_Constructer, self).__init__(class_num=class_num, tf_input=tf_input, tf_label=tf_label, tf_phase=tf_phase,
                                                   config_dict=config_dict)

        # parse conv_struct: e.g. 3-16;5-8;1-32 | 3-8;1-16 | 1
        # => concat[ 3x3 out_channel=16, 5x5 out_channel=8, 1x1 out_channel=32]
        # => followed by inception concat [3x3 out_channel=8, 1x1 out_channel=16] and so on ...
        # => output with a 1x1 conv
        # note: size must be specified for the kernel at output (logits) layer 
        conv_struct = [[[int(x) for x in config.split('-')] for config in layer.split(';')] for layer in conv_struct.split('|')]
        assert len(conv_struct[-1]) == 1 and len(conv_struct[-1][0]) == 1  # assert the kernel at output layer is given
        conv_struct[-1][0].append(class_num)  # output vector with dimension of class_num
        self.conv_struct = conv_struct

        assert self.config['loss_type'] in ['crf', 'xen']
        self._build_graph()

    def _build_graph(self):
        self._build_pipe()
        self._build_logits()
        self._build_loss()
        self._build_output()
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name)
        # print('=================================================================================')
        if self.config['record_summary']:
            self._build_summary()
        
    def _build_pipe(self, net=None, conv_struct=None, scope='pipe'):
        with tf.variable_scope(scope):
            # hidden layer
            if net is None:
                net = self.net
            if conv_struct is None:
                conv_struct = self.conv_struct
            if len(conv_struct) > 1: # if any hidden layer
                for layer_cnt in range(len(conv_struct) - 1):
                    layer_cfg = conv_struct[layer_cnt]

                    with tf.variable_scope('incep_%d' % layer_cnt):
                        # kernel/bias initializer: default to xavier/zeros
                        if len(layer_cfg) > 1:
                            net = tf.concat([tf.layers.conv2d(inputs=net, kernel_size=cfg[0], filters=cfg[1], padding='same',
                                                              activation=tf.nn.relu, kernel_regularizer=self.config['regularizer'],
                                                              name='conv%d-%d' % (cfg[0], cfg[1]))
                                            for cfg in layer_cfg], axis=-1)
                        else:
                            cfg = layer_cfg[0]
                            net = tf.layers.conv2d(inputs=net, kernel_size=cfg[0], filters=cfg[1], strides=1, padding='same',
                                                   activation=tf.nn.relu, kernel_regularizer=self.config['regularizer'],
                                                   name='conv%d-%d' % (cfg[0], cfg[1]))

                        # If anyone has come across any instances where BN before ReLU does better than BN after ReLU, 
                        # please do share with me as I have yet to come across any such instance.
                        if self.config['batchnorm']:
                            net = tf.layers.batch_normalization(net, training=tf.equal(self.tf_phase, 'train'), name='bn')
        self.net = net
        
    def _build_logits(self):
        # logits
        # no non-linearity for last layer, ref: https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt
        self.logits = tf.layers.conv2d(inputs=self.net, kernel_size=self.conv_struct[-1][0], filters=self.class_num,
                                       activation=None, strides=1, padding='same', name='logits')

    def _build_loss(self):
        # loss
        with tf.variable_scope('loss'):
            # logits & label should be of the same shape
            flat_logits = tf.reshape(self.logits, (-1, self.class_num), name='flat_logits')
            flat_labels = tf.reshape(self.tf_label, (-1, self.class_num), name='flat_labels')

            # construct flat weight
            tf_weights = dict(zip(['train', 'val', 'test'], [1] * 3))
            if self.config['weighted_loss'] == 'bal':
                for k in tf_weights.keys():
                    cur_class_weight = tf.expand_dims(tf.constant(self.config['weight'][k], dtype=tf.float32), 0)
                    cur_flat_weight = tf.squeeze(tf.matmul(cur_class_weight, tf.cast(flat_labels, tf.float32), transpose_b=True), 0)
                    tf_weights[k] = cur_flat_weight
            
            self.loss = dict()
            if self.config['loss_type'] == 'crf':
                # TODO: change crf from loss_type into postprocess
                with tf.variable_scope('crf_log_likelihood'):
                    batch_shape = tf.shape(self.logits)
                    batch_size = batch_shape[0]
                    batch_exp_len = tf.reduce_prod(batch_shape[1:-1])
                    flat_logits = tf.reshape(self.logits, (batch_size, batch_exp_len, self.class_num))
                    flat_labels = tf.argmax(tf.reshape(self.tf_label, (batch_size, batch_exp_len, self.class_num)),
                                            axis=-1)  # requie k-class label in [batch_size, seq_len]

                    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(flat_logits,
                                                                                     tf.cast(flat_labels, tf.int32),
                                                                                     tf.fill([batch_size],
                                                                                             batch_exp_len))
                    self.trans_params = trans_params

                    for k in tf_weights.keys():
                        self.loss[k] = tf.reduce_mean(-log_likelihood*tf_weights[k], name='loss') # mean error over batchß
            else:
                for k in tf_weights.keys():
                    cur_loss = tf.losses.softmax_cross_entropy(flat_labels, flat_logits, weights=tf_weights[k], scope='softmax_xen_%s' % k)
                    self.loss[k] = cur_loss

            # assertion for interface
            assert type(self.loss) is dict and all([k in self.loss.keys() for k in ['train', 'val', 'test']])

        # train step
        # ensures we execute the update_ops before performing the train_step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.config['learning_rate']).minimize(self.loss['train'])

    def _build_output(self):
        # output
        with tf.variable_scope('pred'):
            if self.config['loss_type'] == 'crf':
                batch_shape = tf.shape(self.logits)
                batch_size = batch_shape[0]
                batch_exp_len = tf.reduce_prod(batch_shape[1:-1])
                flat_logits = tf.reshape(self.logits, (batch_size, batch_exp_len, self.class_num))

                # decoded into [batch_size, seq_len] with k-class encoding
                prediction, sentence_scores = tf.contrib.crf.crf_decode(flat_logits,
                                                                        self.trans_params,
                                                                        tf.fill([batch_size],
                                                                                batch_exp_len))
                onehot_pred = tf.one_hot(prediction, self.class_num, axis=-1)
                self.pred = tf.reshape(onehot_pred, tf.shape(self.logits), name='pred')
            else:
                self.pred = tf.nn.softmax(self.logits, name='pred')

    def _build_summary(self):
        if self.config['record_summary']:
            with tf.variable_scope('summary'):
                conv_struct = self.conv_struct

                target_tensors = []
                # conv layers params
                if conv_struct != [[[0]]]:
                    for layer_cnt in range(len(conv_struct) - 1):
                        layer_cfg = conv_struct[layer_cnt]

                        for cfg in layer_cfg:
                            target_tensors.extend(['pipe/incep_%d/conv%d-%d/%s' % (layer_cnt, cfg[0], cfg[1], t) for t in ['kernel:0', 'bias:0']])
                        if self.config['batchnorm']:
                            target_tensors.extend(['pipe/incep_%d/bn/%s' % (layer_cnt, t) for t in ['gamma:0', 'beta:0']])
                # logits layer params
                target_tensors.extend(['logits/%s' % t for t in ['kernel:0', 'bias:0']])

                graph = tf.get_default_graph()
                for tensor_name in target_tensors:
                    cur_tensor = graph.get_tensor_by_name(tensor_name)
                    tensor_name = tensor_name.split(':')[0]
                    tf.summary.histogram(tensor_name, cur_tensor)
                    tf.summary.histogram('grad_'+tensor_name, tf.gradients(self.loss['train'], [cur_tensor])[0])

                # loss
                tf.summary.scalar('train_loss', self.loss['train'])
                tf.summary.scalar('val_loss', self.loss['val'])

                self.merged_summary = tf.summary.merge_all()


class Unet_Constructer(FCN_Pipe_Constructer):
    '''
    construct a FCN-pipe (FCN with no downsampling) model
    '''
    def __init__(self, conv_struct, class_num, tf_input, tf_label, tf_phase, config_dict={}):
        super(Unet_Constructer, self).__init__(class_num=class_num, tf_input=tf_input, tf_label=tf_label, tf_phase=tf_phase,
                                               config_dict=config_dict)
        # parse conv_struct: e.g. 3-8;5-8=1-8 | 3-16=1-16 | 3-32=5-8;1-8=3-2
        # => a pipe of inception [ 3x3 out_channel=8, 5x5 out_channel=8 ] -> 1x1 out_channel=8
        # => followed by 2x2 pooling (downsampling)
        # => followed by a tunnel pipe of 3x3 out_channel=16 -> 1x1 out_channel=16]
        # => followed by 2x2 upsampling (unpooling/deconv), with feature map before the corresponding pooling opt concatenated
        # => followed by a pipe of 3x3 out_channel=32 -> inception [5x5 out_channel=8, 1x1 out_channel=8]
        # => the final pipe output logits with -> 3x3 out_channel=2
        self.conv_struct = self._parse_struct(conv_struct)

        # parse config for upsampling
        up_cfg = config_dict['upsample_config'] if 'upsample_config' in config_dict else {}
        self._parse_upsample_config(up_cfg)
        down_cfg = config_dict['downsample_config'] if 'downsample_config' in config_dict else {}
        self._parse_downsample_config(down_cfg)

    def _parse_struct(self, conv_struct):
        conv_struct = [[[(int(n) for n in conv.split('-'))
                          for conv in layer.split(';')]
                          for layer in pipe.split('=')]
                          for pipe in conv_struct.split('|')]

        assert len(conv_struct) % 2 == 1  # even '|' to be fully specified
        assert len(conv_struct[-1][-1]) == 1  # no incep allowed in logit layer
        assert conv_struct[-1][-1][-1][1] == self.class_num  # logit output channel = class num
        return conv_struct

    def _parse_upsample_config(self, config):
        for k, valid_v in zip(['type', 'init'],
                                [('max_unpool', 'avg_unpool', 'deconv'), ('xavier', 'bilinear')]):
            if k in config:
                assert config[k] in valid_v
            else:
                config[k] = valid_v[0]  # 1st val as the default

        if config['type'] == 'max_unpool':
            upsample = tfops.max_unpooling
        elif config['type'] == 'avg_unpool':
            upsample = tfops.average_unpooling
        else:  # deconv needs init
            if config['init'] == 'bilinear':
                init = lambda factor, in_ch, out_ch: tf.constant_initializer(self._get_bilinear_weights(factor, in_ch, out_ch))
            else:
                init = lambda factor, in_ch, out_ch: tf.layers.xavier_initializer()
            upsample = lambda factor, in_ch, out_ch: tf.layers.conv2d_transpose(weight_initializer=init(factor, in_ch, out_ch))

        self.upsample = upsample

    def _parse_downsample_config(self, config):
        for k, valid_v in zip(['type'],
                                [('max', 'avg')]):
            if k in config:
                assert config[k] in valid_v
            else:
                config[k] = valid_v[0]  # 1st val as the default

        if config['type'] == 'max':
            downsample = lambda net, ksize=[2, 2], strides=[1, 1], padding='same': tf.layers.max_pooling2d(net, ksize, strides, padding)
        else:
            downsample = lambda net, ksize=[2, 2], strides=[1, 1], padding='same': tf.layers.average_pooling2d(net, ksize, strides, padding)

        self.downsample = downsample

    @staticmethod
    def _get_deconv_kernel_size(factor):
        return 2 * factor - factor % 2

    @staticmethod
    def _get_biliner_kernel(size):
        '''
        make a 2d bilinear (square) kernel suitable for upsampling of the given size
        '''
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    def _get_bilinear_weights(self, factor, in_channel, out_channel):
        '''
        create weights matrix for transposed convolution with bilinear filter
        initialization.
        '''        
        filter_size = self._get_deconv_kernel_size(factor)

        weights = np.zeros((filter_size, # height
                            filter_size, # width
                            out_channel,
                            in_channel), dtype=np.float32)
        
        upsample_kernel = self._get_biliner_kernel(filter_size)
        
        for in_idx in range(in_channel):
            for out_idx in range(out_channel):
                weights[:, :, out_idx, in_idx] = upsample_kernel

        return weights

    def _build_downsampling(self):
        downsample_track = []
        with tf.variable_scope('downsampling'):
            for pipe in self.conv_struct[: int(len(self.conv_struct) / 2) + 1]:
                print(pipe, pipe + [0])
                self._build_pipe(conv_struct=pipe + [0])  # as the last layer in conv_struct interpreted as the logits here
                downsample_track.append(self.net)
                self.net = self.downsample(self.net)
        return downsample_track

    def _build_tunnel(self):
        pipe = self.conv_struct[int(len(self.conv_struct) / 2) + 1]
        self._build_pipe(conv_struct=pipe + [0], scope='tunnel')

    def _build_upsampling(self, downsample_track):
        with tf.variable_scope('upsampling'):
            for pipe, counter_pipe in zip(self.conv_struct[int(len(self.conv_struct) / 2) + 1 :-1],
                                          reversed(downsample_track)):
                self.net = self.upsample(self.net)
                self.net = tf.concat([counter_pipe, self.net], axis=-1)
                self._build_pipe(conv_struct=pipe + [0])

        with tf.variable_scope('logits'):
            self.net = self.upsample(self.net)
            self.net = tf.concat([downsample_track[0], self.net], axis=-1)
            self._build_pipe(conv_struct=self.conv_struct[-1] + [0], scope=None)
    
    def _build_graph(self):
        downsample_track = self._build_downsampling()
        self._build_tunnel()
        self._build_upsampling(downsample_track)

        self._build_loss()
        self._build_output()
        if self.verbose:
            for n in tf.get_default_graph().as_graph_def().node:
                print(n.name)
            print('=================================================================================')
        if self.config['record_summary']:
            self._build_summary()


class Re3_Tracker(object):
    '''
    replicate the re3 tracking model, original paper:
    Re3 : Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects (https://arxiv.org/abs/1705.06368)
    '''
    def __init__(self, tf_img, tf_label, num_unrolls=2, img_size=227, lstm_size=512,
                 unroll_type='dynamic', bbox_encoding='mask'):
        '''
        tf_img: [batch, time, img_h, img_w, img_channel]
        tf_label: [batch, time, 4] (x,y,w,h)
        '''
        self.tf_img = tf_img
        self.tf_label = tf_label
        imgnet_mean = [123.151630838, 115.902882574, 103.062623801]

        assert bbox_encoding in ['mask', 'corner', 'center']  # mask: no crop; corner/center: crop
        assert unroll_type in ['manual', 'dynamic']  # manual: one step a time; dynamic: unroll the whole sequence
        
        with tf.variable_scope('preprocess'):
            if bbox_encoding == 'mask':  # prepare mask: tf_img contain img & mask
                self.net = tf_img[0] - imgnet_mean
                auxilary_input = tfops.conv_layer(tf_img[1], 96, 11, 4, padding='VALID', activation=None)
                use_spp = True
            else:  # cropping
                self.net = tf_img - imgnet_mean
                auxilary_input = None
                use_spp = False

        with tf.variable_scope('re3'):
            self.net = tfm.alexnet_conv_layers(self.net, auxilary_input=auxilary_input, use_spp=use_spp)  # [B, T, feat]
            if unroll_type == 'manual':
                raise NotImplementedError
            else:
                # late fusion: concat feature from t, t-1
                first_frame = self.net[:, 0:1, ...]
                other_frame = self.net[:, :-1, ...]
                past_frames = tf.concat([first_frame, other_frame], axis=1)  # [B,T,feat]
                self.net = tf.concat([self.net, past_frames], axis=-1)  # [B,T,feat_t-feat_t-1]
                self.net = tfm.re3_lstm_tracker(self.net, num_unrolls, lstm_size=512, prev_state=None)
        self.pred = self.net

        with tf.variable_scope('loss'):
            diff = tf.reduce_sum(tf.abs(self.pred - self.tf_label, name='diff'), axis=-1) # L1 loss
            loss = tf.reduce_mean(diff, name='loss')
            l2_reg = tfops.l2_regularization(scope='l2_weight_penalty')
            self.loss = loss + l2_reg

    def get_train_step(self, learning_rate):
        if not hasattr(self, 'train_step'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            with tf.device('/cpu:0'):  # create cnt on cpu
                global_step = tf.train.create_global_step()
            self.train_step = optimizer.minimize(self.loss, global_step=global_step, var_list=tf.trainable_variables(),
                                                colocate_gradients_with_ops=True)
        return self.train_step