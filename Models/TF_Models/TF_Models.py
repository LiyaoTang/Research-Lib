#!/usr/bin/env python
# coding: utf-8
'''
module: classes to construct TF models
'''

import types
import numpy as np
import tensorflow as tf

from . import TF_Ops as tfops
from . import TF_Modules as tfm

class TF_Model(object):
    '''
    base class to config TF model with common options
    '''
    def __init__(self, class_num, tf_input, tf_label, tf_phase, config_dict={}):
        super(TF_Model, self).__init__()
        
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


class FCN_Pipe(TF_Model):
    '''
    construct a FCN-pipe (FCN with no downsampling) model
    '''
    def __init__(self, conv_struct, class_num, tf_input, tf_label, tf_phase, config_dict={}):
        super(FCN_Pipe, self).__init__(class_num=class_num, tf_input=tf_input, tf_label=tf_label, tf_phase=tf_phase,
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


class Unet(FCN_Pipe):
    '''
    construct a FCN-pipe (FCN with no downsampling) model
    '''
    def __init__(self, conv_struct, class_num, tf_input, tf_label, tf_phase, config_dict={}):
        super(Unet, self).__init__(class_num=class_num, tf_input=tf_input, tf_label=tf_label, tf_phase=tf_phase,
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
    replicate & extend the re3 tracking model from paper:
    Re3 : Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects (https://arxiv.org/abs/1705.06368)
    '''
    def __init__(self, tf_input, tf_label, prev_state=None, feeder=None, lstm_size=512, config={}):
        '''
        tf_input: [batch, time, img_h, img_w, img_channel] for bbox_encoding in ['mask', 'mesh'] => cur img + mask
                  [batch, time, 2, img_h, img_w, img_channel] for bbox_encoding in ['corner', 'center'] => one cur crop, one prev
        tf_label: [batch, time, 4] (x,y,w,h)
        '''
        self.tf_input = tf_input
        self.tf_label = tf_label
        self.prev_state = tuple(prev_state) if prev_state is not None else None
        self.lstm_size = lstm_size
        
        self.learning_rate = None
        self.train_var_list = None
        self._var_scope = tf.get_variable_scope().name
        self.get_trainable_vars = lambda: tf.trainable_variables(self._var_scope)
        
        self.encode_bbox = feeder.encode_bbox_to_img if feeder else None
        self.decode_bbox = feeder.decode_bbox if feeder else None

        self.config = config  # data & encoding associated configuration
        assert config['attention'] in ['hard', 'soft', 'soft_fuse']
        assert config['bbox_encoding'] in ['mask', 'mesh', 'crop']  # mesh/mask: no crop; crop: crop
        assert config['unroll_type'] in ['manual', 'dynamic']  # manual: one step a time; dynamic: unroll the whole sequence
        assert config['label_type'] in ['corner', 'center']  # corner: xyxy; center: xywh
        assert config['label_norm'] in ['fix', 'dynamic', 'raw']  # fix: /2270; dynamic: /img_shape; raw: no division
        assert config['fuse_type'] in ['flat', 'spp', 'resize']

        imgnet_mean = [123.151630838, 115.902882574, 103.062623801]
        with tf.variable_scope('preprocess'):
            shape_tensor = tf.shape(self.tf_input)
            shape_list = self.tf_input.get_shape().as_list()
            input_shape = [shape_list[i] if shape_list[i] is not None else shape_tensor[i] for i in range(len(shape_list))]
            batch_size = input_shape[0]
            num_unrolls = input_shape[1]
            img_size = input_shape[-3:]

            imgnet_mean += [0] * (img_size[-1] - 3)  # fill 0 to fit channel_size
            self.net = self.tf_input - imgnet_mean
            self.net = tf.reshape(self.net, (-1, img_size[0], img_size[1], img_size[2]))  # [-1, img_h, img_w, img_channel]
            if self.config['bbox_encoding'] in ['mask', 'mesh']:  # prepare mask: tf_img contain img & mask
                with tf.variable_scope(self.config['attention']):

                    if self.config['attention'] == 'hard':  # direct embed 0-1 attention
                        mask = tfops.conv_layer(self.net[..., 3:4], 96, 11, 4, padding='VALID', activation=None, scope='mask')
                        auxilary_input = mask
                        if self.config['bbox_encoding'] == 'mesh':
                            mesh = tfops.conv_layer(self.net[..., 4:], 96, 11, 4, padding='VALID', activation=None, scope='mesh')
                            auxilary_input = tf.concat([mask, mesh], axis=-1)
                        self.net = self.net[...,:3]
                        
                    elif self.config['attention'] == 'soft':  # generate attention by 1x1 conv
                        mask = tfops.conv_layer(self.net, 1, 3, 1, padding='SAME', scope='gen_mask')
                        print(mask)
                        mask = tfops.conv_layer(mask, 96, 11, 4, padding='VALID', activation=None, scope='mask')
                        print(mask)
                        auxilary_input = mask
                        if self.config['bbox_encoding'] == 'mesh':  # concat mesh
                            mesh = tfops.conv_layer(self.net[..., 4:], 96, 11, 4, padding='VALID', activation=None, scope='mesh')
                            auxilary_input = tf.concat([mask, mesh], axis=-1)
                        self.net = self.net[..., :3]
                        print(self.net)

                    else:  # soft_fuse: generate (0,1) attention & fuse onto original input by element-size product
                        mask_3 = tfops.conv_layer(self.net, 3, 3, 1, padding='SAME', activation=None, scope='mask_3')
                        mask_1 = tfops.conv_layer(self.net, 3, 1, 1, padding='SAME', activation=None, scope='mask_1')
                        mask = tf.nn.softmax(tf.mask, axis=[-3, -2])  # [-1, img_h, img_w, 1]
                        with tf.variable_scope('fuse'):
                            mask = tf.tile(mask, [1, 1, 1, img_size[-1]])  # [-1, img_h, img_w, img_channel]
                            self.net = self.net * mask  # apply attention onto rgb channel (broadcast to batch)
                        if self.config['bbox_encoding'] == 'mesh':
                            auxilary_input = tfops.conv_layer(self.net[..., 3:], 96, 11, 4, padding='VALID', activation=None, scope='mesh')
            else:  # cropping
                img_size = [227, 227, 3]
                auxilary_input = None
                self.config['fuse_type'] = 'flat'  # other settings not allowed

        with tf.variable_scope('re3'):
            self.net = tfm.alexnet_conv_layers(self.net, auxilary_input=auxilary_input, fuse_type=config['fuse_type'])  # [-1, feat]

            # late fusion of conv features
            feat_len = self.net.get_shape().as_list()[-1]
            if self.config['bbox_encoding'] in ['mask', 'mesh']:
                self.net = tf.reshape(self.net, [-1, feat_len])  # reshaped back to [batch, time, feat]
                # late fusion: concat feature from t, t-1
                first_frame = self.net[:, 0:1, ...]
                prev_frame = self.net[:, :-1, ...]
                past_frames = tf.concat([first_frame, prev_frame], axis=1)  # [batch, time, feat]
                self.net = tf.concat([self.net, past_frames], axis=-1)  # [batch, time, feat_t + feat_t-1]
            else:
                self.net = tf.reshape(self.net, [-1, num_unrolls, 2, feat_len])  # [batch, time, 2, feat]
                self.net = tfops.remove_axis(self.net, 3)  # [batch, time, feat_t + feat_t-1]

            with tf.variable_scope('fc6'):
                feat_len = self.net.get_shape().as_list()[-1]
                self.net = tf.reshape(self.net, [-1, feat_len])
                feat_len = 1024
                self.net = tfops.dense_layer(self.net, feat_len, weight_name='W_fc', bias_name='b_fc')  # [-1,feat=1024]
                self.net = tf.reshape(self.net, [-1, num_unrolls, feat_len]) # [batch, time, feat=1024]

            if self.config['unroll_type'] == 'manual':
                raise NotImplementedError
            else:
                # output as bbox regress: [batch, time, 4]
                self.net, state = tfm.re3_lstm_tracker(self.net, num_unrolls, batch_size, lstm_size=self.lstm_size, prev_state=self.prev_state)
        self.lstm_state = tuple([*state[0], *state[1]])
        if self.prev_state is None:  # bypass placeholder (mask tensor by numpy value in inference)
            self.prev_state = self.lstm_state

        self.logits = self.net  # [batch, time, 4], 4=encoded xywh/xyxy

        with tf.variable_scope('pred'):
            if self.config['label_norm'] == 'raw':
                pred = self.logits
            elif self.config['label_norm'] == 'fix':
                pred = self.logits * 2270
            else:  # dynamic
                multiplier = tf.gather_nd(self.img_size, tf.constant([[1], [0], [1], [0]]))  # tensor [img_w, img_h, img_w, img_h]
                pred = self.logits * multiplier
            self.pred = pred  # [batch, time, 4], 4=xywh/xyxy

    def _build_loss(self, l2_var_list=None):
        if not hasattr(self, 'loss'):
            with tf.variable_scope('loss'):
                diff = tf.reduce_sum(tf.abs(self.pred - self.tf_label, name='diff'), axis=-1)  # L1 loss
                self.reg_loss = tf.reduce_mean(diff, name='loss')
                self.l2_loss = tfops.l2_regularization(var_list=l2_var_list, scope='l2_weight_penalty')
                self.loss = self.reg_loss + self.l2_loss
        
    def build_summary(self, config={}):
        if hasattr(self, 'summary'):
            return
        var_list = config['var_list'] if 'var_list' in config else (self.train_var_list if self.train_var_list else self.get_trainable_vars())
        self._build_loss(var_list)

        self.summary = {'loss': None, 'conv': None, 'lstm': None, 'all': None}
        with tf.variable_scope('summaries'):
            loss_summary = [tf.summary.scalar('reg_loss', self.reg_loss),
                            tf.summary.scalar('l2_loss', self.l2_loss),
                            tf.summary.scalar('full_loss', self.loss)]
            self.summary['loss'] = tf.summary.merge(loss_summary)

            conv_vars = [v for v in var_list if 'conv' in v.name and 'W_conv' in v.name and
                         (v.get_shape().as_list()[0] != 1 or v.get_shape().as_list()[1] != 1)]
            conv_summary = [tfops.get_conv_summaries(var, scope=var.name.replace('/', '_')[:-2]) for var in conv_vars]
            conv_summary = [s for s in conv_summary if s is not None]
            self.summary['conv'] = tf.summary.merge(conv_summary) if conv_summary else None

            lstm_vars = [var for var in var_list if 'lstm1' in var.name or 'lstm2' in var.name]
            lstm_summary = [tfops.get_summary(var, var.name[:-2]) for var in lstm_vars]
            self.summary['lstm'] = tf.summary.merge(lstm_summary) if lstm_summary else None
        self.summary['all'] = tf.summary.merge_all()
        print(self.summary)

    def build_train_step(self, config={}):
        var_list = config['var_list'] if 'var_list' in config else self.get_trainable_vars()
        l2_var_list = config['l2_var_list'] if 'l2_var_list' in config else var_list
        learning_rate = config['learning_rate'] if 'learning_rate' in config else tf.placeholder(tf.float32)

        self._build_loss(l2_var_list)
        renew = config['renew'] if 'renew' in config else False
        if not hasattr(self, 'train_step') or renew:
            self.train_var_list = var_list
            self.learning_rate = learning_rate
            # TODO: TF 2.0: optimizer = tf.keras.optimizers.Adam(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            with tf.device('/cpu:0'):  # create cnt on cpu
                global_step = tf.train.get_or_create_global_step()
            self.train_step = optimizer.minimize(self.loss, global_step=global_step, var_list=var_list)
            # varlist_dict = {
            #     'preprocess': [v for v in var_list if 'preprocess' in v.name],
            #     'conv1': [v for v in var_list if 'preprocess' in v.name or 'conv1' in v.name],
            # }
            # self.stablize_step = {}
            # for k, v_list in varlist_dict.items():
            #     self.stablize_step[k] = optimizer.minimize(self.loss, global_step=global_step, var_list=v_list)

    def inference(self, track, bboxes, sess, display_func=None):
        '''
        given a single track, output inferenced track result
        '''
        prev_state = tuple([np.zeros((1, self.lstm_size)) for _ in range(4)])
        out_bbox = [bboxes[0]]  # the initial box
        prev_input = None
        for img in track:
            if display_func:
                display_func(img, out_bbox[-1])
            cur_input = self.encode_bbox(img, out_bbox[-1])  # encode the prev bbox onto current input (cropping, masking, etc.)
            if self.config['bbox_encoding'] == 'crop' and prev_input is None:
                cur_input = (prev_input, cur_input)
            
            feed_dict = {
                self.tf_input: [[cur_input]], # batch size 1, single step
                self.prev_state: prev_state,
            }
            pred, prev_state = sess.run([self.pred, self.lstm_state], feed_dict=feed_dict)
            prev_state = tuple(prev_state)

            out_bbox.append(self.decode_bbox(out_bbox[-1], pred))  # record bbox under whole img coord
            if self.config['bbox_encoding'] == 'crop':
                prev_input = cur_input
        return out_bbox


class Val_Model(object):
    '''
    run validation for model in a separate thread
    '''
    def __init__(self, sess, model, feeder, recorder, var_dict, config={}):
        '''
        sess: current active session
        model: a graph same as & separate from the one being trained
        var_dict: ckpt var -> var in model for validation
        '''
        self.threading = __import__('threading', fromlist=[''])
        self.sess = sess
        self.model = model
        self.feeder = feeder
        self.recorder = recorder
        self.saver = tf.train.Saver(var_dict)
        self.config = config
        self.eval_cnt = 0

    def inference(self, cur_input, cur_label):
        '''
        get the model inference
        '''
        raise NotImplementedError

    def record_val(self, ckpt_path, global_step=None):
        '''
        run the model inference and record the result
        '''
        self.saver.restore(self.sess, ckpt_path)
        for cur_input, cur_label in self.feeder.iterate_data():
            cur_pred = self.inference(cur_input, cur_label)
            self.recorder.accumulate_rst(cur_label, cur_pred)

        if 'print' in self.config and self.config['print']: # print out
            if global_step is None:
                global_step = self.eval_cnt
            print('====> evaluation %d :', global_step)
            self.recorder.print_result()

        if 'summary' in self.config:  # write to tf summary if provided
            rst_record = self.recorder.get_result()
            feed_dict = {}
            for n in rst_record:
                tf_placeholder = self.config['summary']['placeholder'][n]
                feed_dict[tf_placeholder] = rst_record[n]
            tf_summary_op = self.config['summary']['op']
            cur_summary = self.sess.run([tf_summary_op], feed_dict=feed_dict)

            if global_step is None:
                global_step = self.eval_cnt 
            self.config['summary']['writer'].add_summary(cur_summary, global_step=self.eval_cnt)
            self.config['summary']['writer'].flush()
        self.eval_cnt += 1