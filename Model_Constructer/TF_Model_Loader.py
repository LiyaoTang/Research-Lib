#!/usr/bin/env python
# coding: utf-8
'''
module: utils to load pretrained tf models
'''

import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

'''Examples
pretrained_model = './external/tensorflow-deeplab-resnet/models/deeplab_resnet_init.ckpt'
prefix = 'encoder/visual_info/segnet/'

# create a dict of {tensor name in files : tensor in graph to store the weights}
# note: need to be name of op (no :0 at the end)
load_var = {var.op.name[len(prefix):]: var for var in tf.global_variables()
            if var.name.startswith(prefix) and not var.name[len(prefix):].startswith('conv1')}

snapshot_loader = tf.train.Saver(load_var)
sess = tf.Session()
snapshot_loader.restore(sess, pretrained_model)
'''

