#!/usr/bin/env python
# coding: utf-8
'''
module: construct TFRecord dataset given other formats
'''

import os
import re
import h5py
import numpy as np
import tensorflow as tf
from Data_Feeder.base import TFRecord_Constructer

class txtData_Constructor(TFRecord_Constructer):
    '''
    construct tf record given front-radar txt file
    '''
    def __init__(self, input_dir, output_file, img_shape_2D, class_num,
                 re_expr='dw_((?!_label).)*\.txt', recursive=True, use_one_hot=True):
        super(txtData_Constructor, self).__init__(input_dir, output_file, re_expr, recursive)

        self.img_shape_2D = tuple(img_shape_2D)
        assert len(self.img_shape_2D) == 2  # only need the length & height
        
        self.class_num = class_num
        self.use_one_hot = use_one_hot

    def _separate_coord_feature(self, point_arr):
        # assumed first 2 col is coord, the rest is features in input txt, label in label txt
        return np.array(point_arr[:,:2], dtype=int), point_arr[:, 2:]

    def _fill_in_image(self, coord_arr, feature_arr):
        actual_shape = list(self.img_shape_2D)
        actual_shape.append(feature_arr.shape[-1])

        image = np.zeros(actual_shape)
        for coord, feature in zip(coord_arr, feature_arr):
            image[coord[0], coord[1]] = feature # can not use np array as index for np array!
        # print(image.shape, actual_shape)
        return image

    def _traverse_input_image(self):
        for dirpath, name in self._traverse_file():
            input_path = os.path.join(dirpath, name)
            with open(input_path) as file:
                point_arr = np.array([[float(val) for val in line.split(' ')] for line in file.read().strip('\n').split('\n')])
            coord_arr, feature_arr = self._separate_coord_feature(point_arr)
            yield self._fill_in_image(coord_arr, feature_arr)

    def _traverse_label_array(self):
        for dirpath, name in self._traverse_file():
            label_path = os.path.join(dirpath, name.split('.')[0]+'_label.txt')
            with open(label_path) as file:
                point_arr = np.array([[int(val) for val in line.split(' ')] for line in file.read().strip('\n').split('\n')])
            _, label_arr = self._separate_coord_feature(point_arr)
            yield label_arr

    def _to_onehot(self, label_list):
        height = len(label_list)
        
        label = np.zeros((height, self.class_num), dtype=int)
        label[np.arange(height), label_list] = 1

        return label

    def collect_meta_data(self, norm_param_file=None):
        '''
        collect: mean, std, class ratio
        '''
        class_cnt = np.zeros(self.class_num)
        mean = 0
        std = 0
        img_cnt = 0

        # calculate mean
        for cur_img in self._traverse_input_image():
            mean = mean + cur_img
            img_cnt += 1
        mean = mean / float(img_cnt)
        print("mean[0,0] = ", mean[0, 0], " in ", mean.shape)

        # calculate std
        for cur_img in self._traverse_input_image():
            std = std + (cur_img - mean)**2
        std = np.sqrt(std / img_cnt)
        print('std[0,0] = ', std[0, 0], " in ", std.shape)
        
        self.mean = mean
        self.std = std

        # collect class ratio
        for cur_label_arr in self._traverse_label_array():
            for label in cur_label_arr:
                class_cnt[label] += 1
        total_class_cnt = img_cnt * np.prod(self.img_shape_2D)
        class_ratio = class_cnt / total_class_cnt

        # write into h5 file
        if norm_param_file is None:
            norm_param_file = self.output_file + '_meta'
        h5f = h5py.File(norm_param_file, 'w')
        h5f.create_dataset(name='mean', data=mean)
        h5f.create_dataset(name='std', data=std)
        h5f.create_dataset(name='class_ratio', data=class_ratio)
        h5f.close()

    def _get_raw_input_label_pair(self, dirpath, name):

        input_path = os.path.join(dirpath, name)
        label_path = os.path.join(dirpath, name.split('.')[0]+'_label.txt')

        # read & fill input (read in as float => correspond to tf.float64)
        with open(input_path) as file:
            point_arr = np.array([[float(val) for val in line.split(' ')] for line in file.read().strip('\n').split('\n')])
        coord_arr, feature_arr = self._separate_coord_feature(point_arr)
        input_data = self._fill_in_image(coord_arr, feature_arr)

        # read & fill label (read in as int => correspond to tf.int64)
        with open(label_path) as file:
            point_arr = np.array([[int(val) for val in line.split(' ')] for line in file.read().strip('\n').split('\n')])
        coord_arr, feature_arr = self._separate_coord_feature(point_arr)
        if self.use_one_hot:
            feature_arr = self._to_onehot(feature_arr)
        label_data = self._fill_in_image(coord_arr, feature_arr)

        # print('constructed:', input_data.shape, label_data.shape)
        return input_data.tostring(), label_data.tostring()

    def _construct_example(self, input_data_raw, label_data_raw):
        example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_data_raw])),
        'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_data_raw]))
        }))
        # print('=====================')
        return example

