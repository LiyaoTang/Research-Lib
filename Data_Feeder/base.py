#!/usr/bin/env python
# coding: utf-8
'''
module: base class for feeding data
'''

import os
import re
import sys
import numpy as np
import tensorflow as tf

class File_Traverser(object):
    '''
    traverse file given root dir with an optional fileter
    '''
    def __init__(self, data_dir, re_expr='.*'):
        self.data_dir = data_dir
        self.re_checker = re.compile(re_expr)
        self.chk_re_expr = lambda x: bool(self.re_checker.match(x))  # re checker

    def traverse_file(self):
        '''
        yield dir path and file name
        '''
        for dirpath, dirnames, files in os.walk(self.data_dir):
            for name in files:
                if self.chk_re_expr(name):
                    yield(dirpath, name)

    def traverse_file_path(self):
        '''
        yield joined path
        '''
        for dirpath, dirnames, files in os.walk(self.data_dir):
            for name in files:
                if self.chk_re_expr(name):
                    yield os.path.join(dirpath, name)


class TF_Feeder(object):
    '''
    base class to build a tf.data pipeline
    '''
    def __init__(self, data_path):
        self.data_path = data_path
    
    def _get_dataset(self):
        raise NotImplementedError

    def config_dataset(self, epoch=1, batch=1, shuffle=True, shuffle_buf=10):
        '''
        get a configured dataset
        '''
        dataset = self._get_dataset()
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buf)
        return dataset

    def get_general_iterator(self, epoch=1, batch=1, shuffle=True, shuffle_buf=10):
        '''
        get general iterator => enable to switch dataset 
        '''
        dataset = self.config_dataset(epoch=epoch, batch=batch, shuffle=shuffle, shuffle_buf=shuffle_buf)
        return tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    
    def get_dataset_iterator(self, epoch=1, batch=1, shuffle=True, shuffle_buf=10, one_shot=False):
        '''
        construct tf.train.Dataset iterators to feed batches
        to use, see https://www.tensorflow.org/guide/datasets
        '''
        dataset = self.config_dataset(epoch=epoch, batch=batch, shuffle=shuffle, shuffle_buf=shuffle_buf)

        if one_shot:
            iterator = dataset.make_one_shot_iterator()            
        else:
            iterator = dataset.make_initializable_iterator()

        return iterator    


class TFRecord_Feeder(TF_Feeder):
    '''
    base class to build a tf.data pipeline from a .tfrecord
    '''
    def __init__(self, data_path, features_dict):
        super(TFRecord_Feeder, self).__init__(data_path)
        self.features_dict = features_dict

    # decode an example in the form of bytes
    def _decode_byte(self, example):
        raise NotImplementedError

    def feed(self):
        '''
        to get next example from the tfrecord
        '''
        self.record_iterator = tf.python_io.tf_record_iterator(path=self.data_path)
        for raw_example in self.record_iterator:
            example = tf.train.Example()
            example.ParseFromString(raw_example)

            yield self._decode_byte(example)

    # decode the tensor resulted from reading tfrecord (which read the tfrecord) - mainly by tf.decode_raw(bytes, out_type)
    def _decode_to_tensor(self, serialized_example):
        raise NotImplementedError

    def construct_batch_feeder(self, batch_size, shuffle=True, capacity=None, min_after_dequeue=None, num_threads=1):
        '''
        Deprecate! - replaced by tf.train.Dataset
        construct tensor(s) to feed batches, to start a feeder: 
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
        '''
        if not capacity:
            capacity = 10 * batch_size
        if not min_after_dequeue: # minimal number after dequeue - for the level of mixing
            min_after_dequeue = batch_size

        # generate a queue to read => only single file supported now
        filename_queue = tf.train.string_input_producer([self.data_path])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   # get file name & file
        input_label_tensor_pair = self._decode_to_tensor(serialized_example)

        if shuffle:
            batch_feeder = tf.train.shuffle_batch(input_label_tensor_pair, batch_size=batch_size, capacity=capacity, num_threads=num_threads, min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)
        else:
            batch_feeder = tf.train.batch(input_label_tensor_pair, batch_size=batch_size, capacity=capacity, num_threads=num_threads, allow_smaller_final_batch=True)
        
        return batch_feeder # a list of tensors for corresponding input

    def _get_dataset(self):
        dataset =  tf.contrib.data.TFRecordDataset(self.data_path)
        dataset = dataset.map(self._decode_to_tensor)  # Parse the example protobuf into tensors
        return dataset


class TF_CSV_Feeder(TF_Feeder):
    '''
    base class to construct tf.data pipeline to read, parse & feed CSV data
    '''
    def __init__(self, data_path, record_defaults, select_cols=None, header=True, recursive=True, re_expr='.*'):
        super(TF_CSV_Feeder, self).__init__(data_path)
        self.record_defaults = record_defaults
        self.select_cols = select_cols
        self.header = header
        self.traverser = File_Traverser(data_path, re_expr) # treat data path as dir

        if recursive:
            self.filenames = [path for path in self.traverser.traverse_file_path()]
        else:
            self.filenames = [self.data_path]

    def _get_dataset(self):
        return tf.contrib.data.CsvDataset(filenames=self.filenames, record_defaults=self.record_defaults, header=self.header, select_cols=self.select_cols)


class Gen_Feeder(object):
    '''
    base class to feed from a data dir
    '''
    def __init__(self, data_dir, class_num, re_expr='.*', use_onehot=True):
        self.data_dir = data_dir
        self.traverser = File_Traverser(data_dir, re_expr=re_expr)

        self.class_num = class_num
        self.use_onehot = use_onehot

    def _to_onehot(self, label_list):
        height = len(label_list)
        
        label = np.zeros((height, self.class_num), dtype=int)
        label[np.arange(height), label_list] = 1

        return label

    def _get_input_label_pair(self, dirpath, name):
        raise NotImplementedError

    def feed_one_example(self):
        '''
        iterate through the dataset for once
        feed one example per time
        '''
        for dirpath, name in self.traverser.traverse_file():
            yield self._get_input_label_pair(dirpath, name)

    def get_all_data(self, allowed=False):
        '''
        dangeraous: feed all data at once
        use unless small enough & cannot partial fit
        '''
        if not allowed:
            raise Warning('you sure understand?')
            sys.exit()
        input_list = []
        label_list = []
        for cur_input, cur_label in self.feed_one_example():
            input_list.append(cur_input)
            label_list.append(cur_label)

        return input_list, label_list


class TFRecord_Constructer(object):
    '''
    base class to construct tfrecord
    '''
    def __init__(self, input_dir, output_file, re_expr='.*', recursive=True):
        self.input_dir = input_dir
        self.output_file = output_file
        self.re_checker = re.compile(re_expr)
        self.chk_re_expr = lambda x: bool(self.re_checker.match(x))  # re checker
        
        self.recursive = recursive  # if recursive finding into dir

    def _traverse_file(self):
        assert self.recursive
        
        # traverse
        for dirpath, dirnames, files in os.walk(self.input_dir):
            for name in files:
                if self.chk_re_expr(name):
                    yield(dirpath, name)
    
    def collect_meta_data(self, meta_data_file):
        '''
        implemented in derived class: collect meta data e.g. mean, std, class statistics
        '''
        raise NotImplementedError

    def _get_raw_input_label_pair(self, dirpath, name):
        raise NotImplementedError

    def _construct_example(self, input_data_raw, label_data_raw):
        raise NotImplementedError

    def write_into_record(self):
        '''
        frame work to write into .tfrecord file
        '''        
        with tf.python_io.TFRecordWriter(self.output_file) as writer:
            for dirpath, name in self._traverse_file(): # traverse
                input_data_raw, label_data_raw = self._get_raw_input_label_pair(dirpath, name)
                example = self._construct_example(input_data_raw, label_data_raw)
                writer.write(example.SerializeToString()) # write into tfrecord
