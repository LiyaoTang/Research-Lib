#!/usr/bin/env python
# coding: utf-8
"""
module: base class for using tf dataset
"""

import tensorflow as tf
from .base import File_Traverser

class TF_Feeder(object):
    """
    base class to build a tf.data pipeline
    """
    def __init__(self, data_path):
        self.data_path = data_path
    
    def _get_dataset(self):
        raise NotImplementedError

    def config_dataset(self, epoch=1, batch=1, shuffle=True, shuffle_buf=100):
        """
        get a configured dataset
        """
        dataset = self._get_dataset()
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buf)
        return dataset

    def get_general_iterator(self, epoch=1, batch=1, shuffle=True, shuffle_buf=100):
        """
        get general iterator => enable to switch dataset 
        """
        dataset = self.config_dataset(epoch=epoch, batch=batch, shuffle=shuffle, shuffle_buf=shuffle_buf)
        return tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    
    def get_dataset_iterator(self, epoch=1, batch=1, shuffle=True, shuffle_buf=100, one_shot=False):
        """
        construct tf.train.Dataset iterators to feed batches
        to use, see https://www.tensorflow.org/guide/datasets
        """
        dataset = self.config_dataset(epoch=epoch, batch=batch, shuffle=shuffle, shuffle_buf=shuffle_buf)

        if one_shot:
            iterator = dataset.make_one_shot_iterator()            
        else:
            iterator = dataset.make_initializable_iterator()

        return iterator


class TFRecord_Feeder(TF_Feeder):
    """
    base class to build a tf.data pipeline from a .tfrecord
    """
    def __init__(self, data_path, features_dict):
        super(TFRecord_Feeder, self).__init__(data_path)
        self.features_dict = features_dict

    # decode an example in the form of bytes
    def _decode_byte(self, example):
        raise NotImplementedError

    def feed(self):
        """
        to get next example from the tfrecord
        """
        self.record_iterator = tf.python_io.tf_record_iterator(path=self.data_path)
        for raw_example in self.record_iterator:
            example = tf.train.Example()
            example.ParseFromString(raw_example)

            yield self._decode_byte(example)

    # decode the tensor resulted from reading tfrecord - mainly by tf.decode_raw(bytes, out_type)
    def _decode_to_tensor(self, serialized_example):
        raise NotImplementedError

    def construct_batch_feeder(self, batch_size, shuffle=True, capacity=None, min_after_dequeue=None, num_threads=1):
        """
        Deprecate! - replaced by tf.train.Dataset
        construct tensor(s) to feed batches, to start a feeder: 
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
        """
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
    """
    base class to construct tf.data pipeline to read, parse & feed CSV data
    TODO: implement deprecate pipeline threading for backward compatibility
    """
    def __init__(self, data_path, record_defaults, select_cols=None, header=True, recursive=True, re_expr='.*'):
        super(TF_CSV_Feeder, self).__init__(data_path)
        self.record_defaults = record_defaults
        self.select_cols = select_cols
        self.header = header

        if recursive:
            if type(data_path) is list: # data path as a list of dir
                self.filenames = []
                for dir_p in data_path:
                    traverser = File_Traverser(dir_p, re_expr) 
                    self.filenames.extend([path for path in traverser.traverse_file_path()])
            else:  # data path as dir
                traverser = File_Traverser(data_path, re_expr)
                self.filenames.extend([path for path in traverser.traverse_file_path()])
        else:
            self.filenames = [data_path]

    def _get_dataset(self):
        return tf.contrib.data.CsvDataset(filenames=self.filenames,
                                          record_defaults=self.record_defaults,
                                          header=self.header,
                                          select_cols=self.select_cols)

    def construct_batch_feeder(self, batch_size, shuffle=True, capacity=None, min_after_dequeue=None, num_threads=1):
        """
        Deprecate! - replaced by tf.train.Dataset
        construct tensor(s) to feed batches, to start a feeder: 
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
        """
        if not capacity:
            capacity = 10 * batch_size
        if not min_after_dequeue: # minimal number after dequeue - for the level of mixing
            min_after_dequeue = batch_size

        # generate a queue to read => only single file supported now
        filename_queue = tf.train.string_input_producer(self.data_path)

        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)   # get file name & file
        input_label_tensor_pair = tf.decode_csv(serialized_example,
                                                record_defaults=self.record_defaults,)
                                                # select_cols=self.select_cols)

        if shuffle:
            batch_feeder = tf.train.shuffle_batch(input_label_tensor_pair, batch_size=batch_size, capacity=capacity, num_threads=num_threads, min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)
        else:
            batch_feeder = tf.train.batch(input_label_tensor_pair, batch_size=batch_size, capacity=capacity, num_threads=num_threads, allow_smaller_final_batch=True)
        
        return batch_feeder # a list of tensors for corresponding input


class TF_TXT_Feeder(TF_Feeder):
    """
    base class to construct tf.data pipeline to read, parse & feed txt data
    """
    def __init__(self, data_path, recursive=True, file_re='.*', line_re=None, granularity='file'):
        super(TF_TXT_Feeder, self).__init__(data_path)
        self.traverser = File_Traverser(data_path, file_re)  # treat data path as dir
        self.line_re = line_re

        assert granularity in ('file', 'line')
        self.granularity = granularity

        if recursive:
            self.filenames = [path for path in self.traverser.traverse_file_path()]
        else:
            self.filenames = [self.data_path]

    def _parse_file(self, filename):
        raise NotImplementedError

    def _parse_line(self, line):
        raise NotImplementedError

    def _get_dataset(self):            
        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        if self.granularity == 'file':
            dataset = dataset.map(self._parse_file)
        else:
            dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename))
            dataset = dataset.map(self._parse_line)
        return dataset


class Img_from_Record_Feeder(TFRecord_Feeder):
    """
    construct pipeline (tf.data) to feed image from tfrecord, constructed by txtData_Constructor or similar
    """
    def __init__(self, data_path, img_shape_3D=(1024, 448, 8), label_shape_3D=(1024, 448, 4),
                 meta_data_file=None, norm_type='', use_sparse=False,
                 features_dict={'image_raw': tf.io.FixedLenFeature([], tf.string),
                                'label_raw': tf.io.FixedLenFeature([], tf.string)}):

        super(Img_from_Record_Feeder, self).__init__(data_path, features_dict)
        self.img_shape_3D = tuple(img_shape_3D)
        self.label_shape_3D = tuple(label_shape_3D)
        assert len(self.img_shape_3D) == 3 and len(self.label_shape_3D) == 3
        
        self.use_sparse = use_sparse

        if meta_data_file is not None:
            self._read_meta_data(meta_data_file)
        else:
            assert norm_type == ''
        self._construct_norm_method(norm_type)

    def _construct_norm_method(self, norm_type):
        assert norm_type in ('', 'mean', 'std')
        if not norm_type:
            try:
                if norm_type == 'mean':
                    self.norm_func = lambda x: x - self.mean
                elif norm_type == 'std':
                    self.norm_func = lambda x: (x - self.mean) / self.std
            except:
                print('possibly meta data not read yet')
        else:
            self.norm_func = lambda x: x

    def _read_meta_data(self, meta_data_file):
        h5f = h5py.File(meta_data_file, 'r')
        self.mean = np.array(h5f['mean'])
        self.std = np.array(h5f['std'])
        self.class_ratio = np.array(h5f['class_ratio'])
        h5f.close()
    
    def _decode_to_tensor(self, serialized_example):
        example = tf.parse_single_example(serialized_example, features=self.features_dict)
        
        img = tf.decode_raw(example['image_raw'], tf.float64) # python float <=> tf.float64
        # img = tf.Print(img, [tf.shape(img), img[0]], message='before reshape img')
        img = tf.reshape(img, self.img_shape_3D)
        img = self.norm_func(img)

        label = tf.decode_raw(example['label_raw'], tf.int64) # python int <=> tf.int64
        # label = tf.Print(label, [tf.shape(label), tf.shape(img), label[0]], message='before reshape label')
        label = tf.reshape(label, self.label_shape_3D)

        return [img, label]


class Sparse_Img_from_txt_Feeder(TF_TXT_Feeder):
    def __init__(self, data_path, recursive=True, re_expr='dw_((?!_label).)*\.txt',
                 img_shape_3D=(1024, 448, 8), label_shape_3D=(1024, 448, 4)):
        super(Sparse_Img_from_txt_Feeder, self).__init__(data_path, recursive, re_expr, granularity='file')
        self.img_shape_3D = img_shape_3D
        self.label_shape_3D = label_shape_3D

    def _parse_file(self, filename):
        point_string = tf.read_file(filename).strip().split('\n')
        indice_arr, value_arr = tf.map_fn(lambda line: [line[0:2], line[2:]], point_string, dtype=[tf.int32, tf.float32])
        input_img = tf.SparseTensor(indice_arr, value_arr, self.img_shape_3D)

        label_name = tf.strings.regex_replace(filename, '\.txt', '_label.txt')
        label_string = tf.read_file(label_name).strip().split('\n')
        # plus one => to distinguish unknown from background 0
        label_idx_arr, label_arr = tf.map_fn(lambda line: [line[0:2], line[2:] + 1], label_string, dtype=[tf.int32, tf.int32])
        label_msk = tf.SparseTensor(label_idx_arr, label_arr, self.label_shape_3D)

        return [input_img, label_msk]