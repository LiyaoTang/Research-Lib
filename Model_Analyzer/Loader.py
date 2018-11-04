#!/usr/bin/env python
# coding: utf-8
'''
module: dirty codes to parse project-dependent model / dataset name to model object / data feeder
script: prediction - load model & corresponding feeder, save its prediction onto disk
'''

import sys
root_dir = '../'
sys.path.append(root_dir)

import os
import warnings
import sklearn as sk
import tensorflow as tf
import Data_Feeder as feeder

class Project_Loader(object):
    '''
    parsing project-dependent mapping logic to regarding paths
    '''
    def __init__(self, dataset_name, root_dir='../', verbose=False):
        if dataset_name not in ['corner']:
            raise TypeError('not supported dataset \"%s\"' % dataset_name)

        self.root_dir = root_dir
        self.dataset_name = dataset_name
        if dataset_name == 'corner':
            self.dataset_dir = os.path.join(root_dir, 'Data/', dataset_name, 'dw_19991231_162610_0.000000_0.000000')
            self.model_dir = os.path.join(root_dir, 'Model_Trainer/', dataset_name, 'Model')
            self.pred_dir = os.path.join(root_dir, 'Model_Trainer/', dataset_name, 'Prediction')
        
        model_type_tried = set([])
        for model_group in os.listdir(self.model_dir):
            model_type_tried.add(self.parse_model_type(model_group))
        self.model_type_tried = model_type_tried

        model_saved = []
        for _, _, fname in os.walk(self.model_dir):
            model_saved.extend(fname)
        self.model_saved = model_saved

        if verbose:
            self.print_info()

    def parse_model_type(self, name):
        '''
        get model type given name, raise TypeError if failed
        '''
        if 'sk' in name:
            model_type = 'sk'
        elif 'xgb' in name:
            model_type = 'xg'
        elif 'FCN' in name:
            model_type = 'tf'
        else:
            raise TypeError('not recognized model type from name \"%s\"' % name)

        return model_type

    def print_info(self):
        '''
        print out parsed project info
        '''
        print('root dir \"%s\"' % self.root_dir)
        print('dataset name \"%s\"' % self.dataset_name)
        print('model dir \"%s\"' % self.model_dir)
        print('pred dir \"%s\"' % self.pred_dir)
        print('model type tried %s' % self.model_type_tried)
        print('model saved %s' % self.model_saved)


class Model_Loader(Project_Loader):
    '''
    parsing model-dependent logic to load model & dataset
    '''
    def __init__(self, dataset_name, root_dir='../', model_path=None, verbose=False):
        super(Model_Loader, self).__init__(dataset_name, root_dir, verbose)

        if model_path is None:  # load all models
            self.model = []
            self.pred_func = []
            self.model_name = []
            self.model_type = []
            self.model_path = []
            for cur_name in self.model_saved:
                cur_path = os.path.join(self.model_dir, cur_name.split('_')[0], cur_name)
                cur_type = self.parse_model_type(cur_name)
                cur_model, cur_func = self.load_model(cur_path, cur_type)
                if cur_model is not None and cur_func is not None:
                    self.model.append(cur_model)
                    self.pred_func.append(cur_func)
                    self.model_name.append(cur_name)
                    self.model_type.append(cur_type)
                    self.model_path.append(cur_path)

        else:  # load single model
            self.model_path = model_path
            self.model_name = model_path.split('/')[-1]
            self.model_type = self.parse_model_type(self.model_name)
            if self.model_name not in self.model_saved:
                raise TypeError('specified model name \"%s\" not in found model %s' % (self.model_name, self.model_saved))

            self.model, self.pred_prob = self.load_model(self.model_path, self.model_type)

        if verbose:
            print('loaded model: %s' % self.model_name)

    def load_model(self, model_path, model_type):
        '''
        parse to select a correct loader
        '''
        model = None
        pred_prob = None
        if model_type == 'sk':
            model = sk.externals.joblib.load(model_path)
            pred_prob = model.predict_proba

        elif model_type == 'tf':
            raise NotImplementedError
        elif model_type == 'xg':
            warnings.warn('xgboost currently ignored')
        else:
            raise TypeError('not supported model type: \"%s\"' % self.model_type)

        return model, pred_prob

    def load_dataset(self, verbose=False):
        '''
        parse to select the correct data feeder
        '''
        if type(self.model_name) is list:  # multiple model loaded
            self.dataset = [self._load_single_dataset(name, verbose) for name in self.model_name]
        else: # single model loaded
            self.dataset = self._load_single_dataset(self.model_name, verbose)

    def _load_single_dataset(self, model_name, verbose):
        if self.dataset_name == 'corner':
            select_cols = None
            for param in model_name.split('_'):
                try:
                    select_cols = [int(num) for num in param.split('-')]
                except:
                    pass
            dataset = feeder.Corner_Radar_Points_Gen_Feeder(self.dataset_dir, select_cols=select_cols)
            if verbose:
                print(self.dataset_dir, select_cols, dataset.feature_names)

        else:
            raise TypeError('not supported dataset \"%s\"' % self.dataset_name)
        
        return dataset

    def list_all_file_path(self):
        if type(self.model_name) is list:  # multiple dataset constructed
            all_path = self.dataset[0].traverser.list_all_file_path()
            for dataset in self.dataset:
                if all_path != dataset.traverser.list_all_file_path():
                    raise AssertionError('Data not consistent across multiple models')
        else:
            all_path = self.dataset.traverser.list_all_file_path()

        return all_path

# run as script
if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('--root_dir', dest='root_dir')
    parser.add_option('--dataset_name', dest='dataset_name')
    parser.add_option('--path', dest='model_path')
    parser.add_option('--pred_dir', dest='pred_dir', default=None)
    parser.add_option('--overwrite', dest='overwrite', default=False, action='store_true')
    parser.add_option('--usage', dest='usage')
    (options, args) = parser.parse_args()

    root_dir = options.root_dir
    dataset_name = options.dataset_name
    model_path = options.model_path
    pred_dir = options.pred_dir
    overwrite = options.overwrite
    usage = options.usage

    if usage == 'predict':
        if type(model_path) is not str:
            raise TypeError('prediction for multiple models not supported yet, pls specify model path')
        model = Model_Loader(dataset_name, root_dir=root_dir, model_path=model_path, verbose=True)
        model.load_dataset(verbose=True)
        if pred_dir is None:
            pred_dir = model.pred_dir
        model.dataset.record_prediction(pred_func=model.pred_prob,
                                        model_name=model.model_name,
                                        dataset_name=model.dataset_name,
                                        output_dir=pred_dir,
                                        overwrite=overwrite)
    else:
        raise TypeError('usage \"%s\" not supported' % usage)