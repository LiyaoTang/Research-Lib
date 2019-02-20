#!/usr/bin/env python
# coding: utf-8
'''
module: multiprocessing wrapper for Metric_Recorder
'''

import os
import sys
import dill
import multiprocessing as mp
from .. import Loader as loader

class Parallel_Recorder(object):
    '''
    wrapper class to load the model & evaluate with recorder in separate process
    '''

    def __init__(self, recorder, feeder=None, log_file=None):
        super(Parallel_Recorder, self).__init__()
        if feeder is not None:
            self.feeder = dill.loads(feeder)
        else:
            self.feeder = None
        self.log_file = log_file
        self.recorder = recorder
        self.process = None

    def _evaluate_model(self, model_config):
        '''
        control flow to load & evaluate model (to be run in separate process)
        '''
        model_config = dill.loads(model_config)
        if 'model_func' in model_config and model_config['model_func']:
            model_func = model_config['model_func']
        else:
            project = loader.Project_Loader(dataset_name=model_config['dataset_name'],
                                            root_dir=model_config['root_dir'],
                                            verbose=False)
            # should load only one model
            model_config['feeder'] = self.feeder
            model_config['gpu_num'] = '-1' # NOT using gpu for evaluating, left for training
            project.load_models(model_group='.*', model_name=model_config['name'], config=model_config, verbose=False)
            if len(project.models) != 1:
                raise AssertionError('parallel_recorder %d: len(project.models) = %d, instead of 1' % (os.getpid(), len(project.models)))
            model_func = project.models[0]['pred_prob']

        self.recorder.evaluate_model(model_func=model_func,
                                    input_label_itr=self.feeder)

        if self.log_file:
            sys.stdout = self.log_file
        if 'write_lock' in model_config and model_config['write_lock']:
            write_lock = model_config['write_lock']
            write_lock.acquire()
            print('from subprocess', os.getpid(), '---------')
            self.recorder.print_result()
            print('subprocess ', os.getpid(), 'ends print ---------')
            write_lock.release()
        else:
            self.recorder.print_result()
        sys.stdout = sys.__stdout__

        self.recorder.clear_cur_epoch()

    def evaluate_model(self, model_config):
        '''
        start another process to run
        '''
        if self.process is not None:
            self.process.join()  # end the last round
        self.process = mp.Process(target=self._evaluate_model, args=[], kwargs={'model_config': dill.dumps(model_config)})
        self.process.start()  # start running process
