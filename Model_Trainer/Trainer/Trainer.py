#!/usr/bin/env python
# coding: utf-8
"""
module: utils to config / construct a training pipeline
"""

import os
import numpy as np
from argparse import ArgumentParser

class Default_Argumentparser(ArgumentParser):
    def __init__(self, model_group_name, dataset_name, root_dir='../../', class_num=-1, class_name='', cv_fold=0):
        super(Default_Argumentparser, self).__init__()
        # file related
        self.add_argument('--name',           dest='model_name')
        self.add_argument('--save',           dest='save_path',   default='./Model/%s/' % model_group_name)
        self.add_argument('--analysis_dir',   dest='analysis_dir', default='./Model_Analysis/%s/' % model_group_name)
        self.add_argument('--log_dir',        dest='log_dir',     default='./Log/%s/' % model_group_name)
        self.add_argument('--no_summary',     dest='record_summary',  default=True, action='store_false')
        self.add_argument('--summary_dir',    dest='summary_dir', default='./Summary')
        self.add_argument('--groupby_model',  dest='groupby_model', default=False, action='store_true')

        # model config
        self.add_argument('--loss_type', dest='loss_type', default='xen')
        self.add_argument('--learning_rate',  dest='learning_rate',   default=1e-5, type='float')
        self.add_argument('--reg_scale',      dest='regularizer_scale', default=0.1, type='float')
        self.add_argument('--reg_type',       dest='regularizer_type', default='')
        self.add_argument('--weighted_loss',  dest='weighted_loss',   default='')
        self.add_argument('--batchnorm',      dest='batchnorm',       default=0,  type=int) # 0-disable; 1-enable

        # data feeding
        self.add_argument('--train',      dest='path_train_set', default=os.path.join(root_dir, 'Data/%s/train' % dataset_name))
        self.add_argument('--val',        dest='path_val_set', default=os.path.join(root_dir, 'Data/%s/val' % dataset_name))
        self.add_argument('--test',       dest='path_test_set', default=os.path.join(root_dir, 'Data/%s/test' % dataset_name))
        self.add_argument('--cv_fold',    dest='cv_fold',     default=cv_fold, type=int)
        self.add_argument('--class_num',  dest='class_num',   default=class_num, type=int)
        self.add_argument('--class_name', dest='class_name',  default=class_name)
        self.add_argument('--norm_type',  dest='norm_type',   default='')
        self.add_argument('--use_onehot', dest='use_onehot',  default=False, action='store_true')
        self.add_argument('--rand_seed',  dest='rand_seed',   default=None, type=int)
        self.add_argument('--add_noise',  dest='add_noise',   default=False, action='store_true')

        # model training
        self.add_argument('-e', '--epoch',    dest='epoch', default=20, type='int')
        self.add_argument('-b', '--batch',    dest='batch', default=1,  type='int')
    
    def parse_args(self):
        """
        wrapper function for early assertion & parsing logic - for all model, feeder & dataset
        """
        args = super(Default_Argumentparser, self).parse_args()

        # legal value chk
        # TODO: change crf from loss_type into postprocess
        for arg_n, legal_val in zip(['regularizer_type', 'norm_type', 'weighted_loss', 'batchnorm', 'loss_type'],
                                    [('', 'L1', 'L2'), ('', 'm', 's'), ('', 'bal'), (0, 1), ('xen', 'crf')]):
            cur_val = getattr(args, arg_n)
            if cur_val not in legal_val:
                raise ValueError('supported values for opt %s are %s, but received %s' % (arg_n, str(legal_val), str(cur_val)))
        
        # class name - num chk
        class_name = args.class_name.split(';')
        class_num = args.class_num
        if class_num != len(class_name):
            raise ValueError('specified class num = %d not consistent with num of given class name: %s' % (class_num, class_name))

        # pre-convert
        args.class_name = class_name
        args.batchnorm = (args.batchnorm == 1)

        # configration for model - add model_config
        model_config = {}
        for arg_n in ['learning_rate', 'regularizer_type', 'regularizer_scale', 'weighted_loss', 'batchnorm', 'record_summary', 'loss_type']:
            model_config[arg_n] = getattr(args, arg_n)
        setattr(args, 'model_config', model_config)

        return args


class Cross_Val_Trainer(object):
    """
    base class to train & evaluate model with cross validation
    assume model can be trained & evaluated with each a single function call
    """
    def __init__(self, fold_num, train_func, evaluate_func):
        self.train_func = train_func
        self.evaluate_func = evaluate_func

        self.fold_num = int(fold_num)
        assert self.fold_num > 0
    
    def cross_validate(self, all_input, all_label, verbose=False):
        """
        perform k-fold cross validation given all input, label
        """
        fold_size = int((len(all_input) - 1) / self.fold_num) + 1  # round up if not divisible

        if verbose: print('using', str(self.fold_num) + '-fold', 'corss validation, with fold size of', str(fold_size))

        index_mask = np.arange(all_input.shape[0])
        np.random.shuffle(index_mask)

        for cnt in range(self.fold_num):
            if verbose: print('\nfold', str(cnt))

            # calculate idx for validation fold
            start_idx = cnt * fold_size
            end_idx = ((cnt + 1) * fold_size)

            if start_idx == 0: # first fold
                train_mask = index_mask[end_idx:]
                val_mask = index_mask[:end_idx]
            elif end_idx >= len(all_input): # last fold (val might be smaller)
                train_mask = index_mask[:start_idx]
                val_mask = index_mask[start_idx:]
            else:  # folds in-between
                train_mask = np.concatenate([index_mask[:start_idx], index_mask[end_idx:]])
                val_mask = index_mask[start_idx:end_idx]

            train_fold = {'input': all_input[train_mask], 'label': all_label[train_mask]}
            val_fold = {'input': all_input[val_mask], 'label': all_label[val_mask]}

            self.train_func(train_fold['input'], train_fold['label'])

            # evaluate on train fold
            self.evaluate_func(train_fold, val_fold)

def set_rand_seed(seed, platform):
    assert any([n in str(platform) for n in ['tensorflow', 'torch']])  #
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if 'torch' in str(platform):
        platform.manual_seed(seed)
        platform.cuda.manual_seed(seed)
        platform.backends.cudnn.benchmark = False
        platform.backends.cudnn.deterministic = True
    elif 'tensorflow' in platform:
        platform.set_random_seed(seed)
    else:
        raise Exception('only support tensorflow, torch, but given %s', str(platform))


class Base_Trainer(object):
    def __init__(self):
        raise NotImplementedError

    def load_config(self):
        raise NotImplementedError

    def construct_feeder(self):
        raise NotImplementedError

    def construct_model(self):
        raise NotImplementedError

    def _link_feeder_model_dependency(self):
        raise NotImplementedError

    def prepare_train(self):
        raise NotImplementedError
        
    def train(self):
        raise NotImplementedError
