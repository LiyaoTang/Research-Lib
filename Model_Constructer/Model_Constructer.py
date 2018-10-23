#!/usr/bin/env python
# coding: utf-8
'''
base class to construct a training pipeline
'''

import numpy as np

class Cross_Val_Trainer(object):
    '''
    base class to train & evaluate model with cross validation
    assume model can be trained & evaluated with each a single function call
    '''
    def __init__(self, fold_num, train_func, evaluate_func):
        self.train_func = train_func
        self.evaluate_func = evaluate_func

        self.fold_num = int(fold_num)
        assert self.fold_num > 0
    
    def cross_validate(self, all_input, all_label, verbose=False):
        '''
        perform k-fold cross validation given all input, label
        '''
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
