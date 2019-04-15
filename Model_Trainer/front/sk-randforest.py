#!/usr/bin/env python
# coding: utf-8
'''
construct model for points cloud input based ob sklearn
'''

import sys
root_dir = '../../'
sys.path.append(root_dir + 'Data_Feeder/')
sys.path.append(root_dir + 'Metric_Recorder/')

import os
import psutil
import matplotlib
# matplotlib.use('agg') # so that plt works in command line
import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as sken
import Data_Feeder as feeder
import Metric_Recorder as recorder
from optparse import OptionParser

''' parsing args '''

parser = OptionParser()
parser.add_option('--name', dest='model_name')
parser.add_option('--save', dest='save_path', default='./Model/')
parser.add_option('--analysis_dir', dest='analysis_dir', default='./Model_Analysis/')
parser.add_option('--log_dir', dest='log_dir', default='./Log/')

parser.add_option('--train', dest='path_train_set', default=root_dir + 'Data/front/train/')
parser.add_option('--val', dest='path_val_set', default=root_dir + 'Data/front/val/')
parser.add_option('--test', dest='path_test_set', default=root_dir + 'Data/front/test/')
parser.add_option('--use_cv', dest='use_cv', action='store_true', default=False)
parser.add_option('--feature_num', dest='feature_num', type=int, default=8)
parser.add_option('--class_num', dest='class_num', type=int, default=3)
parser.add_option('--class_name', dest='class_name', default=None)

parser.add_option('-e', '--epoch', dest='epoch', default=20, type='int')
parser.add_option('-b', '--batch', dest='batch', default=1, type='int')
parser.add_option('-t', '--tree_num', dest='tree_num', default=100, type='int')
parser.add_option('--norm', dest='norm_type', default='')
parser.add_option('--learning_rate', dest='learning_rate', default=1e-5, type='float')
parser.add_option('--reg_type', dest='regularizer_type', default='')
parser.add_option('--reg_scale', dest='regularizer_scale', default=0.1, type='float')
parser.add_option('--use_batch_norm', dest='use_batch_norm', action='store_true', default=False)
parser.add_option('--weighted_loss', dest='weighted_loss', default='')

(options, args) = parser.parse_args()

model_name = options.model_name
save_path = options.save_path.rstrip('/') + '/'
analysis_dir = options.analysis_dir.rstrip('/') + '/'
log_dir = options.log_dir.rstrip('/') + '/'

path_train_set = options.path_train_set
path_val_set = options.path_val_set
path_test_set = options.path_test_set
feature_num = options.feature_num
class_num = options.class_num
class_name = options.class_name.split(';')
use_cv = options.use_cv

# epoch = options.epoch
# batch = options.batch
tree_num = options.tree_num
learning_rate = options.learning_rate
norm_type = options.norm_type
regularizer_type = options.regularizer_type
regularizer_scale = options.regularizer_scale
weighted_loss = options.weighted_loss

# assert options to be valid
assert norm_type in ('', 'm', 's')  # null: no norm, m: mean-centered, s: standardized
assert regularizer_type in ('', 'L1', 'L2') # e.g. 'L2-0.1' => use L2 norm with scale of 0.1
assert weighted_loss in ('', 'bal')  # b: balanced
assert len(class_name) == class_num # assert to be corresponding after split by ;

# naming
if not model_name:
    model_name = 'sk-randforest_'
    model_name += 't' + str(tree_num)
    # model_name += 'f' + str(feature_num) + '_'
    if norm_type: model_name += norm_type[0] + '_'
    if regularizer_type: model_name += regularizer_type + '-' + str(regularizer_scale) + '_'
    if weighted_loss: model_name += weighted_loss +'_'
    # model_name += 'e' + str(epoch) + '_'
    # model_name += 'b' + str(batch)

# mkdir if not existed
analysis_dir = analysis_dir + model_name + '/'  # default to folder with same name 
os.makedirs(save_path, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)

# change std out if log file originally given
if options.log_dir:
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = log_dir + model_name
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file

''' constructing dataset '''

train_set = feeder.Front_Radar_Points_from_txt_Gen_Feeder(path_train_set, class_num=class_num, norm_type=norm_type,
                                                          use_onehot=False, weighted=weighted_loss)
val_set = feeder.Front_Radar_Points_from_txt_Gen_Feeder(path_val_set, class_num=class_num, norm_type=norm_type,
                                                        use_onehot=False, weighted=weighted_loss)
test_set = feeder.Front_Radar_Points_from_txt_Gen_Feeder(path_test_set, class_num=class_num, norm_type=norm_type,
                                                         use_onehot=False, weighted=weighted_loss)

# get all data points
train_input, train_label = train_set.get_all_data(allowed=True)
train_input = np.vstack(train_input)
train_label = np.concatenate(train_label)

val_input, val_label = val_set.get_all_data(allowed=True)
val_input = np.vstack(val_input)
val_label = np.concatenate(val_label)

''' constructing model '''

rand_forest = sken.RandomForestClassifier(tree_num)

''' training & monitoring '''

# fit model
rand_forest.fit(train_input, train_label)

# evaluate on train set
print('\ntrain set:')
train_metric = recorder.General_Mertic_Record(class_num=class_num, class_name=class_name)
train_metric.evaluate_model_at_once(prob_pred=rand_forest.predict_proba(train_input),
                                    loss=rand_forest.score(train_input, train_label),
                                    label=train_label,
                                    is_onehot=False)

train_metric.print_result()
train_metric.plot_cur_epoch_curve(save_path=analysis_dir, model_name='train-' + model_name)

# evaluate on val set
print('\nvalidation set:')
val_metric = recorder.General_Mertic_Record(class_num=class_num, class_name=class_name)
val_metric.evaluate_model_at_once(prob_pred=rand_forest.predict_proba(val_input),
                                  loss=rand_forest.score(val_input, val_label),
                                  label=val_label,
                                  is_onehot=False)
val_metric.print_result()
val_metric.plot_cur_epoch_curve(save_path=analysis_dir, model_name='val-' + model_name)
# val_metric.plot_evolve_curve(save_path=analysis_dir, model_name=model_name)

print("finish")
if options.log_dir:
    log_file.close()
sys.exit()