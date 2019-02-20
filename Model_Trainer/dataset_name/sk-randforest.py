#!/usr/bin/env python
# coding: utf-8
'''
script: construct model for points cloud input based on sklearn
'''

import sys
root_dir = '../../'
sys.path.append(root_dir)
sys.path.append(root_dir + 'Data/corner/scripts')

import os
import psutil
import matplotlib
# matplotlib.use('agg') # so that plt works in command line
import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as sken
import Data_Feeder as feeder
import Model_Analyzer as analyzer
import Model_Constructer as constructer

from optparse import OptionParser
from sklearn.externals import joblib

''' parsing args '''

parser = OptionParser()
parser.add_option('--name', dest='model_name')
parser.add_option('--save', dest='save_path', default='./Model/sk-randforest/')
parser.add_option('--analysis_dir', dest='analysis_dir', default='./Model_Analysis/sk-randforest/')
parser.add_option('--log_dir', dest='log_dir', default='./Log/sk-randforest')
parser.add_option('--group_model', dest='group_model', default=False, action='store_true')

parser.add_option('--train', dest='path_train_set', default=root_dir + 'Data/corner/train')
parser.add_option('--val', dest='path_val_set', default=root_dir + 'Data/corner/val')
parser.add_option('--test', dest='path_test_set', default=root_dir + 'Data/corner/test')
parser.add_option('--cv_fold', dest='cv_fold', type='int', default=0)
parser.add_option('--class_num', dest='class_num', type=int, default=2)
parser.add_option('--class_name', dest='class_name', default=None)
parser.add_option('--corner_only', dest='corner_only', action='store_true', default=False)
parser.add_option('--use_onehot', dest='use_onehot', default=False, action='store_true')

parser.add_option('-e', '--epoch', dest='epoch', default=20, type='int')
parser.add_option('-b', '--batch', dest='batch', default=1, type='int')
parser.add_option('-t', '--tree_num', dest='tree_num', default=100, type='int')
parser.add_option('--select_cols', dest='select_cols', default=None)
parser.add_option('--norm', dest='norm_type', default='')
parser.add_option('--learning_rate', dest='learning_rate', default=1e-5, type='float')
parser.add_option('--reg_type', dest='regularizer_type', default='')
parser.add_option('--reg_scale', dest='regularizer_scale', default=0.1, type='float')
parser.add_option('--weighted_loss', dest='weighted_loss', default='')
parser.add_option('--rand_seed', dest='rand_seed', default=None, type=int)
parser.add_option('--add_noise', dest='add_noise', default=False, action='store_true')

(options, args) = parser.parse_args()

model_name = options.model_name
save_path = options.save_path
analysis_dir = options.analysis_dir
log_dir = options.log_dir
group_model = options.group_model

path_train_set = options.path_train_set
path_val_set = options.path_val_set
path_test_set = options.path_test_set
class_num = options.class_num
class_name = options.class_name.split(';')
use_onehot = options.use_onehot
cv_fold = options.cv_fold
corner_only = options.corner_only

# epoch = options.epoch
# batch = options.batch
tree_num = options.tree_num
select_cols = options.select_cols
learning_rate = options.learning_rate
norm_type = options.norm_type
regularizer_type = options.regularizer_type
regularizer_scale = options.regularizer_scale
weighted_loss = options.weighted_loss
rand_seed = options.rand_seed
add_noise = options.add_noise

# assert options to be valid
assert norm_type in ('', 'm', 's')  # null: no norm, m: mean-centered, s: standardized
assert regularizer_type in ('', 'L1', 'L2') # e.g. 'L2-0.1' => use L2 norm with scale of 0.1
assert weighted_loss in ('', 'bal')  # b: balanced
assert len(class_name) == class_num # assert to be corresponding after split by ;

# naming
if not model_name:
    model_name = 'sk-randforest_'
    if corner_only: model_name += 'corner_'
    if select_cols: model_name += select_cols + '_'
    if norm_type: model_name += norm_type[0] + '_'
    if regularizer_type: model_name += regularizer_type + '-' + str(regularizer_scale) + '_'
    if weighted_loss: model_name += weighted_loss +'_'
    model_name += 't' + str(tree_num)
    if rand_seed is not None: model_name += '_r' + str(rand_seed)
    if add_noise: model_name += '_n'
    # model_name += 'e' + str(epoch) + '_'
    # model_name += 'b' + str(batch)

# mkdir if not existed
if group_model:
    analysis_dir = os.path.join(analysis_dir, model_name) + '/'  # group analysis under folder with same name
os.makedirs(analysis_dir, exist_ok=True)
if save_path:
    os.makedirs(save_path, exist_ok=True)

# change std out if log dir originally given
if options.log_dir:
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, model_name)
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file

if rand_seed is not None:
    print('using rand seed =', rand_seed)
    np.random.seed(rand_seed)

# re expr for selecting lines
if corner_only:
    line_re = '\t (?!3).*'
else:
    line_re = '.*'

# parsing cols to be selected
if select_cols is not None:
    select_cols = [int(num) for num in select_cols.split('-')] # str to int

''' constructe dataset '''

train_set = feeder.Corner_Radar_Points_Gen_Feeder(path_train_set, class_num=class_num,
                                                  use_onehot=use_onehot, line_re=line_re, select_cols=select_cols)
if cv_fold <= 0:
    val_set = feeder.Corner_Radar_Points_Gen_Feeder(path_val_set, class_num=class_num,
                                                    use_onehot=use_onehot, line_re=line_re, select_cols=select_cols)
    test_set = feeder.Corner_Radar_Points_Gen_Feeder(path_test_set, class_num=class_num,
                                                     use_onehot=use_onehot, line_re=line_re, select_cols=select_cols)

# get all data points
train_input, train_label = train_set.get_all_data()
train_input = np.vstack(train_input)
train_label = np.concatenate(train_label)
if cv_fold <= 0:
    val_input, val_label = val_set.get_all_data()
    val_input = np.vstack(val_input)
    val_label = np.concatenate(val_label)

if add_noise:
    train_input = np.concatenate([train_input, np.transpose([np.random.rand(train_input.shape[0])])], axis=-1)
    train_set.feature_names.append('noise')
    if cv_fold <= 0:
        val_input = np.concatenate([val_input, np.transpose([np.random.rand(val_input.shape[0])])], axis=-1)
        val_set.feature_names.append('noise')

print("selected cols = ", select_cols)
print(train_set.feature_names)

process = psutil.Process(os.getpid())
print('mem usage after data loaded:', process.memory_info().rss / 1024 / 1024, 'MB')

''' constructing model '''

rand_forest = sken.RandomForestClassifier(tree_num, random_state=rand_seed)

def evaluate_model(train_input, train_label, val_input, val_label):

    # evaluate on train set
    print('\ntrain set:')
    train_metric = analyzer.General_Mertic_Record(class_num=class_num, class_name=class_name)
    train_metric.evaluate_model_at_once(prob_pred=rand_forest.predict_proba(train_input),
                                        loss=rand_forest.score(train_input, train_label),
                                        label=train_label,
                                        is_onehot=use_onehot)

    train_metric.print_result()
    train_metric.plot_cur_epoch_curve(save_path=analysis_dir, model_name='train-' + model_name, use_subdir=(not group_model))

    # evaluate on val set
    print('\nvalidation set:')
    val_metric = analyzer.General_Mertic_Record(class_num=class_num, class_name=class_name)
    val_metric.evaluate_model_at_once(prob_pred=rand_forest.predict_proba(val_input),
                                    loss=rand_forest.score(val_input, val_label),
                                    label=val_label,
                                    is_onehot=use_onehot)
    val_metric.print_result()
    val_metric.plot_cur_epoch_curve(save_path=analysis_dir, model_name='val-' + model_name, use_subdir=(not group_model))
    
    # print & plot feature importance
    importances = rand_forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rand_forest.estimators_],axis=0) # std of importances from each tree
    sort_idx = np.argsort(importances)[::-1] # idx for importantance feature in > sort
    # print
    print('\nfeature importances:\n', 'name\t', [train_set.feature_names[idx] for idx in sort_idx])
    print('score\t', importances[sort_idx], '\n', 'std\t', std[sort_idx])
    # plotd
    plt.bar(range(train_input.shape[1]), importances[sort_idx], yerr=std[sort_idx], align="center")
    plt.xticks(range(train_input.shape[1]), [train_set.feature_names[idx] for idx in sort_idx])
    if group_model:
        plt.savefig(os.path.join(analysis_dir, model_name + '_feature_importances.png'))
    else:
        plt.savefig(os.path.join(analysis_dir, 'feature_importances/', model_name + '.png'))

''' training & monitoring '''

if cv_fold > 0:
    # train with cross validation
    cv_trainer = constructer.Cross_Val_Trainer(cv_fold,
                                               train_func=lambda x, y: rand_forest.fit(x, y),
                                               evaluate_func=lambda train, val: evaluate_model(train['input'],
                                                                                               train['label'],
                                                                                               val['input'],
                                                                                               val['label']))
    cv_trainer.cross_validate(all_input=train_input, all_label=train_label, verbose=True)

else:
    # fit & evaluate model
    rand_forest.fit(train_input, train_label)
    evaluate_model(train_input, train_label, val_input, val_label)

if save_path:
    joblib.dump(rand_forest, filename=save_path + model_name)

print("finish")
if options.log_dir:
    log_file.close()
sys.exit()