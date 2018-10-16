#!/usr/bin/env python
# coding: utf-8
'''
module: record model performance and calculate metrics
'''

import gc
import sys
import numpy as np
import tensorflow as tf
import sklearn.metrics as skmt
import matplotlib.pyplot as plt

class Mertic_Record(object):
    '''
    base class to collect statistics for model evaluation
    '''

    def __init__(self, class_num, class_name=None):

        self.class_num = class_num

        if class_name is None:
            class_name = dict(zip(range(class_num), [str(n) for n in range(class_num)]))
        elif not (type(class_name) is dict):  # treat as iterable
            class_name = dict(zip(range(class_num), class_name))
        self.class_name = class_name

        # cur epoch record
        self.pred_flat = []
        self.label_flat = []
        self.loss_list = []

        # model evolution curve (over epoches)
        self.balanced_acc_curve = []
        self.auc_curve = []
        self.avg_precision_curve = []
        self.loss_curve = []

    def _to_onehot(self, label_list):
        height = len(label_list)
        
        label = np.zeros((height, self.class_num), dtype=int)
        label[np.arange(height), label_list] = 1

        return label

    def _accumulate_rst(self, prob_pred, loss, label, is_onehot=True):
        '''
        prob_pred and label assumed to be np array with label/pred at last axis
        '''
        cur_pred_flat = list(prob_pred.reshape(-1, self.class_num))
        if not is_onehot:
            label = self._to_onehot(label)
        cur_label_flat = list(label.reshape(-1, self.class_num))
        
        self.pred_flat.extend(cur_pred_flat)
        self.label_flat.extend(cur_label_flat)
        self.loss_list.append(loss)

        assert len(self.pred_flat) == len(self.label_flat)

    def _cal_statistics(self):
        average_precision = dict()
        auc_score = dict()
        balanced_acc = dict()
        max_pred_matrix = np.zeros(shape=(self.class_num, self.class_num), dtype=int)
        mean_prob_matrix = np.zeros(shape=(self.class_num, self.class_num), dtype=float)

        for i in range(self.class_num):
            cur_pred_flat = np.array(self.pred_flat)[:, i]
            cur_label_flat = np.array(self.label_flat)[:, i]
            
            balanced_acc[i] = skmt.balanced_accuracy_score(y_pred=cur_pred_flat.round(), y_true=cur_label_flat)
            auc_score[i] = skmt.roc_auc_score(y_score=cur_pred_flat, y_true=cur_label_flat)
            average_precision[i] = skmt.average_precision_score(y_score=cur_pred_flat, y_true=cur_label_flat)
            
        self.avg_precision = average_precision
        self.auc_score = auc_score
        self.balanced_acc = balanced_acc
        self.mean_loss = sum(self.loss_list) / len(self.loss_list)

        pred_class = np.argmax(self.pred_flat, axis=-1)
        label_class = np.argmax(self.label_flat, axis=-1)

        self.balanced_acc_overall = skmt.balanced_accuracy_score(y_pred=pred_class, y_true=label_class)

        # calculate statistical matrix for prediction & prob
        for i, j, cur_pred_prob in zip(label_class, pred_class, self.pred_flat):
            max_pred_matrix[i, j] += 1
            mean_prob_matrix[i] += cur_pred_prob
        self.max_pred_matrix = max_pred_matrix
        self.mean_prob_matrix = mean_prob_matrix / len(self.pred_flat)

    def _cal_curve(self):
        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        threshold = dict()
        
        for i in range(self.class_num):
            cur_pred_flat = np.array(self.pred_flat)[:, i]
            cur_label_flat = np.array(self.label_flat)[:, i]
            
            fpr[i], tpr[i], _ = skmt.roc_curve(y_score=cur_pred_flat, y_true=cur_label_flat)  # roc
            precision[i], recall[i], threshold[i] = skmt.precision_recall_curve(probas_pred=cur_pred_flat, y_true=cur_label_flat)  # precision-recall
            
        self.fpr = fpr
        self.tpr = tpr
        self.precision = precision
        self.recall = recall
        self.threshold = threshold

    def evaluate_model(self):
        '''
        to be implemented for different model
        '''
        raise NotImplementedError

    def clear_cur_epoch(self):
        self.pred_flat = []
        self.label_flat = []
        self.loss_list = []

        self.fpr = dict()
        self.tpr = dict()
        self.auc_score = dict()
        self.precision = dict()
        self.recall = dict()
        self.threshold = dict()

        self.balanced_acc_curve.append(self.balanced_acc)
        self.auc_curve.append(self.auc_score)
        self.avg_precision_curve.append(self.avg_precision)
        self.loss_curve.append(self.mean_loss)

        # clear mem
        gc.collect()

    def plot_evolve_curve(self, show=False, save_path=None, model_name=''):
        '''
        plot curve for model evolution, optionally saved into file
        '''

        save_path = save_path.rstrip('/') + '/'
        plt.close('all')

        for curve_list, curve_name in zip((self.balanced_acc_curve, self.auc_curve, self.avg_precision_curve),
                                          ('balanced_acc_curve', 'auc_curve', 'avg_precision_curve')):
            for i in range(self.class_num):
                plt.plot([snap_shot[i] for snap_shot in curve_list], label=self.class_name[i])
            plt.legend()
            
            if not save_path is None:
                plt.savefig(save_path + model_name + str(curve_name) + '_val.png', bbox_inches='tight')
            if show:
                plt.show()
            plt.close()

        plt.plot(self.loss_curve)
        if not save_path is None:
            plt.savefig(save_path + model_name + 'loss_curve_val.png', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def print_result(self):
        '''
        print cur-epoch statistcs
        '''
        print("mean_loss = ", self.mean_loss, "overall balanced_acc = ", self.balanced_acc_overall,
              "\nbalanced_acc = ", [self.class_name[k] + ' : ' + str(v) for k, v in self.balanced_acc.items()],
              "\nauc = ", [self.class_name[k] + ' : ' + str(v) for k, v in self.auc_score.items()],
              "\navg_precision = ", [self.class_name[k] + ' : ' + str(v) for k, v in self.avg_precision.items()])

        print('max pred matrix:')
        print('\t' + '\t'.join([self.class_name[i] for i in range(self.class_num)]))
        for i in range(self.class_num):
            print(self.class_name[i] + '\t' + '\t'.join([str(cnt) for cnt in self.max_pred_matrix[i]]))
        
        print('mean prop matrix:')
        print('\t' + '\t'.join([self.class_name[i] for i in range(self.class_num)]))
        for i in range(self.class_num):
            print(self.class_name[i] + '\t' + '\t'.join([str(prob) for prob in self.mean_prob_matrix[i]]))

        sys.stdout.flush()

    def plot_cur_epoch_curve(self, show=False, save_path=None, model_name=''):
        '''
        cal & plot curve, optionally saved into file
        '''
        self._cal_curve()

        if save_path:
            save_path = save_path.rstrip('/') + '/'
        plt.close('all')

        # roc
        for i in range(self.class_num):
            plt.plot(self.fpr[i], self.tpr[i], label=self.class_name[i])
        plt.legend()

        if save_path:
            plt.savefig(save_path + model_name + '_roc.png', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

        # precision-recall
        for i in range(self.class_num):
            plt.plot(self.recall[i], self.precision[i], label=self.class_name[i])
        plt.legend()

        if save_path:
            plt.savefig(save_path + model_name + '_pre-recall.png', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


class TF_Metric_Record(Mertic_Record):
    '''
    evaluate model given dataset.iterator & network's loss and pred under TF framework
    an active session must also be given
    '''

    def __init__(self, tf_loss, tf_pred, tf_label, tf_itr_init_op, tf_feed_dict, sess, class_num, class_name=None):
        super(TF_Metric_Record, self).__init__(class_num, class_name)

        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.InteractiveSession(config=config)
        self.sess = sess

        self.tf_pred = tf_pred # assumed to be probablistic output
        self.tf_loss = tf_loss
        self.tf_label = tf_label
        self.tf_itr_init_op = tf_itr_init_op  # assumed to not repeat without exception
        self.tf_feed_dict = tf_feed_dict # should not contain any variable input in test time! (input solved by iterator)

    def evaluate_model(self):
        '''
        evaluate the model on given dataset
        the iterator is assumed to not repeat without exception
        the prediction is assumed to be probablistic prediction for one-hot encoding
        '''

        # per epoch statistic
        self.pred_flat = []
        self.label_flat = []
        self.loss_list = []
        
        self.sess.run(self.tf_itr_init_op)
        try:
            while True:
                [prob_pred, label, xen] = self.sess.run([self.tf_pred, self.tf_label, self.tf_loss],
                                                        feed_dict=self.tf_feed_dict)

                self._accumulate_rst(prob_pred=prob_pred, loss=xen, label=label)

        except tf.errors.OutOfRangeError:
            pass

        self._cal_statistics()


class General_Mertic_Record(Mertic_Record):
    '''
    evaluate model given an arbitrary model
    '''

    def __init__(self, class_num, class_name=None):
        super(General_Mertic_Record, self).__init__(class_num, class_name)
    
    def evaluate_model(self, model_func, input_label_func, is_onehot=True):
        '''
        request an input_label_func to fetch (input, label) pair and will return None when finished;
        a model_func to get probablistic prediction and loss given (input, label) pair
        '''
        while True:
            batch_input, batch_label = input_label_func()
            if batch_input == None:
                break
            
            prob_pred, loss = model_func(batch_input, batch_label)
            self._accumulate_rst(prob_pred=prob_pred, loss=loss, label=batch_label, is_onehot=is_onehot)
        
        self._cal_statistics()

    def evaluate_model_at_once(self, prob_pred, loss, label, is_onehot=True):
        '''
        pass in directly probablistic pred, loss & label
        '''
        self._accumulate_rst(prob_pred=prob_pred, loss=loss, label=label, is_onehot=is_onehot)
        self._cal_statistics()
