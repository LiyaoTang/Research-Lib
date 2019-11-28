#!/usr/bin/env python
# coding: utf-8
"""
module: record model performance and calculate metrics
"""

import os
import gc
import sys
import warnings
import numpy as np
import tensorflow as tf
import sklearn.metrics as skmt
import matplotlib.pyplot as plt
from collections import defaultdict

root_dir = '../'
sys.path.append(root_dir)
import Utilities as utils

class Metric_Record(object):
    def __init__(self):
        pass

    def eval_statistics(self, *args, **kwargs):
        """
        evaluate for the given example (prediction-label pair)
        """
        raise NotImplementedError

    def accumulate_rst(self, *args, **kwargs):
        """
        accumulate the result over the whole dataset
        """
        raise NotImplementedError

    def evaluate_model(self, *args, **kwargs):
        """
        evaluate the model over the whole dataset
        """
        raise NotImplementedError

    def get_result(self, *args, **kwargs):
        """
        calculate and return the result of evaluation
        """
        raise NotImplementedError
    
        
class Classif_Metric_Record(Metric_Record):
    """
    base class to collect statistics for evaluation of classification model 
    """

    def __init__(self, class_num, class_name=None, record_prob=False, title=''):
        super(Classif_Metric_Record, self).__init__()
        self.class_num = class_num
        if class_name is None:
            class_name = dict(zip(range(class_num), [str(n) for n in range(class_num)]))
        elif not (type(class_name) is dict):  # treat as iterable
            class_name = dict(zip(range(class_num), class_name))
        self.class_name = class_name

        self.record_prob = record_prob  # record config

        # cur epoch record
        self.example_cnt = 0
        self.pred_flat = []
        self.label_flat = []
        self.rst_record = defaultdict(lambda: 0)
        self.title = title

        # model evolution curve (over epoches)
        self.balanced_acc_curve = []

    def _to_onehot(self, label_list):
        height = len(label_list)
        label = np.zeros((height, self.class_num), dtype=int)
        label[np.arange(height), label_list] = 1
        return label

    def _cal_confusion_matrix(self, pred_flat, label_flat):
        """
        calculate statistical matrix for prediction & prob
        """
        pred_class = np.argmax(pred_flat, axis=-1)
        label_class = np.argmax(label_flat, axis=-1)

        confusion_matrix = np.zeros(shape=(self.class_num, self.class_num), dtype=int)
        mean_prob_matrix = np.zeros(shape=(self.class_num, self.class_num), dtype=float)
        for i, j, cur_pred_prob in zip(label_class, pred_class, pred_flat):
            confusion_matrix[i, j] += 1
            mean_prob_matrix[i] = mean_prob_matrix[i] + cur_pred_prob
        mean_prob_matrix = mean_prob_matrix / len(pred_flat)

        return confusion_matrix, mean_prob_matrix


    def eval_statistics(self, pred_flat, label_flat):
        """
        evaluate the statistics on the given prediction and label (both assume to be 1d array)
        """
        balanced_acc = dict()
        auc_score = dict()
        average_precision = dict()

        for i in range(self.class_num):
            cur_pred_flat = np.array(pred_flat)[:, i]
            cur_label_flat = np.array(label_flat)[:, i]
            
            balanced_acc[i] = skmt.balanced_accuracy_score(y_pred=cur_pred_flat.round(), y_true=cur_label_flat)
            auc_score[i] = skmt.roc_auc_score(y_score=cur_pred_flat, y_true=cur_label_flat)
            average_precision[i] = skmt.average_precision_score(y_score=cur_pred_flat, y_true=cur_label_flat)

        pred_class = np.argmax(pred_flat, axis=-1)
        label_class = np.argmax(label_flat, axis=-1)
        overall_balanced_acc = skmt.balanced_accuracy_score(y_pred=pred_class, y_true=label_class)
        confusion_matrix, mean_prob_matrix = self._cal_confusion_matrix(pred_flat, label_flat)
        
        return {'avg_precision': average_precision, 'auc_score': auc_score, 'balanced_acc': balanced_acc,
                'overall_balanced_acc': overall_balanced_acc, 'confusion_matrix': confusion_matrix, 'mean_prob_matrix': mean_prob_matrix}

    def accumulate_rst(self, prob_pred, label, is_onehot=True, record_prob=False):
        """
        prob_pred and label assumed to be np array with pred/label at last axis
        prob_pred is assumed to be one-hot; label encoding specified by `is_onehot`
        """
        cur_pred_flat = list(prob_pred.reshape(-1, self.class_num))
        if not is_onehot:
            label = self._to_onehot(label)
        cur_label_flat = list(label.reshape(-1, self.class_num))
        assert len(cur_pred_flat) == len(cur_label_flat)
        
        self.example_cnt += len(cur_pred_flat)
        if record_prob:  # all pred-label reserved
            self.pred_flat.extend(cur_pred_flat)
            self.label_flat.extend(cur_label_flat)
        else:
            confusion_matrix, mean_prob_matrix = self._cal_confusion_matrix(cur_pred_flat, cur_label_flat)
            self.rst_record['confusion_matrix'] += confusion_matrix
            self.rst_record['mean_prob_matrix'] += mean_prob_matrix * len(cur_pred_flat)

    def _cal_stat_on_rstrecord(self, record_prob=False):
        if record_prob:
            stat_record = self.eval_statistics(self.pred_flat, self.label_flat)
            self.rst_record = stat_record
        else:
            self.rst_record['mean_prob_matrix'] /= self.example_cnt

        self.rst_record['example_cnt'] = self.example_cnt
        self.rst_record['precision'] = {}
        self.rst_record['recall'] = {}
        cf_mat = self.rst_record['confusion_matrix']
        for k in self.class_name:
            self.rst_record['precision'][k] = cf_mat[k, k] / cf_mat[:, k].sum()
            self.rst_record['recall'][k] = cf_mat[k, k] / cf_mat[k, :].sum()
        # balanced acc = mean of class recall
        self.rst_record['overall_balanced_acc'] = sum(self.rst_record['recall'].values()) / self.class_num

    def evaluate_model(self):
        """
        to be implemented for different model
        """
        raise NotImplementedError

    def clear_cur_epoch(self):
        """
        clear intermediate result & record overall analysis of current epoch
        """
        self.balanced_acc_curve.append(self.rst_record['overall_balanced_acc'])
        self.rst_record = defaultdict(lambda: 0)

        self.pred_flat = []
        self.label_flat = []

        # clear mem
        gc.collect()

    def plot_evolve_curve(self, show=False, save_path=None, model_name=''):
        """
        plot curve for model evolution, optionally saved into file
        """
        plt.close('all')
        for curve_list, curve_name in zip((self.balanced_acc_curve),
                                          ('balanced_acc_curve')):
            for i in range(self.class_num):
                plt.plot([snap_shot[i] for snap_shot in curve_list], label=self.class_name[i])
            plt.legend()
            
            if not save_path is None:
                plt.savefig(os.path.join(save_path, '%s_%s.png' % (model_name, curve_name)), bbox_inches='tight')
            if show:
                plt.show()
            plt.close()

    def print_result(self, rst_record=None):
        """
        print cur-epoch statistcs
        """
        if self.title:
            print(self.title)
        if not rst_record:
            rst_record = self.rst_record
        print('overall balanced acc =', rst_record['overall_balanced_acc'],
              '\nprecision =', *[self.class_name[k] + ' : ' + str(v) for k, v in rst_record['precision'].items()],
              '\nrecall =', *[self.class_name[k] + ' : ' + str(v) for k, v in rst_record['recall'].items()],
              '\nexample cnt =', self.example_cnt)

        name_max_len = max([len(self.class_name[i]) for i in range(self.class_num)])
        name_str = '\t'.join([self.class_name[i] for i in range(self.class_num)])
        title_str = '%s\t%s' % (' ' * name_max_len, name_str)
        print('confusion_matrix:')
        print(title_str)
        for i in range(self.class_num):
            print(self.class_name[i] + '\t' + '\t'.join([str(cnt) for cnt in rst_record['confusion_matrix'][i]]))
        
        print('mean prob matrix:')
        print(title_str)
        for i in range(self.class_num):
            print(self.class_name[i] + '\t' + '\t'.join([str(prob) for prob in rst_record['mean_prob_matrix'][i]]))

        sys.stdout.flush()

    def plot_cur_epoch_curve(self, show=False, save_path=None, model_name='', use_subdir=True):
        """
        cal & plot curve, optionally saved into file
        """
        def _cal_curve():
            fpr = dict()
            tpr = dict()
            precision = dict()
            recall = dict()
            pr_thr = dict()
            
            for i in range(self.class_num):
                cur_pred_flat = np.array(self.pred_flat)[:, i]
                cur_label_flat = np.array(self.label_flat)[:, i]
                
                fpr[i], tpr[i], _ = skmt.roc_curve(y_score=cur_pred_flat, y_true=cur_label_flat)  # roc
                precision[i], recall[i], pr_thr[i] = skmt.precision_recall_curve(probas_pred=cur_pred_flat, y_true=cur_label_flat)  # precision-recall
                
            return fpr, tpr, precision, recall, pr_thr

        if not self.record_prob:
            warnings.warn('prob (pred-label) not recorded, not able to plot')
            return

        fpr, tpr, precision, recall, pr_thr = _cal_curve()
        plt.close('all')

        # roc
        for i in range(self.class_num):
            plt.plot(fpr[i], tpr[i], label=self.class_name[i])
        plt.legend()

        if save_path:
            if use_subdir:
                os.makedirs(os.path.join(save_path, 'roc'), exist_ok=True)
                plt.savefig(os.path.join(save_path, 'roc/%s.png' % model_name), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(save_path, '%s_roc.png' % model_name), bbox_inches='tight')
        if show:
            plt.title('roc - %s' % model_name)
            plt.show()
        plt.close()

        # precision-recall
        for i in range(self.class_num):
            plt.plot(recall[i], precision[i], label=self.class_name[i])
            for thr_val in [0.03, 0.25, 0.5, 0.75, 0.97]:
                idx = np.argmin(abs(pr_thr[i] - thr_val))
                r = recall[i][idx]
                p = precision[i][idx]
                plt.scatter(r, p, marker='.')
                plt.annotate(str(thr_val), (r,p))

        plt.legend()

        if save_path:
            if use_subdir:
                os.makedirs(os.path.join(save_path, 'pre-recall'), exist_ok=True)
                plt.savefig(os.path.join(save_path, 'pre-recall/%s.png' % model_name), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(save_path, '%s_pre-recall.png' % model_name), bbox_inches='tight')
        if show:
            plt.title('pre-recall - %s' % model_name)
            plt.show()
        plt.close()


class TF_Classif_Record(Classif_Metric_Record):
    """
    evaluate model given dataset.iterator & network's pred under TF framework;
    an active session will be created if not given given
    """

    def __init__(self, tf_pred, tf_label, tf_itr_init_op, tf_feed_dict, sess, class_num, class_name=None, record_prob=False, title=''):
        super(TF_Classif_Record, self).__init__(class_num, class_name, record_prob, title)

        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.InteractiveSession(config=config)
        self.sess = sess

        self.tf_pred = tf_pred # assumed to be probablistic output
        self.tf_label = tf_label
        self.tf_itr_init_op = tf_itr_init_op  # assumed to not repeat without exception
        self.tf_feed_dict = tf_feed_dict # should not contain any variable input in test time! (require input solved by iterator)

    def evaluate_model(self):
        """
        evaluate the model on given dataset
        the iterator is assumed to not repeat without exception
        the prediction is assumed to be probablistic prediction for one-hot encoding
        """
        # per epoch statistic
        self.pred_flat = []
        self.label_flat = []
        
        self.sess.run(self.tf_itr_init_op)
        try:
            while True:
                [prob_pred, label] = self.sess.run([self.tf_pred, self.tf_label],
                                                   feed_dict=self.tf_feed_dict)

                self.accumulate_rst(prob_pred=prob_pred, label=label)

        except tf.errors.OutOfRangeError:
            pass
        self._cal_stat_on_rstrecord(self.record_prob)


class General_Classif_Record(Classif_Metric_Record):
    """
    evaluate model given an arbitrary model
    """

    def __init__(self, class_num, class_name=None, record_prob=False, title=''):
        super(General_Classif_Record, self).__init__(class_num, class_name, record_prob, title)

    def evaluate_model(self, model_func, input_label_itr, is_onehot=True, print_rst=True):
        """
        request an input_label_itr to fetch (input, label) pair in a for loop;
        a model_func to get probablistic prediction given (input, label) pair
        """
        for batch_input, batch_label in input_label_itr():
            prob_pred = model_func(batch_input, batch_label)
            self.accumulate_rst(prob_pred=prob_pred, label=batch_label, is_onehot=is_onehot, record_prob=self.record_prob)
        self._cal_stat_on_rstrecord(self.record_prob)

    def evaluate_model_at_once(self, prob_pred, label, is_onehot=True):
        """
        pass in directly probablistic pred & label
        """
        self.accumulate_rst(prob_pred=prob_pred, label=label, is_onehot=is_onehot, record_prob=True)
        self._cal_stat_on_rstrecord(record_prob=True)


class Bbox_Metric_Record(Classif_Metric_Record):
    """
    detection regarded as multiple classification + bbox localization
    """
    def __init__(self, class_num, class_name=None, elem='pixel', match_score='IoU', match_threshold='0.5-0.95/0.05',
                 confidence_type='prob', filter_mode=None, config={}):
        super(Bbox_Metric_Record, self).__init__(class_num, class_name=class_name)

        assert confidence_type in ['prob', 'score']

        # configure elem type
        assert elem in ['pixel', 'point', 'realxy']
        if elem == 'point':
            self.bi_intersection = lambda b1, b2: utils.Det_Bounding_Box.elem_intersection(b1, b2)
            self.get_size = lambda b: max(b.elem_size, 1)  # in case of empty set (divided by 0)
        else:
            self.bi_intersection = lambda b1, b2: utils.Det_Bounding_Box.xy_intersection(b1, b2)
            self.get_size = lambda b: b.size
        self.elem = elem

        # configure match score type
        assert match_score in ['IoU', 'pred_overlap', 'label_overlap']
        if match_score == 'pred_overlap':
            self._cal_match_score = self._cal_pred_overlap
        elif match_score == 'label_overlap':
            self._cal_match_score = self._cal_label_overlap
        else:
            self._cal_match_score = self._cal_IoU
        self.match_score = match_score

        # configure match threshold
        if type(match_threshold) == str and '-' in match_threshold:
            # 'min-max/res' as a list of float
            if '/' in match_threshold:
                match_threshold = match_threshold.split('/')
                res = float(match_threshold[-1])
                match_threshold = match_threshold[0]
            else:
                res = 0.05  # default resolution to 0.05
            rg = [float(i) for i in match_threshold.split('-')]
            self.match_threshold = np.arange(rg[0], rg[1] + res, res)  # include end point
        else:
            self.match_threshold = [float(match_threshold)]  # single float


        # configure filter mode
        assert filter_mode is None or set(filter_mode.keys()).issubset(set(['blockage', 'focus_size']))
        self.filter_mode = filter_mode
        self._gen_bbox_filter()

        self.config = config

        # cur epoch record
        self.example_cnt = 0  # cnt of examples
        self.prob_list = []  # list of prob for each pred bbox, consistent with those in score_matrix_list
        self.rst_record = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # {[class]: {[thr]:{stat of the class given thr}, ...}, ...}

        # model evolution curve (over epoches)
        self.balanced_acc_curve = []
        self.precision_curve = []
        self.recall_curve = []

    def _cal_pred_overlap(self, pred_bbox, label_bbox):
        intersection = self.bi_intersection(pred_bbox, label_bbox)
        return intersection / self.get_size(pred_bbox)
        
    def _cal_label_overlap(self, pred_bbox, label_bbox):
        intersection = self.bi_intersection(pred_bbox, label_bbox)
        return intersection / self.get_size(label_bbox)

    def _cal_IoU(self, pred_bbox, label_bbox):
        intersection = self.bi_intersection(pred_bbox, label_bbox)
        union = self.get_size(pred_bbox) + self.get_size(label_bbox) - intersection
        return intersection / union

    def _cal_matchscore_matrix(self, pred_bboxlist, label_bboxlist):
        """
        both detect_bbox & label bbox assumed to be a list of Det_Bounding_Box obj
        matrix are constructed as [pred][label]
        """
        matrix = np.zeros(shape=(len(pred_bboxlist), len(label_bboxlist)))
        for i in range(len(pred_bboxlist)):
            for j in range(len(label_bboxlist)):
                matrix[i, j] = self._cal_match_score(pred_bboxlist[i], label_bboxlist[j])
        return matrix

    def __blocking_bbox(self, fixed_bbox, bbox):
        # no intersection & at least farther & angle overlaped 
        return self.bi_intersection(fixed_bbox, bbox) < 1e-5 and \
               fixed_bbox.dist_var[0] < bbox.dist_var[0] and \
               utils.calc_overlap_interval(fixed_bbox.angle_var, bbox.angle_var) / (bbox.angle_var[1] - bbox.angle_var[0]) > self.filter_mode['blockage']['overlap_threshold']

    def __blockage_filtering(self, pred_bboxlist, label_bboxlist):
        # use polar coord for convenience
        origin = self.filter_mode['blockage']['origin']

        for bbox in label_bboxlist:
            bbox.to_polar(origin)
        for bbox in pred_bboxlist:
            bbox.to_polar(origin)

        for label_idx in range(len(label_bboxlist)):
            label_bbox = label_bboxlist[label_idx]
            if label_bbox.valid and label_bbox.blockage > self.filter_mode['blockage']['overlap_threshold']:
                label_bbox.valid = False
                
            # farther away (no overlap) & angle overlaped > thr
            for bbox in pred_bboxlist:  # chk all pred bbox
                if bbox.valid and self.__blocking_bbox(label_bbox, bbox):
                    bbox.valid = False
            for other_label_idx in range(len(label_bboxlist)):  # chk other label bbox
                if other_label_idx != label_idx:
                    bbox = label_bboxlist[other_label_idx]
                    if bbox.valid and self.__blocking_bbox(label_bbox, bbox):
                        bbox.valid = False

    def __infocus_bbox(self, bbox):
        return bbox.y_var[0] > self.filter_mode['focus_size']['y'][0] and \
               bbox.y_var[1] < self.filter_mode['focus_size']['y'][1] and \
               bbox.x_var[0] > self.filter_mode['focus_size']['x'][0] and \
               bbox.x_var[1] < self.filter_mode['focus_size']['x'][1]

    def __infocus_filtering(self, pred_bboxlist, label_bboxlist):
        for bbox in pred_bboxlist:
            if bbox.valid and not self.__infocus_bbox(bbox):
               bbox.valid = False
        for bbox in label_bboxlist:
            if bbox.valid and not self.__infocus_bbox(bbox):
               bbox.valid = False

    def _gen_bbox_filter(self):
        """
        generate bbox filter function given specified config
        """
        filter_list = []
        if self.filter_mode is not None:
            if 'focus_size' in self.filter_mode:
                # select only the bbox in focus size
                filter_list.append(self.__infocus_filtering)
            if 'blockage' in self.filter_mode:
                # select only the bbox can be directly seen from my position
                filter_list.append(self.__blockage_filtering)
        self.filter_list = filter_list

    def _filtering_bbox(self, pred_bboxlist, label_bboxlist):
        """
        filtering bbox by setting the bbox.valid flag
        """
        for func in self.filter_list:
            func(pred_bboxlist, label_bboxlist)

    @staticmethod
    def _cal_stat_at_threshold(score_matrix, pred_flag, label_flag, thr):
        match_matrix = (score_matrix > thr).astype(int)
        label_match = match_matrix.sum(axis=0)
        pred_match = match_matrix.sum(axis=1)

        for l_idx in np.where(np.logical_not(label_flag))[0]: # invalid label
            for p_idx in np.where(pred_match == 1)[0]:  # pred uniquely matching
                if pred_flag[p_idx] and match_matrix[p_idx][l_idx] == 1:  # uniquely matched (valid pred - invalid label)
                    pred_flag[p_idx] = False  # flag as invalid (as algorithm try matching invalid label)

        match_matrix = match_matrix[np.where(pred_flag)[0], :][:, np.where(label_flag)[0]]  # slicing according to validity
        label_match = match_matrix.sum(axis=0)
        pred_match = match_matrix.sum(axis=1)
        
        rst_stat = {
            'fp': (pred_match == 0).sum(),  # fp: false positive - unmatched pred
            'fn': (label_match == 0).sum(),  # fn: false_negative - unmatched label
            'matched_pred': (pred_match >= 1).sum(),  # matched_pred: as long as the pred matched to a label
            'matched_label': (label_match >= 1).sum(),  # matched_label: as long as the label matched by a pred
            'bestmatch_tp': min((pred_match >= 1).sum(), (label_match >= 1).sum()),  # tp: true_positive - best matched pred-label (each can be matched at most once)
            'strict_tp': min((pred_match == 1).sum(), (label_match == 1).sum()), # tp: true_positive - strictly 1-1 matched pred-label
            'merge': (pred_match > 1).sum(),    # merge: merged_target - one pred matching multi label
            'split': (label_match > 1).sum(),  # split: split_target - multi pred matching one label
            'predbox_cnt': len(pred_match),
            'labelbox_cnt': len(label_match)
        }
        return rst_stat

    def _cal_stat_from_rstrecord(self, rst_record):
        for cls_idx in self.class_name:
            cls_record = rst_record[cls_idx]
            for thr in self.match_threshold:
                cur_record = cls_record[thr]
                predbox_cnt = cur_record['predbox_cnt']
                labelbox_cnt = cur_record['labelbox_cnt']

                cur_stat = {}
                cur_stat['strict_precision'] = cur_record['strict_tp'] / predbox_cnt
                cur_stat['bestmatch_precision'] = cur_record['bestmatch_tp'] / predbox_cnt
                cur_stat['matched_precision'] = (cur_record['matched_pred']) / predbox_cnt

                cur_stat['strict_recall'] = cur_record['strict_tp'] / labelbox_cnt
                cur_stat['bestmatch_recall'] = cur_record['bestmatch_tp'] / labelbox_cnt
                cur_stat['matched_recall'] = (cur_record['matched_label']) / labelbox_cnt
                
                cur_stat['totally_miss'] = cur_record['fn'] / labelbox_cnt
                cur_stat['totally_false_alarm'] = cur_record['fp'] / predbox_cnt
                cur_stat['merge'] = cur_record['merge']
                cur_stat['split'] = cur_record['split']

                cur_record['stat'] = cur_stat
                
            # mean over thr
            for k in cur_stat.keys():
                val = 0
                for thr in self.match_threshold:
                    val += cls_record[thr]['stat'][k]
                cls_record['mean_stat'][k] = val / len(self.match_threshold)
        
        # mean over class
        for k in cur_stat.keys():
            val = 0
            for cls_idx in self.class_name:
                val += rst_record[cls_idx]['mean_stat'][k]
            rst_record['mean_stat'][k] = val / self.class_num

    def _cal_clsrecord_from_scorematrix(self, score_matrix, pred_flag, label_flag, cls_record):
        for thr in self.match_threshold:
            cur_record = cls_record[thr]
            rst_stat = self._cal_stat_at_threshold(score_matrix, pred_flag.copy(), label_flag.copy(), thr)
            for k, v in rst_stat.items():
                cur_record[k] += v

    def eval_statistics(self, pred_bboxlist, label_bboxlist):
        """
        evaluate model for the given example (prediction-label pair)
        both pred_bboxlist & label_bboxlist assumed to be a list of definition for class Det_Bounding_Box 
        """
        rst_record = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.accumulate_rst(pred_bboxlist, label_bboxlist, rst_record=rst_record)
        self._cal_stat_from_rstrecord(rst_record=rst_record)

        return rst_record

    def accumulate_rst(self, pred_bboxlist, label_bboxlist, rst_record):
        """
        accumulate the result over the whole dataset
        """
        pred_list = [utils.Det_Bounding_Box(bbox) for bbox in pred_bboxlist]
        label_list = [utils.Det_Bounding_Box(bbox) for bbox in label_bboxlist]
        self._filtering_bbox(pred_list, label_list)

        for n_idx in self.class_name:  # idx of name
            pred_flag = np.array([b.valid for b in pred_list if b.prob[n_idx] > 0])
            label_flag = np.array([b.valid for b in label_list if b.prob[n_idx] == 1])
            score_matrix = self._cal_matchscore_matrix(pred_list, label_list)

            self.example_cnt += 1
            cls_record = rst_record[n_idx]
            # cls_record['cnt']['predbox'] += pred_flag.sum()  # cnt of pred bbox
            # cls_record['cnt']['labelbox'] += label_flag.sum()  # cnt of label bbox
            self._cal_clsrecord_from_scorematrix(score_matrix, pred_flag, label_flag, cls_record=cls_record)

    def evaluate_model(self, model_func, input_label_itr, is_onehot=True):
        """
        request an input_label_itr to fetch (input, label) pair in a for loop;
        a model_func to get probablistic prediction given (input, label) pair
        """
        for batch_input, label_bboxlist in input_label_itr():
            pred_bboxlist = model_func(batch_input, label_bboxlist) # pred_bboxlist should be a valid definition
            self.accumulate_rst(pred_bboxlist=pred_bboxlist, label_bboxlist=label_bboxlist, rst_record=self.rst_record)
        self._cal_stat_from_rstrecord(self.rst_record)

    def clear_cur_epoch(self):
        """
        clear the remaining result & stat
        """
        self.example_cnt = 0
        self.prob_list = []
        self.rst_record = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        gc.collect()

    def print_result(self, mean_over_thr=True, mean_over_class=True, rst_record=None, raw_stat=False):
        """
        print the calculated stat record
        """
        def __print_dict(d, except_k=[]):
            for k, v in d.items():
                if k not in except_k:
                    print('%s\t%s' % (str(k), str(v)))

        if rst_record is None:
            rst_record = self.rst_record
        
        for cls_idx in self.class_name:
            print('\n === result on class %s ===' % self.class_name[cls_idx])
            cls_record = rst_record[cls_idx]

            for thr in self.match_threshold:
                print('\nresult at thr=%.3f :' % thr)
                __print_dict(cls_record[thr]['stat'])
                if raw_stat:
                    __print_dict(cls_record[thr], except_k=['stat'])

        
        # mean stat over thr
        if mean_over_thr and len(self.match_threshold) > 1:
            print('\n\n === mean stat over threshold ===')
            for cls_idx in self.class_name:
                print('\nmean result on class %s' % self.class_name[cls_idx])
                __print_dict(rst_record[cls_idx]['mean_stat'])

        # cal mean stat over class
        if mean_over_class and self.class_num > 1:
            print('\n\n === mean stat over class ===')
            __print_dict(rst_record['mean_stat'])

        print('total examples\t%d' % self.example_cnt)


class Tracking_SOT_Record(Metric_Record):
    def __init__(self):
        super(Tracking_SOT_Record, self).__init__()
        self.rst_record = {'frame_cnt': 0, 'lost_target': 0, 'iou_sum': 0}
        self.track_record = {}

    def accumulate_rst(self, label, pred, track_key=None, rst_record=None):
        """
        accumulate the result over the whole dataset, one track a time
        """
        if rst_record is None:
            rst_record = self.rst_record

        lost_target = 0
        iou_sum = 0
        for label_box, pred_box in zip(label, pred):
            label_box = utils.Bounding_Box(label_box, def_type='xywh')
            pred_box = utils.Bounding_Box(pred_box, def_type='xywh')
            iou = pred_box.IoU(label_box)
            iou_sum += iou
            if iou == 0:
                lost_target += 1
        rst_record['frame_cnt'] += len(label)
        rst_record['iou_sum'] += iou_sum
        rst_record['lost_target'] += lost_target

        if track_key:
            self.track_record[track_key] = {'frame_cnt': len(label), 'lost_target': lost_target, 'iou_sum': iou_sum}

    def _cal_stat(self, record):
        record['robustness'] = np.exp(-30.0 * record['lost_target'] / record['frame_cnt'])
        record['mean_iou'] = record['iou_sum'] / record['frame_cnt']

    def get_result(self, rst_record=None):
        """
        calculate and return the result of evaluation
        """
        if rst_record is None:
            rst_record = self.rst_record
        self._cal_stat(rst_record)

        if self.track_record:
            for r in self.track_record.values():
                self._cal_stat(r)
        return rst_record