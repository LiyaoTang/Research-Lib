#!/usr/bin/env python
# coding: utf-8
"""
module: dirty codes to parse log for each project-dependent model
script: show the result of parsing
"""

import os
import re
import warnings
import collections

class Log_Parser(object):
    def __init__(self, log_dir, log_name='.*', verbose=True):
        self.log_files = [os.path.join(d, f) for d, _, fs in os.walk(log_dir) for f in fs if re.fullmatch(log_name, f)]
        self.log_files = sorted(self.log_files)
        if verbose:
            print('\n'.join(self.log_files))
        
    def _parse_sklog(self, log):
        raise NotImplementedError

    def _parse_tflog(self, log):
        """
        return a dict with 'train': [ep0 metric dict, ep1 metric dict, ...], 'val': [ep0 metric dict, ep1 metric dict, ...]
        return None if log file not tagged as finished at the last line
        """

        def _parse_into_metrics(line, ep_metric_dict):
            line = line.split()
            if len(line) == 0: return
            
            if line[0] == 'mean_loss':
                ep_metric_dict['mean_loss'] = float(line[2])
                ep_metric_dict['balanced_acc'] = float(line[-1])
            elif line[0] in ['auc', 'avg_precision']:
                ep_metric_dict[line[0]] = float(line[-1].strip('\']'))

        cnt = 0
        log = log.strip('\n').split('\n')  # split into lines
        if log[-1] != 'finish':
            warnings.warn('Log for model %s NOT finished, skipped...' % log[0])
            return None
        else:
            print('parsing ', log[0])

        metric_dict = {'train': [], 'val': []}
        while not log[cnt].startswith('epoch =====>>>>>> 0'): cnt += 1;  # move to start of the 1st epoch
        while (cnt < len(log)):
            while not log[cnt].startswith('train eval:'): cnt += 1  # move to train eval

            cur_dict = collections.defaultdict(lambda: 0)
            metric_dict['train'].append(cur_dict)
            while not log[cnt].startswith('val eval'): cnt += 1; _parse_into_metrics(log[cnt], cur_dict)  # parse train till val eval

            cur_dict = collections.defaultdict(lambda: 0)
            metric_dict['val'].append(cur_dict)
            while not (log[cnt] == 'finish' or log[cnt].startswith('epoch =====>>>>>>')): cnt += 1; _parse_into_metrics(log[cnt], cur_dict)  # parse val
            cnt += 1
            
        return metric_dict

    def _find_parser(self, log_f):
        if 'sk' in log_f:
            parser = self._parse_sklog
        elif 'tf' in log_f:
            parser = self._parse_tflog
        else:
            raise TypeError('no parser for log file / model type: \"%s\"' % log_f)
        return parser

    def parse_logs(self, model_type=None, label_type='text', show_num=True, plot_singlemodel=True, plot_allmodel=False, verbose=False):
        metric_list = []
        for log_f in self.log_files:
            with open(log_f, 'r') as f:
                log = f.read()
            if model_type:
                cur_parser = self._find_parser(model_type)
            else:
                cur_parser = self._find_parser(log_f)

            cur_metric_dict = cur_parser(log)
            if cur_metric_dict is not None:
                metric_list.append(cur_metric_dict)

        plt = __import__('matplotlib.pyplot', fromlist=[''])
        keys = metric_list[0]['train'][0].keys() # probe the sample keys

        if show_num:
            print('numeric result: ')
            for metric, log_f in zip(metric_list, self.log_files):
                print(log_f) # log file path
                for k in keys:
                    train_curve = [d[k] for d in metric['train']]
                    val_curve = [d[k] for d in metric['val']]
                    print('max %s = train: %f; val: %f' % (str(k), max(train_curve), max(val_curve)))

        fig_idx = 0
        # train vs. val - single model
        if plot_singlemodel:
            for metric, log_f in zip(metric_list, self.log_files):
                for k in keys:
                    train_curve = [d[k] for d in metric['train']]
                    val_curve = [d[k] for d in metric['val']]
                    plt.figure(fig_idx)
                    plt.plot(train_curve, label='train')
                    plt.plot(val_curve, label='val')
                    plt.legend()
                    plt.title('%s : %s' % (log_f.split('/')[-1], str(k)))  # log name as title
                    fig_idx += 1

        # val - across models
        if plot_allmodel:
            for k in keys:
                plt.figure(fig_idx, figsize=(20,20))
                for metric, log_f in zip(metric_list, self.log_files):
                    val_curve = [d[k] for d in metric['val']]
                    name = log_f.split('/')[-1]
                    if label_type == 'box':
                        plt.plot(val_curve, label=name)
                    else:
                        plt.plot(val_curve)
                        an = plt.annotate(name, xy=(len(val_curve) - 1, val_curve[-1]))  # annotate log name to curve
                        an.draggable()
                plt.legend()
                plt.title('val set cmp : %s' % str(k))
                fig_idx += 1
        
        # enter GUI main loop in the end
        plt.show()

# run as script
if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('--log_dir', dest='log_dir', default='./Log')
    parser.add_option('--log_name', dest='log_name', default='.*')
    parser.add_option('--model_type', dest='model_type', default=None)
    parser.add_option('--label_type', dest='label_type', default='text')
    parser.add_option('--show_num',         dest='show_num',         default=1)  # 1 as True & 0 as False
    parser.add_option('--plot_singlemodel', dest='plot_singlemodel', default=0)
    parser.add_option('--plot_allmodel',    dest='plot_allmodel',    default=1)
    (options, args) = parser.parse_args()

    for k in ['plot_singlemodel', 'plot_allmodel', 'show_num']:
        cur_val = getattr(options, k)
        assert cur_val in (0, 1)
        setattr(options, k, bool(cur_val))

    logparser = Log_Parser(options.log_dir, options.log_name, verbose=True)
    print(options)
    logparser.parse_logs(model_type=options.model_type,
                         label_type=options.label_type,
                         show_num=options.show_num,
                         plot_singlemodel=options.plot_singlemodel,
                         plot_allmodel=options.plot_allmodel,
                         verbose=True)
