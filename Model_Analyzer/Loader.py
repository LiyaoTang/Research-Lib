#!/usr/bin/env python
# coding: utf-8
"""
module: dirty codes to parse project-dependent model / dataset name to model object / data feeder
script: prediction - load model & corresponding feeder, save its prediction onto disk
"""

import sys
root_dir = '../'
sys.path.append(root_dir)

import os
import re
import warnings
import collections
import sklearn as sk
import Data_Feeder as feeder


class Model_Info(object):
    """
    class to store info of each model
    """
    def __init__(self, model_type, model_group, model_name, dirpath, log_dir, epoch=-1):
        super(Model_Info, self).__init__()
        self.type = model_type
        self.group = model_group
        self.name = model_name
        self.dirpath = dirpath
        self.log_dir = log_dir
        self.epoch = epoch

    def __str__(self):
        return '%s %s %s %s %s' % (self.type, self.group, self.name, self.dirpath, self.log_dir)

class Model_Group(object):
    """
    class to specify model group info
    """
    def __init__(self, group_name, dirpath, log_dir):
        super(Model_Group, self).__init__()
        self.group_type = self.parse_model_type(group_name)
        self.group_name = group_name
        self.dirpath = dirpath
        self.log_dir = log_dir
        self.models_info = [Model_Info(self.group_type, self.group_name, name, self.dirpath, self.log_dir) for name in self._find_models()]

    def parse_model_type(self, name):
        """
        get model type given name, raise TypeError if failed
        """
        if 'sk' in name:
            cur_type = 'sk'
        elif 'xgb' in name:
            cur_type = 'xg'
        elif 'tf' in name:
            cur_type = 'tf'
        elif name == 'ext_model':
            cur_type = 'ext_model'
        else:
            raise TypeError('not recognized model type from name \"%s\"' % name)

        return cur_type
        
    def _find_models(self):
        if self.group_type == 'tf':
            model_names = []
            for dirpath, curdir, fnames in os.walk(self.dirpath):
                for fn in fnames:
                    cur_name = '-'.join('.'.join(fn.split('.')[:-1]).split('-')[:-1])
                    if cur_name.startswith(self.group_name) and cur_name not in model_names:
                        model_names.append(cur_name)
        elif self.group_type == 'ext_model':  # external model - only prediction provided, no true model to load
            model_names = ['ext_model']
        else:
            model_names = []
            for dirpath, curdir, fnames in os.walk(self.dirpath):
                model_names.extend(fnames)
        return model_names

    def __str__(self):
        return '\n'.join([md_info.__str__() for md_info in self.models_info])

class Project_Loader(object):
    """
    parsing project-dependent mapping logic to regarding paths
    """
    def __init__(self, dataset_name, root_dir='../', verbose=False):
        if dataset_name not in ['corner', 'back']:
            raise TypeError('not supported dataset \"%s\"' % dataset_name)

        # project level paths
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(root_dir, 'Data/', dataset_name)
        self.model_dir = os.path.join(root_dir, 'Model_Trainer/', dataset_name, 'Model')
        self.pred_dir = os.path.join(root_dir, 'Model_Trainer/', dataset_name, 'Prediction')
        self.log_dir = os.path.join(root_dir, 'Model_Trainer/', dataset_name, 'Log')
        
        self.model_groups = [Model_Group(gname, os.path.join(self.model_dir, gname), os.path.join(self.log_dir, gname))
                             for gname in os.listdir(self.model_dir)]

        # project level info
        if dataset_name == 'corner':
            self.input_type = 'point_cloud'
            self.output_type = 'point_cloud'
        elif dataset_name == 'back':
            self.input_type = 'point_cloud'
            self.output_type = 'bounding_box'

        if verbose:
            self.print_info()

    def print_info(self):
        """
        print out parsed project info
        """
        print('root dir \"%s\"' % self.root_dir)
        print('dataset name \"%s\"' % self.dataset_name)
        print('model dir \"%s\"' % self.model_dir)
        print('pred dir \"%s\"' % self.pred_dir)
        print('log dir \"%s\"' % self.log_dir)
        print('model groups\n%s' % '\n\n'.join([group.__str__() for group in self.model_groups]) )

    def select_model(self, model_group='.*', model_name='.*', sort=True):
        """
        select models with corresponding model group / name
        Args:
            model_group: search models in all groups that model_group can match
            model_name:  select models that model_name can match
            => to load all models, leave both as '.*'
        """
        selected_groups = [g for g in self.model_groups if re.fullmatch(model_group, g.group_name)]
        assert len(selected_groups) > 0

        model_name = model_name.replace('|', '\|')
        selected_models = [m for m in sum([cur_group.models_info for cur_group in selected_groups], []) if re.fullmatch(model_name, m.name)]
        assert len(selected_models) > 0

        if sort:
            selected_models = sorted(selected_models, key=lambda md_info: md_info.name)
        return selected_models

    def load_models(self, model_group='.*', model_name='.*', verbose=False, config={}):
        """
        load a selected model / model group with corresponding feeder,
        where config should be a dict containg all necessary config dict for all model**s**
        """
        selected_models = self.select_model(model_group, model_name)

        if 'feeder' in config and config['feeder'] is not None:
            self.feeder = config['feeder']
        else:
            common_feeder = set()
            gen_feeder = lambda: None
            for cur_info in selected_models:
                cur_feeder_class, gen_feeder = self._find_feeder(cur_info)
                common_feeder.add(cur_feeder_class)
            assert len(common_feeder) == 1
            self.feeder = gen_feeder()

        if hasattr(self, 'models'):
            loaded_models = []
        loaded_models = []
        for md_info in selected_models:
            if md_info.name in config:
                cur_config = config[md_info.name]
            else:
                cur_config = config
            if verbose: print('loading model %s with config %s' % (md_info.name, cur_config))
            loaded_models.append(dict(zip(['model', 'pred_prob', 'info'], [*self._load_model(md_info, cur_config), md_info])))
        self.models = loaded_models

        return self.models, self.feeder

    def _load_model(self, model_info, config):
        """
        parse to select a correct loader, where config contains special concern for models
        """
        model = None
        pred_prob = None
        if model_info.type == 'sk':
            model = sk.externals.joblib.load(os.path.join(model_info.dirpath, model_info.name))

            cols = [s for s in model_info.name.split('_') if '-' in set(s) and s != model_info.group]
            assert len(cols) == 1
            cols = [int(s) for s in cols[0].split('-')]
            pred_prob = lambda x, md=model, cols=cols: md.predict_proba(x[:, cols])

        elif model_info.type == 'tf':
            md_path = os.path.join(model_info.dirpath, model_info.name)
            if 'epoch' in config.keys():
                ep_num = config['epoch']
            else:
                meta_files = [i for i in os.listdir(model_info.dirpath) if i.startswith(model_info.name) and i.endswith('.meta')]
                ep_num = max([int(i.rstrip('.meta').split('-')[-1]) for i in meta_files])

            if 'gpu_num' in config:
                os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_num']
            import tensorflow as tf

            # session config
            gpu_options = tf.GPUOptions(allow_growth=True,
                                        per_process_gpu_memory_fraction=config['gpu_mem'] if 'gpu_mem' in config else 0.99,
                                        visible_device_list=config['gpu_num'] if 'gpu_num' in config else '0')
            sess_cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options,
                                      device_count={'CPU': 1, 'GPU': int(config['gpu_num']) if 'gpu_num' in config else 0})
            
            # new graph & session
            tf_g = tf.Graph()
            sess = tf.InteractiveSession(graph=tf_g, config=sess_cfg)
            model = sess

            saver = tf.train.import_meta_graph('%s-%d.meta' % (md_path, ep_num))
            saver.restore(sess, '%s-%d' % (md_path, ep_num))
            tf_g.finalize()

            tf_prob = sess.graph.get_tensor_by_name('prob_out/prob_out:0')
            tf_phase = sess.graph.get_tensor_by_name('input/phase:0')
            tf_X = sess.graph.get_tensor_by_name('input/X:0')
            cols = [s for s in model_info.name.split('_') if '-' in set(s) and s != model_info.group and ';' not in set(s)]
            assert len(cols) == 1
            cols = [int(s) for s in cols[0].split('-')]
            print('creating pred_prob')
            pred_prob = lambda img, cols=cols, sess=model, X=tf_X, prob=tf_prob, p=tf_phase: sess.run([prob], feed_dict={X: [img[..., cols]], p: 'train-eval'})[0]
            print('finish loading -----------------------------------')

        elif model_info.type == 'xg':
            warnings.warn('xgboost currently ignored')
        elif model_info.type == 'ext_model':
            warnings.warn('specified to be external model, no true model to load')
        else:
            raise TypeError('not supported model type: \"%s\"' % model_info.type)

        return [model, pred_prob]

    def _find_feeder(self, model_info):
        if self.dataset_name == 'corner':
            if model_info.group == 'sk-randforest':
                feeder_class = feeder.Corner_Radar_Points_Gen_Feeder
                gen_feeder = lambda: feeder.Corner_Radar_Points_Gen_Feeder(data_dir=self.dataset_dir,
                                                                           select_cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            elif model_info.group == 'tf-fcnpipe':
                feeder_class = feeder.Corner_Radar_Boxcenter_Gen_Feeder
                gen_feeder = lambda: feeder.Corner_Radar_Boxcenter_Gen_Feeder(data_dir=self.dataset_dir,
                                                                              select_cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # preserve enough indexes
            else:
                raise TypeError('not supported model: \"%s\"' % model_info)
        elif self.dataset_name == 'back':
            feeder_class = feeder.Back_Radar_Bbox_Gen_Feeder
            gen_feeder = lambda: feeder.Back_Radar_Bbox_Gen_Feeder(data_dir=self.dataset_dir, resolution=4,
                                                                   config={'ext_module': {'radar_label_pb2': self.dataset_dir},
                                                                           'offset': {'pred': 5, 'label': 52}})
        else:
            raise TypeError('not supported dataset \"%s\"' % self.dataset_name)
        
        return feeder_class, gen_feeder

    def make_prediction(self, pred_dir=None, overwrite=False, options=dict()):
        """
        make prediction after model selected
        """
        if pred_dir is None:
            pred_dir = self.pred_dir
        for md in self.models:
            self.feeder.record_prediction(pred_func=md['pred_prob'],
                                          model_name=md['info'].name,
                                          output_dir=pred_dir,
                                          dataset_name=self.dataset_name,
                                          overwrite=overwrite,
                                          options=options)

# run as script
if __name__ == '__main__':
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('--dataset', dest='dataset')
    parser.add_option('--root_dir', dest='root_dir', default='../')
    parser.add_option('--model_group', dest='model_group', default='.*')
    parser.add_option('--model_name', dest='model_name', default='.*')
    parser.add_option('--epoch', dest='epoch', type=int, default=None)

    # prediction
    parser.add_option('--pred_dir', dest='pred_dir', default=None)
    parser.add_option('--overwrite', dest='overwrite', default=False, action='store_true')
    parser.add_option('--pred_type', dest='pred_type', default='csv')
    parser.add_option('--protobuf_path', dest='protobuf_path', default='../../radarfusion/radar-data-processing/scripts')
    parser.add_option('--proto_post', dest='proto_post', default='.prototxt')

    # model export
    parser.add_option('--language', dest='language', default=None)
    parser.add_option('--dest', dest='out_dir', default='./')

    # evaluate
    parser.add_option_group

    parser.add_option('--usage', dest='usage')
    (options, args) = parser.parse_args()

    if options.usage == 'predict':
        pred_opt = dict()
        for k in ['pred_type', 'protobuf_path', 'proto_post']:
            pred_opt[k] = getattr(options, k)
        model_cfg = dict()
        for k in ['epoch']:
            val = getattr(options, k)
            if val is not None: model_cfg[k] = val
        project = Project_Loader(options.dataset, root_dir=options.root_dir, verbose=True)
        project.load_models(model_group=options.model_group, model_name=options.model_name, config=model_cfg, verbose=True)
        project.make_prediction(pred_dir=options.pred_dir, overwrite=options.overwrite, options=pred_opt)

    elif options.usage == 'export':
        sklearn_porter = __import__('sklearn_porter', fromlist=[''])

        model_cfg = dict()
        for k in ['epoch']:
            val = getattr(options, k)
            if val is not None: model_cfg[k] = val
        project = Project_Loader(options.dataset, root_dir=options.root_dir, verbose=True)
        project.load_models(model_group=options.model_group, model_name=options.model_name, config=model_cfg, verbose=True)
        for md in project.models:
            if md['info'].group == 'sk-randforest':
                porter = sklearn_porter.Porter(md['model'], language=options.language)
                output_model = porter.export(embed_data=True)
                with open(os.path.join(options.out_dir, md['info'].name+'.h'), 'w') as f:
                    f.write(output_model)
            else:
                raise NotImplementedError
                
    elif options.usage == 'evaluate':
        from multiprocessing import Pool

        project = Project_Loader(options.dataset, root_dir=options.root_dir, verbose=True)
        project.load_models(model_group=options.model_group, model_name=options.model_name, verbose=True)
        if len(project.models) != 1:
            raise AssertionError('parallel_recorder %d: len(project.models) = %d, instead of 1' % (os.getpid(), len(project.models)))
        
        model_list = project.models.copy()  # shallow copy enough

        cur_cfg = {'feeder': project.feeder}
        for ep in range(project.models[0]['info'].epoch):
            cur_cfg['epoch'] = ep
            cur_model = project.load_models(model_group=options.model_group, model_name=options.model_name, config=cur_cfg, verbose=True)
            assert len(cur_model) == 1
            model_list.append(cur_model[0])
        
    else:
        raise TypeError('usage \"%s\" not supported' % options.usage)