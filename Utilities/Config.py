#!/usr/bin/env python
# coding: utf-8
"""
module: utilities for managing (dcit-based) configration, including:
    convert argparse into config dict
    convert yaml into config dict
    merge multiple config (with override)
    specify argparser by yaml file
"""
# __all__ = ('load_config_into_argparser',
#            'load_yaml_into_argparser',
#            'merge_config',
#            'merge_yaml_into_cfg',
#            'merge_args_into_cfg')
__all__ = ('Frozen_Config', 'Config')

import yaml
import argparse

class Frozen_Config(object):
    def __init__(self, config):
        super(Frozen_Config, self).__init__()
        self.__config = config.copy()
        assert type(self.__config) is dict, 'not able to freeze from ' + repr(config)

    def __getitem__(self, k):
        if type(self.__config[k]) is dict:
            return Frozen_Config(self.__config[k])
        return self.__config[k]
    def __iter__(self):
        return iter(self.__config.copy())
    def __repr__(self):
        return repr(self.__config)
    def keys(self):
        return self.__config.keys()
    def items(self):
        return self.__config.copy().items()
    def values(self):
        return self.__config.copy().values()
    def copy(self):
        return Frozen_Config(self.__config.copy())

class Config(object):
    def __init__(self, config=None):
        super(Config, self).__init__()
        if type(config) is dict:
            self.config = config
        elif type(config) is str and config.endswith('yaml'):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        elif not config:
            self.config = {}
        else:
            raise TypeError('not supported to have', repr(obj), 'as config')
        self.arg_cfg_map = {}  # args parser attr -> key in self.config

    def construct_argparser(self):
        parser, self.arg_cfg_map = load_config_into_argparser(self.config)
        return parser

    def merge_yaml(self, path):
        self.config = merge_yaml_into_cfg(self.config, path)

    def merge_args(self, args):
        self.config = merge_args_into_cfg(self.config, args, self.arg_cfg_map)

    def __getitem__(self, k):
        # if type(self.config[k]) is dict:
        #     return Config(self.config[k])
        return self.config[k]
    def __setitem__(self, k, v):
        self.config[k] = v
    def __delitem__(self, k):
        del self.config[k]
    def __iter__(self):  # used by dict(c:'Config')
        return iter(self.config)
    def __repr__(self):
        return repr(self.config)
    def keys(self):
        return self.config.keys()
    def items(self):
        return self.config.items()
    def values(self):
        return self.config.values()
    def copy(self):
        return Config(self.config.copy())
    def freeze(self):
        return Frozen_Config(self.config)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected, given ', type(v))

def load_config_into_argparser(cfg):
    def _add_cfg(cfg, parser, cfg_keys, arg_cfg_map):
        for k, v in cfg.items():
            if type(v) is dict:
                _add_cfg(cfg[k], parser, cfg_keys.copy() + [k], arg_cfg_map)
            else:
                if type(v) is bool:
                    parser.add_argument('--' + k, dest=k, type=str2bool, default=v)
                else:
                    parser.add_argument('--' + k, dest=k, type=type(v), default=v)
                arg_cfg_map[k] = cfg_keys.copy()  # shallow copy on each element
        return parser, arg_cfg_map

    parser = argparse.ArgumentParser()
    return _add_cfg(cfg, parser, [], {})

def load_yaml_into_argparser(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return load_config_into_argparser(cfg)

def merge_config(cfg, ext):
    for k in ext:
        if k in cfg and type(ext[k]) == type(cfg[k]):
            if type(ext[k]) is dict:
                merge_config(cfg[k], ext[k])
            else:  # override
                cfg[k] = ext[k]
        elif k in cfg:
            raise TypeError('2 configs not compatible: got %s vs. %s' % (str(cfg[k]), str(ext[k])))
        else:  # insert
            cfg[k] = ext[k]
    return cfg

def merge_args_into_cfg(cfg, args, arg_cfg_map={}):
    attr_list = [attr for attr in dir(args) if not attr.startswith('_')]
    args_dict = {attr: getattr(args, attr) for attr in attr_list}

    if not arg_cfg_map:
        return merge_config(cfg, args_dict)

    for k in args_dict:
        cfg_path = arg_cfg_map[k]
        sub_cfg = cfg
        for cfg_k in cfg_path:
            sub_cfg = sub_cfg[cfg_k]
        if sub_cfg[k] is None or type(sub_cfg[k]) == type(args_dict[k]):
            sub_cfg[k] = args_dict[k]  # override & insert
        else:
            raise TypeError('inconsistent config: ' + str(sub_cfg[k]) + ' vs. ' + str(args_dict[k]))
    return cfg

def merge_yaml_into_cfg(cfg, path):
    with open(path, 'r') as f:
        ext = yaml.safe_load(f)
    return merge_config(cfg, ext)