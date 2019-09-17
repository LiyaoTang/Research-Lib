#!/usr/bin/env python
# coding: utf-8
'''
module: utilities for managing (dcit-based) configration, including:
    convert argparse into config dict
    convert yaml into config dict
    merge multiple config (with override)
    specify argparser by yaml file
'''
__all__ = ('load_config_into_argparser',
           'load_yaml_into_argparser',
           'merge_config',
           'merge_yaml_into_cfg',
           'merge_args_into_cfg')

import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected, given ', type(v))

def load_config_into_argparser(cfg, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    for k, v in cfg.items():
        if type(v) is bool:
            parser.add_argument('--' + k, dest=k, type=str2bool, default=v)
        elif type(v) is dict:
            load_config_into_argparser(cfg[k], parser)
        else:
            parser.add_argument('--' + k, dest=k, type=type(v), default=v)
    return parser

def load_yaml_into_argparser(path):
    import yaml
    with open(path, 'r') as f:
        cfg = yaml.load(f)
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

def merge_yaml_into_cfg(cfg, path):
    import yaml
    with open(path, 'r') as f:
        ext = yaml.load(f)
    return merge_config(cfg, ext)

def merge_args_into_cfg(args, cfg):
    attr_list = [attr for attr in dir(args) if '_' not in attr]
    args_dict = {attr: getattr(args, attr) for attr in attr_list}
    return merge_config(cfg, args_dict)