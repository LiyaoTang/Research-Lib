#!/usr/bin/env python
# coding: utf-8
'''
import control for folder Data_Feeder
'''

# __all__ = ['Data_Feeder']

from .base import Parallel_Feeder
from .Data_Feeder import *
from . import Torch_Feeder as torch