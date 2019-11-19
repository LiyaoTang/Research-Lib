#!/usr/bin/env python
# coding: utf-8
'''
module: given a saved model & weights, enable conversion of pytorch <-> tensorflow
        (using onnx as interface in the mid: pytorch <-> onnx <-> tf)
'''

import onnx
import pytorch
import argparse
import tensorflow as tf

