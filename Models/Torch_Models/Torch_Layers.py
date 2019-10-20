#!/usr/bin/env python
# coding: utf-8
'''
module: some self-constructed torch layers as classes
'''

import torch.nn as nn
import torch.nn.functional as F
import Torch_Ops as torch_ops
from torch.nn.modules.utils import _pair

class Conv2d(nn.modules.conv._ConvNd):
    '''
    enable padding with channel-wise average
    (following pytorch conv2d implementation)
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        assert padding_mode in ['zeros', 'avg', 'circular']
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        if self.padding_mode not in ['avg', 'zeros']:
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        elif self.padding_mode == 'avg':
            return F.conv2d(F.pad(input, self.padding, mode='constant', value=input.mean(dims=(-1,-2))),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)
