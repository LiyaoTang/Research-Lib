#!/usr/bin/env python
# coding: utf-8
'''
module: some torch modules/blocks from classic networks
'''

import torch.nn as nn
import torch.nn.functional as F
from . import Torch_Ops as torch_ops
from . import Torch_Layers as torch_layers


''' backbone '''


class AlexNet(nn.Module):
    '''
    self-implemented AlexNet
    careful with: conv stride, group num, batch- vs. lrn-norm, pooling, relu vs. prelu
    '''
    def __init__(self, width_mult=1, auxilary_loss=None):
        super(AlexNet, self).__init__()

        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x == 3 else int(x * width_mult), configs))
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
            )
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class AlexNetLegacy(nn.Module):

    def __init__(self, width_mult=1):
        super(AlexNetLegacy, self).__init__()
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x == 3 else int(x * width_mult), configs))
        self.extract_features = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),

            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),

            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.extract_features(x)
        return x


class MobileNetV2(nn.Sequential):
    def __init__(self, width_mult=1, used_layers=[3, 5, 7]):
        super(MobileNetV2, self).__init__()

        self.interverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 2, 1],
            [6, 96, 3, 1, 1],
            [6, 160, 3, 2, 1],
            [6, 320, 1, 1, 1],
        ]
        # 0,2,3,4,6

        self.interverted_residual_setting = [
            # t, c, n, s, d = expand_ratio, channel, block num, stride, dilation
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 1, 2],
            [6, 96, 3, 1, 2],
            [6, 160, 3, 1, 4],
            [6, 320, 1, 1, 4],
        ]

        self.channels = [24, 32, 96, 320]
        self.channels = [int(c * width_mult) for c in self.channels]

        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult)

        # torch_ops.conv_bn(3, input_channel, 2, 0)
        layer0 = nn.Sequential(
            nn.Conv2d(3, out_channels=input_channel, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=input_channel),
            nn.ReLU6(inplace=True)  # min(max(0,x),6): maximal response=6
        )
        self.add_module('layer0', layer0)


        last_dilation = 1
        self.used_layers = used_layers
        for idx, (t, c, n, s, d) in enumerate(self.interverted_residual_setting, start=1):
            output_channel = int(c * width_mult)

            layers = []
            for i in range(n):  # num of inverted res block
                if i == 0:
                    dd = d if d == last_dilation else max(d // 2, 1)
                    layers.append(InvertedResidual(input_channel, output_channel, s, t, dd))
                else:
                    layers.append(InvertedResidual(input_channel, output_channel, 1, t, d))
                input_channel = output_channel

            last_dilation = d
            self.add_module('layer%d' % (idx), nn.Sequential(*layers))

    def forward(self, x):
        outputs = []
        for idx in range(8):
            name = "layer%d" % idx
            x = getattr(self, name)(x)
            outputs.append(x)
        p0, p1, p2, p3, p4 = [outputs[i] for i in [1, 2, 3, 5, 7]]
        out = [outputs[i] for i in self.used_layers]
        return out


''' Residual block '''


class InvertedResidual(nn.Module):
    '''
    inverted residual blocks for mobile net v2
    TODO: a block, to be placed into layers or else-where
    '''

    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        padding = dilation if dilation > 1 else 2 - stride
        expanded_ch = in_channels * expand_ratio
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, expanded_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(expanded_ch),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(expanded_ch, expanded_ch, 3, stride, padding, dilation=dilation, groups=expanded_ch, bias=False),
            nn.BatchNorm2d(expanded_ch),
            nn.ReLU6(inplace=True),
            # pw-linear 1x1 conv
            nn.Conv2d(expanded_ch, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:  # with shortcut
            return x + self.conv(x)
        else:
            return self.conv(x)


class DepthwiseXCorr(nn.Module):
    '''
    perform cross-relation (as conv) with one feature map (as kernel) on the other (as search)
    '''
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(  # regress the conv kernel
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(  # regress the search region
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(  # refine for output head
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = torch_ops.xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


''' RPN block '''


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelRPN(RPN):
    def __init__(self, anchor_num=5, feature_in=256):
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseRPN(RPN):
    '''
    perform depth-wise cross-correlation to obtain both classification & regression branch
    '''
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)  # classification branch
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)  # regression branch

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MultiRPN(RPN):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn%d' % (i + 2), DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))

        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    @staticmethod
    def _avg(lst):
        return sum(lst) / len(lst)

    @staticmethod
    def _weighted_avg(lst, weight):
        s = 0
        for i in range(len(weight)):
            s += lst[i] * weight[i]
        return s

    def forward(self, z_fs, x_fs):
        cls_branch = []
        loc_branch = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
            c, l = rpn(z_f, x_f)
            cls_branch.append(c)
            loc_branch.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            cls_branch, loc_branch = self._weighted_avg(cls_branch, cls_weight), self._weighted_avg(loc_branch, loc_weight)
        else:
            cls_branch, loc_branch = self._avg(cls_branch), self._avg(loc_branch)
        return cls_branch, loc_branch
