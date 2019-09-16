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
    moddified AlexNet where, 
        reduced conv stride & pooling => better spatial resolution for tracking
        removed group in conv2 & conv4 => more parameters
        batchnorm => believed to be better than lrn-norm

    width_mult: to double the channel
    '''

    def __init__(self, width_mult=1):
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
    '''
    mobile net backbone as feature extractor
    width_mult: to obtain more channels (thus more representative)
    '''

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
            # t, c, n, s, d = expand_ratio, out channel, block num, stride, dilation
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 1, 2],
            [6, 96, 3, 1, 2],
            [6, 160, 3, 1, 4],
            [6, 320, 1, 1, 4],
        ]

        self.channels = [int(c * width_mult) for c in [24, 32, 96, 320]]
        self.last_channel = int(1280 * width_mult)

        # first layer
        input_channel = int(32 * width_mult)
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
        out = tuple([outputs[i] for i in self.used_layers])
        return out


class ResNet(nn.Module):
    '''
    resnet with dilated-conv
    '''

    def __init__(self, block, layers, used_layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)  # 3
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        layer3 = True if 3 in used_layers else False
        layer4 = True if 4 in used_layers else False

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.maxpool(x_)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        out = [x_, p1, p2, p3, p4]
        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out


def ResNet18(**kwargs):
    ''' resnet-18 model '''
    model = ResNet(Residual_Block, [2, 2, 2, 2], **kwargs)
    return model


def ResNet34(**kwargs):
    ''' resnet-34 model '''
    model = ResNet(Residual_Block, [3, 4, 6, 3], **kwargs)
    return model


def ResNet50(**kwargs):
    ''' resnet-50 model '''
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


''' Residual block '''


class Residual_Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        padding = 2 - stride

        if dilation > 1:
            padding = dilation

        dd = dilation
        pad = padding
        if downsample is not None and dilation > 1:
            dd = dilation // 2
            pad = dd

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=pad, dilation=dd, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class InvertedResidual(nn.Module):
    '''
    inverted residual blocks for mobile net v2
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, 'stride and dilation must have one equals to 1 at least'

        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


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
    def __init__(self, anchor_num=5, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn%d' % (i + 2), DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))

        if self.weighted:  # trainable weights
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
