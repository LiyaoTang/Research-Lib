#!/usr/bin/env python
# coding: utf-8
"""
module: resnet backbone use in siamRPN++, augmented with self-constructed ops
"""
import math
import torch

import torch.nn as nn
import Torch_Layers as torch_layers

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


def conv2d(in_planes, out_planes, kernel_size, bias=True,
           padding=0, stride=1, dilation=1, padding_mode='zeros'):
    return torch_layers.Conv2d(in_planes, out_planes, kernel_size=kernel_size, bias=bias,
                               stride=stride, padding=padding, dilation=dilation,
                               padding_mode=padding_mode)

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return conv2d(in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=dilation, dilation=dilation)

class BasicBlock(nn.Module):
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

        self.conv1 = conv2d(inplanes, planes, kernel_size=3, bias=False,
                            stride=stride, dilation=dd, padding=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv2d(planes, planes * 4, kernel_size=1, bias=False)
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


class ResNet(nn.Module):
    def __init__(self, block, layers, used_layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        layer3 = 3 in used_layers or 4 in used_layers
        layer4 = 4 in used_layers

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):  # use the base class
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
                    conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes*block.expansion),
                )
            else:
                if dilation > 1:  # triggered in layer-3/4
                    dd = dilation // 2
                    padding = dd
                else:  # triggered in layer-2
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    conv2d(self.inplanes, planes*block.expansion, kernel_size=3,
                           stride=stride, bias=False, padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))  # 1st block
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


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


# if __name__ == '__main__':
#     net = resnet50(used_layers=[4])
#     print(net)
#     net = net.cuda()

#     var = torch.FloatTensor(1, 3, 127, 127).cuda()
#     # var = Variable(var)

#     net(var)
#     print('*************')
#     var = torch.FloatTensor(1, 3, 255, 255).cuda()
#     # var = Variable(var)
#     net(var)
