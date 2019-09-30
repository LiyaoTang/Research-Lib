import torch
import torch.nn as nn
from . import Torch_Layers as torch_layers
# from .utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=None, padding_mode='zeros'):
    if padding is None:
        padding = dilation
    return torch_layers.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=padding, groups=groups, bias=False, dilation=dilation,
                               padding_mode=padding_mode)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)  # potentially adjust channel 
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)  # stride 1
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):  # 3x3-bn-relu-3x3-bn | identity }-add-relu
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups  # channel num inside bottle-neck
        width = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1=conv1x1(inplanes, width)  # reduce channel
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)  # expand channel num => to a multiplication of 'planes'
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):  # 1x1-bn-relu -3x3-bn-relu -1x1-bn | identity }-add-relu
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,  # layers: num of res-blocks at each stage
                 groups=1, norm_layer=None,
                 stride2dilation=None, stride2pool=None, used_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if used_layer is None:
            used_layer = [4]
        assert not max(used_layer) < 4, 'need to always use feature from last layer(stage)'
        assert not max(used_layer) > 4, 'last layer(stage) is layer-4, but required layer-%d' % max(used_layer)

        if stride2dilation is None:
            # each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution
            # => retain the spatial size & use dilated conv to increase receptive field
            stride2dilation = [False, False, False]
        if len(stride2dilation) != 3:
            raise ValueError('stride2dilation should be None or a 3-element tuple, got {}'.format(stride2dilation))

        if stride2pool is None:
            # each element in the tuple indicates if we should downsample shortcut with pooling, instead of stride-2 1x1 conv
            stride2pool = [False, False, False]
        if len(stride2pool) != 3:
            raise ValueError('stride2pool should be None or a 3-element tuple, got {}'.format(stride2pool))

        self.used_layer = used_layer
        self.norm_layer = norm_layer
        self.inplanes = 64  # input channel for next block
        self.dilation = 1
        self.groups = groups
        self.base_width = 64

        # expand channel num to 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # finish 1st stage (downsample to 1/4)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=stride2dilation[0], pool=stride2pool[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=stride2dilation[1], pool=stride2pool[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=stride2dilation[2], pool=stride2pool[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, pool=False):
        assert not (dilate and pool)  # dilate and pool can not be used together
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            # use dilated-conv to maintain the receptive field 
            # => do NOT downsample: maintain spatial size & resolution (starts from the 2nd block)
            self.dilation *= stride
            stride = 1
        if pool:
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            # need to adjust identity as the main path
            # => can be spatial downsample & channel adjust (originally, by 1x1 conv with stride 2...)
            # => only once in each stage (at the 1st block)
            downsample = [
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ]
            if pool:
                downsample.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            downsample = nn.Sequential(*downsample)

        layers = []  # at least 1 block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        # the input of next block & stage is expanded (a class-level constant)
        # => only expand channel once in each stage (at 1st block, more specifically, the 1st 3x3 conv)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):  # if required more
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # stage-1: stride-2 7x7conv + stride-2 3x3maxpool => downsample to 1/4
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)  # finish stage-1
        f2 = self.layer2(f1)  # downsample to 1/8
        f3 = self.layer3(f2)  # downsample to 1/16
        f4 = self.layer4(f3)  # downsample to 1/32

        out = [x, f1, f2, f3, f4]
        out = [out[i] for i in self.used_layer]
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return tuple(out)

class ResNet_SiamTracker(nn.Module):
    '''
    modified to incooperate input size = 127 / 255 s.t. 
    1. there is no need to drop border info, due to [size - kernel + 2*padding] // stride != 0
        => remove padding at 7x7 conv (consider also the following max-pool)
        => replace 1x1 conv with 3x3 conv in downsample
    2. make the template image output odd kernel
        => remove padding at stage-2 3x3 stride-2 conv
    '''

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, norm_layer=None,
                 stride2dilation=None, used_layer=None):
        super(ResNet_SiamTracker, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if used_layer is None:
            used_layer = [4]
        assert not max(used_layer) < 4, 'need to always use feature from last layer(stage)'
        assert not max(used_layer) > 4, 'last layer(stage) is layer-4, but required layer-%d' % max(used_layer)

        if stride2dilation is None:
            stride2dilation = [False, True, True]  # default setting in siamrpn++
        if len(stride2dilation) != 3:
            raise ValueError('stride2dilation should be None or a 3-element tuple, got {}'.format(stride2dilation))

        self.used_layer = used_layer
        self.norm_layer = norm_layer
        self.inplanes = 64  # input channel for next block
        self.dilation = 1
        self.groups = groups
        self.base_width = 64

        # expand channel num to 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, bias=False)  # even-padding
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # finish 1st stage (downsample to 1/4)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=stride2dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=stride2dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=stride2dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, pool=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            # use dilated-conv to maintain the receptive field 
            # => do NOT downsample: maintain spatial size & resolution (starts from the 2nd block)
            self.dilation *= stride
            stride = 1
        if pool:
            # use stride-1 1x1 conv + max-pool (instead of stride-2 1x1 conv)
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            dilation = previous_dilation
            downsample = [
                conv1x1(self.inplanes, planes*block.expansion, stride=stride),
                norm_layer(planes*block.expansion),
            ]
            if pool:
                downsample.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            downsample = nn.Sequential(*downsample)

        layers = []  # at least 1 block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        # the input of next block & stage is expanded (a class-level constant)
        # => only expand channel once in each stage (at 1st block, more specifically, the 1st 3x3 conv)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):  # if required more
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # stage-1: stride-2 7x7conv + stride-2 3x3maxpool => downsample to 1/4
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)  # finish stage-1
        f2 = self.layer2(f1)  # downsample to 1/8
        f3 = self.layer3(f2)  # downsample to 1/16
        f4 = self.layer4(f3)  # downsample to 1/32

        out = [x, f1, f2, f3, f4]
        out = [out[i] for i in self.used_layer]
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return tuple(out)


def _resnet(type, block, layers, **kwargs):
    if type == 'original':
        class_type = ResNet
    elif type == 'siam':
        class_type = ResNet_SiamTracker

    model = class_type(block, layers, **kwargs)
    return model


def resnet18(type='original', **kwargs):
    '''
	ResNet-18 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    '''
    return _resnet(type, BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(type='original', **kwargs):
    '''
	ResNet-34 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    '''
    return _resnet(type, BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(type='original', **kwargs):
    '''
	ResNet-50 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    '''
    return _resnet(type, Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(type='original', **kwargs):
    '''
    ResNet-101 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    '''
    return _resnet(type, Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(type='original', **kwargs):
    '''
	ResNet-152 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    '''
    return _resnet(type, Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(type='original', **kwargs):
    '''
	ResNeXt-50 32x4d model from
    "Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>
    '''
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(type='original', **kwargs):
    '''
	ResNeXt-101 32x8d model from
    "Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>
    '''
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(type='original', **kwargs):
    '''
	Wide ResNet-50-2 model from
    "Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048
    '''
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(type='original', **kwargs):
    '''
	Wide ResNet-101-2 model from
    "Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048
    '''
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
