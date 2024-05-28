import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .classification import ClassificationModuleNet
from .resnet_utils import BasicBlock, Bottleneck

# logger object
import logging
logger = logging.getLogger('main_all')


class ResNet_s(ClassificationModuleNet):
    """
    ResNet model from Scratch
    """

    def __init__(self, block, num_blocks, input_shape, nb_classes,
                 device='cpu', planes=[64, 128, 256, 512],
                 strides=[1, 2, 2, 2], conv_SN=False,
                 lin_SN=False, flexible_shortcut=False, **kwargs):
        from .utils import apply_SN

        super().__init__(input_shape, nb_classes, device)
        self.in_planes = planes[0]
        self.conv_SN = conv_SN

        assert len(input_shape) == 3

        self.conv1 = apply_SN(nn.Conv2d(
            input_shape[0], planes[0], kernel_size=3,
            stride=1, padding=1, bias=False), conv_SN)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.layers = nn.Sequential(*[
            self._make_layer(block, plane, n_block, stride=stride,
                             flexible_shortcut=flexible_shortcut)
            for (plane, n_block, stride) in zip(planes, num_blocks, strides)])

        nb_features = np.prod(self.features_forward(
            torch.randn(1, *input_shape)).shape)
        self.linear = apply_SN(nn.Linear(nb_features, nb_classes), lin_SN)

        # to device
        self.to(device)

    def _make_layer(self, block, planes, num_blocks,
                    stride, flexible_shortcut):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.conv_SN,
                                flexible_shortcut=flexible_shortcut))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.features_forward(x)
        out = self.linear(out)
        return out


class ResNet18_s(ResNet_s):
    """
    ResNet18 model from Scratch
    """

    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], input_shape,
                         nb_classes, device, **kwargs)


class ResNet34_s(ResNet_s):
    """
    ResNet34 model from Scratch
    """

    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(BasicBlock, [3, 4, 6, 3], input_shape,
                         nb_classes, device, **kwargs)


class ResNet50_s(ResNet_s):
    """
    ResNet50 model from Scratch
    """

    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(Bottleneck, [3, 4, 6, 3], input_shape,
                         nb_classes, device, **kwargs)


class ResNet101_s(ResNet_s):
    """
    ResNet101 model from Scratch
    """

    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(Bottleneck, [3, 4, 23, 3], input_shape,
                         nb_classes, device, **kwargs)


class ResNet152_s(ResNet_s):
    """
    ResNet152 model from Scratch
    """

    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(Bottleneck, [3, 8, 36, 3], input_shape,
                         nb_classes, device, **kwargs)
