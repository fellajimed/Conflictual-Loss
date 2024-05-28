'''
source: https://github.com/kuangliu/pytorch-cifar
'''
import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_SN=False,
                 expansion=1, flexible_shortcut=False):
        from .utils import apply_SN

        super().__init__()
        self.expansion = expansion
        self.conv1 = apply_SN(nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False), conv_SN)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = apply_SN(nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False), conv_SN)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut_coef = torch.tensor(1., dtype=float)
        if flexible_shortcut:
            self.shortcut_coef = nn.Parameter(self.shortcut_coef)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                apply_SN(nn.Conv2d(in_planes, self.expansion*planes,
                                   kernel_size=1, stride=stride,
                                   bias=False), conv_SN),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut_coef * self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, conv_SN=False,
                 expansion=4, flexible_shortcut=False):
        from .utils import apply_SN

        super().__init__()
        self.expansion = expansion
        self.conv1 = apply_SN(nn.Conv2d(
            in_planes, planes, kernel_size=1, bias=False), conv_SN)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = apply_SN(nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False), conv_SN)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = apply_SN(nn.Conv2d(planes, self.expansion * planes,
                                        kernel_size=1, bias=False), conv_SN)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut_coef = torch.tensor(1., dtype=float)
        if flexible_shortcut:
            self.shortcut_coef = nn.Parameter(self.shortcut_coef)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                apply_SN(nn.Conv2d(in_planes, self.expansion*planes,
                                   kernel_size=1, stride=stride,
                                   bias=False), conv_SN),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut_coef * self.shortcut(x)
        out = F.relu(out)
        return out
