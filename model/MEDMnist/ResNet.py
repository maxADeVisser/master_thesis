"""
source: https://github.com/MedMNIST/experiments/blob/main/MedMNIST3D/models.py
This is the ResNet benchmarkmodel for the MedMNIST3D dataset.

Adapted from kuangliu/pytorch-cifar (https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py?)
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# TODO convert this into a native 3D ResNet implementation
# see 3D conv docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html


class BasicBlock(nn.Module):
    expansion = (
        1  # 1 for BasicBlock, meaning the output size is the same as the input size
    )

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
                # nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        # self.bn3 = nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
                # nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = F.adaptive_avg_pool3d(out, output_size=4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_channels, num_classes):
    return ResNet(
        BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes
    )


def ResNet50(in_channels, num_classes):
    return ResNet(
        Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes
    )


# %%
if __name__ == "__main__":
    img_dim = 64  # 64x64x64; depth, height, width
    channels = 1  # 1 for grayscale (or one-dimensional data), 3 for RGB
    batch_size = 8
    n_classes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(channels, num_classes=n_classes).to(device)
    model

    test_input = Variable(torch.randn(batch_size, channels, img_dim, img_dim, img_dim))

    output = model(test_input)
    print(output.shape)
