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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Stride 1 means no downsampling, stride higher than 1 means downsampling
        When expansion = 4, the number of output channels is 4 times the number of input channels
        """
        super(Bottleneck, self).__init__()

        n_out_channels = self.expansion * out_channels
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        # self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.conv3 = nn.Conv2d(
            out_channels, n_out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_features=n_out_channels)
        # self.bn3 = nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != n_out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    n_out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(n_out_channels),
                # nn.GroupNorm(num_groups=2, num_channels=self.expansion*planes)
            )

    def forward(self, x: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_blocks: list[int],  # list of number of blocks in each layer
        in_channels: int = 1,  # number of input channels (1 for grayscale)
        num_classes: int = 2,
    ):
        """ResNet Architecture Class"""
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Initial Conv Layer (no downsampling):
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        """
        Create a layer with multiple blocks
        @planes is the number of output channels of the first block in the layer
        @num_blocks is the number of @BottleNeck blocks in the layer
        @stride is the stride of the first block in the layer (rest of the blocks have stride 1)
        """
        # [stride, 1, 1, 1, ...]
        strides: list[int] = [stride] + ([1] * (num_blocks - 1))
        layers = []
        for stride in strides:
            layers.append(
                Bottleneck(
                    in_channels=self.in_planes, out_channels=planes, stride=stride
                )
            )
            self.in_planes = planes * Bottleneck.expansion
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
        out = out.view(out.size(0), -1)  # flatten
        out = self.linear(out)
        return out


def ResNet50(in_channels, num_classes):
    return ResNet(
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
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

    # Test 3D input
    test_input = Variable(torch.randn(batch_size, channels, img_dim, img_dim, img_dim))

    # Test 2D input
    test_input = Variable(torch.randn(batch_size, channels, img_dim, img_dim))
    output = model(test_input)
    print(output.shape)
