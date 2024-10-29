"""
source: https://github.com/MedMNIST/experiments/blob/main/MedMNIST3D/models.py
This is the ResNet50 benchmark model for the MedMNIST3D dataset.
Adapted from kuangliu/pytorch-cifar.
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from acsconv.converters import Conv3dConverter

from project_config import SEED, pipeline_config

NUM_CLASSES = pipeline_config.model.num_classes
IN_CHANNELS = pipeline_config.model.in_channels
torch.manual_seed(SEED)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        in_channels: int = IN_CHANNELS,  # number of input channels (1 for grayscale)
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """
        ResNet Architecture Class altered so it is fit to use the CORN loss for ordinal regression
        """
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
        self.corn_output_layer = nn.Linear(
            in_features=512 * Bottleneck.expansion, out_features=num_classes - 1
        )

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
        flattened = out.view(out.size(0), -1)
        logits = self.corn_output_layer(flattened)
        return logits


def ResNet50(in_channels, num_classes) -> ResNet:
    return ResNet(
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
    )


def compute_class_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Given the logits, compute the class probabilities.
    Returns tensor of shape (batch_size, num_classes)
    """
    with torch.no_grad():
        sigmoid_output = torch.sigmoid(logits)
        cum_probas = torch.cumprod(sigmoid_output, dim=1)
        class_probas = torch.stack(
            [
                1 - cum_probas[:, 0],  # P(y = 1) = 1 - P(y > 1)
                cum_probas[:, 0] - cum_probas[:, 1],  # P(y = 2) = P(y > 1) - P(y > 2)
                cum_probas[:, 1] - cum_probas[:, 2],  # P(y = 3) = P(y > 2) - P(y > 3)
                cum_probas[:, 2] - cum_probas[:, 3],  # P(y = 4) = P(y > 3) - P(y > 4)
                cum_probas[:, 3],  # P(y = 5) = P(y > 4)
            ],
            dim=1,
        )
        return class_probas


def predict_rank_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Given output logits, return the malignancy rank.

    Params
    ---
    @logits: The model's output logits
        tensor of shape (batch_size, num_classes)

    Returns
    ---
    @predicted_rank: The predicted rank
        tensor of shape (batch_size,)
    """
    with torch.no_grad():
        probas = torch.sigmoid(logits)
        probas = torch.cumprod(probas, dim=1)
        predicted_rank = torch.sum(probas > 0.5, dim=1) + 1
    return predicted_rank.float().tolist()


def predict_binary_from_logits(
    logits: torch.Tensor,
    return_probabilites: bool = False,
) -> list[int] | list[float]:
    """
    Given output logits, return the binary classification based on the classification threshold

    Params
    ---
    @logits: The model's output logits
        tensor of shape (batch_size, num_classes)
    @classification_threshold: The threshold for binary classification

    Returns
    ---
    @binary_prediction: The binary prediction
        list of length (batch_size,)
    """
    with torch.no_grad():
        probas = torch.sigmoid(logits)
        probas = torch.cumprod(probas, dim=1)
        greater_than_3_idx = 2  # P(y > 3) - threshold for binary classification
        if return_probabilites:
            binary_prediction = probas[:, greater_than_3_idx]
            return binary_prediction.tolist()
        else:
            binary_prediction = 0.5 <= probas[:, greater_than_3_idx]
            return binary_prediction.float().tolist()


def convert_model_to_3d(model: nn.Module) -> nn.Module:
    """
    Uses the ACSConv library to convert a 2D model to its 3D counterpart
    see https://github.com/M3DV/ACSConv for more information

    TODO: I am omitting to use the SyncBN conversion function for now (they do this in their training script)
    SyncBN is useful in distributed training (e.g., training on multiple GPUs), where it synchronizes the
    batch statistics across all GPUs to make batch normalization work more effectively with small batch sizes
    split across devices.
    (refer to their training script for more information. The func is in utils.py of the ACSConv repo. I have a chatGPT conversation explaining what the code does)
    """
    return Conv3dConverter(model)


# %%
if __name__ == "__main__":
    img_dim = 64  # 64x64x64; depth, height, width
    channels = 1  # 1 for grayscale (or one-dimensional data)
    batch_size = 2
    n_classes = 5

    # Test 3D input
    model = ResNet50(in_channels=1, num_classes=5)  # 2D model
    model = convert_model_to_3d(model)  # 3D model
    test_input = torch.randn(batch_size, channels, img_dim, img_dim, img_dim)
    logits = model(test_input)
    logits

    # Get predicted rank probabilities
    # from coral_pytorch.dataset import corn_label_from_logits
    # corn_label_from_logits(logits).float()
    predict_rank_from_logits(logits)

    # Binary inference (this gives the binary classification)
    # predict_binary_from_logits(logits)

    compute_class_probs_from_logits(logits)
