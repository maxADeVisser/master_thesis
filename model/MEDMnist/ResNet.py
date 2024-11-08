"""
source: https://github.com/MedMNIST/experiments/blob/main/MedMNIST3D/models.py
This is the ResNet50 benchmark model for the MedMNIST3D dataset.
Adapted from kuangliu/pytorch-cifar.
"""

# %%
from typing import Literal

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
        in_channels: int,  # number of input channels (1 for grayscale)
        num_classes: int,
    ) -> None:
        """
        ResNet Architecture Class altered so it is fit to use the CORN loss for ordinal regression
        """
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.GroupNorm(num_groups=2, num_channels=64),
            self._make_layer(64, num_blocks[0], stride=1),
            self._make_layer(128, num_blocks[1], stride=2),
            self._make_layer(256, num_blocks[2], stride=2),
            self._make_layer(512, num_blocks[3], stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.corn_classifier_layer = nn.Linear(
            in_features=512 * Bottleneck.expansion, out_features=num_classes - 1
        )

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
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

    def forward(self, x) -> torch.Tensor:
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        logits = self.corn_classifier_layer(flattened)
        return logits

    def get_feature_vector(self, x) -> torch.Tensor:
        """
        Return the feature vector before the classifier layer.
        Returns tensor of shape (batch_size, 512 * Bottleneck.expansion)
        """
        with torch.no_grad():
            # flatten the images in the batch
            return self.features(x).view(x.size(0), -1)


def ResNet50(
    in_channels: int, num_classes: int, dims: Literal["2D", "3D"] = "3D"
) -> ResNet:
    """
    Return a ResNet50 model for the given number of input channels and classes.
    The ResNet50 model has 3 layers with 3, 4, 6, and 3 blocks each.
    """
    if dims == "3D":
        return convert_model_to_3d(
            ResNet(
                num_blocks=[3, 4, 6, 3],
                in_channels=in_channels,
                num_classes=num_classes,
            )
        )
    elif dims == "2D":
        return ResNet(
            num_blocks=[3, 4, 6, 3],
            in_channels=in_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError("Invalid value for dims. Must be '2D' or '3D'")


def get_conditional_probas(logits: torch.Tensor) -> torch.Tensor:
    """
    Process the output logits to obtain the conditional probabilities for each rank.
    That is, the probability of the rank being greater than or equal to the current rank,
    given that it is greater than the previous rank:
    For node k in the output layer, the probability of the rank being k or greater for input x is:
    f_k(x_i) = P(y_i > r_k | y_i > r_k-1)
    """
    return torch.sigmoid(logits)


def get_unconditional_probas(logits: torch.Tensor) -> torch.Tensor:
    """
    Obtain the unconditional probabilities for each rank using the conditional probabilities
    and the chain rule of probability.
    That is, return P(y_i > r_k) for each rank k.
    """
    conditional_probas = get_conditional_probas(logits)
    # chain rule of probability:
    unconditional_probas = torch.cumprod(conditional_probas, dim=1)
    # (ensure that rank-monotonicity is maintained)
    return unconditional_probas


def compute_class_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Given the logits, return the class probabilities.
    Returns tensor of shape (batch_size, num_classes)
    """
    uncond_probas = get_unconditional_probas(logits)
    class_probas = torch.stack(
        [
            1 - uncond_probas[:, 0],  # P(y = 1) = 1 - P(y > 1)
            uncond_probas[:, 0] - uncond_probas[:, 1],  # P(y = 2) = P(y > 1) - P(y > 2)
            uncond_probas[:, 1] - uncond_probas[:, 2],  # P(y = 3) = P(y > 2) - P(y > 3)
            uncond_probas[:, 2] - uncond_probas[:, 3],  # P(y = 4) = P(y > 3) - P(y > 4)
            uncond_probas[:, 3],  # P(y = 5) = P(y > 4)
        ],
        dim=1,
    )
    return class_probas.float()


def get_pred_malignancy_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
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
    uncond_probas = get_unconditional_probas(logits)
    predicted_levels = 0.5 <= uncond_probas
    predicted_rank = torch.sum(predicted_levels, dim=1) + 1
    # NOTE: i think that the +1 is not added in the @corn_label_from_logits function, because the function
    # returns the rank index, and not the rank value itself. But we can do this here, since we are predicting a
    # score from 1 to 5
    return predicted_rank.int()


def predict_binary_from_logits(
    logits: torch.Tensor,
    return_probs: bool = False,
) -> torch.Tensor:
    """
    Given output logits, return the binary classification based on the classification threshold

    Params
    ---
    @logits: The model's output logits
        tensor of shape (batch_size, num_classes)

    Returns
    ---
    @binary_prediction: The binary prediction (0 or 1) or the probability of the positive class (P(y > 3))
        tensor of size (batch_size,)
    """
    uncond_probas = get_unconditional_probas(logits)
    greater_than_3_idx = 2  # index for P(y > 3)
    if return_probs:
        binary_prediction = uncond_probas[:, greater_than_3_idx]
        return binary_prediction.float()
    else:
        binary_prediction = 0.5 <= uncond_probas[:, greater_than_3_idx]
        return binary_prediction.int()


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


def load_resnet_model(
    weights_path: str,
    in_channels: int,
    dims: Literal["2D", "2.5D" "3D"],
) -> nn.Module:
    """
    Load a ResNet model from a checkpoint file.
    """
    model = ResNet50(in_channels=in_channels, num_classes=5, dims=dims)
    model.load_state_dict(torch.load(weights_path))
    return model


# %%
if __name__ == "__main__":
    img_dim = 10
    channels = 1  # 1 for grayscale (or one-dimensional data)
    batch_size = 2
    n_classes = 5

    # Test 3D input
    model = ResNet50(in_channels=1, num_classes=5, dims="3D")
    test_input = torch.randn(batch_size, channels, img_dim, img_dim, img_dim)
    test_input.shape

    features = model.get_feature_vector(test_input)

    logits = model(test_input)
    logits

    # Get predicted rank probabilities
    # from coral_pytorch.dataset import corn_label_from_logits
    # corn_label_from_logits(logits).float()

    get_unconditional_probas(logits)

    get_pred_malignancy_score_from_logits(logits)

    # Binary inference (this gives the binary classification)
    # predict_binary_from_logits(logits)

    compute_class_probs_from_logits(logits)

    # Model only trained for 1 epoch
    model = load_resnet_model(
        "out/model_runs/testing_training_flow_0811_1113/model_fold_0.pth",
        in_channels=1,
        dims="3D",
    )
    logits = model(test_input)

    with torch.no_grad():
        uncond = get_unconditional_probas(logits)
        preds = get_pred_malignancy_score_from_logits(logits)
        class_probas = compute_class_probs_from_logits(logits)

    uncond
    preds
    class_probas
