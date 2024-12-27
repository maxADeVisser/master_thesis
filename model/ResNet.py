"""
source: https://github.com/MedMNIST/experiments/blob/main/MedMNIST3D/models.py
This is the ResNet50 model implemenation for the MedMNIST3D dataset.
"""

# %%
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from acsconv.converters import Conv3dConverter

from project_config import SEED, pipeline_config

DROPOUT_RATE = pipeline_config.training.dropout_rate

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
        # 1x1 convolution:
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        # 3x3 convolution:
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        # 1x1 convolution:
        self.conv3 = nn.Conv2d(
            out_channels, n_out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_features=n_out_channels)

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
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
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
            self._make_layer(64, num_blocks[0], stride=1),
            self._make_layer(128, num_blocks[1], stride=2),
            self._make_layer(256, num_blocks[2], stride=2),
            self._make_layer(512, num_blocks[3], stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling layer
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),  # added dropout layer
            # corn classifier layer:
            nn.Linear(
                in_features=512 * Bottleneck.expansion, out_features=num_classes - 1
            ),
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
        logits = self.classifier(flattened)
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
    in_channels: int, num_classes: int, dims: Literal["2.5D", "3D"] = "3D"
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
    elif dims == "2.5D":
        return ResNet(
            num_blocks=[3, 4, 6, 3],
            in_channels=in_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError("Invalid value for dims. Must be '2D' or '3D'")


def get_conditional_probas(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the conditional probabilities P(y_i > r_k | y_i > r_k-1) for each rank k -> the probability of the rank being greater than or equal to the current rank, given that it is greater than the previous rank
    """
    return torch.sigmoid(logits)


def get_unconditional_probas(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the unconditional probabilities P(y_i > r_k) for each rank k using the chain rule:
    P(y_i > r_k) = P(y_i > r_k | y_i > r_k-1) * P(y_i > r_k-1).
    """
    conditional_probas = get_conditional_probas(logits)
    # chain rule of probability:
    unconditional_probas = torch.cumprod(conditional_probas, dim=1)
    # (ensure that rank-monotonicity is maintained)
    return unconditional_probas


def compute_class_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Given the logits, return the class probabilities.

    Params
    ---
    @logits: The model's output logits
        tensor of shape (batch_size, num_classes)
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


def get_malignancy_rank_confidence(
    logits: torch.Tensor, predicted_ranks: torch.Tensor
) -> torch.Tensor:
    """
    Given the logits and the predicted rank, return the confidence of the predicted rank.
    The confidence is the probability of the predicted rank being correct:
    P(y_i = r_i) = P(y_i > r_i-1) - P(y_i > r_i)
    where r_i is the predicted rank.
    For rank 0 (r_i = 1), the confidence is 1 - P(y_i > 1),
    and for rank 4 (r_i = 5), the confidence is P(y_i > 4).

    @logits is the model's output logits
    @predicted_rank is the rank index (1 to 5)
    """
    class_probas = compute_class_probs_from_logits(logits)
    batch_size = logits.size(0)
    confidences = torch.zeros(batch_size)
    for i in range(batch_size):
        confidences[i] = class_probas[i, predicted_ranks[i] - 1]
    return confidences


def get_pred_malignancy_from_logits(logits: torch.Tensor) -> torch.Tensor:
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
    predicted_levels = 0.5 < uncond_probas
    predicted_rank = torch.sum(predicted_levels, dim=1) + 1
    """NOTE: i think that the +1 is not added in the @corn_label_from_logits function, because the function returns the rank index, and not the rank value itself. But we can do this here, since we are predicting a score from 1 to 5"""
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
    """
    return Conv3dConverter(model)


def load_resnet_model(
    weights_path: str, in_channels: int, dims: Literal["2.5D", "3D"]
) -> nn.Module:
    """
    Load a ResNet model from a checkpoint file.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet50(in_channels=in_channels, num_classes=5, dims=dims)
    model.load_state_dict(
        torch.load(f=weights_path, map_location=torch.device(device), weights_only=True)
    )
    return model


# %%
if __name__ == "__main__":
    img_dim = 10
    channels = 1  # 1 for grayscale (or one-dimensional data)
    batch_size = 2
    n_classes = 5

    # Test 3D input
    model = ResNet50(in_channels=3, num_classes=5, dims="2.5D")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    test_input = torch.randn(batch_size, channels, img_dim, img_dim, img_dim)
    test_input.shape

    features = model.get_feature_vector(test_input)

    logits = model(test_input)
    logits

    # Get predicted rank probabilities
    # from coral_pytorch.dataset import corn_label_from_logits
    # corn_label_from_logits(logits).float()

    get_unconditional_probas(logits)

    get_pred_malignancy_from_logits(logits)

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
        preds = get_pred_malignancy_from_logits(logits)
        class_probas = compute_class_probs_from_logits(logits)

    uncond
    preds
    class_probas
