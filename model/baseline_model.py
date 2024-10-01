"""Source: https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py"""

# %%
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


class ConvNet3D(nn.Module):
    """
    This is the c3d implementation with batch norm.

    [1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
    Proceedings of the IEEE international conference on computer vision. 2015.
    Link: https://ieeexplore.ieee.org/document/7410867
    """

    def __init__(self, slice_size: int, n_slices: int, num_classes=5, in_channels=1):

        super(ConvNet3D, self).__init__()

        # Convolutional Blocks
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.block5 = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)),
        )

        final_feature_map_depth = int(math.floor(n_slices / 16))

        final_feature_map_size = int(math.ceil(slice_size / 32))

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=(
                    512
                    * final_feature_map_depth
                    * final_feature_map_size
                    * final_feature_map_size
                ),
                out_features=4096,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5))
        self.fc = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = out.view(out.size(0), -1)  # flatten
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append("fc")

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({"params": v})
                    break
            else:
                parameters.append({"params": v, "lr": 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def print_sizes(model, input_tensor):
    output = input_tensor
    for m in model.children():
        output = m(output)
        print(m, output.shape)
    return output


# %%
if __name__ == "__main__":
    img_dim = 64  # 64x64x64; depth, height, width
    channels = 1  # 1 for grayscale (or one-dimensional data), 3 for RGB
    batch_size = 8
    n_classes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet3D(
        slice_size=img_dim,
        n_slices=img_dim,
        num_classes=n_classes,
        in_channels=channels,
    ).to(device)

    # model = nn.DataParallel(model, device_ids=None)  # Use all available GPUs
    # print(model)

    input_var = Variable(torch.randn(batch_size, channels, img_dim, img_dim, img_dim))

    print_sizes(model, input_var)

    output = model(input_var)
    print(output.shape)
