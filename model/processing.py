"""Here we define functions to process the data. These functions will be used when building training and validation datasets.
source: https://keras.io/examples/vision/3D_image_classification/"""

import numpy as np
import torch
import torch.nn.functional as F


def clip_and_normalise_volume(
    scan: torch.Tensor, min_bound: int = -1000, max_bound: int = 400
) -> np.ndarray:
    """
    Normalize the scan to be in the range of [0, 1] by clipping
    - CT scans store raw voxel intensity in Hounsfield units (HU). Above 400 are bones with different radiointensity, so this is used as a higher bound. A threshold between -1000 and 400 is commonly used to normalize CT scans.
    - source: https://keras.io/examples/vision/3D_image_classification/
    """
    scan = torch.clamp(scan, min_bound, max_bound)
    return (scan - min_bound) / (max_bound - min_bound)


def add_dialation(scan: torch.Tensor, dilation: int = 1) -> torch.Tensor:
    """
    Add dilation to the scan
    """
    # TODO does not work yet
    # Define a kernel (for example, a 3x3 kernel)
    kernel = torch.ones(
        (1, 3, 3), dtype=torch.float32
    )  # Shape (out_channels, in_channels, height, width)

    # Perform a dilated convolution with a dilation factor of 2
    dilation = 2
    output_tensor = F.conv2d(scan, kernel, padding=dilation, dilation=dilation)
    return output_tensor
