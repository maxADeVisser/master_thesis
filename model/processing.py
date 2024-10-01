"""Here we define functions to process the data. These functions will be used when building training and validation datasets.
source: https://keras.io/examples/vision/3D_image_classification/"""

import numpy as np
import torch
import torch.nn.functional as F

from project_config import pipeline_config

LOWER_BOUND, HIGHER_BOUND = pipeline_config.dataset.normalisation_bounds


def clip_and_normalise_volume(
    scan: torch.Tensor, min_bound: int = LOWER_BOUND, max_bound: int = HIGHER_BOUND
) -> np.ndarray:
    """
    Normalize the scan to be in the range of [0, 1] by clipping
    - CT scans store raw voxel intensity in Hounsfield units (HU). Above 400 are bones with different radiointensity, so this is used as a higher bound. A threshold between -1000 and 400 is commonly used to normalize CT scans.
    - source: https://keras.io/examples/vision/3D_image_classification/
    """
    scan = torch.clamp(scan, min_bound, max_bound)
    return (scan - min_bound) / (max_bound - min_bound)
