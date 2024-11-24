"""Here we define functions to process the data. These functions will be used when building training and validation datasets.
source: https://keras.io/examples/vision/3D_image_classification/"""

import numpy as np
import torch

from project_config import SEED, pipeline_config

torch.manual_seed(SEED)
np.random.seed(SEED)

LOWER_BOUND, HIGHER_BOUND = pipeline_config.dataset.clipping_bounds


def clip_and_normalise_volume(
    nodule_scan: torch.Tensor,
    min_bound: int = LOWER_BOUND,
    max_bound: int = HIGHER_BOUND,
) -> torch.Tensor:
    """
    Clip range to @min_bound and @max_bound and normalize the scan to be in the range of [0, 1].
    - CT scans store raw voxel intensity in Hounsfield units (HU). Above 400 are bones with different radiointensity, so this is used as a higher bound. A threshold between -1000 and 400 is commonly used to normalize CT scans.
    - source: https://keras.io/examples/vision/3D_image_classification/
    """
    clipped = torch.clamp(nodule_scan, min_bound, max_bound)
    clipped_min_bound = torch.min(clipped)
    clipped_max_bound = torch.max(clipped)
    normalised = (clipped - clipped_min_bound) / (clipped_max_bound - clipped_min_bound)
    return normalised


if __name__ == "__main__":
    # test clip_and_normalise_volume
    nodule_scan = torch.tensor([[-1000, 400], [0, 1000]])
    print(nodule_scan)
    print(clip_and_normalise_volume(nodule_scan))
