"""Here we define functions to process the data. These functions will be used when building training and validation datasets.
source: https://keras.io/examples/vision/3D_image_classification/"""

import numpy as np
import torch
from scipy.ndimage import zoom

from project_config import SEED, pipeline_config

torch.manual_seed(SEED)
np.random.seed(SEED)

LOWER_BOUND, HIGHER_BOUND = pipeline_config.dataset.clipping_bounds


def resample_voxel_size(
    volume: np.ndarray,
    pixel_spacing: float,
    slice_thickness: float,
    target_spacing: tuple = (1.0, 1.0, 1.0),  # Default is 1mm x 1mm x 1mm
    verbose: bool = False,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Resample the voxel size of a 3D volume to 1mm x 1mm x 1mm.
    NOTE: The shape of the volume likely changes.

    Parameters:
        - volume: The 3D volume to normalize. Shape (x, y, z). Fetch using `load_scan` from `utils.utils`.
        - pixel_spacing: The spacing in the transverse plane (x and y dimensions) in mm.
        - slice_thickness: The thickness of each slice in mm (z dimension).
        - target_spacing: The desired voxel spacing in mm.

    Returns:
        - The resampled volume with the desired voxel spacing.
        - The scale factors used to resample the volume.
    """
    current_spacing = (pixel_spacing, pixel_spacing, slice_thickness)

    # Calculate the resampling factors and new shape:
    scale_factors = [
        current / target for current, target in zip(current_spacing, target_spacing)
    ]

    # Resample the volume using bilinear interpolation:
    resampled_volume = zoom(volume, scale_factors, order=1)

    if verbose:
        new_shape = [
            round(dim * scale) for dim, scale in zip(volume.shape, scale_factors)
        ]
        print(f"Input shape: {volume.shape}")
        print(f"Target spacing: {target_spacing}")
        print(f"Current spacing: {current_spacing}")
        print(f"Scale factors: {scale_factors}")
        print(f"Expected output shape: {new_shape}")
        print(f"Actual output shape: {resampled_volume.shape}")

    return resampled_volume, scale_factors


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
