# %%
"""Here we define functions to process the data. These functions will be used when building training and validation datasets.
source: https://keras.io/examples/vision/3D_image_classification/"""

from typing import Literal

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


def mask_out_center(
    roi: torch.Tensor, pixel_diameter: int, dim: Literal["2.5D", "3D"]
) -> torch.Tensor:
    """
    Masks out the center of the ROI to remove the nodule only keep context information.
    Assumes the ROI is either a 3D uniform cube with 1 channel or 2.5D uniform square with 3-channels

    Args:
        pixel_diameter (int): the diameter of the cutout in pixels. Is unified across all dimensions.
    """
    assert (
        pixel_diameter < roi.shape[-1]
    ), "Pixel diameter must be smaller than the ROI."

    center = roi.shape[-1] // 2
    start = center - (pixel_diameter // 2)
    end = center + (pixel_diameter // 2)

    if dim == "3D":
        roi[0, start:end, start:end, start:end] = 0
        return roi

    elif dim == "2.5D":
        roi[0, start:end, start:end] = 0
        roi[1, start:end, start:end] = 0
        roi[2, start:end, start:end] = 0
        return roi
    else:
        raise ValueError("Invalid dimension. Choose either '2.5D' or '3D'.")


# %%
if __name__ == "__main__":
    # test clip_and_normalise_volume
    # nodule_scan = torch.tensor([[-1000, 400], [0, 1000]])
    # print(nodule_scan)
    # print(clip_and_normalise_volume(nodule_scan))

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from data.dataset import PrecomputedNoduleROIs

    # 3D test of mask_out_center
    loader = DataLoader(
        PrecomputedNoduleROIs(
            "/Users/newuser/Documents/ITU/master_thesis/data/precomputed_resampled_rois_30C_3D",
            data_augmentation=False,
            remove_center=True,
        ),
        batch_size=2,
        shuffle=True,
    )
    feature, _, _ = next(iter(loader))
    feature = feature[0]
    middle = feature.shape[-1] // 2
    plt.imshow(feature[0, :, :, middle], cmap="gray")
    feature.shape

    masked = mask_out_center(feature, pixel_diameter=25, dim="3D")
    masked.shape
    plt.imshow(masked[0, :, :, middle], cmap="gray")

    # 2.5D test of mask_out_center
    loader = DataLoader(
        PrecomputedNoduleROIs(
            "/Users/newuser/Documents/ITU/master_thesis/data/precomputed_resampled_rois_30C_2.5D",
            data_augmentation=False,
            remove_center=True,
        ),
        batch_size=1,
        shuffle=True,
    )
    feature, _, _ = next(iter(loader))
    feature = feature[0]
    plt.imshow(feature[1, :, :], cmap="gray")
    feature.shape

    masked = mask_out_center(feature, pixel_diameter=20, dim="2.5D")
    masked.shape
    plt.imshow(masked[1, :, :], cmap="gray")
