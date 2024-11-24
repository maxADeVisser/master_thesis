import matplotlib.pyplot as plt
import pandas as pd
import pylidc as pl
import torch
from scipy.ndimage import zoom

from utils.utils import load_scan

# %%
# inspect distribution of pixel spacing and slice thickness
pixel_spacings = [s.pixel_spacing for s in pl.query(pl.Scan).all()]
plt.hist(pixel_spacings)
plt.show()

slice_thickness = [s.slice_thickness for s in pl.query(pl.Scan).all()]
plt.hist(slice_thickness, bins=20)
plt.show()
# %%

# Load a nodule
id = "LIDC-IDRI-0001"
scan = torch.from_numpy(load_scan(id, to_numpy=True))
scan_info = pl.query(pl.Scan).filter(pl.Scan.patient_id == id).first()
scan.shape


# %%
def resample_voxel_size(
    volume: torch.Tensor,
    pixel_spacing: float,
    slice_thickness: float,
    target_spacing: tuple = (1.0, 1.0, 1.0),  # Default is 1mm x 1mm x 1mm
    verbose: bool = False,
) -> torch.Tensor:
    """
    Resample the voxel size of a 3D volume to 1mm x 1mm x 1mm.
    NOTE: The shape of the volume likely changes.

    Parameters:
        volume: The 3D volume to normalize. Shape (x, y, z).
        pixel_spacing: The spacing in the transverse plane (x and y dimensions) in mm.
        slice_thickness: The thickness of each slice in mm (z dimension).
        target_spacing: The desired voxel spacing in mm.

    Returns:
        The resampled volume with the desired voxel spacing.
    """
    current_spacing = (pixel_spacing, pixel_spacing, slice_thickness)

    # Calculate the resampling factors and new shape:
    scale_factors = [
        current / target for current, target in zip(current_spacing, target_spacing)
    ]

    # Resample the volume using bilinear interpolation:
    resampled_volume = zoom(input=volume, zoom=scale_factors, order=1)

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

    return resampled_volume


# %%

normalised = resample_voxel_size(
    scan, scan_info.pixel_spacing, scan_info.slice_thickness, verbose=True
)
normalised.shape
plt.imshow(normalised[:, :, 150], cmap="gray")


torch.all(scan == normalised[0])
