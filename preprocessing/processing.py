"""Here we define functions to process the data. These functions will be used when building training and validation datasets.
source: https://keras.io/examples/vision/3D_image_classification/"""

import numpy as np
import pydicom
from PIL import Image
from scipy import ndimage


def convert_dicom_to_png(
    dicom_file_path: str, output_dir: str, normalise: bool = True
) -> None:
    """Takes a DICOM file path and converts it to the file ending provided in the output_dir (.png or .jpg)
    see https://www.youtube.com/watch?v=k6hD0xNp2B8&list=PLQCkKRar9trMY2qJAU6H4nZQwTfZc91Oq
    """
    dicom_data = pydicom.dcmread(dicom_file_path)
    image = dicom_data.pixel_array.astype(float)
    if normalise:
        image = np.maximum(image, 0) / np.maximum(image.max(), 1)

    # Convert to 8-bit unsigned integer (0-255)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(output_dir)
    return None


def normalise_volume(volume: np.ndarray) -> np.ndarray:
    """CT scans store raw voxel intensity in Hounsfield units (HU). They range from -1024 to above 2000 in this dataset. Above 400 are bones with different radiointensity, so this is used as a higher bound. A threshold between -1000 and 400 is commonly used to normalize CT scans.
    source: https://keras.io/examples/vision/3D_image_classification/
    """
    min = -1000  # lower bound. corresponds to air
    max = 400  # higher bound. corresponds to bones
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(
    img, desired_depth: int = 64, desired_width: int = 128, desired_height: int = 128
) -> np.ndarray:
    """Resize across z-axis
    source: https://keras.io/examples/vision/3D_image_classification/"""
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
