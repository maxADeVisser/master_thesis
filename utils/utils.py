import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom

from project_config import config

PATH = str  # type alias


def get_scan_ids(data_dir: PATH = config.DATA_DIR) -> list[PATH]:
    scan_ids = sorted(
        [
            pid
            for pid in os.listdir(config.DATA_DIR)
            if os.path.isdir(os.path.join(config.DATA_DIR, pid))
        ]
    )
    return scan_ids


def get_ct_scan_slices(scan_id: str) -> list[PATH]:
    """Get all the full CT scan file paths for a given patient."""
    directory = os.path.join(config.DATA_DIR, scan_id)

    if not os.path.isdir(directory):
        raise ValueError(f"Provided path '{directory}' is not a valid directory.")

    entries = sorted([f for f in os.listdir(directory) if f not in [".DS_Store"]])

    # expects that the file structure is sorted in ascending order
    sub_dir = os.path.join(directory, entries[0])
    if os.path.isdir(sub_dir):
        return get_ct_scan_slices(sub_dir)
    else:
        # If no subdirectory is found, filter out .dcm files in the current directory
        dcm_files = sorted([f for f in entries if f.endswith(".dcm")])

        # Return the .dcm files (with full path)
        return [os.path.join(directory, f) for f in dcm_files]


def get_scan_directory_path_by_patient_id(patient_id_dir: str) -> list[PATH]:
    """Returns the directory path of the scan for a given patient ID."""
    directory = os.path.join(config.DATA_DIR, patient_id_dir)

    if not os.path.isdir(directory):
        raise ValueError(f"Provided path '{directory}' is not a valid directory.")

    excluded_files = [".DS_Store"]
    entries = sorted([f for f in os.listdir(directory) if f not in excluded_files])

    # expects that the file structure is sorted in ascending order
    sub_dir = os.path.join(directory, entries[0])
    if os.path.isdir(sub_dir):
        return get_scan_directory_path_by_patient_id(sub_dir)  # recursive call
    else:
        return directory


def load_dicom_image(dicom_file_path: PATH) -> pydicom.dataset.FileDataset:
    """Returns a pydicom object from a DICOM file path."""
    dicom = pydicom.dcmread(dicom_file_path)
    return dicom.pixel_array


def load_dicom_images_from_folder(folder_path: PATH) -> np.ndarray:
    dicom_files = [
        pydicom.dcmread(os.path.join(folder_path, f))
        for f in sorted(os.listdir(folder_path))
        if f.endswith(".dcm")
    ]
    images = np.stack([f.pixel_array for f in dicom_files], axis=0)
    return images


def normalize_image(image):
    """Normalize an image's pixel values to the range [0, 255]."""
    # Convert image to float for accurate division
    image = image.astype(np.float32)
    # Find the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    # Normalize the image to the range [0, 1]
    normalized_image = (image - min_val) / (max_val - min_val)

    # Scale to the range [0, 255]:
    return (normalized_image * 255).astype(np.uint8)


def show_image(slice_index, images):
    plt.imshow(images[slice_index], cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    scan_ids = get_scan_ids(config.DATA_DIR)
    get_ct_scan_slices(scan_ids[1])
