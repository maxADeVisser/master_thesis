# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom

from project_config import config

PATH = str  # type alias


def get_ct_scan_slice_paths(
    patient_id_dir: PATH, return_parent_dir: bool = False
) -> list[PATH] | PATH:
    """Dending on @return_parent_dir_only flag:
    - Returns ALL (multiple) the full CT scan file paths for a given patient
    - OR Returns the parent directory path of the CT scan for a given patient"""
    directory = os.path.join(config.DATA_DIR, patient_id_dir)

    if not os.path.isdir(directory):
        raise ValueError(f"Provided path '{directory}' is not a valid directory.")

    entries = sorted([f for f in os.listdir(directory) if f not in [".DS_Store"]])

    # expects that the file structure is sorted in ascending order
    sub_dir = os.path.join(directory, entries[0])
    if os.path.isdir(sub_dir):
        return get_ct_scan_slice_paths(sub_dir, return_parent_dir)
    else:
        if return_parent_dir == True:
            return directory
        else:
            # If no subdirectory is found, filter out .dcm files in the current directory
            dcm_files = sorted([f for f in entries if f.endswith(".dcm")])

            # Return the .dcm files (with full path)
            return [os.path.join(directory, f) for f in dcm_files]


def load_dicom_images_from_folder(scan_parent_dir: PATH) -> np.ndarray:
    """Returns a 3D numpy array of the CT scan images in the given directory."""
    dicom_files = [
        pydicom.dcmread(os.path.join(scan_parent_dir, f))
        for f in sorted(os.listdir(scan_parent_dir))
        if f.endswith(".dcm")
    ]
    images = np.stack([f.pixel_array for f in dicom_files], axis=0)
    return images


# %%
if __name__ == "__main__":
    get_ct_scan_slice_paths("LIDC-IDRI-0001", return_parent_dir=True)
