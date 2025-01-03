# %%
import os

import numpy as np
import pydicom
import pylidc as pl

from project_config import env_config

# to avoid error in pylidc due to deprecated types:
np.int = int
np.float = float


def get_ct_scan_slice_paths(
    patient_id_dir: str, return_parent_dir: bool = False
) -> list[str] | str:
    """Dending on @return_parent_dir_only flag:
    - Returns ALL (multiple) the full CT scan file paths for a given patient
    - OR Returns the parent directory path of the CT scan for a given patient"""
    # BUG this sometimes does not return the first folder in the list??? Sometimes the annnotations folder is returned. I think this was the case for LIDC-IDRI-0068
    directory = os.path.join(env_config.RAW_DATA_DIR, patient_id_dir)

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


def load_scan(patient_id: str, to_numpy: bool = True) -> list[pl.Scan] | np.ndarray:
    """Returns the first scan for a given patient_id (i think there is only one scan per patient)"""
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    return scan.to_volume(verbose=False) if to_numpy else scan


# TODO this can be replaced by @get_scans_by_patient_id
def load_dicom_images_from_folder(scan_parent_dir: str) -> np.ndarray:
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
