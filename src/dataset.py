import os

import numpy as np
import pydicom
import pylidc as pl
import torch
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

load_dotenv(".env")
DATA_DIR = os.getenv("LIDC_IDRI_DIR")


def load_dicom_file(dicom_file_path: str) -> pydicom.dataset.FileDataset:
    """Returns a pydicom object from a DICOM file path."""
    return pydicom.dcmread(dicom_file_path)


class LIDC_IDRI_DATASET(Dataset):
    """Custom dataset class for DICOM files."""

    def __init__(self, dicom_dir_path: str = DATA_DIR):
        self.dicom_dir_path = dicom_dir_path
        # This is only for individual slices:
        # self.dicom_files = [
        #     os.path.join(dicom_dir_path, f)
        #     for f in os.listdir(dicom_dir_path)
        #     if f.endswith(".dcm")
        # ]
        self.patient_ids = sorted(
            [
                pid
                for pid in os.listdir(DATA_DIR)
                if os.path.isdir(os.path.join(DATA_DIR, pid))
            ]
        )

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, patient_id: str) -> tuple[np.ndarray, str]:
        # TODO this needs to eventually return a tensor
        # TODO add normalising the images and preprocessing them
        volume = (
            pl.query(pl.Scan)
            .filter(pl.Scan.patient_id == patient_id)
            .first()
            .to_volume()
        )

        # Normalise the volume
        # TODO maybe we need to normalise the slices individually? the outcome images are very dark
        volume = np.maximum(volume, 0) / np.maximum(volume.max(), 1)
        # Convert to 8-bit unsigned integer (0-255)
        volume = (volume * 255).astype(np.uint8)

        return volume

    def visualise(self, patient_id: str, slice_number: int) -> None:
        volume = self.__getitem__(patient_id)
        print(f"Patient ID: {patient_id} has {volume.shape[2] - 1} slices.")
        plt.imshow(volume[:, :, slice_number], cmap=plt.cm.bone)
        plt.show()


if __name__ == "__main__":
    dataset = LIDC_IDRI_DATASET()
    selected_patient = dataset.patient_ids[28]
    image = dataset.__getitem__(patient_id=selected_patient)
    # get unique values
    dataset.visualise(patient_id=selected_patient, slice_number=0)

    raw_image = load_dicom_file(
        "data/lung_data/manifest-1725363397135/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-001.dcm"
    ).pixel_array

    plt.hist(np.unique(raw_image))

    dataset.visualise(patient_id=selected_patient, slice_number=50)
