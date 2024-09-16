import numpy as np
import pydicom
import pylidc as pl
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from project_config import config
from utils.pylidc_utils import get_scans_by_patient_id


def load_dicom_file(dicom_file_path: str) -> pydicom.dataset.FileDataset:
    """Returns a pydicom object from a DICOM file path."""
    return pydicom.dcmread(dicom_file_path)


class LIDC_IDRI_DATASET(Dataset):
    def __init__(self, dicom_dir_path: str = config.DATA_DIR) -> None:
        self.dicom_dir_path = dicom_dir_path
        self.patient_ids = config.patient_ids

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, patient_id: str) -> tuple[np.ndarray, str]:
        """Retrieves a single scan and its corresponding label"""
        # TODO this needs to eventually return a tensor
        # TODO add normalising the images and preprocessing them
        scan = get_scans_by_patient_id(patient_id, to_numpy=False)
        # TODO needs to return label as well
        malignancy_scores = get_malignancy_by_scan(scan)

        scan = get_scans_by_patient_id(patient_id, to_numpy=True)
        # # Normalise the volume
        # volume = np.maximum(volume, 0) / np.maximum(volume.max(), 1)
        # # Convert to 8-bit unsigned integer (0-255)
        # volume = (volume * 255).astype(np.uint8)

        return scan, malignancy_scores


if __name__ == "__main__":
    dataset = LIDC_IDRI_DATASET()
    selected_patient = dataset.patient_ids[15]
    scan, malignancy_scores = dataset.__getitem__(patient_id=selected_patient)
