"""Project config file"""

import os

from dotenv import load_dotenv

load_dotenv(".env")


class _Config:
    """
    Config class for the project. Intended to be a singleton

    Attributes:
        @DATA_DIR: str, the root directory path to the LIDC-IDRI dataset
        @patient_ids: list, the list of patient ids in the dataset
        @dicom_encoding_mapping_file: str, the path to the dicom encoding mapping file
    """

    def __init__(self):
        self.DATA_DIR = os.getenv("LIDC_IDRI_DIR")
        self.patient_ids = sorted(
            [
                pid
                for pid in os.listdir(self.DATA_DIR)
                if os.path.isdir(os.path.join(self.DATA_DIR, pid))
            ]
        )
        self.dicom_encoding_mapping_file = "utils/dicom_encoding_mapping.pkl"
        self.meta_dataframe_file = "utils/meta_dataframe.parquet"


# Singleton pattern: only one instance of the Config class is created
config = _Config()

if __name__ == "__main__":
    config = _Config()
