"""Project config file"""

import os

from dotenv import load_dotenv

from utils.experiments import create_experiment_from_json

load_dotenv(".env")  # Load environment variables from .env file


SEED = 39  # Seed for reproducibility


class _EnvConfig:
    """
    Config class for the project. Intended to be a singleton

    Attributes:
        @DATA_DIR: str, the root directory path to the LIDC-IDRI dataset
        @patient_ids: list, the list of patient ids in the dataset
        @dicom_encoding_mapping_file: str, the path to the dicom encoding mapping file
    """

    def __init__(self):
        assert os.getenv(
            "LIDC_IDRI_DIR"
        ), "Please set the LIDC_IDRI_DIR env var in a .env file in the root directory of the project"
        self.DATA_DIR = os.getenv("LIDC_IDRI_DIR")
        self.patient_ids = sorted(
            [
                pid
                for pid in os.listdir(self.DATA_DIR)
                if os.path.isdir(os.path.join(self.DATA_DIR, pid))
            ]
        )
        self.OUT_DIR = os.getenv("OUTPUT_DIR") or "out"
        self.dicom_encoding_mapping_file = "utils/dicom_encoding_mapping.pkl"
        self.meta_dataframe_file = "utils/meta_dataframe.parquet"
        self.nodule_df_file = "out/nodule_df.csv"
        self.excluded_dicom_attributes = [
            "Acquisition DateTime",
            "Study Time",
            "Acquisition Time",
            "Content Time",
            "Study Date",
            "Series Date",
            "Acquisition Date",
            "Content Date",
            "Overlay Date",
            "Curve Date",
            "SOP Class UID",
            "SOP Instance UID",
            "Accession Number",
            "Referring Physician's Name",
            "Referenced SOP Instance UID",
            "Patient's Name",
            "Study Instance UID",
            "Series Instance UID",
            "Study ID",
            "Series Number",
            "Instance Number",
            "Frame of Reference UID",
            "Position Reference Indicator",
            "Slice Location",
            "Rows",
            "Columns",
            "Bits Allocated",
            "Bits Stored",
            "High Bit",
            "Longitudinal Temporal Information M",
            "Admitting Date",
            # "Scheduled Procedure Step Start Date",
            # "Scheduled Procedure Step End Date",
            # "Performed Procedure Step Start Date",
            "Placer Order Number / Imaging Servi",
            "Filler Order Number / Imaging Servi",
            "Verifying Observer Name",
            "Person Name",
            "Content Creator's Name",
            "Storage Media File-set UID",
            "Pixel Data",
            "UID",
            "Private tag data",
            "Specific Character Set",
        ]


# Singleton pattern: only one instance of the Config class is created
env_config = _EnvConfig()

pipeline_config = create_experiment_from_json(name="test", out_dir=env_config.OUT_DIR)
