"""Project config file"""

import json
import os

from dotenv import load_dotenv

load_dotenv(".env")  # Load environment variables from .env file

# TODO make the pipeline config into a pydantic class. This way, I can verify the parameters and types provided in the configuration
with open("pipeline_parameters.json", "r") as f:
    pipeline_config = json.load(f)


class _Config:
    """
    Config class for the project. Intended to be a singleton

    Attributes:
        @DATA_DIR: str, the root directory path to the LIDC-IDRI dataset
        @patient_ids: list, the list of patient ids in the dataset
        @dicom_encoding_mapping_file: str, the path to the dicom encoding mapping file
    """

    def __init__(self):
        # TODO write these attributes in a configuration file and load them here instead
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
config = _Config()

if __name__ == "__main__":
    config = _Config()
