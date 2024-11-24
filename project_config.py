# %%
"""Project config file"""

import os

from dotenv import load_dotenv

from utils.data_models import create_experiment_from_json

load_dotenv(".env")


SEED = 39  # Seed for reproducibility

pipeline_config = create_experiment_from_json("pipeline_parameters.json")


class _EnvConfig:
    """
    Config class for the project. Intended to be a singleton

    Attributes:
        @DATA_DIR: str, the root directory path to the LIDC-IDRI dataset
        @patient_ids: list, the list of patient ids in the dataset
        @dicom_encoding_mapping_file: str, the path to the dicom encoding mapping file
    """

    def __init__(self):
        project_dir = os.getenv("PROJECT_DIR")
        lidc_idri_dir = os.getenv("LIDC_IDRI_DIR")
        assert (
            project_dir is not None
        ), "Please set the PROJECT_DIR env var in a .env file in the root directory of the project"
        assert (
            lidc_idri_dir is not None
        ), "Please set the LIDC_IDRI_DIR env var in a .env file in the root directory of the project"
        self.PROJECT_DIR = project_dir
        self.RAW_DATA_DIR = lidc_idri_dir
        self.PREPROCESSED_DATA_DIR = f"{project_dir}/data/precomputed_resampled_rois_{pipeline_config.dataset.context_window}C_{pipeline_config.dataset.dimensionality}"
        self.OUT_DIR = f"{project_dir}/out"
        try:
            self.patient_ids = sorted(
                [
                    pid
                    for pid in os.listdir(lidc_idri_dir)
                    if os.path.isdir(os.path.join(lidc_idri_dir, pid))
                ]
            )
        except FileNotFoundError:
            print(
                "WARNING: The LIDC-IDRI dataset directory is not found.\nMost of the project will not work.\nPlease set the LIDC_IDRI_DIR env var in a .env file in the root directory of the project"
            )

        self.nodule_df_file = f"{project_dir}/preprocessing/nodule_df.csv"
        self.processed_nodule_df_file = (
            f"{project_dir}/preprocessing/processed_nodule_df.csv"
        )
        self.hold_out_nodule_df_file = (
            f"{project_dir}/preprocessing/hold_out_nodule_df.csv"
        )
        # self.dicom_encoding_mapping_file = "utils/dicom_encoding_mapping.pkl"
        # self.meta_dataframe_file = "utils/meta_dataframe.parquet"
        # self.excluded_dicom_attributes = [
        #     "Acquisition DateTime",
        #     "Study Time",
        #     "Acquisition Time",
        #     "Content Time",
        #     "Study Date",
        #     "Series Date",
        #     "Acquisition Date",
        #     "Content Date",
        #     "Overlay Date",
        #     "Curve Date",
        #     "SOP Class UID",
        #     "SOP Instance UID",
        #     "Accession Number",
        #     "Referring Physician's Name",
        #     "Referenced SOP Instance UID",
        #     "Patient's Name",
        #     "Study Instance UID",
        #     "Series Instance UID",
        #     "Study ID",
        #     "Series Number",
        #     "Instance Number",
        #     "Frame of Reference UID",
        #     "Position Reference Indicator",
        #     "Slice Location",
        #     "Rows",
        #     "Columns",
        #     "Bits Allocated",
        #     "Bits Stored",
        #     "High Bit",
        #     "Longitudinal Temporal Information M",
        #     "Admitting Date",
        #     # "Scheduled Procedure Step Start Date",
        #     # "Scheduled Procedure Step End Date",
        #     # "Performed Procedure Step Start Date",
        #     "Placer Order Number / Imaging Servi",
        #     "Filler Order Number / Imaging Servi",
        #     "Verifying Observer Name",
        #     "Person Name",
        #     "Content Creator's Name",
        #     "Storage Media File-set UID",
        #     "Pixel Data",
        #     "UID",
        #     "Private tag data",
        #     "Specific Character Set",
        # ]


# Singleton pattern: only one instance of the Config class is created
env_config = _EnvConfig()

# %%
