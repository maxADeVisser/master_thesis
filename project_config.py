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
    Config class for the project. Intended to be a singleton.
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
            print("WARNING: The LIDC-IDRI dataset directory is not found.")

        self.nodule_df_file = f"{project_dir}/preprocessing/nodule_df.csv"
        self.processed_nodule_df_file = (
            f"{project_dir}/preprocessing/processed_nodule_df.csv"
        )
        self.hold_out_nodule_df_file = (
            f"{project_dir}/preprocessing/hold_out_nodule_df.csv"
        )


# Singleton pattern: only one instance of the Config class is created
env_config = _EnvConfig()

# %%
