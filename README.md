# Analyzing Shortcuts in Computed Tomography Scans of Lung Nodules

## Abstract

## Repository Overview

The project uses [Poetry](https://python-poetry.org/) for dependency management and the requirements are specified in the `pyproject.toml` file.

Below is a explanation of what each non-self-explanatory root level file is for:

- `experiment_analysis_parameters.json`: a json file with experiment analysis configurations used for analysing the results of the conducted experiments
- `pipeline_parameters.json`: a json file for training pipeline configurations
- `project_config.py`: contains a singleton `_EnvConfig` class to use across the project with configuration specifications as well as loads the `pipeline_parameters.json` file as a python dictionary.
- `training_requirements.txt`: the specific requries we used to install dependencies on ITUs HPC as it did not support Poetry.

Below is a overview of the structure of the repository and what each folder contains:

### `adhoc`

Exploraty and adhoc script and notebooks.

### `analysis`

Contains:

- analysis work such as exploraty data analysis on the data that was used for developement of the model
- our setup for using [FiftyOne](https://docs.voxel51.com/) for the qualitative analysis.
- the script for doing the annotator agreement analysis by computing Kendall's W (`annotater_agreement.py`).
- a .csv file with the exploraty shortcut annotations we did (`exploraty_shortcut_annotations.csv`).
- the beginning of a script for creating SHAP explanations (this is however not used in the report.) (`shap_explanations.py`).

### `data`

Data folder and data handling related code. Contains:

- the ROI middle slice .jpg images computed for display in the FiftyOne application as well as the script for creating them (`create_img_dataset_version.py`).
- the precomputed ROIs at different context window sizes that was used during training to speed up data loading as well as the script to compute them (`precompute_nodule_dataset.py`).
- Helper classes to process the LIDC-IDRI dataset and individual nodules (`dataset.py`)
- NOTE: this folder should also contain the full [LIDC-IDRI dataset](https://www.cancerimagingarchive.net/collection/lidc-idri/)

### `hpc`

HPC related code and files downloaded from the HPC. Contains:

- `jobs/`: the fetched outputs from the HPC used to run the experiments.
- `fetch_hpc_output.py`: a script to fetch the outputs from the HPC.
- `.job` scripts for running the jobs on the HPC.

### `model`

Contains:

- the computed embeddings and predictions for the different models in .csv files.
- the scripts to compute the embeddings and predictions.
- the script we used to benchmark the model on the holdout set as well as the results in a .csv and .json file.
- the ResNet ordinal regression model implementation (`ResNet.py`). This file also contains all utility functions for the model (e.g. computing confidence scores, retrieving predicted malignancy, converting model to 3D, etc..)

### `preprocessing`

Contains:

- scripts and a notebook for creating a nodule dataframe as well as the holdout set (`create_nodule_df.py` and `process_nodule_df.ipynb`)
- .csv files for the processed nodule dataframe used for model developement (`processed_nodule_df.csv`) and the holdout set nodule dataframe used for final evaluation (`hold_out_nodule_df.csv`)
- a .csv with the extract DICOM meta data attributes (`dicom_meta_data.csv`)
- script for extracting annotations used for inter-annotator reliability analysis (`create_annotations.py`).
- contains a `processing.py` module with util functions used for data processing.
- contains a `data_augmentations.py` module with the function for data augmentations used during training.
- a script to extract the DICOM meta-data attributes from the DICOM files.

### `report`

Contains:

- a notebook to create more or less all the plots and tables we have in the report (`plots_and_tables.ipynb`)
- the saved plots and other screenshots that went into the report.

### `test`

Contains some testing code to:

- overfit to a single batch to verify the model works as expected (`overfit_to_single_batch.py`)
- verify that the precomputed ROIs are the correct shape and dimensionality (`test_precomputed_nodules.py`)
- test the preprocessing functions in the preprocessing module (`test_preprocessing.py`)
- test the util functions we use with the ResNet model (`test_ResNet.py`)
- test aspects of the training pipeline (`test_training.py`)

### `train`

Contains

- the training script (`train.py`)
- a implementation of early stopping used during training (`early_stopping.py`)
- implementation of the model evaluation metrics (`metrics.py`)

### `utils`

Contains various utility modules, data models, and functions used across the project.

## Project Setup

1. Install poetry (refer to [docs](https://python-poetry.org/docs/#installing-with-the-official-installer)):

   `curl -sSL https://install.python-poetry.org | python3 -`

2. Once installed, run `poetry install` which will install all dependencies specified in the `pyproject.toml` file

3. `pylidc` requires a configuration file in the home directory called `.pylidcrc` on Mac or `pylidc.conf` on Windows. Follow the instructions at [pylidc documentation](https://pylidc.github.io/install.html).

4. add a `.env` file in the root of the working directory and specify `LIDC_IDRI_DIR` as the directory path for the downloaded dataset. The path should be specified the same way that is specified for the configuration of [pylidc](https://pylidc.github.io/install.html)

5. create a `out/` directory at the repository root level.

Example setup of the `.env` file:

```
LIDC_IDRI_DIR = .../master_thesis/data/lung_data/manifest-1600709154662/LIDC-IDRI
PROJECT_DIR = .../master_thesis
LOG_LEVEL = DEBUG
FIFTYONE_APP_CONFIG = "analysis/fiftyone_app/app_config.json"
```
