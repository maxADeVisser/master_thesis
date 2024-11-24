"""Script for precomputing the nodule ROIs and their corresponding labels for the LIDC-IDRI dataset."""

# %%
import os
import sys
from typing import Literal

import numpy as np
from dotenv import load_dotenv

load_dotenv(".env")
sys.path.append(os.getenv("PROJECT_DIR"))

import ast
import json

import pandas as pd
import torch
from tqdm import tqdm

from data.dataset import transform_3d_to_25d
from preprocessing.processing import clip_and_normalise_volume
from project_config import env_config
from utils.data_models import ExperimentAnalysis
from utils.utils import load_scan

# SCRIPT PARAMS ---------
with open("experiment_analysis_parameters.json", "r") as f:
    config = ExperimentAnalysis.model_validate(json.load(f))


# --- SCRIPT PARAMS ---
context_windows = config.precompute_nodule_dataset.context_windows
dimensionalities = config.precompute_nodule_dataset.dimensionalities
holdout_set: bool = config.precompute_nodule_dataset.holdout_set
# ---------------------

dataset_version: Literal["hold_out", "train"] = "hold_out" if holdout_set else "train"
_holdout_indicator = "hold_out" if dataset_version == "hold_out" else ""

# Load the preprocessed nodule dataframe
PROJECT_DIR = os.getenv("PROJECT_DIR")
if dataset_version == "train":
    nodule_df = pd.read_csv(env_config.processed_nodule_df_file)
else:
    nodule_df = pd.read_csv(env_config.hold_out_nodule_df_file)

# Cast the string representations of the consensus bounding boxes to tuples:
for cws in context_windows:
    nodule_df[f"consensus_bbox_{cws}"] = nodule_df[f"consensus_bbox_{cws}"].apply(
        ast.literal_eval
    )

# Create the directories for the precomputed ROIs:
# First check if the directories already exist:
for cws in context_windows:
    for dim in dimensionalities:
        OUT_DIR = (
            f"{PROJECT_DIR}/data/precomputed_rois_{cws}C_{dim}{_holdout_indicator}"
        )
        if os.path.exists(OUT_DIR):
            raise FileExistsError(f"{OUT_DIR} already exists. Reset first")
# Then create the directories:
for cws in context_windows:
    for dim in dimensionalities:
        OUT_DIR = (
            f"{PROJECT_DIR}/data/precomputed_rois_{cws}C_{dim}{_holdout_indicator}"
        )
        os.makedirs(OUT_DIR)


def _process_loaded_nodule(
    input_nodule: torch.Tensor,
    cws: Literal[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    dim: Literal["2.5D", "3D"],
) -> torch.Tensor:
    """Util func to process the loaded input_nodule."""
    if dim == "2.5D":
        processed_nodule = transform_3d_to_25d(input_nodule)
        processed_nodule = clip_and_normalise_volume(processed_nodule)
        assert processed_nodule.shape == (
            3,
            cws,
            cws,
        ), f"Incorrect shape for 2.5D ROI: {processed_nodule.shape}. Should be ({(3, cws, cws)})"
    elif dim == "3D":
        processed_nodule = clip_and_normalise_volume(input_nodule)
        assert processed_nodule.shape == (
            1,
            cws,
            cws,
            cws,
        ), f"Incorrect shape for 3D ROI: {processed_nodule.shape}. Should be ({(1, cws, cws, cws)})"
    return processed_nodule


# loop over nodules and precompute the ROIs at the specified context windows and dimensionalities:
for _, row in tqdm(
    nodule_df.iterrows(),
    desc="Precomputing ROIs",
    total=len(nodule_df),
):
    # NOTE: Load the scan once per nodule (a bottleneck operation we want to avoid doing to many times)
    scan: np.ndarray = load_scan(row["scan_id"], to_numpy=True)

    label = row["malignancy_consensus"]

    # For each context, precompute the ROI
    for cws in context_windows:
        x_bounds, y_bounds, z_bounds = row[f"consensus_bbox_{cws}"]
        nodule_roi = scan[
            x_bounds[0] : x_bounds[1],
            y_bounds[0] : y_bounds[1],
            z_bounds[0] : z_bounds[1],
        ]
        nodule_roi: torch.Tensor = torch.from_numpy(nodule_roi).unsqueeze(0).float()

        for dim in dimensionalities:
            OUT_DIR = (
                f"{PROJECT_DIR}/data/precomputed_rois_{cws}C_{dim}{_holdout_indicator}"
            )
            processed_nodule_roi = _process_loaded_nodule(nodule_roi, cws, dim)

            torch.save(
                (processed_nodule_roi, label),
                f"{OUT_DIR}/{row['nodule_id']}.pt",
            )
