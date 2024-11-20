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
from utils.utils import load_scan

with open("experiment_analysis_parameters.json", "r") as f:
    config = json.load(f)

# --- SCRIPT PARAMS ---
context_window: list[int] = config["context_sizes"]
dimensionality: list[str] = config["dimensionalities"]
holdout_set: bool = config["holdout_set"]
# ---------------------

dataset_version: Literal["hold_out", "train"] = "hold_out" if holdout_set else "train"

# Load the preprocessed nodule dataframe
PROJECT_DIR = os.getenv("PROJECT_DIR")
_holdout_indicator = "hold_out" if dataset_version == "hold_out" else ""
if dataset_version == "train":
    nodule_df = pd.read_csv(env_config.processed_nodule_df_file)
else:
    nodule_df = pd.read_csv(env_config.hold_out_nodule_df_file)
for cws in context_window:
    nodule_df[f"consensus_bbox_{cws}"] = nodule_df[f"consensus_bbox_{cws}"].apply(
        ast.literal_eval
    )

# create the directories for the precomputed ROIs
for cws in context_window:
    for dim in dimensionality:
        OUT_DIR = (
            f"{PROJECT_DIR}/data/precomputed_rois_{cws}C_{dim}{_holdout_indicator}"
        )
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        else:
            raise FileExistsError(f"{OUT_DIR} already exists. Reset first")

# loop over nodules and precompute the ROIs at different context windows sizes:
for _, row in tqdm(
    nodule_df.iterrows(),
    desc="Precomputing ROIs",
    total=len(nodule_df),
):
    # NOTE: Load the scan once per nodule (a bottleneck operation we want to avoid doing to many times)
    scan: np.ndarray = load_scan(row["scan_id"], to_numpy=True)
    malignancy_consensus = row["malignancy_consensus"]

    # For each context, precompute the ROI
    for cws in context_window:
        x_bounds, y_bounds, z_bounds = row[f"consensus_bbox_{cws}"]
        nodule_roi = scan[
            x_bounds[0] : x_bounds[1],
            y_bounds[0] : y_bounds[1],
            z_bounds[0] : z_bounds[1],
        ]
        assert nodule_roi.shape == (
            cws,
            cws,
            cws,
        ), f"ROI shape is not correct for context window {cws}"

        nodule_roi: torch.Tensor = torch.from_numpy(nodule_roi).unsqueeze(0).float()

        for dim in dimensionality:
            OUT_DIR = (
                f"{PROJECT_DIR}/data/precomputed_rois_{cws}C_{dim}{_holdout_indicator}"
            )
            if dim == "2.5D":
                nodule_roi = transform_3d_to_25d(nodule_roi)
            nodule_roi = clip_and_normalise_volume(nodule_roi)

            torch.save(
                (nodule_roi, malignancy_consensus), f"{OUT_DIR}/{row['nodule_id']}.pt"
            )
