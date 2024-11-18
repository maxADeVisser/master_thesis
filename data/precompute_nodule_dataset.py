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

import pandas as pd
import torch
from tqdm import tqdm

from data.dataset import transform_3d_to_25d
from preprocessing.processing import clip_and_normalise_volume
from project_config import env_config
from utils.utils import load_scan


def main():
    # SCRIPT PARAMS (which dataset configurations to precompute)
    CONTEXT_WINDOW_SIZES = [30, 50, 70]
    DIMENSIONALITY = ["2.5D", "3D"]
    DATASET_VERSION: Literal["hold_out", "full"] = "full"

    # Load the preprocessed nodule dataframe
    PROJECT_DIR = os.getenv("PROJECT_DIR")
    _holdout_indicator = "hold_out" if DATASET_VERSION == "hold_out" else ""
    if DATASET_VERSION == "full":
        nodule_df = pd.read_csv(env_config.processed_nodule_df_file)
    else:
        nodule_df = pd.read_csv(env_config.hold_out_nodule_df_file)
    for cws in CONTEXT_WINDOW_SIZES:
        nodule_df[f"consensus_bbox_{cws}"] = nodule_df[f"consensus_bbox_{cws}"].apply(
            ast.literal_eval
        )
    # Create nodule IDs:
    nodule_df["nodule_id"] = (
        nodule_df["pid"].astype(str) + "_" + nodule_df["nodule_idx"].astype(str)
    )

    # create the directories for the precomputed ROIs
    for cws in CONTEXT_WINDOW_SIZES:
        for dim in DIMENSIONALITY:
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
        scan: np.ndarray = load_scan(row["pid"], to_numpy=True)
        nodule_id = row["pid"] + "_" + str(row["nodule_idx"])
        malignancy_consensus = row["malignancy_consensus"]

        # For each context, precompute the ROI
        for cws in CONTEXT_WINDOW_SIZES:
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

            for dim in DIMENSIONALITY:
                OUT_DIR = f"{PROJECT_DIR}/data/precomputed_rois_{cws}C_{dim}{_holdout_indicator}"
                if dim == "2.5D":
                    nodule_roi = transform_3d_to_25d(nodule_roi)
                nodule_roi = clip_and_normalise_volume(nodule_roi)

                torch.save(
                    (nodule_roi, malignancy_consensus), f"{OUT_DIR}/{nodule_id}.pt"
                )


# %%
if __name__ == "__main__":
    main()
