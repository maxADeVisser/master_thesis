"""Script for computing the middle slice of the nodule ROIs as .jpg images"""

# %%
import ast
import json
import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from preprocessing.processing import clip_and_normalise_volume
from project_config import env_config
from utils.utils import load_scan


def main():
    # SCRIPT PARAMS (which dataset configurations to precompute)
    CONTEXT_WINDOW_SIZES = [30, 50, 70]
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
    out_dirs = {}
    for cws in CONTEXT_WINDOW_SIZES:
        OUT_DIR = f"{PROJECT_DIR}/data/middle_slice_images_c{cws}{_holdout_indicator}"
        out_dirs[cws] = OUT_DIR
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        else:
            raise FileExistsError(f"{OUT_DIR} already exists. Reset first")

    # loop over nodules and compute the middle slice .jpg images of the nodule ROIs
    annotations = {"c30": {}, "c50": {}, "c70": {}}
    for _, row in tqdm(
        nodule_df.iterrows(),
        desc="Precomputing ROIs",
        total=len(nodule_df),
    ):
        # NOTE: Load the scan once per nodule (a bottleneck operation we want to avoid doing to many times)
        scan: np.ndarray = load_scan(row["pid"], to_numpy=True)
        nodule_id = row["pid"] + "_" + str(row["nodule_idx"])
        malignancy_consensus = row["malignancy_consensus"]
        subtlety_consensus = row["subtlety_consensus"]
        cancer_label = row["cancer_label"]
        ann_mean_volume = row["ann_mean_volume"]
        ann_mean_diameter = row["ann_mean_diameter"]

        # For each context, precompute the ROI
        for cws in CONTEXT_WINDOW_SIZES:
            img_path = f"{out_dirs[cws]}/{nodule_id}.jpg"
            annotations[f"c{cws}"][img_path] = {
                "nodule_id": nodule_id,
                "malignancy": malignancy_consensus,
                "subtlety": subtlety_consensus,
                "cancer_label": cancer_label,
                "ann_mean_volume": ann_mean_volume,
                "ann_mean_diameter": ann_mean_diameter,
            }

            # Extract the nodule ROI:
            x_bounds, y_bounds, z_bounds = row[f"consensus_bbox_{cws}"]
            nodule_roi = scan[
                x_bounds[0] : x_bounds[1],
                y_bounds[0] : y_bounds[1],
                z_bounds[0] : z_bounds[1],
            ]
            nodule_roi = clip_and_normalise_volume(torch.from_numpy(nodule_roi)).numpy()
            nodule_roi = (255 * nodule_roi).astype(np.uint8)
            middle_slice = nodule_roi.shape[-1] // 2
            nodule_roi = nodule_roi[:, :, middle_slice]

            # save the middle slice as a .jpg image
            image = Image.fromarray(nodule_roi)
            image.save(img_path)

    for cws in CONTEXT_WINDOW_SIZES:
        with open(f"{out_dirs[cws]}/annotations.json", "w") as f:
            json.dump(annotations[f"c{cws}"], f)


# %%
if __name__ == "__main__":
    main()
