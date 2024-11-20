import json

import fiftyone as fo
import numpy as np
import pandas as pd

from project_config import env_config

with open("experiment_analysis_parameters.json", "r") as f:
    config = json.load(f)

# --- SCRIPT PARAMS ---
context_size = config["context_size"]
experiment_id = config["experiment_id"]
fold = config["fold"]
# ---------------------

nodule_roi_jpg_dir = (
    f"{env_config.PROJECT_DIR}/data/middle_slice_images_c{context_size}"
)

# load predictions and embeddings
pred_df_path = (
    f"{env_config.OUT_DIR}/predictions/{experiment_id}/pred_nodule_df_fold{fold}.csv"
)
pred_nodule_df = pd.read_csv(pred_df_path).set_index("nodule_id")
dataset_name = f"C{context_size}_Nodule_ROIs"

# --- CREATE DATASET ---
# (Only run once - stores in a MongoDB database)
dataset = fo.Dataset.from_images_patt(
    images_patt=f"{nodule_roi_jpg_dir}/*.jpg", name=dataset_name, persistent=True
)

for sample in dataset:
    nodule_id = sample.filename.split(".")[0]
    row = pred_nodule_df.loc[nodule_id]

    # Store classification in a field name of your choice
    sample["nodule_id"] = nodule_id
    sample["malignancy_consensus"] = fo.Classification(
        label=str(row["malignancy_consensus"])
    )
    sample["malignancy_scores"] = row["malignancy_scores"]
    sample["prediction"] = fo.Classification(
        label=str(row["pred"])
    )  # TODO add confidence?
    sample["subtlety"] = row["subtlety_consensus"]
    sample["subtlety_scores"] = row["subtlety_scores"]
    sample["cancer_label"] = row["cancer_label"]
    sample["ann_mean_volume"] = row["ann_mean_volume"]
    sample["ann_mean_diameter"] = row["ann_mean_diameter"]
    sample.save()
