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


def create_fiftyone_nodule_dataset(delete_if_exists: bool = False) -> None:
    nodule_roi_jpg_dir = (
        f"{env_config.PROJECT_DIR}/data/middle_slice_images_c{context_size}"
    )

    # load predictions and embeddings
    pred_df_path = f"{env_config.OUT_DIR}/predictions/{experiment_id}/pred_nodule_df_fold{fold}.csv"
    try:
        pred_nodule_df = pd.read_csv(pred_df_path).set_index("nodule_id")
    except FileNotFoundError:
        raise FileNotFoundError(f"Predictions not found. Run get_predictions.py first")

    dataset_name = f"C{context_size}_Nodule_ROIs"

    # --- CREATE DATASET ---
    if not delete_if_exists and dataset_name in fo.list_datasets():
        raise FileExistsError(
            f"Dataset {dataset_name} already exists in fiftyone database. Reset first"
        )

    if delete_if_exists and dataset_name in fo.list_datasets():
        try:
            fo.delete_dataset(dataset_name)
        except:
            raise Exception(f"Could not delete dataset {dataset_name}")

    # (Only run once - stores in a MongoDB database)
    dataset = fo.Dataset.from_images_patt(
        images_patt=f"{nodule_roi_jpg_dir}/*.jpg", name=dataset_name, persistent=True
    )

    for sample in dataset:
        nodule_id = sample.filename.split(".")[0]
        row = pred_nodule_df.loc[nodule_id]

        # Store classification in a field name of your choice
        sample["nodule_id"] = nodule_id

        # TODO add confidence? can we get the confidence?:
        sample["prediction"] = fo.Classification(label=str(row["pred"]))
        sample["malignancy_consensus"] = fo.Classification(
            label=str(row["malignancy_consensus"])
        )
        sample["malignancy_scores"] = row["malignancy_scores"]
        sample["cancer_label"] = row["cancer_label"]

        sample["subtlety_consensus"] = row["subtlety_consensus"]
        sample["subtlety_scores"] = row["subtlety_scores"]

        sample["volume_mean"] = row["ann_mean_volume"]
        sample["diameter_mean"] = row["ann_mean_diameter"]

        sample["lobulations_consensus"] = row["lobulation_consensus"]
        sample["lobulation_scores"] = row["ann_lobulation_scores"]

        sample["internalStructure_consensus"] = row["internalStructure_consensus"]
        sample["internalStructure_scores"] = row["ann_internalStructure_scores"]

        sample["calcification_consensus"] = row["calcification_consensus"]
        sample["calcification_scores"] = row["ann_calcification_scores"]

        sample["sphericity_consensus"] = row["sphericity_consensus"]
        sample["sphericity_scores"] = row["ann_sphericity_scores"]

        sample["margin_consensus"] = row["margin_consensus"]
        sample["margin_scores"] = row["ann_margin_scores"]

        sample["spiculation_consensus"] = row["spiculation_consensus"]
        sample["spiculation_scores"] = row["ann_spiculation_scores"]

        sample["texture_consensus"] = row["texture_consensus"]
        sample["texture_scores"] = row["ann_texture_scores"]

        # Scan-level fields
        sample["scan_contrast_used"] = row["scan_contrast_used"]
        sample["scan_slice_thickness"] = row["scan_slice_thickness"]
        sample["scan_slice_spacing"] = row["scan_slice_spacing"]
        sample["scan_pixel_spacing"] = row["scan_pixel_spacing"]

        sample.save()


if __name__ == "__main__":
    create_fiftyone_nodule_dataset(delete_if_exists=False)
