import json

import fiftyone as fo
import pandas as pd

from project_config import env_config
from utils.data_models import ExperimentAnalysis

with open("experiment_analysis_parameters.json", "r") as f:
    config = ExperimentAnalysis.model_validate(json.load(f))

# --- SCRIPT PARAMS ---
context_size = config.analysis.context_size
experiment_id = config.experiment_id
fold = config.analysis.fold
# ---------------------


def create_fiftyone_nodule_dataset(
    dataset_name: str, overwrite_if_exists: bool = False
) -> None:
    nodule_roi_jpg_dir = (
        f"{env_config.PROJECT_DIR}/data/middle_slice_images_c{context_size}"
    )

    # load predictions and embeddings
    pred_df_path = f"{env_config.OUT_DIR}/predictions/{experiment_id}/pred_nodule_df_fold{fold}.csv"
    try:
        pred_nodule_df = pd.read_csv(pred_df_path).set_index("nodule_id")
    except FileNotFoundError:
        raise FileNotFoundError(f"Predictions not found. Run get_predictions.py first")

    dataset = fo.Dataset.from_images_patt(
        images_patt=f"{nodule_roi_jpg_dir}/*.jpg",
        name=dataset_name,
        persistent=True,
        overwrite=overwrite_if_exists,
        progress=True,
    )

    for sample in dataset:
        nodule_id = sample.filename.split(".")[0]
        row = pred_nodule_df.loc[nodule_id]

        # Store classification in a field name of your choice
        sample["nodule_id"] = nodule_id

        sample["prediction"] = fo.Classification(
            label=str(row["pred"]),
            confidence=float(row["confidence"]),
        )
        sample["abs_error"] = abs(row["pred"] - row["malignancy_consensus"])

        sample["malignancy_consensus"] = fo.Classification(
            label=str(row["malignancy_consensus"]),
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
    dataset.save()


if __name__ == "__main__":
    create_fiftyone_nodule_dataset(f"{experiment_id}", overwrite_if_exists=False)
