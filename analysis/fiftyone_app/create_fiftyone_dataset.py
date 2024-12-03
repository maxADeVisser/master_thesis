import json

import fiftyone as fo
import pandas as pd

from project_config import env_config
from utils.data_models import ExperimentAnalysis

with open("experiment_analysis_parameters.json", "r") as f:
    config = ExperimentAnalysis.model_validate(json.load(f))

# --- SCRIPT PARAMS ---
context_size = config.analysis.context_size
dataset_name = "c30"
# ---------------------

all_experiment_ids = {
    "c30": {
        "2.5D": "c30_25D_2411_1543",
        "3D": "c30_3D_2411_1947",
    },
    "c50": {
        "2.5D": "c50_25D_2411_1812",
        "3D": "c50_3D_2411_1831",
    },
    "c70": {
        "2.5D": "c70_25D_2411_1705",
        "3D": "c70_3D_2411_1824",
    },
}
fold = 0


def create_fiftyone_nodule_dataset(
    dataset_name: str, overwrite_if_exists: bool = False
) -> None:
    nodule_roi_jpg_dir = (
        f"{env_config.PROJECT_DIR}/data/middle_slice_images_c{context_size}"
    )

    experiment_ids = all_experiment_ids[dataset_name]

    # load predictions and embeddings
    pred_25D_path = f"{env_config.PROJECT_DIR}/model/predictions/{experiment_ids['2.5D']}/pred_nodule_df_fold{fold}.csv"
    pred_3D_path = f"{env_config.PROJECT_DIR}/model/predictions/{experiment_ids['3D']}/pred_nodule_df_fold{fold}.csv"
    try:
        pred_25D_df = pd.read_csv(pred_25D_path)
        # add dicom meta data attributes
        dicom_meta_df = pd.read_csv(
            f"{env_config.PROJECT_DIR}/preprocessing/dicom_meta_data.csv"
        )
        pred_25D_df = pd.merge(
            pred_25D_df, dicom_meta_df, on="scan_id", how="left"
        ).set_index("nodule_id")

        pred_3D_df = pd.read_csv(pred_3D_path).set_index("nodule_id")

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
        row25D = pred_25D_df.loc[nodule_id]
        row3D = pred_3D_df.loc[nodule_id]

        # Store classification in a field name of your choice
        sample["nodule_id"] = nodule_id

        sample["pred25D"] = fo.Classification(
            label=str(row25D["pred"]),
            confidence=float(row25D["confidence"]),
        )
        sample["abs_error25D"] = abs(row25D["pred"] - row25D["malignancy_consensus"])

        sample["pred3D"] = fo.Classification(
            label=str(row3D["pred"]),
            confidence=float(row3D["confidence"]),
        )
        sample["abs_error3D"] = abs(row3D["pred"] - row3D["malignancy_consensus"])

        # Get the rest of the fields from the 2.5D model (they are the same for both models)
        sample["malignancy_consensus"] = fo.Classification(
            label=str(row25D["malignancy_consensus"]),
        )
        sample["malignancy_scores"] = row25D["malignancy_scores"]
        sample["cancer_label"] = row25D["cancer_label"]

        sample["subtlety_consensus"] = row25D["subtlety_consensus"]
        sample["subtlety_scores"] = row25D["subtlety_scores"]

        sample["volume_mean"] = row25D["ann_mean_volume"]
        sample["diameter_mean"] = row25D["ann_mean_diameter"]

        sample["lobulations_consensus"] = row25D["lobulation_consensus"]
        sample["lobulation_scores"] = row25D["ann_lobulation_scores"]

        sample["internalStructure_consensus"] = row25D["internalStructure_consensus"]
        sample["internalStructure_scores"] = row25D["ann_internalStructure_scores"]

        sample["calcification_consensus"] = row25D["calcification_consensus"]
        sample["calcification_scores"] = row25D["ann_calcification_scores"]

        sample["sphericity_consensus"] = row25D["sphericity_consensus"]
        sample["sphericity_scores"] = row25D["ann_sphericity_scores"]

        sample["margin_consensus"] = row25D["margin_consensus"]
        sample["margin_scores"] = row25D["ann_margin_scores"]

        sample["spiculation_consensus"] = row25D["spiculation_consensus"]
        sample["spiculation_scores"] = row25D["ann_spiculation_scores"]

        sample["texture_consensus"] = row25D["texture_consensus"]
        sample["texture_scores"] = row25D["ann_texture_scores"]

        # Scan-level fields
        sample["scan_contrast_used"] = row25D["scan_contrast_used"]
        sample["scan_slice_thickness"] = row25D["scan_slice_thickness"]
        sample["scan_slice_spacing"] = row25D["scan_slice_spacing"]
        sample["scan_pixel_spacing"] = row25D["scan_pixel_spacing"]

        # Dicom meta data fields
        sample["manufacturer"] = row25D["manufacturer"]
        sample["manufacturer_model_name"] = row25D["manufacturer_model_name"]
        sample["x-ray tube current"] = row25D["x-ray tube current"]
        sample["kvp"] = row25D["kvp"]
        sample["exposure time"] = row25D["exposure time"]
        sample["exposure"] = row25D["exposure"]
        sample["software_versions"] = str(row25D["software_versions"])

        sample.save()
    dataset.save()


if __name__ == "__main__":
    create_fiftyone_nodule_dataset("c30", overwrite_if_exists=True)
    create_fiftyone_nodule_dataset("c50", overwrite_if_exists=True)
    create_fiftyone_nodule_dataset("c70", overwrite_if_exists=True)
