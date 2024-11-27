"""Script is just for adding any additional sample info to the dataset"""

import json

import fiftyone as fo
import pandas as pd
from tqdm import tqdm

from project_config import env_config
from utils.data_models import ExperimentAnalysis

with open("experiment_analysis_parameters.json", "r") as f:
    config = ExperimentAnalysis.model_validate(json.load(f))

# --- SCRIPT PARAMS ---
experiment_ids = ["c30_25D_2411_1543", "c50_25D_2411_1812", "c70_25D_2411_1705"]
# ---------------------
fold = 0

for exp_id in tqdm(
    experiment_ids, total=len(experiment_ids), desc="Adding sample info"
):
    dataset = fo.load_dataset(f"{exp_id}")

    pred_df_path = (
        f"{env_config.OUT_DIR}/predictions/{exp_id}/pred_nodule_df_fold{fold}.csv"
    )
    try:
        pred_nodule_df = pd.read_csv(pred_df_path).set_index("nodule_id")
    except FileNotFoundError:
        raise FileNotFoundError(f"Predictions not found. Run get_predictions.py first")

    for sample in dataset:
        nodule_id = sample.filename.split(".")[0]
        row = pred_nodule_df.loc[nodule_id]

        sample["abs_error"] = abs(row["pred"] - row["malignancy_consensus"])

        sample.save()
    dataset.save()
