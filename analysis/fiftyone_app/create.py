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
dataset_name = "test"
overwrite_if_exists = False
persist = False
# ---------------------

nodule_roi_jpg_dir = (
    f"{env_config.PROJECT_DIR}/data/middle_slice_images_c{context_size}"
)

# load predictions and embeddings
pred_df_path = (
    f"{env_config.OUT_DIR}/predictions/{experiment_id}/pred_nodule_df_fold{fold}.csv"
)
try:
    pred_nodule_df = pd.read_csv(pred_df_path).set_index("nodule_id")
except FileNotFoundError:
    raise FileNotFoundError(f"Predictions not found. Run get_predictions.py first")

dataset = fo.Dataset.from_images_patt(
    images_patt=f"{nodule_roi_jpg_dir}/*.jpg",
    name=dataset_name,
    persistent=persist,
    overwrite=overwrite_if_exists,
    progress=True,
)
