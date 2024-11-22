import json
import os

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PrecomputedNoduleROIs
from model.ResNet import (
    get_malignancy_rank_confidence,
    get_pred_malignancy_from_logits,
    load_resnet_model,
)
from project_config import env_config

with open("experiment_analysis_parameters.json", "r") as f:
    config = json.load(f)

# --- SCRIPT PARAMS ---
context_size = config["analysis"]["context_size"]
experiment_id = config["experiment_id"]
dimensionality = config["analysis"]["dimensionality"]
fold = config["analysis"]["fold"]

precomputed_dir = (
    f"{env_config.PROJECT_DIR}/data/precomputed_rois_{context_size}C_{dimensionality}"
)
assert os.path.exists(
    precomputed_dir
), f"Precomputed ROIs not found at {precomputed_dir}. Run precomputed_nodule_dataset.py first"

pred_out_dir = f"{env_config.OUT_DIR}/predictions/{experiment_id}"

if not os.path.exists(pred_out_dir):
    os.makedirs(pred_out_dir, exist_ok=True)

pred_out_file = f"{pred_out_dir}/pred_nodule_df_fold{fold}.csv"
# if os.path.exists(pred_out_file):
#     raise FileExistsError(f"Predictions already exist at {pred_out_file}. Reset first")

weights_path = (
    f"{env_config.PROJECT_DIR}/hpc/jobs/{experiment_id}/fold_{fold}/model.pth"
)

nodule_df = pd.read_csv(env_config.processed_nodule_df_file)

in_channels = 1 if dimensionality == "3D" else 3
model = load_resnet_model(
    weights_path=weights_path, in_channels=in_channels, dims=dimensionality
)
model.eval()

dataset = PrecomputedNoduleROIs(precomputed_dir, data_augmentation=False)
loader = DataLoader(dataset, batch_size=16, shuffle=False)
all_preds = []
all_confidence = []
all_nodule_ids = []
for i, (nodule_roi, label, nodule_id) in tqdm(
    enumerate(loader), total=len(loader), desc="Predicting on Batches"
):
    logits = model(nodule_roi)
    preds = get_pred_malignancy_from_logits(logits)
    confidence = get_malignancy_rank_confidence(logits, preds).tolist()
    preds = preds.tolist()

    all_preds.extend(preds)
    all_confidence.extend(confidence)
    all_nodule_ids.extend(nodule_id)

pred_df = pd.DataFrame(
    {
        "nodule_id": all_nodule_ids,
        "pred": all_preds,
        "confidence": all_confidence,
    }
)
pred_nodule_df = pd.merge(nodule_df, pred_df, on="nodule_id", how="left")
assert len(pred_nodule_df) == len(
    nodule_df
), "Length of prediction dataframe does not match length of nodule dataframe"

pred_nodule_df.to_csv(pred_out_file, index=False)
