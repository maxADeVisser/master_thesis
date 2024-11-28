import os
import sys

import torch
from dotenv import load_dotenv

load_dotenv(".env")
sys.path.append(os.getenv("PROJECT_DIR"))

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
from utils.data_models import ExperimentAnalysis
from utils.logger_setup import logger

with open("experiment_analysis_parameters.json", "r") as f:
    config = ExperimentAnalysis.model_validate(json.load(f))

# --- SCRIPT PARAMS ---
context_size = config.analysis.context_size
experiment_id = config.experiment_id
dimensionality = config.analysis.dimensionality
fold = 0
batch_size = 2
num_workers = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


precomputed_dir = f"{env_config.PROJECT_DIR}/data/precomputed_resampled_rois_{context_size}C_{dimensionality}"
assert os.path.exists(
    precomputed_dir
), f"Precomputed ROIs not found at {precomputed_dir}. Run precomputed_nodule_dataset.py first"

pred_out_dir = f"{env_config.OUT_DIR}/predictions/{experiment_id}"
if not os.path.exists(pred_out_dir):
    os.makedirs(pred_out_dir, exist_ok=True)
pred_out_file = f"{pred_out_dir}/pred_nodule_df_fold{fold}.csv"

weights_path = (
    f"{env_config.PROJECT_DIR}/out/model_runs/{experiment_id}/fold{fold}/model.pth"
)

logger.info(
    f"""
    Running predictions for experiment {experiment_id} on fold {fold}
    with context size {context_size} and dimensionality {dimensionality}

    Using precomputed ROIs from {precomputed_dir}
    with model weights from {weights_path}
    Saving predictions to {pred_out_file}

    Batch size: {batch_size}
    Num workers: {num_workers}
    Using GPU: {torch.cuda.get_device_name(0)}
    """
)

nodule_df = pd.read_csv(env_config.processed_nodule_df_file)

in_channels = 1 if dimensionality == "3D" else 3
model = load_resnet_model(
    weights_path=weights_path, in_channels=in_channels, dims=dimensionality
)
model.to(DEVICE)
logger.info(f"Loaded model weights from {weights_path} to device {DEVICE}")
model.eval()

dataset = PrecomputedNoduleROIs(
    precomputed_dir, data_augmentation=False, dimensionality=dimensionality
)
loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
all_preds = []
all_confidence = []
all_nodule_ids = []
for i, (nodule_roi, label, nodule_id) in tqdm(
    enumerate(loader), total=len(loader), desc="Predicting on Batches"
):
    # transfer to device
    nodule_roi = nodule_roi.to(DEVICE)

    # forward pass
    logits = model(nodule_roi)
    logits = logits.to("cpu")

    # get predictions:
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
