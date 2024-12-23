import os
import sys

from dotenv import load_dotenv

load_dotenv(".env")
sys.path.append(os.getenv("PROJECT_DIR"))

import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PrecomputedNoduleROIs
from model.ResNet import load_resnet_model
from project_config import env_config
from train.train import evaluate_model

experiments = [
    "c30_25D_2411_1543",
    "c50_25D_2411_1812",
    "c70_25D_2411_1705",
    "c30_3D_2411_1947",
    "c50_3D_2411_1831",
    "c70_3D_2411_1824",
]
folds = [0, 1, 2, 3, 4]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_exp_results = {}
for exp in tqdm(experiments, desc="Evaluating models"):
    all_exp_results[exp] = {}
    for fold in tqdm(folds, desc=f"Folds for {exp}"):
        weights_path = f"out/model_runs/{exp}/fold{fold}/model.pth"

        in_channels = 1 if "3D" in exp else 3
        dims = "3D" if "3D" in exp else "2.5D"
        context = int(exp.split("_")[0][1:])

        model = load_resnet_model(weights_path, in_channels=in_channels, dims=dims)

        precomputed_holdout_path = f"{env_config.PROJECT_DIR}/data/precomputed_resampled_rois_{context}C_{dims}hold_out"
        holdout_dataset = PrecomputedNoduleROIs(
            precomputed_holdout_path,
            dims,
            data_augmentation=False,
            center_mask_size=None,
        )
        holdout_loader = DataLoader(
            holdout_dataset, batch_size=4, shuffle=False, num_workers=4
        )
        metric_results = evaluate_model(model, holdout_loader)

        all_exp_results[exp][fold] = {
            "avg_val_loss": metric_results["avg_val_loss"],
            "accuracy": metric_results["accuracy"],
            "binary_accuracy": metric_results["binary_accuracy"],
            "AUC_ovr": metric_results["AUC_ovr"],
            "AUC_filtered": metric_results["AUC_filtered"],
            "mae": metric_results["mae"],
            "mse": metric_results["mse"],
            "cwce": metric_results["cwce"],
            "fold_errors": metric_results["errors"],
        }

    # Write incremental results to file
    with open(f"{env_config.PROJECT_DIR}/model/model_benchmark_results.json", "w") as f:
        json.dump(all_exp_results, f)
