import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PrecomputedNoduleROIs
from model.ResNet import load_resnet_model
from project_config import env_config, pipeline_config
from train.train import evaluate_model

dimensionality = pipeline_config.dataset.dimensionality
context_window = pipeline_config.dataset.context_window

out_dir = f"{env_config.PROJECT_DIR}"

experiments = [
    "c30_25D_2411_1543",
    "c50_25D_2411_1812",
    "c70_25D_2411_1705",
    "c30_3D_2411_1947",
    "c50_3D_2411_1831",
    "c70_3D_2411_1824",
]
fold = 0

all_exp_results = {}

for exp in tqdm(experiments, desc="Evaluating models"):
    weights_path = f"hpc/jobs/{exp}/fold_{fold}/model.pth"

    in_channels = 1 if "3D" in exp else 3
    dims = "3D" if "3D" in exp else "2.5D"
    context = int(exp.split("_")[0][1:])

    model = load_resnet_model(weights_path, in_channels=in_channels, dims=dims)
    precomputed_holdout_path = f"{env_config.PROJECT_DIR}/data/precomputed_resampled_rois_{context}C_{dims}hold_out"
    holdout_dataset = PrecomputedNoduleROIs(
        precomputed_holdout_path, dims, data_augmentation=False, center_mask_size=None
    )
    holdout_loader = DataLoader(
        holdout_dataset, batch_size=4, shuffle=False, num_workers=0
    )
    metric_results = evaluate_model(model, holdout_loader)

    # errors = metric_results["errors"]
    all_exp_results[exp] = {
        "avg_val_loss": metric_results["avg_val_loss"],
        "accuracy": metric_results["accuracy"],
        "binary_accuracy": metric_results["binary_accuracy"],
        "AUC_ovr": metric_results["AUC_ovr"],
        "AUC_filtered": metric_results["AUC_filtered"],
        "mae": metric_results["mae"],
        "mse": metric_results["mse"],
        "cwce": metric_results["cwce"],
    }

# make a dataframe
results_df = pd.DataFrame(all_exp_results).T
results_df.to_csv(f"{out_dir}/model/model_benchmark_results.csv")