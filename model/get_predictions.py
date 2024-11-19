import os

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PrecomputedNoduleROIs
from model.ResNet import get_pred_malignancy_from_logits, load_resnet_model
from project_config import env_config

# --- SCRIPT PARAMS ---
CONTEXT_WINDOW_SIZE = 30
experiment_id = "c30_3D_1711_1513"
dimensionality = "3D"
fold = 3

precomputed_dir = f"{env_config.PROJECT_DIR}/data/precomputed_rois_{CONTEXT_WINDOW_SIZE}C_{dimensionality}"
pred_out_dir = f"{env_config.OUT_DIR}/predictions/{experiment_id}"
pred_out_file = f"{pred_out_dir}/pred_nodule_df_fold{fold}.csv"
if os.path.exists(pred_out_file):
    raise FileExistsError(f"Predictions already exist at {pred_out_file}. Reset first")
if not os.path.exists(pred_out_dir):
    os.makedirs(pred_out_dir, exist_ok=True)

weights_path = (
    f"{env_config.PROJECT_DIR}/hpc/jobs/{experiment_id}/fold_{fold}/model.pth"
)

nodule_df = pd.read_csv(env_config.processed_nodule_df_file)
nodule_df["nodule_id"] = (
    nodule_df["pid"].astype(str) + "_" + nodule_df["nodule_idx"].astype(str)
)

model = load_resnet_model(weights_path=weights_path, in_channels=1, dims=dimensionality)
model.eval()

dataset = PrecomputedNoduleROIs(precomputed_dir)
loader = DataLoader(dataset, batch_size=16, shuffle=False)
all_preds = []
all_labels = []
all_nodule_ids = []
for i, (nodule_roi, label, nodule_id) in tqdm(enumerate(loader), total=len(loader)):
    logits = model(nodule_roi)
    preds = get_pred_malignancy_from_logits(logits).tolist()
    all_preds.extend(preds)
    all_labels.extend(label.tolist())
    all_nodule_ids.extend(nodule_id)

pred_df = pd.DataFrame({"nodule_id": all_nodule_ids, "pred": all_preds})
pred_nodule_df = pd.merge(nodule_df, pred_df, on="nodule_id", how="left")
assert len(pred_nodule_df) == len(
    nodule_df
), "Length of prediction dataframe does not match length of nodule dataframe"

pred_nodule_df.to_csv(pred_out_file, index=False)
