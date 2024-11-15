import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import LIDC_IDRI_DATASET
from model.ResNet import load_resnet_model

# TODO seperate the train and validation sets when getting the embeddings!
experiment_id = "c50_25d_1311_1450"
fold = 0
out_dir = f"hpc/jobs/{experiment_id}/fold_{fold}"
weights_path = f"{out_dir}/model_fold{fold}.pth"
n_dims = "2.5D"
embeddings_out = f"out/embeddings/fold{fold}/embeddings.npy"
if not os.path.exists(embeddings_out):
    os.makedirs(os.path.dirname(embeddings_out), exist_ok=True)

model = load_resnet_model(weights_path, in_channels=3, dims=n_dims)
model.eval()

# QUESTION which data to do the embedding on?
dataset = LIDC_IDRI_DATASET(
    context_size=50,
    segmentation_configuration="none",
    n_dims=n_dims,
    # nodule_df_path="preprocessing/hold_out_nodule_df.csv",
)
loader = DataLoader(dataset, batch_size=8, shuffle=False)
n_samples = len(dataset)
n_batches = len(loader)
print("dataset:", n_samples)
print("batches:", n_batches)

all_embeddings = torch.empty(n_samples, 8, 2048, dtype=torch.float, device="cpu")
start_idx = 0
with torch.no_grad():
    for i, (inputs, _) in tqdm(
        enumerate(loader), desc="Getting embeddings", total=n_batches
    ):
        batch_embeddings = model.get_feature_vector(inputs)

        batch_size = inputs.size(0)
        end_idx = start_idx + batch_size
        all_embeddings[start_idx:end_idx] = batch_embeddings
        start_idx += batch_size
# BUG i think there is a problem with the last batch. It fails there

# reshape lthe tensor to (n_samples, 2048):
all_embeddings = all_embeddings.view(n_samples, -1).numpy()
np.save(embeddings_out, all_embeddings)
