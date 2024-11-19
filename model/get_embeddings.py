import os

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PrecomputedNoduleROIs
from model.ResNet import load_resnet_model
from project_config import SEED, env_config

# TODO seperate the train and validation sets when getting the embeddings!
# TODO also add the embeddings for the holdout set and make a flag indicating them

# QUESTION which data to do the embedding on? (using train and validation now for testing)

# --- SCRIPT PARAMS ---
experiment_id = "c30_3D_1711_1513"
context_size = 30
fold = 3
n_dims = "3D"
batch_size = 8
processed_dir_path = f"{env_config.PROJECT_DIR}/data/precomputed_rois_30C_3D"
# ---------------------

if not os.path.exists(processed_dir_path):
    raise FileNotFoundError(
        f"Precomputed ROIs found for {context_size} | {n_dims}. Precompute the nodule ROIs first"
    )

weights_path = (
    f"{env_config.PROJECT_DIR}/hpc/jobs/{experiment_id}/fold_{fold}/model.pth"
)
embeddings_out = f"{env_config.PROJECT_DIR}/out/embeddings/{experiment_id}/fold{fold}"
if not os.path.exists(embeddings_out):
    os.makedirs(embeddings_out)
ic = 1 if n_dims == "3D" else 3  # for 2.5D or 3D

model = load_resnet_model(weights_path, in_channels=ic, dims=n_dims)
model.eval()

dataset = PrecomputedNoduleROIs(prepcomputed_dir=processed_dir_path)
loader = DataLoader(dataset, batch_size, shuffle=False)
n_samples = len(dataset)
n_batches = len(loader)
print("dataset samples:", n_samples)
print("batches:", n_batches)

feature_vector_size = 2048  # size of the feature vector from the ResNet model
all_embeddings = torch.empty(
    size=(n_samples, feature_vector_size), dtype=torch.float, device="cpu"
)
all_labels = []
all_nodule_ids = []

start_idx = 0
with torch.no_grad():
    for _, (inputs, labels, nodule_ids) in tqdm(
        enumerate(loader), desc="Creating Embeddings (batches)", total=n_batches
    ):
        batch_embeddings = model.get_feature_vector(inputs)

        cur_batch_size = inputs.size(0)
        end_idx = start_idx + cur_batch_size

        all_embeddings[start_idx:end_idx, :] = batch_embeddings
        all_labels.extend(labels.tolist())
        all_nodule_ids.extend(nodule_ids)
        start_idx += cur_batch_size

# Reduce embeddings to 2 dimensions using t-SNE:
tnse = TSNE(n_components=2, perplexity=30, random_state=SEED)
dim_reduced_embeddings = tnse.fit_transform(all_embeddings.numpy())

embeddings_df = pd.DataFrame(
    {
        "nodule_id": all_nodule_ids,
        "label": all_labels,
        "x_embed": dim_reduced_embeddings[:, 0],
        "y_embed": dim_reduced_embeddings[:, 1],
    }
).set_index("nodule_id")

# save the embeddings, nodule_id and labels in a pandas dataframe
embeddings_out_file = f"{embeddings_out}/embeddings_df.csv"
embeddings_df.to_csv(embeddings_out_file, index=True)
