import json
import os

import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PrecomputedNoduleROIs
from model.ResNet import load_resnet_model
from project_config import SEED, env_config
from utils.data_models import ExperimentAnalysis

# TODO also add the embeddings for the holdout set and make a flag indicating them

# SCRIPT PARAMS ---------
with open("experiment_analysis_parameters.json", "r") as f:
    config = ExperimentAnalysis.model_validate(json.load(f))

# --- SCRIPT PARAMS ---
experiment_id = config.experiment_id
context_size = config.analysis.context_size
fold = config.analysis.fold
n_dims = config.analysis.dimensionality
batch_size = 4
# ---------------------

processed_dir_path = (
    f"{env_config.PROJECT_DIR}/data/precomputed_resampled_rois_{context_size}C_{n_dims}"
)

if not os.path.exists(processed_dir_path):
    raise FileNotFoundError(
        f"Precomputed ROIs not found for {context_size} | {n_dims} (both holdout and full needs to be there). Precompute the nodule ROIs first using precomputed_nodule_dataset.py"
    )

embeddings_out = f"{env_config.PROJECT_DIR}/out/embeddings/{experiment_id}/fold{fold}"
if not os.path.exists(embeddings_out):
    os.makedirs(embeddings_out)
embeddings_out_file = f"{embeddings_out}/embeddings_df.csv"

model = load_resnet_model(
    f"{env_config.PROJECT_DIR}/hpc/jobs/{experiment_id}/fold_{fold}/model.pth",
    in_channels=1 if n_dims == "3D" else 3,  # for 2.5D or 3D
    dims=n_dims,
)
model.eval()

dataset = PrecomputedNoduleROIs(
    prepcomputed_dir=processed_dir_path, data_augmentation=False, remove_center=False
)

feature_vector_size = 2048  # size of the feature vector from the model
loader = DataLoader(dataset, batch_size, shuffle=False)
n_samples = len(dataset)
n_batches = len(loader)
print(f"dataset samples:", n_samples)
print(f"batches:", n_batches)

# Compute embeddings for the datasets and store results in the @embeddings_results dict:
all_dataset_embeddings = torch.empty(
    size=(n_samples, feature_vector_size), dtype=torch.float, device="cpu"
)
all_labels = []
all_nodule_ids = []
start_idx = 0
embeddings_results = {}
with torch.no_grad():
    for _, (inputs, labels, nodule_ids) in tqdm(
        enumerate(loader),
        desc=f"Creating Embeddings (batches)",
        total=n_batches,
    ):
        batch_embeddings = model.get_feature_vector(inputs)

        cur_batch_size = inputs.size(0)
        end_idx = start_idx + cur_batch_size

        all_dataset_embeddings[start_idx:end_idx, :] = batch_embeddings
        all_labels.extend(labels.tolist())
        all_nodule_ids.extend(nodule_ids)

        start_idx += cur_batch_size

embeddings_results["embeddings"] = all_dataset_embeddings
embeddings_results["labels"] = all_labels
embeddings_results["nodule_ids"] = all_nodule_ids

# Reduce embeddings to 2 dimensions using t-SNE:
tnse = TSNE(n_components=2, perplexity=30, random_state=SEED)

# TODO, we cant compute the t-sne on the two datasets seperately, we need to combine them and then compute the t-sne. This is because the t-sne is a non-linear dimensionality reduction technique that tries to preserve the local structure of the data. If we compute the t-sne on the two datasets seperately, we will not be able to compare the embeddings of the two datasets.

train_reduced_embeddings = tnse.fit_transform(embeddings_results["embeddings"].numpy())
train_embeddings_df = pd.DataFrame(
    {
        "nodule_id": embeddings_results["nodule_ids"],
        "label": embeddings_results["labels"],
        "x_embed": train_reduced_embeddings[:, 0],
        "y_embed": train_reduced_embeddings[:, 1],
    }
).set_index("nodule_id")

# save the embeddings, nodule_id and labels in a pandas dataframe
train_embeddings_df.to_csv(embeddings_out_file, index=True)
