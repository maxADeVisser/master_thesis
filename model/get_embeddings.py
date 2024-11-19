import os

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
processed_holdout_dir_path = f"{processed_dir_path}hold_out"
# ---------------------

if not os.path.exists(processed_dir_path) or not os.path.exists(
    processed_holdout_dir_path
):
    raise FileNotFoundError(
        f"Precomputed ROIs not found for {context_size} | {n_dims}. Precompute the nodule ROIs first"
    )

embeddings_out = f"{env_config.PROJECT_DIR}/out/embeddings/{experiment_id}/fold{fold}"
if not os.path.exists(embeddings_out):
    os.makedirs(embeddings_out)

model = load_resnet_model(
    f"{env_config.PROJECT_DIR}/hpc/jobs/{experiment_id}/fold_{fold}/model.pth",
    in_channels=1 if n_dims == "3D" else 3,  # for 2.5D or 3D
    dims=n_dims,
)
model.eval()

datasets = {
    "train": PrecomputedNoduleROIs(
        prepcomputed_dir=processed_dir_path, data_augmentation=False
    ),
    "holdout": PrecomputedNoduleROIs(
        prepcomputed_dir=processed_holdout_dir_path, data_augmentation=False
    ),
}

feature_vector_size = 2048  # size of the feature vector from the model
embeddings_results = {
    "train": {"embeddings": None, "labels": None, "nodule_ids": None},
    "holdout": {"embeddings": None, "labels": None, "nodule_ids": None},
}
for dataset_type, dataset in datasets.items():
    # dataset_type = "holdout"
    # dataset = PrecomputedNoduleROIs(
    #     prepcomputed_dir=processed_holdout_dir_path, data_augmentation=False
    # )
    loader = DataLoader(dataset, batch_size, shuffle=False)
    n_samples = len(dataset)
    n_batches = len(loader)
    print(f"{dataset_type} dataset samples:", n_samples)
    print(f"{dataset_type} batches:", n_batches)

    # Compute embeddings for the datasets and store results in the @embeddings_results dict:
    all_dataset_embeddings = torch.empty(
        size=(n_samples, feature_vector_size), dtype=torch.float, device="cpu"
    )
    all_labels = []
    all_nodule_ids = []
    start_idx = 0
    with torch.no_grad():
        for _, (inputs, labels, nodule_ids) in tqdm(
            enumerate(loader),
            desc=f"Creating {dataset_type} Embeddings (batches)",
            total=n_batches,
        ):
            batch_embeddings = model.get_feature_vector(inputs)

            cur_batch_size = inputs.size(0)
            end_idx = start_idx + cur_batch_size

            all_dataset_embeddings[start_idx:end_idx, :] = batch_embeddings
            all_labels.extend(labels.tolist())
            all_nodule_ids.extend(nodule_ids)

            start_idx += cur_batch_size

    embeddings_results[dataset_type]["embeddings"] = all_dataset_embeddings
    embeddings_results[dataset_type]["labels"] = all_labels
    embeddings_results[dataset_type]["nodule_ids"] = all_nodule_ids

# Reduce embeddings to 2 dimensions using t-SNE:
tnse = TSNE(n_components=2, perplexity=30, random_state=SEED)

# TODO, we cant compute the t-sne on the two datasets seperately, we need to combine them and then compute the t-sne. This is because the t-sne is a non-linear dimensionality reduction technique that tries to preserve the local structure of the data. If we compute the t-sne on the two datasets seperately, we will not be able to compare the embeddings of the two datasets.

# Train
train_reduced_embeddings = tnse.fit_transform(
    embeddings_results["train"]["embeddings"].numpy()
)
train_embeddings_df = pd.DataFrame(
    {
        "nodule_id": embeddings_results["train"]["nodule_ids"],
        "label": embeddings_results["train"]["labels"],
        "x_embed": train_reduced_embeddings[:, 0],
        "y_embed": train_reduced_embeddings[:, 1],
        "dataset": "train",
    }
).set_index("nodule_id")

# Holdout
holdout_reduced_embeddings = tnse.fit_transform(
    embeddings_results["holdout"]["embeddings"].numpy()
)
holdout_embeddings_df = pd.DataFrame(
    {
        "nodule_id": embeddings_results["holdout"]["nodule_ids"],
        "label": embeddings_results["holdout"]["labels"],
        "x_embed": holdout_reduced_embeddings[:, 0],
        "y_embed": holdout_reduced_embeddings[:, 1],
        "dataset": "holdout",
    }
).set_index("nodule_id")

# Combine the embeddings dataframes
combined = pd.concat([train_embeddings_df, holdout_embeddings_df], axis=0)

# save the embeddings, nodule_id and labels in a pandas dataframe
embeddings_out_file = f"{embeddings_out}/embeddings_df.csv"
combined.to_csv(embeddings_out_file, index=True)
