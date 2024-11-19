from typing import Literal

import fiftyone as fo
import fiftyone.brain as fob
import numpy as np
import pandas as pd

from project_config import env_config

# SCRIPT PARAMS:
context_size: Literal[30, 50, 70] = 30
experiment_id = "c30_3D_1711_1513"
fold = 3
# ---

nodule_roi_jpg_path = (
    f"{env_config.PROJECT_DIR}/data/middle_slice_images_c{context_size}"
)

# load predictions and embeddings
pred_df_path = (
    f"{env_config.OUT_DIR}/predictions/{experiment_id}/pred_nodule_df_fold{fold}.csv"
)
embeddings_df_path = (
    f"{env_config.OUT_DIR}/embeddings/{experiment_id}/fold{fold}/embeddings_df.csv"
)
pred_nodule_df = pd.read_csv(pred_df_path).set_index("nodule_id")
embeddings_df = pd.read_csv(embeddings_df_path).set_index("nodule_id")

# load embeddings
# TODO improve how embeddings are computed!
# embeddings_path = f"{env_config.OUT_DIR}/embeddings/{experiment_id}/fold{fold}"
# embeddings = np.load(f"{embeddings_path}/tnse_embeddings.npy")
# labels = np.load(f"{embeddings_path}/labels.npy")

dataset_name = f"C{context_size}_Nodule_ROIs"
if dataset_name in fo.list_datasets():
    fo.delete_dataset(dataset_name)


# --- CREATE DATASET ---
# (Only run once - stores in a MongoDB database)
dataset = fo.Dataset.from_images_patt(
    images_patt=f"{nodule_roi_jpg_path}/*.jpg", name=dataset_name
)
# dataset = fo.load_dataset(dataset_name)

for sample in dataset:
    nodule_id = sample.filename.split(".")[0]
    row = pred_nodule_df.loc[nodule_id]
    row_embed = embeddings_df.loc[nodule_id]

    # Store classification in a field name of your choice
    sample["nodule_id"] = pred_nodule_df.loc[nodule_id].name
    sample["malignancy_consensus"] = fo.Classification(
        label=str(row["malignancy_consensus"])
    )
    sample["malignancy_scores"] = row["malignancy_scores"]
    sample["prediction"] = fo.Classification(
        label=str(row["pred"])
    )  # TODO add confidence?
    sample["subtlety"] = row["subtlety_consensus"]
    sample["subtlety_scores"] = row["subtlety_scores"]
    sample["cancer_label"] = row["cancer_label"]
    sample["ann_mean_volume"] = row["ann_mean_volume"]
    sample["ann_mean_diameter"] = row["ann_mean_diameter"]
    sample.save()

nodule_id_sample_id_mapping = {sample["nodule_id"]: sample.id for sample in dataset}
nodule_id_sample_id_mapping = (
    pd.DataFrame.from_dict(nodule_id_sample_id_mapping, orient="index")
    .reset_index()
    .rename(columns={"index": "nodule_id", 0: "sample_id"})
)
fiftyone_df = pd.merge(pred_nodule_df, nodule_id_sample_id_mapping, on="nodule_id")
fiftyone_df = pd.merge(fiftyone_df, embeddings_df, on="nodule_id")

fiftyone_df[["sample_id", "x_embed", "y_embed"]]

sample_id_embedding_mapping = dict(
    zip(fiftyone_df["sample_id"], zip(fiftyone_df["x_embed"], fiftyone_df["y_embed"]))
)
# Needs to be in numpy format:
sample_id_embedding_mapping = {
    k: np.array(v) for k, v in sample_id_embedding_mapping.items()
}

fob.compute_visualization(
    samples=dataset, points=sample_id_embedding_mapping, brain_key="img_viz"
)

# --- LAUNCH APP ---
session = fo.launch_app(dataset=dataset)
session.wait()
