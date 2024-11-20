import json

import fiftyone as fo
import fiftyone.brain as fob
import numpy as np
import pandas as pd

from project_config import env_config

# TODO put the path to the experiment_analysis_parameters.json file into the env_config in all places where used
with open("experiment_analysis_parameters.json", "r") as f:
    config = json.load(f)

context_size = config["context_size"]
experiment_id = config["experiment_id"]
fold = config["fold"]

dataset = fo.load_dataset(f"C{context_size}_Nodule_ROIs")

embeddings_df_path = (
    f"{env_config.OUT_DIR}/embeddings/{experiment_id}/fold{fold}/embeddings_df.csv"
)
embeddings_df = pd.read_csv(embeddings_df_path).set_index("nodule_id")

nodule_id_sample_id_mapping = {sample["nodule_id"]: sample.id for sample in dataset}
nodule_id_sample_id_mapping = (
    pd.DataFrame.from_dict(nodule_id_sample_id_mapping, orient="index")
    .reset_index()
    .rename(columns={"index": "nodule_id", 0: "sample_id"})
)

fiftyone_df = pd.merge(nodule_id_sample_id_mapping, embeddings_df, on="nodule_id")

# get the format needed for the compute_visualization function:
sample_id_embedding_mapping = dict(
    zip(fiftyone_df["sample_id"], zip(fiftyone_df["x_embed"], fiftyone_df["y_embed"]))
)
# Needs to be in numpy format:
sample_id_embedding_mapping = {
    k: np.array(v) for k, v in sample_id_embedding_mapping.items()
}

fob.compute_visualization(
    samples=dataset, points=sample_id_embedding_mapping, brain_key="resnet50_embeddings"
)

session = fo.launch_app(dataset=dataset, port=5151)
session.wait()
