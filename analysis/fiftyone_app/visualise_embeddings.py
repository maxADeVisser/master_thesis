from typing import Literal

import fiftyone as fo
import fiftyone.brain as fob
import numpy as np
import pandas as pd

from project_config import env_config

context_size: Literal[30, 50, 70] = 30
experiment_id = "c30_3D_1711_1513"
fold = 3

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
    samples=dataset, points=sample_id_embedding_mapping, brain_key="img_viz"
)

session = fo.launch_app(dataset=dataset, port=5151)
session.wait()
