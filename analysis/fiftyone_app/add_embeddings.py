import json
from typing import Literal

import fiftyone as fo
import fiftyone.brain as fob
import numpy as np
import pandas as pd

from project_config import env_config

all_experiment_ids = {
    "c30": {
        "2.5D": "c30_25D_2411_1543",
        "3D": "c30_3D_2411_1947",
    },
    "c50": {
        "2.5D": "c50_25D_2411_1812",
        "3D": "c50_3D_2411_1831",
    },
    "c70": {
        "2.5D": "c70_25D_2411_1705",
        "3D": "c70_3D_2411_1824",
    },
}
fold = 0


def add_embeddings(
    dataset_name: Literal["c30", "c50", "c70"],
    model_dimensionality: Literal["2.5D", "3D"],
) -> None:
    dataset = fo.load_dataset(dataset_name)
    experiment_id = all_experiment_ids[dataset_name][model_dimensionality]

    embeddings_df_path = f"{env_config.PROJECT_DIR}/model/embeddings/{experiment_id}/fold{fold}/embeddings_df.csv"
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
        zip(
            fiftyone_df["sample_id"],
            zip(fiftyone_df["x_embed"], fiftyone_df["y_embed"]),
        )
    )
    # Needs to be in numpy format:
    sample_id_embedding_mapping = {
        k: np.array(v) for k, v in sample_id_embedding_mapping.items()
    }

    fob.compute_visualization(
        samples=dataset,
        points=sample_id_embedding_mapping,
        brain_key=f"embeddings_{model_dimensionality.replace('.', '')}",
    )


if __name__ == "__main__":
    add_embeddings("c30", "2.5D")
    add_embeddings("c30", "3D")
    add_embeddings("c50", "2.5D")
    add_embeddings("c50", "3D")
    add_embeddings("c70", "2.5D")
    add_embeddings("c70", "3D")
