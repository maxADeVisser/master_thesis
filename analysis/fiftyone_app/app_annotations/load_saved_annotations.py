import ast
import json
from typing import Literal

import fiftyone as fo
import pandas as pd
from tqdm import tqdm

all_experiment_ids = {
    "c30": "c30_25D_2411_1543",
    "c50": "c50_3D_2411_1831",
    "c70": "c70_25D_2411_1705",
}
fold = 0


def load_tags(dataset_name: Literal["c30", "c50", "c70"]) -> None:
    experiment_id = all_experiment_ids[dataset_name]

    # Load and parse the tags:
    tags_df = pd.read_csv(
        f"analysis/fiftyone_app/app_annotations/pandas_{experiment_id}_fold0.csv"
    ).set_index("nodule_ids")
    assert len(tags_df) == 2113, f"Expected 2113, got {len(tags_df)}"
    tags_df["tags"] = tags_df["tags"].apply(ast.literal_eval)

    dataset = fo.load_dataset(dataset_name)
    for sample in tqdm(dataset, total=len(dataset), desc="Loading tags"):
        row = tags_df.loc[sample.nodule_id]
        sample.tags = row["tags"]
        sample.save()
    dataset.save()


if __name__ == "__main__":
    load_tags("c30")
    load_tags("c50")
    load_tags("c70")
