"""Exports tags from FiftyOne dataset to CSV file."""

import os

import fiftyone as fo
import pandas as pd
from tqdm import tqdm

experiments = [
    "c30_25D_2411_1543",
    # "c50_25D_2411_1812",
    "c70_25D_2411_1705",
    "c50_3D_2411_1831",
]
fold = 0

for experiment_id in tqdm(experiments, total=len(experiments), desc="Exporting tags"):
    export_dir = f"/Users/maxvisser/Documents/ITU/master_thesis/analysis/fiftyone_app/app_annotations"
    out_file_name = f"{experiment_id}_fold{fold}.csv"
    dataset_name = f"{experiment_id}"

    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    dataset = fo.load_dataset(dataset_name)
    tags_export = [
        {
            "sample_id": sample.id,
            "filepath": sample.filepath,
            "tags": sample.tags,
            "nodule_ids": sample.nodule_id,
        }
        for sample in dataset
    ]

    df = pd.DataFrame(tags_export)
    df.to_csv(f"{export_dir}/pandas_{out_file_name}", index=False)
