"""Exports tags from FiftyOne dataset to CSV file."""

import csv
import json
import os

import fiftyone as fo
from tqdm import tqdm

from utils.data_models import ExperimentAnalysis

with open("experiment_analysis_parameters.json", "r") as f:
    config = ExperimentAnalysis.model_validate(json.load(f))

experiments = ["c30_25D_2411_1543", "c50_25D_2411_1812", "c70_25D_2411_1705"]
fold = 0

for experiment_id in tqdm(experiments, total=len(experiments), desc="Exporting tags"):
    export_dir = f"/Users/newuser/Documents/ITU/master_thesis/analysis/fiftyone_app/app_annotations"
    file_name = f"{experiment_id}_fold{fold}.csv"
    dataset_name = f"{experiment_id}"
    # ----

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

    with open(f"{export_dir}/{file_name}", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "filepath", "tags", "nodule_ids"]
        )
        writer.writeheader()
        for item in tags_export:
            writer.writerow(item)
