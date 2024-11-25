"""Exports tags from FiftyOne dataset to CSV file."""

import csv

import fiftyone as fo

# SCRIPT PARAMS:
export_dir = "/Users/newuser/Documents/ITU/master_thesis/out/fiftyone/test"
dataset_name = "test"
# ----

dataset = fo.load_dataset(dataset_name)
tags_export = [
    {"filepath": sample.filepath, "tags": sample.tags, "nodule_ids": sample.nodule_id}
    for sample in dataset
]


with open(f"{export_dir}/tags_export.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filepath", "tags"])
    writer.writeheader()
    for item in tags_export:
        writer.writerow(item)
