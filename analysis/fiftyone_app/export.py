import csv

import fiftyone as fo

# The Dataset or DatasetView containing the samples you wish to export
dataset = fo.load_dataset("test")
tags_export = [{"filepath": sample.filepath, "tags": sample.tags} for sample in dataset]

# The directory to which to write the exported dataset
export_dir = "/Users/newuser/Documents/ITU/master_thesis/out/fiftyone/test"

# Write tags to CSV
with open(f"{export_dir}/tags_export.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filepath", "tags"])
    writer.writeheader()
    for item in tags_export:
        writer.writerow(item)
