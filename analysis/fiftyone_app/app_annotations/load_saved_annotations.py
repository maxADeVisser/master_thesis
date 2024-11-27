import ast
import json

import fiftyone as fo
import pandas as pd
from tqdm import tqdm

from utils.data_models import ExperimentAnalysis

# PARAMS
with open("experiment_analysis_parameters.json", "r") as f:
    config = ExperimentAnalysis.model_validate(json.load(f))


context_size = config.analysis.context_size
experiment_id = config.experiment_id
fold = 0

# Load and parse the tags:
tags_df = pd.read_csv(
    f"analysis/fiftyone_app/app_annotations/pandas_{experiment_id}_fold0.csv"
).set_index("nodule_ids")
assert len(tags_df) == 2113, f"Expected 2113, got {len(tags_df)}"
tags_df["tags"] = tags_df["tags"].apply(ast.literal_eval)

dataset = fo.load_dataset(f"{experiment_id}")
for sample in tqdm(dataset, total=len(dataset), desc="Loading tags"):
    row = tags_df.loc[sample.nodule_id]
    sample.tags = row["tags"]
    sample.save()
dataset.save()
