import json

import fiftyone as fo

from utils.data_models import ExperimentAnalysis

with open("experiment_analysis_parameters.json", "r") as f:
    config = ExperimentAnalysis.model_validate(json.load(f))

experiment_id = config.experiment_id
fold = config.analysis.fold

dataset = fo.load_dataset(f"{experiment_id}")

session = fo.launch_app(dataset=dataset, port=5151)
session.wait()
