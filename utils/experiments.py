# %%
import datetime as dt
import os
from typing import Optional

from pydantic import BaseModel, Field

from utils.common_imports import *


class ExperimentDataset(BaseModel):
    """Base Experiment Dataset for input validation"""

    dataset_desc: str = Field(..., description="Description of the dataset.")
    image_dim: int = Field(
        ..., description="Uniform dimensions of the images: (H, W, D)"
    )
    consensus_level: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Consensus level used for computing the consensus mask of the segmentations",
    )
    normalisation_bounds: tuple[int, int] = Field(
        [-1000, 400],
        description="Bounds for normalisation of Hounsfield Units in the CT scans. Default corresponds to air (lowerbound) and bone (upperbound)",
    )
    segment_nodule: Literal["none", "remove_background", "remove_nodule"] = Field(
        "none", description="Segmentation setting for the nodules."
    )
    validation_split: float = Field(  # TODO not used yet ...
        0.2, description="Fraction of data to use for validation split."
    )


class ExperimentModel(BaseModel):
    """Base Experiment Model for input validation"""

    name: str = Field(..., description="Name of the model.")
    description: str = Field(..., description="Description of the model.")
    num_classes: int = Field(
        ..., ge=2, description="Number of output classes for classification."
    )
    in_channels: int = Field(
        ..., ge=1, description="Number of input channels. 1 for grayscale, 3 for RGB."
    )


class ExperimentTraining(BaseModel):
    batch_size: int = Field(..., ge=1, description="Batch size for training.")
    optimiser: str = Field("adam", description="Optimiser to use for training.")
    learning_rate: float = Field(
        ..., gt=0, description="Learning rate for the optimizer."
    )
    num_epochs: int = Field(..., ge=1, description="Number of epochs for training.")
    epoch_print_interval: int = Field(
        10, description="Interval for printing epoch info."
    )
    cross_validation_folds: Optional[int] = Field(
        None, description="If provided, determines number of CV folds. If None, no CV."
    )

    # TODO add for moving around nodule?
    # augmentation: bool = Field(False, description="Whether to apply data augmentation.")


class ExperimentEvaluation(BaseModel):
    """Base Experiment Evaluation for input validation"""

    eval_metrics: list[str] = Field(
        ..., description="List of evaluation metrics to use."
    )
    metrics_results: dict = Field({}, description="Results of the evaluation.")
    cv_epoch_losses: dict = Field({}, description="List of epoch losses.")


class BaseExperimentConfig(BaseModel):
    """Base Experiment Configuration for input validation"""

    # Experiment metadata
    name: str = Field(..., description="Name of the experiment.")
    description: str = Field(..., description="Description of the experiment.")
    start_time: dt.datetime = Field(
        dt.datetime.now(), description="Start time of the experiment."
    )
    end_time: dt.datetime | None = Field(
        None, description="End time of the experiment."
    )
    duration: dt.timedelta | None = Field(
        None, description="Duration of the experiment."
    )
    # TODO add logs from the experiment
    # logs: dict = Field({}, description="Logs of the experiment.")
    dataset: ExperimentDataset = Field(..., description="Dataset configuration.")
    model: ExperimentModel = Field(..., description="Model configuration.")
    training: ExperimentTraining = Field(..., description="Training configuration.")
    evaluation: ExperimentEvaluation = Field(
        ..., description="Evaluation configuration."
    )

    def write_experiment_to_json(self, out_dir: str) -> None:
        """Write the experiment configuration to a JSON file."""
        experiment_file_name = self.start_time.strftime("%d/%m/%Y-%H:%M:%S")
        with open(f"{out_dir}/{experiment_file_name}", "w") as f:
            json.dump(self.model_dump(), f)


def create_experiment_from_json(
    name: str,
    desc: str,
    out_dir: str,
    config_json_path: str = "pipeline_parameters.json",
) -> BaseExperimentConfig:
    """Load the configuration from the pipeline parameters JSON file."""
    with open(config_json_path, "r") as f:
        config = json.load(f)
    return BaseExperimentConfig(name=name, description=desc, out_dir=out_dir, **config)


class ContextExperimentConfig(BaseExperimentConfig):
    """Context Experiment Configuration"""

    image_dims: list[int] = [8, 16, 32, 64, 128]


# %%
if __name__ == "__main__":
    experiment1 = create_experiment_from_json(name="test", desc="first test experiment")
