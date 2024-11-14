# %%
import datetime as dt

from pydantic import BaseModel, Field

from utils.common_imports import *


class ExperimentDataset(BaseModel):
    """Base Experiment Dataset for input validation"""

    context_window: Literal[10, 20, 30, 40, 50, 60, 70] = Field(
        ..., description="The uniform size of the context window for the nodule ROI."
    )
    clipping_bounds: list[int, int] = Field(
        [-1000, 400],
        description="Bounds for normalisation of Hounsfield Units in the CT scans. Default corresponds to air (lowerbound) and bone (upperbound)",
    )
    segment_nodule: Literal["none", "remove_background", "remove_nodule"] = Field(
        "none", description="Segmentation setting for the nodules."
    )
    dimensionality: Literal["2.5D", "3D"] = Field(
        "3D",
        description="Dimensionality. If 2.5D, the middle +/- one slice is used. That is, it mimics a RGB image with 3 channels",
    )


class ExperimentModel(BaseModel):
    """Base Experiment Model for input validation"""

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
    do_cross_validation: bool = Field(
        ...,
        description="Whether to do cross-validation or only train on a single fold.",
    )
    cross_validation_folds: int | None = Field(
        None, description="If provided, determines number of CV folds. If None, no CV."
    )
    cv_train_folds: int = Field(
        ...,
        description="Number of folds to train. If train on all folds, set equal to cross_validation_folds.",
    )
    num_workers: int | None = Field(
        None, ge=0, description="Number of workers for data loading."
    )
    patience: int | None = Field(
        5,
        ge=1,
        description="Early stopping parameter: number of epochs to wait for improvement.",
    )
    min_delta: float | None = Field(
        0.5,
        ge=0,
        description="Early stopping parameter: minimum change in the monitored metric to qualify as an improvement.",
    )
    gpu_used: str | None = Field(None, description="Name of GPU used for training.")


class BaseExperimentConfig(BaseModel):
    """Base Experiment Configuration for input validation"""

    config_name: str = Field(..., description="Name of the experiment.")
    id: str | None = Field(None, description="ID of the experiment.")
    start_time: dt.datetime | str = Field(
        dt.datetime.now(), description="Start time of the experiment."
    )
    end_time: dt.datetime | str | None = Field(
        None, description="End time of the experiment."
    )
    duration: dt.timedelta | str | None = Field(
        None, description="Duration of the experiment."
    )
    dataset: ExperimentDataset = Field(..., description="Dataset configuration.")
    model: ExperimentModel = Field(..., description="Model configuration.")
    training: ExperimentTraining = Field(..., description="Training configuration.")
    results: dict | None = Field(None, description="Results")

    def write_experiment_to_json(self, out_dir: str) -> None:
        """Write the experiment to a JSON file."""
        self.start_time = str(self.start_time)
        if self.end_time:
            self.end_time = str(self.end_time)
        if self.duration:
            self.duration = str(self.duration)
        with open(f"{out_dir}/run_{self.id}.json", "w") as f:
            json.dump(self.model_dump(), f)


def create_experiment_from_json(
    name: str,
    out_dir: str,
    config_json_path: str = "pipeline_parameters.json",
) -> BaseExperimentConfig:
    """Load the configuration from the pipeline parameters JSON file."""
    with open(config_json_path, "r") as f:
        config = json.load(f)
    return BaseExperimentConfig(name=name, out_dir=out_dir, **config)


def load_experiment_from_json(
    experiment_file: str,
) -> BaseExperimentConfig:
    """Load the configuration from the pipeline parameters JSON file."""
    with open(experiment_file, "r") as f:
        config = json.load(f)
    return BaseExperimentConfig(**config)


# %%
if __name__ == "__main__":
    name = "test"
    out_dir = "out"
    config_json_path = "pipeline_parameters.json"
    test = create_experiment_from_json(name, out_dir)
