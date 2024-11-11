# %%
import datetime as dt

from pydantic import BaseModel, Field

from utils.common_imports import *


class ExperimentDataset(BaseModel):
    """Base Experiment Dataset for input validation"""

    image_dims: list[int] = Field(
        ..., description="Uniform dimensions of the images: (H, W, D)"
    )
    clipping_bounds: list[int, int] = Field(
        [-1000, 400],
        description="Bounds for normalisation of Hounsfield Units in the CT scans. Default corresponds to air (lowerbound) and bone (upperbound)",
    )
    segment_nodule: Literal["none", "remove_background", "remove_nodule"] = Field(
        "none", description="Segmentation setting for the nodules."
    )


class ExperimentModel(BaseModel):
    """Base Experiment Model for input validation"""

    name: str = Field(..., description="Name of the model.")
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
    cross_validation_folds: int | None = Field(
        None, description="If provided, determines number of CV folds. If None, no CV."
    )
    cv_train_folds: int = Field(
        ...,
        description="Number of folds to train. If train on all folds, set equal to cross_validation_folds.",
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
    context_window_size: int | None = Field(None, description="Image dimension.")


class BaseExperimentConfig(BaseModel):
    """Base Experiment Configuration for input validation"""

    name: str = Field(..., description="Name of the experiment.")
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
