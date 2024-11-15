# %%
import datetime as dt

from pydantic import BaseModel, Field

from utils.common_imports import *


class TrainingFold(BaseModel):
    fold_id: str = Field(..., description="ID of the fold (e.g. fold0_experimentID)")
    train_idxs: list[int] = Field(
        ..., description="List of training indicies used for the fold"
    )
    val_idxs: list[int] = Field(
        ..., description="List of validation indicies used for the fold"
    )
    start_time: dt.datetime | str = Field(..., description="Start time of the fold.")
    duration: dt.timedelta | str | None = Field(
        None, description="Duration of the fold"
    )
    train_losses: list[float] = Field(
        [], description="List of average training loss for all epochs"
    )
    val_losses: list[float] = Field(
        [], description="List of average validation loss for all epochs"
    )
    val_accuracies: list[float] = Field(
        [], description="List of average validation accuracies for all epochs"
    )
    val_AUC_filtered: list[float] = Field(
        [], description="List of average validation AUC for non-ambiguous cases"
    )
    val_AUC_ovr: list[float] = Field(
        [], description="List of average validation AUC for all classes"
    )
    val_maes: list[float] = Field(
        [], description="List of average validation MAE for all epochs"
    )
    val_mses: list[float] = Field(
        [], description="List of average validation MSE for all epochs"
    )
    val_cwces: list[float] = Field(
        [], description="List of average validation CWCE for all epochs"
    )
    best_loss: float | None = Field(None, description="Best loss for the fold")
    epoch_stopped: int | None = Field(
        None, description="Epoch where early stopping stopped training"
    )

    def write_fold_to_json(self, out_dir: str) -> None:
        """Write the experiment to a JSON file."""
        copy = self.model_copy()
        copy.start_time = str(copy.start_time)
        if copy.duration:
            copy.duration = str(copy.duration)
        with open(f"{out_dir}/{copy.fold_id}.json", "w") as f:
            json.dump(copy.model_dump(), f)
        del copy


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
    data_augmentation: bool = Field(
        ..., description="Whether to use data augmentation."
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

    config_name: str = Field(None, description="Name of the experiment.")
    id: str | None = Field(None, description="ID of the experiment.")
    start_time: dt.datetime | str = Field(
        dt.datetime.now(),
        description="Start time of the experiment (initialised to now to also act as created_at).",
    )
    duration: dt.timedelta | str | None = Field(
        None, description="Duration of the experiment."
    )
    dataset: ExperimentDataset = Field(..., description="Dataset configuration.")
    model: ExperimentModel = Field(..., description="Model configuration.")
    training: ExperimentTraining = Field(..., description="Training configuration.")
    fold_results: list[TrainingFold] = Field([], description="Results for each fold.")

    def __init__(self, **data):
        super().__init__(**data)
        self.config_name = f"c{self.dataset.context_window}_{self.dataset.dimensionality.replace('.', '')}"

    def write_experiment_to_json(self, out_dir: str) -> None:
        """Write the experiment to a JSON file."""
        copy = self.model_copy()
        # cast all dates to str for JSON serialisation
        copy.start_time = str(copy.start_time)
        if copy.duration:
            copy.duration = str(copy.duration)
        if copy.fold_results:
            for fold_res in copy.fold_results:
                fold_res.start_time = str(fold_res.start_time)
                if fold_res.duration:
                    fold_res.duration = str(fold_res.duration)
        with open(f"{out_dir}/run_{copy.id}.json", "w") as f:
            json.dump(copy.model_dump(), f)
        del copy


def create_experiment_from_json(
    config_json_path: str = "pipeline_parameters.json",
) -> BaseExperimentConfig:
    """Load the configuration from the pipeline parameters JSON file."""
    with open(config_json_path, "r") as f:
        config = json.load(f)
    return BaseExperimentConfig(**config)


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
    test = create_experiment_from_json("pipeline_parameters.json")
    from project_config import env_config

    test.write_experiment_to_json(env_config.OUT_DIR)

    # config = load_experiment_from_json(
    #     "out/model_runs/c30_25d_1411_1620/run_c30_25d_1411_1620.json"
    # )
    # config.fold_results[0].latest_eval_metrics
