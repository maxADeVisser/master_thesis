"""Script for training a model"""

# %%
import os
import sys

from dotenv import load_dotenv

load_dotenv(".env")
sys.path.append(os.getenv("PROJECT_DIR"))

import datetime as dt
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from coral_pytorch.losses import CornLoss
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from model.dataset import LIDC_IDRI_DATASET
from model.MEDMnist.ResNet import (
    ResNet50,
    compute_class_probs_from_logits,
    get_pred_malignancy_score_from_logits,
    predict_binary_from_logits,
)
from project_config import SEED, env_config, pipeline_config
from utils.common_imports import *
from utils.early_stopping import EarlyStopping
from utils.logger_setup import logger
from utils.metrics import (
    compute_accuracy,
    compute_errors,
    compute_filtered_AUC,
    compute_mae,
    compute_mse,
    compute_ovr_AUC,
)
from utils.visualisation import plot_loss, plot_val_error_distribution

torch.manual_seed(SEED)
np.random.seed(SEED)

# LOAD SCRIPT PARAMS:
LR = pipeline_config.training.learning_rate
NUM_EPOCHS = pipeline_config.training.num_epochs
IMAGE_DIMS = pipeline_config.dataset.image_dims
NUM_CLASSES = pipeline_config.model.num_classes
IN_CHANNELS = pipeline_config.model.in_channels
CV_FOLDS = pipeline_config.training.cross_validation_folds
CV_TRAIN_FOLDS = pipeline_config.training.cv_train_folds
BATCH_SIZE = pipeline_config.training.batch_size
NUM_WORKERS = pipeline_config.training.num_workers
PATIENCE = pipeline_config.training.patience
MIN_DELTA = pipeline_config.training.min_delta
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    Trains the model for one epoch.
    Returns the average batch loss for the epoch.
    """
    model.to(DEVICE)  # move model to GPU
    model.train()
    running_epoch_loss = 0.0
    n_batches = len(train_loader)

    for inputs, labels in tqdm(train_loader, desc="Epoch Batches"):
        # Move data to GPU (if available):
        inputs, labels = inputs.float().to(DEVICE), labels.int().to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass and loss calculation
        logits = model(inputs)
        # class labels should start at 0 according to documentation:
        # https://raschka-research-group.github.io/coral-pytorch/api_subpackages/coral_pytorch.losses/
        loss = criterion(logits, labels - 1)

        # Backward pass and optimisation
        loss.backward()
        optimizer.step()

        running_epoch_loss += loss.item()

    average_epoch_loss = running_epoch_loss / n_batches
    return average_epoch_loss


def validate_model(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    validation_loader: DataLoader,
) -> dict:
    """Validates the model according performance metrics using the provided data loader."""
    logger.info("Validating model ...")
    model.to(DEVICE)
    model.eval()

    # Preallocate tensors for storing predictions and labels (on the CPU):
    num_val_samples = len(validation_loader.dataset)
    all_true_labels = torch.empty(num_val_samples, dtype=torch.int, device="cpu")
    all_binary_prob_predictions = torch.empty(
        num_val_samples, dtype=torch.float, device="cpu"
    )
    all_malignancy_predictions = torch.empty(
        num_val_samples, dtype=torch.int, device="cpu"
    )
    all_class_proba_preds = torch.empty(
        num_val_samples, NUM_CLASSES, dtype=torch.float, device="cpu"
    )

    start_idx = 0
    running_val_loss = 0.0
    number_of_batches = len(validation_loader)

    # (no gradient calculations needed during validation)
    with torch.no_grad():
        for inputs, labels in tqdm(validation_loader, "Validation Batches"):
            # move features and labels to GPU, get logits and compute loss:
            inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE).int()
            logits = model(inputs)
            # (class labels should start at 0 according to corn loss documentation)
            loss = criterion(logits, labels - 1)
            running_val_loss += loss.item()

            # move to CPU for metrics calculations:
            logits, labels = logits.cpu().float(), labels.cpu().int()

            # Get predictions and labels:
            binary_probas = predict_binary_from_logits(logits, return_probs=True)
            malignancy_scores = get_pred_malignancy_score_from_logits(logits)
            class_probas = compute_class_probs_from_logits(logits)

            # Fill the preallocated tensors with the predictions and labels
            batch_size = inputs.size(0)
            end_idx = start_idx + batch_size
            all_binary_prob_predictions[start_idx:end_idx] = binary_probas
            all_malignancy_predictions[start_idx:end_idx] = malignancy_scores
            all_class_proba_preds[start_idx:end_idx] = class_probas
            all_true_labels[start_idx:end_idx] = labels

            start_idx += batch_size

    # convert to numpy:
    all_true_labels = all_true_labels.numpy()
    all_binary_prob_predictions = all_binary_prob_predictions.numpy()
    all_malignancy_predictions = all_malignancy_predictions.numpy()
    all_class_proba_preds = all_class_proba_preds.numpy()

    # --- COMPUTE METRICS ---
    errors = compute_errors(all_true_labels, all_malignancy_predictions)
    val_metrics = {
        "avg_val_loss": running_val_loss / number_of_batches,
        "accuracy": compute_accuracy(all_true_labels, all_malignancy_predictions),
        "AUC_ovr": compute_ovr_AUC(all_true_labels, all_class_proba_preds),
        "AUC_filtered": compute_filtered_AUC(
            all_true_labels, all_binary_prob_predictions
        ),
        "errors": errors,
        "mae": compute_mae(errors),
        "mse": compute_mse(all_true_labels, all_malignancy_predictions),
    }
    return val_metrics


def train_model(
    model_name: str, context_window_size: int, cross_validation: bool = False
) -> None:
    """
    Trains the model.

    Params
    ---
        @model_name: name of the model trained.
        @context_window_size: size of the context window of the nodule ROI used for training.
        @cv: whether to train the model using cross-validation.
    """
    # Log experiment:
    experiment = pipeline_config.model_copy()
    start_time = dt.datetime.now()
    experiment.start_time = start_time
    experiment.name = model_name
    experiment.training.context_window_size = context_window_size
    experiment.id = f"{experiment.name}_{start_time.strftime('%d%m_%H%M')}"

    # Create output directory for experiment:
    exp_out_dir = f"{env_config.OUT_DIR}/model_runs/{experiment.id}"
    if not os.path.exists(exp_out_dir):
        os.makedirs(exp_out_dir)

    logger.info(
        f"""
        [[--- Training model: {experiment.name} ---]]
        LR: {LR}
        EPOCHS: {NUM_EPOCHS}
        BATCH_SIZE: {BATCH_SIZE}
        CONTEXT_WINDOW_SIZE: {context_window_size}
        CROSS_VALIDATION: {cross_validation}
        ES_PATIENCE: {PATIENCE}
        ES_MIN_DELTA: {MIN_DELTA}
        NUM_WORKERS: {NUM_WORKERS}

        Output directory: {exp_out_dir}

        Device used: {DEVICE}
        GPU: {torch.cuda.get_device_name(0)}
        """
    )

    dataset = LIDC_IDRI_DATASET(
        img_dim=context_window_size,
        segmentation_configuration="none",
    )

    # --- Cross Validation ---
    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_results = {}
    for fold, (train_ids, val_ids) in enumerate(
        sgkf.split(
            X=dataset.nodule_df,
            y=dataset.nodule_df["malignancy_consensus"],
            groups=dataset.nodule_df["pid"],
        )
    ):
        logger.info(f"\nStarting Fold {fold + 1}/{CV_FOLDS}")
        fold_start_time = dt.datetime.now()
        fold_out_dir = f"{exp_out_dir}_fold{fold}"
        if not os.path.exists(fold_out_dir):
            os.makedirs(fold_out_dir)

        # Initialize model and move to GPU (if available)
        model = ResNet50(
            in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, dims="3D"
        ).to(DEVICE)
        criterion = CornLoss(num_classes=NUM_CLASSES)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        train_subset = Subset(dataset, indices=train_ids)
        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_subset = Subset(dataset, indices=val_ids)
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        early_stopper = EarlyStopping(
            checkpoint_path=f"{fold_out_dir}/model_fold{fold}.pth",
            patience=PATIENCE,
            min_delta=MIN_DELTA,
        )

        # --- Training Loop ---
        avg_epoch_losses = []
        val_losses = []
        for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epoch"):
            if epoch == 50 or epoch == 75:
                # decreasing learning rate at specified epochs:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.1
                logger.info(
                    f"Learning rate adjusted to: {optimizer.param_groups[0]['lr']}"
                )

            avg_epoch_train_loss = train_epoch(
                model, train_loader, optimizer, criterion
            )
            avg_epoch_losses.append(avg_epoch_train_loss)

            # Validate model:
            val_metrics = validate_model(model, criterion, val_loader)
            val_losses.append(val_metrics["avg_val_loss"])

            # Log epoch results:
            plot_loss(avg_epoch_losses, val_losses, out_dir=fold_out_dir)
            plot_val_error_distribution(val_metrics["errors"], out_dir=fold_out_dir)
            logger.info(
                f"""
                [[Fold {fold + 1}/{CV_FOLDS}]] - [Epoch {epoch}]
                Average Training Loss: {avg_epoch_train_loss:.4f}
                Average Validation Loss: {val_metrics['avg_val_loss']:.4f}
                Val Accuracy: {val_metrics['accuracy']:.4f}
                Val AUC_filtered: {val_metrics['AUC_filtered']:.4f}
                Val AUC_ovr: {val_metrics['AUC_ovr']:.4f}
                Val MAE: {np.mean(np.abs(val_metrics['errors'])):.4f}
                Val MSE: {val_metrics['mse']:.4f}
                """
            )

            # (NOTE: Checkpointing the model if it improves is handled by the EarlyStopping)
            early_stopper(val_loss=val_metrics["avg_val_loss"], model=model)
            if early_stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        fold_end_time = dt.datetime.now()
        fold_duration_time = fold_end_time - fold_start_time

        # Store fold results:
        fold_results = {}
        fold_results["fold_duration_seconds"] = fold_duration_time.total_seconds()
        fold_results["avg_epoch_train_losses"] = avg_epoch_losses
        fold_results["avg_epoch_val_losses"] = val_losses
        fold_results["latest_eval_metrics"] = val_metrics
        fold_results["best_loss"] = early_stopper.best_loss
        fold_results["epoch_stopped"] = epoch
        fold_results["train_ids"] = train_ids.tolist()
        fold_results["val_ids"] = val_ids.tolist()

        # Write fold results to JSON:
        with open(f"{fold_out_dir}/fold_results.json", "w") as f:
            json.dump(fold_results, f)

        # Store fold results in cv_results:
        cv_results[fold + 1] = fold_results

        if not cross_validation:
            # do not do cross-validation (train on one fold only)
            break

        if fold + 1 == CV_TRAIN_FOLDS:
            # train on a specified subset of the folds only
            break

    # Log results of experiment:
    experiment.end_time = dt.datetime.now()
    experiment.duration = experiment.end_time - experiment.start_time
    # cast datetimes to strings for JSON serialization:
    experiment.start_time = str(experiment.start_time)
    experiment.end_time = str(experiment.end_time)
    experiment.duration = str(experiment.duration)
    experiment.results = cv_results
    experiment.write_experiment_to_json(out_dir=f"{exp_out_dir}")


# %%
if __name__ == "__main__":
    model_name = "c40"
    context_window_size = 40
    cross_validation = False
    train_model(
        model_name=model_name,
        context_window_size=context_window_size,
        cross_validation=cross_validation,
    )
