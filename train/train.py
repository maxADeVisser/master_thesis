"""Script for training a model"""

# %%
import os
import sys

from dotenv import load_dotenv

load_dotenv(".env")
sys.path.append(os.getenv("PROJECT_DIR"))

import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
from coral_pytorch.losses import corn_loss
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from data.dataset import PrecomputedNoduleROIs
from model.ResNet import (
    ResNet50,
    compute_class_probs_from_logits,
    get_pred_malignancy_score_from_logits,
    predict_binary_from_logits,
)
from project_config import SEED, env_config, pipeline_config
from utils.common_imports import *
from utils.early_stopping import EarlyStopping
from utils.experiments import TrainingFold
from utils.logger_setup import logger
from utils.metrics import (
    compute_accuracy,
    compute_cwce,
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
CONFIG_NAME = pipeline_config.config_name
LR = pipeline_config.training.learning_rate
NUM_EPOCHS = pipeline_config.training.num_epochs
NUM_CLASSES = pipeline_config.model.num_classes
IN_CHANNELS = pipeline_config.model.in_channels
DATA_DIMENSIONALITY = pipeline_config.dataset.dimensionality
CONTEXT_WINDOW_SIZE = pipeline_config.dataset.context_window
DO_CROSS_VALIDATION = pipeline_config.training.do_cross_validation
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
) -> float:
    """
    Trains the model for one epoch.
    Returns the average batch loss for the epoch.
    """
    model.to(DEVICE)  # move model to GPU
    model.train()
    running_epoch_loss = 0.0
    n_batches = len(train_loader)

    for inputs, labels in tqdm(train_loader, desc="Batches"):
        # Move data to GPU (if available):
        inputs, labels = inputs.float().to(DEVICE), labels.int().to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass and loss calculation
        logits = model(inputs)
        # class labels should start at 0 according to documentation:
        # https://raschka-research-group.github.io/coral-pytorch/api_subpackages/coral_pytorch.losses/
        loss = corn_loss(logits, torch.squeeze(labels) - 1, num_classes=NUM_CLASSES)

        # Backward pass and optimisation
        loss.backward()
        optimizer.step()

        running_epoch_loss += loss.item()

    average_epoch_loss = running_epoch_loss / n_batches
    return average_epoch_loss


def evaluate_model(
    model: nn.Module,
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
        for inputs, labels in validation_loader:
            # move features and labels to GPU, get logits and compute loss:
            inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE).int()
            logits = model(inputs)
            # (class labels should start at 0 according to corn loss documentation)
            loss = corn_loss(logits, labels - 1, num_classes=NUM_CLASSES)
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

    # --- COMPUTE METRICS ---
    np_all_true_labels = all_true_labels.numpy()
    np_all_class_proba_preds = all_class_proba_preds.numpy()
    np_all_binary_prob_predictions = all_binary_prob_predictions.numpy()
    val_metrics = {
        "avg_val_loss": running_val_loss / number_of_batches,
        "accuracy": compute_accuracy(all_true_labels, all_malignancy_predictions),
        "AUC_ovr": compute_ovr_AUC(np_all_true_labels, np_all_class_proba_preds),
        "AUC_filtered": compute_filtered_AUC(
            np_all_true_labels, np_all_binary_prob_predictions
        ),
        "errors": compute_errors(all_true_labels, all_malignancy_predictions),
        "mae": compute_mae(all_true_labels, all_malignancy_predictions),
        "mse": compute_mse(all_true_labels, all_malignancy_predictions),
        "cwce": compute_cwce(all_true_labels, all_class_proba_preds, M=10),
    }
    return val_metrics


def train_model(
    context_window_size: int = CONTEXT_WINDOW_SIZE,
    data_dimensionality: Literal["2.5D", "3D"] = DATA_DIMENSIONALITY,
    cross_validation: bool = DO_CROSS_VALIDATION,
) -> None:
    """
    Trains the model.

    Params
    ---
        @context_window_size: size of the context window of the nodule ROI used for training.
        @data_dimensionality: whether to use 2.5D or 3D data.
        @cross_validation: whether to train the model using cross-validation.
    """
    # Log experiment:
    experiment = pipeline_config.model_copy()
    start_time = dt.datetime.now()
    experiment.start_time = start_time
    experiment.dataset.context_window = context_window_size
    experiment.id = f"{experiment.config_name}_{start_time.strftime('%d%m_%H%M')}"
    experiment.training.gpu_used = torch.cuda.get_device_name(0)

    # Create output directory for experiment:
    exp_out_dir = f"{env_config.OUT_DIR}/model_runs/{experiment.id}"
    if not os.path.exists(exp_out_dir):
        os.makedirs(exp_out_dir)
    experiment.write_experiment_to_json(out_dir=f"{exp_out_dir}")

    logger.info(
        f"""
        [[--- Training model: {experiment.config_name} ---]]
        LR: {LR}
        EPOCHS: {NUM_EPOCHS}
        BATCH_SIZE: {BATCH_SIZE}
        CONTEXT_WINDOW_SIZE: {context_window_size}
        DATA DIMENSIONALITY: {data_dimensionality}
        DO_CROSS_VALIDATION: {DO_CROSS_VALIDATION}
        CROSS_VALIDATION: {cross_validation}
        CV_FOLDS: {CV_FOLDS}
        ES_PATIENCE: {PATIENCE}
        ES_MIN_DELTA: {MIN_DELTA}
        NUM_WORKERS: {NUM_WORKERS}

        Output directory: {exp_out_dir}

        GPU: {torch.cuda.get_device_name(0)}
        Device used: {DEVICE}
        """
    )

    preprocessed_data_dir = f"{env_config.PROJECT_DIR}/data/precomputed_rois_{CONTEXT_WINDOW_SIZE}C_{DATA_DIMENSIONALITY}"
    assert os.path.exists(
        preprocessed_data_dir
    ), f"Precomputed ROIs do not exist for {CONTEXT_WINDOW_SIZE}C_{DATA_DIMENSIONALITY}"
    dataset = PrecomputedNoduleROIs(preprocessed_dir=preprocessed_data_dir)
    nodule_df = pd.read_csv(env_config.processed_nodule_df_file)

    # --- Cross Validation ---
    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idxs, val_idxs) in enumerate(
        sgkf.split(
            X=nodule_df,
            y=nodule_df["malignancy_consensus"],
            groups=nodule_df["pid"],
        )
    ):
        logger.info(
            f"""
            [[Starting Fold {fold + 1}/{CV_FOLDS}]]
            Train instances: {len(train_idxs)}
            Validation instances: {len(val_idxs)}
            """
        )

        fold_start_time = dt.datetime.now()
        fold_out_dir = f"{exp_out_dir}/fold{fold}"
        if not os.path.exists(fold_out_dir):
            os.makedirs(fold_out_dir)

        fold_results = TrainingFold(
            fold_id=f"fold{fold}_{experiment.id}",
            train_idxs=train_idxs.tolist(),
            val_idxs=val_idxs.tolist(),
            start_time=fold_start_time,
        )
        # save initial fold information
        fold_results.write_fold_to_json(out_dir=f"{fold_out_dir}")

        # Initialize model and move to GPU (if available)
        model = ResNet50(
            in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, dims=DATA_DIMENSIONALITY
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        train_subset = Subset(dataset, indices=train_idxs)
        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_subset = Subset(dataset, indices=val_idxs)
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        early_stopper = EarlyStopping(
            checkpoint_path=f"{fold_out_dir}/model.pth",
            patience=PATIENCE,
            min_delta=MIN_DELTA,
        )

        # --- Training Loop ---
        for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epoch"):
            if epoch == 50 or epoch == 75:
                # decreasing learning rate at specified epochs:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.1
                logger.info(
                    f"Learning rate adjusted to: {optimizer.param_groups[0]['lr']}"
                )

            avg_epoch_train_loss = train_epoch(model, train_loader, optimizer)
            fold_results.train_losses.append(avg_epoch_train_loss)

            # Evaluate model:
            val_metrics = evaluate_model(model, val_loader)
            # (NOTE: Checkpointing the model if it improves is handled by the EarlyStopping)
            early_stopper(val_loss=val_metrics["avg_val_loss"], model=model)

            fold_results.best_loss = early_stopper.best_loss
            fold_results.val_losses.append(round(val_metrics["avg_val_loss"], 4))
            fold_results.val_accuracies.append(round(val_metrics["accuracy"], 4))
            fold_results.val_AUC_filtered.append(round(val_metrics["AUC_filtered"], 4))
            fold_results.val_AUC_ovr.append(round(val_metrics["AUC_ovr"], 4))
            fold_results.val_maes.append(round(val_metrics["mae"], 4))
            fold_results.val_mses.append(round(val_metrics["mse"], 4))
            fold_results.val_cwces.append(round(val_metrics["cwce"], 4))
            # Write incremental results out to JSON:
            fold_results.write_fold_to_json(out_dir=f"{fold_out_dir}")

            # Log epoch results:
            plot_loss(
                fold_results.train_losses, fold_results.val_losses, out_dir=fold_out_dir
            )
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
                Val CWCE: {val_metrics['cwce']:.4f}
                """
            )

            if early_stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                fold_results.epoch_stopped = epoch
                break

        fold_results.duration = dt.datetime.now() - fold_start_time
        fold_results.write_fold_to_json(out_dir=f"{fold_out_dir}")

        experiment.fold_results.append(fold_results)

        if not cross_validation:
            # do not do cross-validation (train on one fold only)
            break

        if fold + 1 == CV_TRAIN_FOLDS:
            # train on a specified subset of the folds only
            break

    # Log results of experiment:
    experiment.duration = dt.datetime.now() - experiment.start_time
    experiment.write_experiment_to_json(out_dir=f"{exp_out_dir}")


# %%
if __name__ == "__main__":
    train_model(
        context_window_size=CONTEXT_WINDOW_SIZE,
        cross_validation=DO_CROSS_VALIDATION,
        data_dimensionality=DATA_DIMENSIONALITY,
    )
