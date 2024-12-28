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
from torch.utils.data import DataLoader

from data.dataset import PrecomputedNoduleROIs
from model.ResNet import (
    ResNet50,
    compute_class_probs_from_logits,
    get_pred_malignancy_from_logits,
    predict_binary_from_logits,
)
from project_config import SEED, env_config, pipeline_config
from train.early_stopping import EarlyStopping
from train.metrics import (
    compute_accuracy,
    compute_binary_accuracy,
    compute_cwce,
    compute_errors,
    compute_filtered_AUC,
    compute_mae,
    compute_mse,
    compute_ovr_AUC,
)
from utils.common_imports import *
from utils.data_models import TrainingFold
from utils.logger_setup import logger
from utils.visualisation import plot_loss, plot_val_error_distribution

# Set random seeds for reproducibility:
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# LOAD SCRIPT PARAMS:
CONFIG_NAME = pipeline_config.config_name
LR = pipeline_config.training.learning_rate
NUM_EPOCHS = pipeline_config.training.num_epochs
NUM_CLASSES = pipeline_config.model.num_classes
IN_CHANNELS = pipeline_config.model.in_channels
DATA_DIMENSIONALITY = pipeline_config.dataset.dimensionality
CONTEXT_WINDOW_SIZE = pipeline_config.dataset.context_window
CENTER_MASK_SIZE = pipeline_config.dataset.center_mask_size
DATA_AUGMENTATION = pipeline_config.dataset.data_augmentation
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
    Trains the @model for one epoch.
    Returns the average batch loss for the epoch.
    """
    model.to(DEVICE)  # move model to GPU if available
    model.train()
    running_epoch_loss = 0.0
    n_batches = len(train_loader)

    for inputs, labels, _ in train_loader:
        # move data to GPU (if available):
        inputs, labels = inputs.float().to(DEVICE), labels.int().to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass and loss calculation
        logits = model(inputs)
        # NOTE: class labels are treated as indices, and should therefore start at 0 according to documentation:
        # https://raschka-research-group.github.io/coral-pytorch/api_subpackages/coral_pytorch.losses/
        loss = corn_loss(logits, labels - 1, num_classes=NUM_CLASSES)

        # backward pass and optimisation
        loss.backward()
        optimizer.step()

        running_epoch_loss += loss.item()

    average_epoch_loss = running_epoch_loss / n_batches
    return average_epoch_loss


def evaluate_model(
    model: nn.Module,
    validation_loader: DataLoader,
) -> dict:
    """
    Evaluates the model according to performance metrics using the provided data loader.
    Returns a dictionary of the computed metrics.
    """
    logger.info("Validating model ...")
    model.to(DEVICE)
    model.eval()

    # preallocate tensors for storing predictions and labels (on the CPU):
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
        for inputs, labels, _ in validation_loader:
            # move features and labels to GPU, get logits and compute loss:
            inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE).int()
            logits = model(inputs)
            # (class labels should start at 0 according to corn loss documentation)
            loss = corn_loss(logits, labels - 1, num_classes=NUM_CLASSES)
            running_val_loss += loss.item()

            # move to CPU for metrics calculations:
            logits, labels = logits.cpu().float(), labels.cpu().int()

            # get predictions and labels:
            binary_probas = predict_binary_from_logits(logits, return_probs=True)
            malignancy_scores = get_pred_malignancy_from_logits(logits)
            class_probas = compute_class_probs_from_logits(logits)

            # fill the preallocated tensors with the predictions and labels
            batch_size = inputs.size(0)
            end_idx = start_idx + batch_size
            all_binary_prob_predictions[start_idx:end_idx] = binary_probas
            all_malignancy_predictions[start_idx:end_idx] = malignancy_scores
            all_class_proba_preds[start_idx:end_idx] = class_probas
            all_true_labels[start_idx:end_idx] = labels

            start_idx += batch_size

    # --- COMPUTE EVALUATION METRICS ---
    np_all_true_labels = all_true_labels.numpy()
    np_all_class_proba_preds = all_class_proba_preds.numpy()
    np_all_binary_prob_predictions = all_binary_prob_predictions.numpy()
    val_metrics = {
        "avg_val_loss": running_val_loss / number_of_batches,
        "accuracy": compute_accuracy(all_true_labels, all_malignancy_predictions),
        "binary_accuracy": compute_binary_accuracy(
            np_all_true_labels, np_all_binary_prob_predictions
        ),
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
    Trains the model using the specified configuration in the pipeline_config.

    Params:
        @context_window_size: size of the context window of the nodule ROI used for training.
        @data_dimensionality: whether to use 2.5D or 3D data.
        @cross_validation: whether to train the model using cross-validation.
    """
    # log experiment:
    experiment = pipeline_config.model_copy()
    start_time = dt.datetime.now()
    experiment.start_time = start_time
    experiment.dataset.context_window = context_window_size
    experiment.id = f"{experiment.config_name}_{start_time.strftime('%d%m_%H%M')}"
    experiment.training.gpu_used = torch.cuda.get_device_name(0)

    assert os.path.exists(
        env_config.PREPROCESSED_DATA_DIR
    ), f"Precomputed ROIs do not exist for {CONTEXT_WINDOW_SIZE}C_{DATA_DIMENSIONALITY}"

    # create output directory for experiment:
    exp_out_dir = f"{env_config.OUT_DIR}/model_runs/{experiment.id}"
    if not os.path.exists(exp_out_dir):
        os.makedirs(exp_out_dir)

    experiment.write_experiment_to_json(out_dir=f"{exp_out_dir}")
    nodule_df = pd.read_csv(env_config.processed_nodule_df_file)

    logger.info(
        f"""
        [[--- Training model: {experiment.id} ---]]
        LR: {LR}
        EPOCHS: {NUM_EPOCHS}
        BATCH SIZE: {BATCH_SIZE}
        CONTEXT_WINDOW_SIZE: {context_window_size}
        DATA DIMENSIONALITY: {data_dimensionality}
        DATA AUGMENTATION: {DATA_AUGMENTATION}
        CENTER MASK SIZE: {CENTER_MASK_SIZE}
        DO CROSS VALIDATION: {DO_CROSS_VALIDATION}
        TOTAL NODULES USED FOR TRAINING: {len(nodule_df)}
        CROSS VALIDATION: {cross_validation}
        CV FOLDS: {CV_FOLDS}
        ES PATIENCE: {PATIENCE}
        ES MIN_DELTA: {MIN_DELTA}
        NUM WORKERS: {NUM_WORKERS}

        Experiment output directory: {exp_out_dir}

        Device used: {DEVICE}
        GPU used: {experiment.training.gpu_used}
        """
    )

    # --- cross validation ---
    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idxs, val_idxs) in enumerate(
        sgkf.split(
            X=nodule_df,
            y=nodule_df["malignancy_consensus"],
            groups=nodule_df["scan_id"],
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

        # initialize model and move to GPU if available
        model = ResNet50(
            in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, dims=DATA_DIMENSIONALITY
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        train_dataset = PrecomputedNoduleROIs(
            prepcomputed_dir=env_config.PREPROCESSED_DATA_DIR,
            data_augmentation=DATA_AUGMENTATION,
            indices=train_idxs,
            dimensionality=data_dimensionality,
            center_mask_size=CENTER_MASK_SIZE,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )

        validation_dataset = PrecomputedNoduleROIs(
            prepcomputed_dir=env_config.PREPROCESSED_DATA_DIR,
            data_augmentation=DATA_AUGMENTATION,
            indices=val_idxs,
            dimensionality=data_dimensionality,
            center_mask_size=CENTER_MASK_SIZE,
        )
        val_loader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        early_stopper = EarlyStopping(
            checkpoint_path=f"{fold_out_dir}/model.pth",
            patience=PATIENCE,
            min_delta=MIN_DELTA,
        )

        # --- training loop ---
        for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epoch"):
            avg_epoch_train_loss = train_epoch(model, train_loader, optimizer)
            fold_results.train_losses.append(avg_epoch_train_loss)

            # evaluate model:
            val_metrics = evaluate_model(model, val_loader)
            # (NOTE: checkpointing the model if it improves is handled by the EarlyStopping)
            early_stopper(
                val_loss=val_metrics["avg_val_loss"], epoch=epoch, model=model
            )

            fold_results.best_loss = early_stopper.best_loss
            fold_results.best_loss_epoch = early_stopper.best_loss_epoch
            fold_results.val_losses.append(round(val_metrics["avg_val_loss"], 6))
            fold_results.val_accuracies.append(round(val_metrics["accuracy"], 6))
            fold_results.val_binary_accuracies.append(
                round(val_metrics["binary_accuracy"], 6)
            )
            fold_results.val_AUC_filtered.append(round(val_metrics["AUC_filtered"], 6))
            fold_results.val_AUC_ovr.append(round(val_metrics["AUC_ovr"], 6))
            fold_results.val_maes.append(round(val_metrics["mae"], 6))
            fold_results.val_mses.append(round(val_metrics["mse"], 6))
            fold_results.val_cwces.append(round(val_metrics["cwce"], 6))
            # incrementally write results out to JSON:
            fold_results.write_fold_to_json(out_dir=f"{fold_out_dir}")

            # log epoch results:
            plot_loss(
                fold_results.train_losses, fold_results.val_losses, out_dir=fold_out_dir
            )
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
            # used for debugging or quick model training
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
