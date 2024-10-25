"""Script for training a model"""

# %%
import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
from coral_pytorch.losses import CornLoss

# import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from model.dataset import LIDC_IDRI_DATASET
from model.MEDMnist.ResNet import (
    ResNet50,
    convert_model_to_3d,
    predict_binary_from_logits,
    predict_rank_from_logits,
)
from project_config import SEED, env_config, pipeline_config
from utils.common_imports import *
from utils.logger_setup import logger
from utils.metrics import compute_aes

torch.manual_seed(SEED)

# SCRIPT PARAMS:
LR = pipeline_config.training.learning_rate
NUM_EPOCHS = pipeline_config.training.num_epochs
EPOCH_PRINT_INTERVAL = pipeline_config.training.epoch_print_interval
NUM_CLASSES = pipeline_config.model.num_classes
IN_CHANNELS = pipeline_config.model.in_channels
CV_FOLDS = pipeline_config.training.cross_validation_folds
BATCH_SIZE = pipeline_config.training.batch_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    Trains the model for one epoch.
    Returns the average loss for the epoch.
    """
    model.to(DEVICE)
    model.train()
    epoch_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}")  # DEBUGGING
        inputs, labels = inputs.float().to(DEVICE), labels.float().to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass and loss calculation
        logits = model(inputs)
        loss = criterion(logits, labels)

        # Backward pass and optimisation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # TODO Also calculate metrics for training set??
    average_epoch_loss = epoch_loss / len(train_loader)
    return average_epoch_loss


def validate_model(model: nn.Module, validation_loader: DataLoader) -> dict:
    """Validates the model according performance metrics using the provided data loader."""
    print("Validating model ...")
    model.to(DEVICE)
    model.eval()

    all_binary_predictions = []
    all_rank_predictions = []
    all_true_labels = []

    with torch.no_grad():  # No gradient calculation during validation
        for batch_idx, (inputs, labels) in enumerate(validation_loader):
            print(f"Batch: {batch_idx + 1}/{len(validation_loader)}")
            logits = model(inputs)
            logits = logits.cpu()

            # Get binary prediction
            all_binary_predictions.extend(
                predict_binary_from_logits(logits, classification_threshold=0.5)
            )

            # Get rank prediction
            all_rank_predictions.extend(predict_rank_from_logits(logits))

            # Get true labels
            all_true_labels.extend(labels.cpu())

    # convert to int
    all_true_labels = [int(x) for x in all_true_labels]
    all_rank_predictions = [int(x) for x in all_rank_predictions]

    # Filter out ambiguous cases for binary (AUC) evaluation:
    non_ambiguous_idxs = [i for i, label in enumerate(all_true_labels) if label != 3]
    binary_predictions_filtered = [
        int(all_binary_predictions[i]) for i in non_ambiguous_idxs
    ]
    labels_filtered = [all_true_labels[i] for i in non_ambiguous_idxs]
    binary_labels = [1 if label > 3 else 0 for label in labels_filtered]

    # Compute all metrics
    metric_results = {
        "accuracy": accuracy_score(y_true=all_true_labels, y_pred=all_rank_predictions),
        "f1": f1_score(
            y_true=all_true_labels, y_pred=all_rank_predictions, average="weighted"
        ),
        "AUC_filtered": roc_auc_score(
            y_true=binary_labels, y_score=binary_predictions_filtered
        ),
        "AUC_n_samples": len(binary_labels),
        "mae": mean_absolute_error(y_true=all_true_labels, y_pred=all_rank_predictions),
        "aes": compute_aes(y_true=all_true_labels, y_pred=all_rank_predictions),
        "rmse": root_mean_squared_error(
            y_true=all_true_labels, y_pred=all_rank_predictions
        ),
    }
    return metric_results


def train_model(experiment_name: str, cv: bool = False) -> None:
    """
    Trains the model.
    @experiment_name: name of the experiment.
    @cv: whether to train the model using cross-validation.
    """
    # Log experiment
    experiment = pipeline_config.model_copy()
    experiment.name = experiment_name
    start_time = dt.datetime.now()
    experiment.start_time = start_time
    logger.info(
        f"""
        Running experiment '{experiment.name}'
        LR: {LR}
        EPOCHS: {NUM_EPOCHS}
        BATCH_SIZE: {BATCH_SIZE}
        """
    )
    experiment.id = f"{experiment.name}_{start_time.strftime('%d%m_%H%M')}"
    exp_out_dir = f"{env_config.OUT_DIR}/model_runs/{experiment.id}"

    if not os.path.exists(exp_out_dir):
        os.makedirs(exp_out_dir)

    dataset = LIDC_IDRI_DATASET()

    # --- Cross Validation ---
    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_results = {}
    for fold, (train_ids, test_ids) in enumerate(
        sgkf.split(
            X=dataset.nodule_df,
            y=dataset.nodule_df["malignancy_consensus"],
            groups=dataset.nodule_df["pid"],
        )
    ):
        model = convert_model_to_3d(
            ResNet50(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
        ).to(DEVICE)
        criterion = CornLoss(num_classes=NUM_CLASSES)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        logger.info(f"Fold {fold + 1}/{CV_FOLDS}")
        cv_fold_info = {}

        # Define train and test loaders
        train_subsampler = SubsetRandomSampler(indices=train_ids)
        train_loader = dataset.get_dataloader(data_sampler=train_subsampler)
        test_subsampler = SubsetRandomSampler(indices=test_ids)
        test_loader = dataset.get_dataloader(data_sampler=test_subsampler)
        logger.info(
            f"Train batch size: {len(train_loader)} | Test batch size: {len(test_loader)}"
        )

        avg_fold_epoch_losses = []

        # --- Training Loop ---
        for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc=f"Epoch"):
            average_epoch_loss = train_epoch(model, train_loader, optimizer, criterion)
            avg_fold_epoch_losses.append(average_epoch_loss)

            if epoch % EPOCH_PRINT_INTERVAL == 0:
                # Print training info ...
                metrics = validate_model(model, test_loader)
                logger.info(
                    f"Epoch {epoch} | Average Loss: {average_epoch_loss:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['AUC_filtered']:.4f}"
                )

        # Save fold model
        torch.save(model.state_dict(), f"{exp_out_dir}/model_fold_{fold}.pth")

        # Save fold results
        cv_fold_info["cv_epoch_losses"] = avg_fold_epoch_losses
        cv_fold_info["eval_metrics"] = metrics
        cv_results[fold] = cv_fold_info

        if not cv:
            # only train for one fold (do not do CV)
            break

    # Log results of experiment:
    experiment.end_time = dt.datetime.now()
    experiment.duration = experiment.end_time - experiment.start_time
    experiment.results = cv_results
    experiment.write_experiment_to_json(out_dir=f"{env_config.OUT_DIR}/model_runs")


# %%
if __name__ == "__main__":
    train_model(experiment_name="testing_training_flow", cv=False)

    # dataset = LIDC_IDRI_DATASET()
    # sgkf = StratifiedGroupKFold(n_splits=30, shuffle=True, random_state=SEED)
    # cv_fold_losses = {}
    # for fold, (train_ids, test_ids) in enumerate(
    #     sgkf.split(
    #         X=dataset.nodule_df,
    #         y=dataset.nodule_df["malignancy_consensus"],
    #         groups=dataset.nodule_df["pid"],
    #     )
    # ):
    #     logger.info(f"Fold {fold + 1}/{CV_FOLDS}")

    #     train_subsampler = SubsetRandomSampler(train_ids)
    #     test_subsampler = SubsetRandomSampler(test_ids)
    #     train_loader = dataset.get_dataloader(data_sampler=train_subsampler)
    #     test_loader = dataset.get_dataloader(data_sampler=test_subsampler)
    #     break

    # # Testing validation:
    # model = convert_model_to_3d(
    #     ResNet50(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    # )
    # metrics = validate_model(model, test_loader)
    # metrics

    # plt.hist(metrics["aes"], bins=5)
    # plt.title("Absolute Errors")
