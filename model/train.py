"""Script for training a model"""

import datetime as dt

# import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import CornLoss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from model.dataset import LIDC_IDRI_DATASET
from model.MEDMnist.ResNet import ResNet50, convert_model_to_3d
from project_config import SEED, env_config, pipeline_config
from utils.common_imports import *
from utils.logger_setup import logger

torch.manual_seed(SEED)

# SCRIPT PARAMS:
LR = pipeline_config.training.learning_rate
NUM_EPOCHS = pipeline_config.training.num_epochs
EPOCH_PRINT_INTERVAL = pipeline_config.training.epoch_print_interval
NUM_CLASSES = pipeline_config.model.num_classes
IN_CHANNELS = pipeline_config.model.in_channels
CV_FOLDS = pipeline_config.training.cross_validation_folds
BATCH_SIZE = pipeline_config.training.batch_size


def validate_model(
    model: nn.Module, data_loader: DataLoader, device: Literal["cuda", "cpu"]
) -> None:
    """Validates the model with @data_loader on @device"""
    model.eval()

    # Containers for all predicted and true labels
    all_predicted_labels = []
    all_true_labels = []
    classification_threshold = 0.5
    greater_than_3_idx = 2

    c = 0
    with torch.no_grad():  # No gradient calculation during validation
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            print(f"Processing batch: {batch_idx}/{len(data_loader)}")
            logits = model(inputs)

            # Get predicted rank probabilities:
            rank_probas = torch.sigmoid(logits)
            rank_probas = torch.cumprod(rank_probas, dim=1)

            # Make prediction binary:
            binary_prediction = (
                classification_threshold <= rank_probas[:, greater_than_3_idx]
            )

            all_predicted_labels.extend(binary_prediction.cpu())
            all_true_labels.extend(labels.cpu())
            c += 1

            if c == 10:  # DEBUGGING
                break

    # convert to int
    all_true_labels = [int(x) for x in all_true_labels]
    all_predicted_labels = [int(x) for x in all_predicted_labels]

    return roc_auc_score(y_true=all_true_labels, y_score=all_predicted_labels)


def train_model(experiment_name: str, cv: bool = False) -> None:
    """
    Trains the model.
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
        START TIME: {start_time}
        LR: {LR}
        EPOCHS: {NUM_EPOCHS}
        """
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = convert_model_to_3d(
        ResNet50(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    ).to(device)
    criterion = CornLoss(num_classes=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    dataset = LIDC_IDRI_DATASET()

    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_fold_losses = {}
    for fold, (train_ids, test_ids) in enumerate(
        sgkf.split(
            X=dataset.nodule_df,
            y=dataset.nodule_df["malignancy_consensus"],
            groups=dataset.nodule_df["pid"],
        )
    ):
        logger.info(f"Fold {fold + 1}/{CV_FOLDS}")

        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        train_loader = dataset.get_dataloader(data_sampler=train_subsampler)
        test_loader = dataset.get_dataloader(data_sampler=test_subsampler)
        logger.info(
            f"Train batch size: {len(train_loader)} | Test batch size: {len(test_loader)}"
        )

        all_epoch_losses = []

        # Training loop
        for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc=f"Epoch"):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                print(f"Batch {batch_idx}")
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass, backward pass, and optimize
                logits = model(inputs)
                loss = criterion(logits, labels)

                # Backward pass and optimisation
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            average_epoch_loss = epoch_loss / len(train_loader)
            all_epoch_losses.append(average_epoch_loss)

            if epoch % EPOCH_PRINT_INTERVAL == 0:
                # Print statistics
                logger.info(f"Epoch {epoch} | Average Loss: {average_epoch_loss:.4f}")

                # Evaluate model on test set
                model.eval()
                all_preds = []

        cv_fold_losses[fold] = all_epoch_losses

        if not cv:
            # only train for one fold
            # TODO save model
            break

    # Log results of experiment:
    experiment.end_time = dt.datetime.now()
    experiment.duration = experiment.end_time - experiment.start_time
    # experiment.evaluation.metrics_results = {}  # TODO add more metrics
    experiment.cv_epoch_losses = cv_fold_losses
    experiment.write_experiment_to_json(out_dir=f"{env_config.OUT_DIR}")


if __name__ == "__main__":
    # train_model(experiment_name="Testing training flow", cv=False)

    dataset = LIDC_IDRI_DATASET()
    sgkf = StratifiedGroupKFold(n_splits=30, shuffle=True, random_state=SEED)
    cv_fold_losses = {}
    for fold, (train_ids, test_ids) in enumerate(
        sgkf.split(
            X=dataset.nodule_df,
            y=dataset.nodule_df["malignancy_consensus"],
            groups=dataset.nodule_df["pid"],
        )
    ):
        logger.info(f"Fold {fold + 1}/{CV_FOLDS}")

        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        train_loader = dataset.get_dataloader(data_sampler=train_subsampler)
        test_loader = dataset.get_dataloader(data_sampler=test_subsampler)
        break

    # Testing validation:
    model = convert_model_to_3d(
        ResNet50(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    )
    auc = validate_model(model, test_loader, "cpu")
    auc
