"""Script for training a model"""

import datetime as dt

# import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from coral_pytorch.losses import CornLoss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import SubsetRandomSampler

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

        # Training loop
        all_epoch_losses = []
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (inputs, labels) in tqdm(
                enumerate(train_loader), desc=f"Epoch {epoch}/{NUM_EPOCHS}"
            ):
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass, backward pass, and optimize
                logits = model(inputs)
                loss = criterion(logits, labels)

                # Back pass and optimisation
                loss.backward()
                optimizer.step()

                # Print statistics
                epoch_loss += loss.item()

            if epoch % EPOCH_PRINT_INTERVAL == 0:
                logger.info(
                    f"Epoch {epoch}, Average Loss: {sum(epoch_loss)/len(epoch_loss):.4f}"
                )

            all_epoch_losses.append(epoch_loss)

        cv_fold_losses[fold] = all_epoch_losses

        if not cv:
            # onlt train for one fold
            # TODO save model
            break

    # Log results of experiment:
    experiment.end_time = dt.datetime.now()
    experiment.duration = experiment.end_time - experiment.start_time
    # experiment.evaluation.metrics_results = {}  # TODO add more metrics
    experiment.cv_epoch_losses = cv_fold_losses
    experiment.write_experiment_to_json(out_dir=f"{env_config.OUT_DIR}")


if __name__ == "__main__":
    train_model(experiment_name="Testing training flow")
