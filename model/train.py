"""Script for training a model"""

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from model.dataset import LIDC_IDRI_DATASET
from model.MEDMnist.ResNet_baseline import ResNet50, convert_model_to_3d
from project_config import SEED, env_config, pipeline_config
from utils.common_imports import *
from utils.logger_setup import logger

LR = 0.001
NUM_EPOCHS = 10
EPOCH_PRINT_INTERVAL = pipeline_config["training"]["epoch_print_interval"]


def main() -> None:
    # Define the loss function and optimizer
    # 1. Load training configuration
    # 2. Create the training files (nodule_df)
    # 3. Instantiate the dataloader
    # 4. Instantiate the model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = convert_model_to_3d(ResNet50(in_channels=1, num_classes=5)).to(device)
    criterion = nn.CrossEntropyLoss()  # Replace with the appropriate loss function
    optimizer = optim.Adam(model.parameters(), lr=LR)

    dataset = LIDC_IDRI_DATASET()
    train_loader = dataset.get_train_loader()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Back pass and optimisation
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (batch_idx + 1) % EPOCH_PRINT_INTERVAL == 0:
                logger.info(
                    f"[{epoch + 1}, {batch_idx + 1}] loss: {running_loss / EPOCH_PRINT_INTERVAL}"
                )
                running_loss = 0.0
        break  # DEBUGGING (only run one epoch for now)


if __name__ == "__main__":
    main()
