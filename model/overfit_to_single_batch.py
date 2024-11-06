"""
Check the model and see if we can overfit on a single batch of data. This is to check if the model is working as expected.
"""

import torch
import torch.optim as optim
from coral_pytorch.losses import CornLoss
from torch.utils.data import DataLoader

from model.dataset import LIDC_IDRI_DATASET
from model.MEDMnist.ResNet import ResNet50
from project_config import SEED, pipeline_config
from utils.common_imports import *

torch.manual_seed(SEED)
np.random.seed(SEED)

# LOAD SCRIPT PARAMS:
LR = pipeline_config.training.learning_rate
NUM_EPOCHS = pipeline_config.training.num_epochs
IMAGE_DIMS = pipeline_config.dataset.image_dims
NUM_CLASSES = pipeline_config.model.num_classes
IN_CHANNELS = pipeline_config.model.in_channels
CV_FOLDS = pipeline_config.training.cross_validation_folds
BATCH_SIZE = pipeline_config.training.batch_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL:
model = ResNet50(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, dims="3D").to(DEVICE)
criterion = CornLoss(num_classes=NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=LR)

# get a single batch of data:
dataset = LIDC_IDRI_DATASET(img_dim=20, augment_scans=False)
batch_features, batch_labels = next(iter(DataLoader(dataset, batch_size=16)))
batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)

model.train()
iterations = len(dataset)

losses = []

for i in range(iterations):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass and loss calculation
    logits = model(batch_features)
    # class labels should start at 0 according to documentation:
    # https://raschka-research-group.github.io/coral-pytorch/api_subpackages/coral_pytorch.losses/
    loss = criterion(logits, batch_labels - 1)

    # Backward pass and optimisation
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # average loss so far
    print(f"batch loss so far: {sum(losses) / (i + 1)}")

    # plot loss batch loss every 5 iterations
    if i % 5 == 0:
        plt.plot(losses, label="single batch loss")
        plt.title("Single Batch Loss")
        plt.xlabel("Training Iterations")
        plt.ylabel("CORN Loss")
        plt.show()
