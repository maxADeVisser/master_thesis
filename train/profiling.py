import torch
import torch.profiler

"""
Check the model and see if we can overfit on a single batch of data. This is to check if the model is working as expected.
"""

import torch
import torch.optim as optim
from coral_pytorch.losses import CornLoss
from torch.utils.data import DataLoader

from data.dataset import LIDC_IDRI_DATASET
from model.ResNet import ResNet50
from project_config import SEED, pipeline_config
from utils.common_imports import *

torch.manual_seed(SEED)
np.random.seed(SEED)

# LOAD SCRIPT PARAMS:
LR = pipeline_config.training.learning_rate
NUM_EPOCHS = pipeline_config.training.num_epochs
NUM_CLASSES = pipeline_config.model.num_classes
DATA_DIMENSIONALITY = pipeline_config.dataset.dimensionality
IN_CHANNELS = pipeline_config.model.in_channels
CV_FOLDS = pipeline_config.training.cross_validation_folds
BATCH_SIZE = pipeline_config.training.batch_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL:
model = ResNet50(
    in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, dims=DATA_DIMENSIONALITY
).to(DEVICE)
criterion = CornLoss(num_classes=NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=LR)

# get a single batch of data:
dataset = LIDC_IDRI_DATASET(
    context_size=30, n_dims=DATA_DIMENSIONALITY, segmentation_configuration="none"
)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        # torch.profiler.ProfilerActivity.CUDA  # TODO enable
    ],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    with_stack=True,
) as profiler:
    for input, labels in loader:  # Replace with your actual data loader
        output = model(input)  # Forward pass
        loss = criterion(output, labels - 1)
        loss.backward()  # Backward pass
        optimizer.step()
        optimizer.zero_grad()

        # Step the profiler after each iteration
        profiler.step()
