"""Precompute the nodule ROIs and their corresponding labels for the LIDC-IDRI dataset."""

import os

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import LIDC_IDRI_DATASET

load_dotenv()

# SCRIPT PARAMS
CONTEXT_WINDOW_SIZE = 70
DIMENSIONALITY = "3D"
PROJECT_DIR = os.getenv("PROJECT_DIR")
OUT_DIR = f"{PROJECT_DIR}/data/precomputed_rois_{CONTEXT_WINDOW_SIZE}C_{DIMENSIONALITY}"

# Constants
batch_size = 1

dataset = LIDC_IDRI_DATASET(context_size=CONTEXT_WINDOW_SIZE, n_dims=DIMENSIONALITY)
N = len(dataset)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

if os.path.exists(OUT_DIR):
    raise FileExistsError(
        f"Output directory {OUT_DIR} already exists. Please remove it before running this script."
    )
else:
    os.makedirs(OUT_DIR)

for i, instance in tqdm(enumerate(loader), desc="Precomputing ROIs", total=N):
    torch.save(
        instance,
        f"{OUT_DIR}/instance{i}.pt",
    )
