"""Script for precomputing the nodule ROIs and their corresponding labels for the LIDC-IDRI dataset."""

# %%
import os
import sys

from dotenv import load_dotenv

load_dotenv(".env")
sys.path.append(os.getenv("PROJECT_DIR"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import LIDC_IDRI_DATASET
from utils.logger_setup import logger


def main() -> None:
    # SCRIPT PARAMS
    CONTEXT_WINDOW_SIZE = 50
    DIMENSIONALITY = "2.5D"

    # Constants
    batch_size = 1
    PROJECT_DIR = os.getenv("PROJECT_DIR")
    OUT_DIR = (
        f"{PROJECT_DIR}/data/precomputed_rois_{CONTEXT_WINDOW_SIZE}C_{DIMENSIONALITY}"
    )

    dataset = LIDC_IDRI_DATASET(context_size=CONTEXT_WINDOW_SIZE, n_dims=DIMENSIONALITY)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if os.path.exists(OUT_DIR):
        logger.info(
            f"Precomputed ROIs for {CONTEXT_WINDOW_SIZE}C_{DIMENSIONALITY} already exist."
        )
        return None
    else:
        logger.info(
            f"Precomputed ROIs for {CONTEXT_WINDOW_SIZE}C_{DIMENSIONALITY} does not exists."
        )
        os.makedirs(OUT_DIR)
        for i, (roi, label) in tqdm(
            enumerate(loader), desc="Precomputing ROIs", total=len(loader)
        ):
            torch.save(
                (roi[0], label),
                f"{OUT_DIR}/instance{i}.pt",
            )


# %%
if __name__ == "__main__":
    main()

    # t = torch.load(
    #     "/Users/newuser/Documents/ITU/master_thesis/data/precomputed_rois_50C_2.5D/instance0.pt"
    # )
    # t[0].shape
    # t[1].shape
