# %%
import ast
import os

import numpy as np
import pandas as pd
import pylidc as pl
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from project_config import config
from utils.pylidc_utils import get_scans_by_patient_id


class LIDC_IDRI_DATASET(Dataset):
    def __init__(self, main_dir_path: str = config.DATA_DIR) -> None:
        self.main_dir_path = main_dir_path
        self.patient_ids = config.patient_ids

        assert os.path.exists(
            "out/nodule_df.csv"
        ), "Nodule dataframe not found. Please run create_nodule_df.py script to create"

        self.nodule_df = pd.read_csv("out/nodule_df.csv").assign(
            consensus_bbox=lambda x: x["consensus_bbox"].apply(ast.literal_eval)
        )

        self.transform = transforms.Compose(transforms.ToTensor())

    def __len__(self) -> int:
        return len(self.nodule_df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieves a single nodule roi in size 64x64x64 and its cancer label"""
        # TODO standardise the size of the nodule roi
        nodule_row = self.nodule_df.iloc[idx]
        scan: np.ndarray = get_scans_by_patient_id(nodule_row["pid"], to_numpy=True)
        x_bounds, y_bounds, z_bounds = nodule_row["consensus_bbox"]
        nodule_roi = scan[
            x_bounds[0] : x_bounds[1],
            y_bounds[0] : y_bounds[1],
            z_bounds[0] : z_bounds[1],
        ]

        # scan = self.transform(scan)
        cancer_label = nodule_row["cancer_label"]

        return nodule_roi, cancer_label


# %%
if __name__ == "__main__":
    dataset = LIDC_IDRI_DATASET()
    dataset.nodule_df

    idx = 700
    roi, label = dataset.__getitem__(idx)
    import matplotlib.pyplot as plt

    plt.imshow(roi[:, :, 0], cmap="gray")
    plt.title(f"Label: {label}")
    plt.show()
