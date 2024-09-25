# %%
import ast

import numpy as np
import pandas as pd
import pylidc as pl
import torch
from pylidc.utils import volume_viewer
from torch.utils.data import Dataset
from torchvision import transforms

from project_config import config
from utils.common_imports import pipeline_config
from utils.utils import get_scans_by_patient_id


class LIDC_IDRI_DATASET(Dataset):
    """
    Custom dataset for the LIDC-IDRI dataset containing nodule ROIs and their cancer labels.
    - NOTE: Relies on having the nodule dataframe created by create_nodule_df.py script
    """

    def __init__(self, main_dir_path: str = config.DATA_DIR) -> None:
        self.main_dir_path = main_dir_path
        self.patient_ids = config.patient_ids

        try:
            # Read in the nodule dataframe and convert the string representations to python objects
            self.nodule_df = pd.read_csv(
                f"out/{pipeline_config['preprocessing']['nodule_df']['nodule_df_csv_name']}.csv"
            ).assign(
                consensus_bbox=lambda x: x["consensus_bbox"].apply(ast.literal_eval),
                nodule_annotation_ids=lambda x: x["nodule_annotation_ids"].apply(
                    ast.literal_eval
                ),
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "The nodule dataframe was not found. Please run the create_nodule_df.py script first."
            )

        # self.transform = transforms.Compose(transforms.ToTensor())

    def __len__(self) -> int:
        return len(self.nodule_df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieves a single nodule roi in size 64x64x64 and its cancer label"""
        # TODO standardise the size of the nodule roi to 64x64x64
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

    def visualise_nodule_bbox(self, nodule_idx: int, annotation_idx: int = 0) -> None:
        """
        - Interactivelt visualise the consensus bbox of @nodule_idx (idxd in @self.nodule_df) as created by the @create_nodule_df.py script.
        - Also show the segmentation mask of the nodule as defined by @annotation_idx.
        NOTE: this function needs to be run from the console in order to work interactively
        """
        annotation_id = self.nodule_df.iloc[nodule_idx]["nodule_annotation_ids"][
            annotation_idx
        ]
        ann = pl.query(pl.Annotation).filter(pl.Annotation.id == annotation_id).first()
        x, y, z = self.nodule_df.iloc[nodule_idx]["consensus_bbox"]

        # NOTE: pad is set to a large number to include the entire scan:
        ann_mask = ann.boolean_mask(pad=100_000)[x[0] : x[1], y[0] : y[1], z[0] : z[1]]
        ann_cutout = ann.scan.to_volume(verbose=False)[
            x[0] : x[1], y[0] : y[1], z[0] : z[1]
        ]

        # Uncomment the following to visualise the nodule bbox and mask in jupyter notebook
        # %matplotlib widget
        volume_viewer(
            vol=ann_cutout,
            mask=ann_mask,
            ls="--",
            c="r",
        )


# %%
if __name__ == "__main__":
    dataset = LIDC_IDRI_DATASET()
    # pd.cut(dataset.nodule_df["consensus_diameter"], bins=[0, 3, 100]).value_counts()

    roi_consensus, label = dataset.__getitem__(0)
    roi_consensus.shape

    # dataset.nodule_df.query("pid == 'LIDC-IDRI-0101'")

    dataset.visualise_nodule_bbox(nodule_idx=0, annotation_idx=0)
    # dataset.visualise_nodule_bbox(nodule_idx=292, annotation_idx=0)

# %%
