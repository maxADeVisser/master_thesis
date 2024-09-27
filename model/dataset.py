# %%
import ast
from typing import Literal

import numpy as np
import pandas as pd
import pylidc as pl
import torch
from pylidc.utils import consensus, volume_viewer
from torch.utils.data import Dataset

from model.processing import clip_and_normalise_volume
from project_config import config, pipeline_config
from utils.logger_setup import logger
from utils.utils import get_scans_by_patient_id

dataset_config = pipeline_config["preprocessing"]["nodule_dataset"]
nodule_segmentation_config = dataset_config["segment_nodule"]
logger.info(f"Dataset config: {dataset_config}")


class Nodule:
    """
    Helper class to store and manipulate nodule information from nodule_df.
    @segmentation_setting is either 0, 1 or 2. 0 means no segmentation, 1 means segment the nodule, 2 means segment the background
    """

    def __init__(
        self,
        nodule_record: pd.Series,
        segmentation_setting: (
            Literal["none", "remove_nodule", "remove_background"] | None
        ) = None,
    ) -> None:
        self.nodule_record = nodule_record
        self.annotation_ids = nodule_record["nodule_annotation_ids"]
        self.pylidc_annotations = [
            pl.query(pl.Annotation).filter(pl.Annotation.id == ann_id).first()
            for ann_id in self.annotation_ids
        ]
        self.malignancy_consensus = nodule_record["malignancy_consensus"]
        self.nodule_consensus_bbox = nodule_record["consensus_bbox"]
        self.nodule_roi = self.get_nodule_roi()

        self.nodule_roi = clip_and_normalise_volume(self.nodule_roi)

        if segmentation_setting == "remove_background":
            self.nodule_roi = self.segment_nodule(invert=False)
        elif segmentation_setting == "remove_nodule":
            self.nodule_roi = self.segment_nodule(invert=True)

    def get_nodule_roi(self) -> torch.Tensor:
        """Returns the nodule region of interest from the scan based on the consensus bbox from the 4 annotators"""
        scan: np.ndarray = get_scans_by_patient_id(
            self.nodule_record["pid"], to_numpy=True
        )
        x_bounds, y_bounds, z_bounds = self.nodule_record["consensus_bbox"]
        nodule_roi = scan[
            x_bounds[0] : x_bounds[1],
            y_bounds[0] : y_bounds[1],
            z_bounds[0] : z_bounds[1],
        ]
        # Convert to pytorch tensor
        nodule_roi: torch.Tensor = torch.from_numpy(nodule_roi).float()
        return nodule_roi

    def segment_nodule(
        self, consensus_level: float = 0.5, invert: bool = False
    ) -> torch.Tensor:
        """
        Either cut out the nodule from the scan using the consensus boundary or remove the nodule from the scan.
        @nodule_roi: the nodule region of interest
        @nodule_idx: the index of the nodule in the nodules dataframe
        @invert: if True, the nodule will be removed only instead of the background
        """
        x, y, z = self.nodule_consensus_bbox
        # NOTE: Using 100_000 as padding to include the entire scan in the mask (for alignment with the nodule consensus bbox)
        consensus_mask, _ = consensus(
            self.pylidc_annotations,
            clevel=consensus_level,
            pad=100_000,
            ret_masks=False,
        )
        consensus_mask = consensus_mask[x[0] : x[1], y[0] : y[1], z[0] : z[1]]
        consensus_mask = torch.from_numpy(consensus_mask).to(dtype=torch.bool)
        # consensus_mask = add_dialation(consensus_mask, dilation=1) # TODO does not work yet (needed?)
        if invert:
            consensus_mask = torch.logical_not(consensus_mask)

        return self.nodule_roi * consensus_mask

    def visualise_nodule_bbox(self) -> None:
        """
        - Interactivelt visualise the consensus bbox of @nodule_idx (idxd in @self.nodule_df) as created by the @create_nodule_df.py script.
        - Also show the segmentation mask of the nodule as defined by @annotation_idx.
        NOTE: this function needs to be run from the console in order to work interactively
        """
        ann = self.pylidc_annotations[0]
        x, y, z = self.nodule_consensus_bbox

        # NOTE: pad is set to a large number to include the entire scan in the mask.
        # Then we can index the full scan with the consensus bbox for alignment.
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


class LIDC_IDRI_DATASET(Dataset):
    """
    Custom dataset for the LIDC-IDRI dataset containing nodule ROIs and their cancer labels.
    NOTE: Relies on having the nodule dataframe created by create_nodule_df.py script.
    The file is saved in the projects out/ directory as "nodule_df_all.csv"
    """

    def __init__(self, main_dir_path: str = config.DATA_DIR) -> None:
        self.main_dir_path = main_dir_path
        self.patient_ids = config.patient_ids

        try:
            # Read in the nodule dataframe and convert the string representations to python objects
            self.nodule_df = pd.read_csv(f"out/nodule_df_32.csv").assign(
                consensus_bbox=lambda x: x["consensus_bbox"].apply(ast.literal_eval),
                nodule_annotation_ids=lambda x: x["nodule_annotation_ids"].apply(
                    ast.literal_eval
                ),
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "The nodule dataframe was not found. Please run the create_nodule_df.py script first."
            )

    def __len__(self) -> int:
        return len(self.nodule_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Retrieves a single nodule volume and its
        corresponding consensus malignancy score
        """
        nodule = Nodule(
            self.nodule_df.iloc[idx],
            segmentation_setting=nodule_segmentation_config,
        )

        # TODO implement random translation of the nodule in the scan to augment the dataset

        return nodule.nodule_roi, nodule.malignancy_consensus


# %%
if __name__ == "__main__":
    dataset = LIDC_IDRI_DATASET()
    test_nodule = Nodule(dataset.nodule_df.iloc[0])
    # import matplotlib.pyplot as plt

    # roi_consensus, label = dataset.__getitem__(0)
    # plt.imshow(roi_consensus[:, :, 20], cmap="gray")
    # plt.show()

    test_nodule.visualise_nodule_bbox()

# %%
