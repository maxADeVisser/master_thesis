# %%
import ast
from typing import Literal

import numpy as np
import pandas as pd
import pylidc as pl
import torch
from pylidc.utils import consensus, volume_viewer
from torch.utils.data import DataLoader, Dataset

from model.processing import clip_and_normalise_volume
from project_config import env_config, pipeline_config
from utils.logger_setup import logger
from utils.utils import get_scans_by_patient_id

# SCRIPT_PARAMS:
IMAGE_DIM = pipeline_config.dataset.image_dim
NODULE_SEGMENTATION = pipeline_config.dataset.segment_nodule
CONSENSUS_LEVEL = pipeline_config.dataset.consensus_level
BATCH_SIZE = pipeline_config.training.batch_size


class Nodule:
    def __init__(
        self,
        nodule_record: pd.Series,
        nodule_context_size: Literal[8, 16, 32, 64, 128],
        segmentation_setting: (
            Literal["remove_nodule", "remove_background"] | None
        ) = None,
    ) -> None:
        """
        Helper class to store and manipulate nodule information from nodule df created by create_nodule_df.py script.
        @nodule_record: a single row from the nodule dataframe.
        @nodule_context_size: the size of the nodule context to be used for the nodule ROI.
        @segmentation_setting determines if/how the nodule is segmented from the scan.
        """
        self.patient_id = nodule_record["pid"]
        self.annotation_ids = nodule_record["nodule_annotation_ids"]
        self.pylidc_annotations = [
            pl.query(pl.Annotation).filter(pl.Annotation.id == ann_id).first()
            for ann_id in self.annotation_ids
        ]
        self.malignancy_consensus = nodule_record["malignancy_consensus"]
        self.nodule_consensus_bbox = nodule_record[
            f"consensus_bbox_{nodule_context_size}"
        ]
        self.nodule_roi = self.get_nodule_roi()
        self.nodule_roi = clip_and_normalise_volume(self.nodule_roi)

        if segmentation_setting == "remove_background":
            self.nodule_roi = self.segment_nodule(invert=False)
        elif segmentation_setting == "remove_nodule":
            self.nodule_roi = self.segment_nodule(invert=True)

    def get_nodule_roi(self) -> torch.Tensor:
        """Returns the nodule region of interest from the scan based on the consensus bbox from the 4 annotators"""
        scan: np.ndarray = get_scans_by_patient_id(self.patient_id, to_numpy=True)
        x_bounds, y_bounds, z_bounds = self.nodule_consensus_bbox
        nodule_roi = scan[
            x_bounds[0] : x_bounds[1],
            y_bounds[0] : y_bounds[1],
            z_bounds[0] : z_bounds[1],
        ]
        # Convert to pytorch tensor
        nodule_roi: torch.Tensor = torch.from_numpy(nodule_roi).float()
        return nodule_roi

    def get_consensus_mask(
        self, consensus_level: float = CONSENSUS_LEVEL
    ) -> torch.Tensor:
        """Returns the consensus mask of the nodule based on the annotations of the 4 radiologists. Refer to the documentation of pylidc.utils.consensus for more info."""
        # NOTE: Using 100_000 as padding to include the entire scan in the mask (for alignment with the nodule consensus bbox)
        consensus_mask, _ = consensus(
            self.pylidc_annotations,
            clevel=consensus_level,
            pad=100_000,
            ret_masks=False,
        )
        return consensus_mask

    def segment_nodule(self, invert: bool = False) -> torch.Tensor:
        """
        Either cut out the nodule from the scan using the consensus boundary or remove the nodule from the scan.
        @consensus_level: the level of agreement between the annotators to be used for the
        consensus mask of the nodule.
        @invert: if True, the nodule will be removed only instead of the background
        """
        x, y, z = self.nodule_consensus_bbox
        consensus_mask = self.get_consensus_mask()[
            x[0] : x[1], y[0] : y[1], z[0] : z[1]
        ]
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
        consensus_mask = self.get_consensus_mask()[
            x[0] : x[1], y[0] : y[1], z[0] : z[1]
        ]

        # Uncomment the following to visualise the nodule bbox and mask in jupyter notebook
        # %matplotlib widget
        volume_viewer(
            vol=self.nodule_roi,
            mask=consensus_mask,
            ls="--",
            c="r",
        )


class LIDC_IDRI_DATASET(Dataset):
    """
    Custom dataset for the LIDC-IDRI dataset containing nodule ROIs and their cancer labels.
    NOTE: Relies on having the nodule dataframe created by create_nodule_df.py script.
    The file is saved in the projects out/ directory as "nodule_df_all.csv"
    """

    def __init__(self, dataset_dir_path: str = env_config.DATA_DIR) -> None:
        self.dataset_dir_path = dataset_dir_path
        self.patient_ids = env_config.patient_ids

        try:
            # Read in the nodule dataframe and convert the string representations to python objects
            self.nodule_df = pd.read_csv(f"out/nodule_df.csv")
            self.nodule_df[f"consensus_bbox_{IMAGE_DIM}"] = self.nodule_df[
                f"consensus_bbox_{IMAGE_DIM}"
            ].apply(ast.literal_eval)
            self.nodule_df["nodule_annotation_ids"] = self.nodule_df[
                "nodule_annotation_ids"
            ].apply(ast.literal_eval)

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
            segmentation_setting=NODULE_SEGMENTATION,
            nodule_context_size=IMAGE_DIM,
        )

        # TODO implement random translation of the nodule in the scan to augment the dataset

        return nodule.nodule_roi, nodule.malignancy_consensus


def get_train_loader(dataset: LIDC_IDRI_DATASET) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def get_test_loader(self):
    pass


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # testing dataloader
    dataset = LIDC_IDRI_DATASET()
    train_loader = get_train_loader(dataset)

    for i, (roi, label) in enumerate(train_loader):
        print(roi.shape)
        # plt.imshow(roi[:, :, 35], cmap="gray")
        # plt.title(f"Label malignancy score: {label}")
        # plt.show()
        break

    # Testing data set:
    # dataset = LIDC_IDRI_DATASET()
    # roi_consensus, label = dataset.__getitem__(0)
    # plt.imshow(roi_consensus[:, :, 35], cmap="gray")
    # plt.title(f"Label malignancy score: {label}")
    # plt.show()

    # Test Nodule
    # test_nodule = Nodule(
    #     dataset.nodule_df.iloc[0],
    #     nodule_context_size=IMAGE_DIM,
    #     segmentation_setting="remove_nodule",
    # )
    # plt.imshow(test_nodule.nodule_roi[:, :, 35], cmap="gray")
    # plt.show()
    # test_nodule.visualise_nodule_bbox()

    # Validate that all ROIs have standardise shape:
    dataset = LIDC_IDRI_DATASET()
    from tqdm import tqdm

    for i in tqdm(range(len(dataset))):
        roi, _ = dataset.__getitem__(i)
        assert roi.shape == (
            IMAGE_DIM,
            IMAGE_DIM,
            IMAGE_DIM,
        ), f"ROI shape is not standardised. ROI {i} has shape {roi.shape}"

# %%
