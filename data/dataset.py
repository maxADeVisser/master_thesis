# %%
import ast

import torch
from pylidc.utils import consensus, volume_viewer
from torch.utils.data import DataLoader, Dataset

from data.data_augmentations import apply_augmentations
from preprocessing.processing import clip_and_normalise_volume
from project_config import SEED, env_config, pipeline_config
from utils.common_imports import *
from utils.logger_setup import logger
from utils.utils import get_scans_by_patient_id

torch.manual_seed(SEED)
np.random.seed(SEED)


def transform_3d_to_25d(volume: torch.Tensor) -> torch.Tensor:
    """
    Transform a 3D volume with 1 channel to a 2.5D volume with 3 channels by selecting the three middle slices and stacking them along a new axis for the channel (mimics a RGB image)
    """
    volume = volume.squeeze(0)  # Remove channel dimension
    middle_slice_idx = volume.shape[0] // 2
    z_indices = [middle_slice_idx - 1, middle_slice_idx, middle_slice_idx + 1]
    # Change the order of the dimensions to match the expected input shape of the model:
    transformed = volume[:, :, z_indices].permute(2, 0, 1)
    return transformed


class Nodule:
    """
    Helper class to store and manipulate nodule information from nodule df created by create_nodule_df.py script.
    @nodule_record: a single row from the nodule dataframe.
    @nodule_context_size: the size of the nodule context to be used for the nodule ROI.
    @segmentation_setting determines if/how the nodule is segmented from the scan.
    """

    def __init__(
        self,
        nodule_record: pd.Series,
        nodule_context_size: int,
        # TODO clean up if unused
        segmentation_setting: Literal["none", "remove_background", "remove_nodule"],
        nodule_dim: Literal["2.5D", "3D"] = "3D",
    ) -> None:
        self.patient_id = nodule_record["pid"]
        self.nodule_dim = nodule_dim
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
        if self.nodule_dim == "2.5D":
            self.nodule_roi = transform_3d_to_25d(self.nodule_roi)
        self.nodule_roi = clip_and_normalise_volume(self.nodule_roi)

        # if segmentation_setting == "remove_background":
        #     self.nodule_roi = self.segment_nodule(invert=False)
        # elif segmentation_setting == "remove_nodule":
        #     self.nodule_roi = self.segment_nodule(invert=True)
        # elif segmentation_setting == "none":
        #     pass
        # else:
        #     raise ValueError(
        #         f"Segmentation setting {segmentation_setting} not recognised. Please choose one of: ['none', 'remove_nodule', 'remove_background']"
        #     )

    def get_nodule_roi(self) -> torch.Tensor:
        """Returns the nodule region of interest from the scan based on the consensus bbox from the 4 annotators"""
        scan: np.ndarray = get_scans_by_patient_id(self.patient_id, to_numpy=True)
        x_bounds, y_bounds, z_bounds = self.nodule_consensus_bbox
        nodule_roi = scan[
            x_bounds[0] : x_bounds[1],
            y_bounds[0] : y_bounds[1],
            z_bounds[0] : z_bounds[1],
        ]
        # Convert to pytorch tensor and add channel dimension
        nodule_roi: torch.Tensor = torch.from_numpy(nodule_roi).unsqueeze(0).float()
        return nodule_roi

    def get_consensus_mask(self) -> torch.Tensor:
        """Returns the consensus mask of the nodule based on the annotations of the 4 radiologists. Refer to the documentation of pylidc.utils.consensus for more info."""
        # NOTE: Using 100_000 as padding to include the entire scan in the mask (for alignment with the nodule consensus bbox)
        consensus_mask, _ = consensus(
            self.pylidc_annotations,
            clevel=0.5,
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
        if invert:
            consensus_mask = torch.logical_not(consensus_mask)

        return self.nodule_roi * consensus_mask

    def visualise_nodule_bbox(self) -> None:
        """
        - Interactivelt visualise the consensus bbox of @nodule_idx (idxd in @self.nodule_df) as created by the @create_nodule_df.py script.
        - Also show the segmentation mask of the nodule as defined by @annotation_idx.
        NOTE: this function needs to be run from the console in order to work interactively
        """
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

    def __init__(
        self,
        context_size: Literal[10, 20, 30, 40, 50, 60, 70],
        segmentation_configuration: Literal[
            "none", "remove_background", "remove_nodule"
        ] = "none",
        n_dims: Literal["2.5D", "3D"] = "3D",
        nodule_df_path: str = env_config.processed_nodule_df_file,
    ) -> None:
        self.context_size = context_size
        self.n_dims = n_dims
        self.segmentation_configuration = segmentation_configuration
        self.patient_ids = env_config.patient_ids

        try:
            # Read in the nodule dataframe and convert the string representations to python objects
            self.nodule_df = pd.read_csv(nodule_df_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "The nodule dataframe was not found. Please run the create_nodule_df.py script first"
            )
        self.nodule_df[f"consensus_bbox_{self.context_size}"] = self.nodule_df[
            f"consensus_bbox_{self.context_size}"
        ].apply(ast.literal_eval)
        self.nodule_df["nodule_annotation_ids"] = self.nodule_df[
            "nodule_annotation_ids"
        ].apply(ast.literal_eval)

        logger.info(
            f"""
            LIDC-IDRI nodule dataset loaded successfully with parameters:
            CONTEXT SIZE: {self.context_size}
            DATA DIMENSIONALITY: {self.n_dims}
            NODULE SEGMENTATION: {self.segmentation_configuration}
            """
        )

    def __len__(self) -> int:
        return len(self.nodule_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Retrieves a single nodule volume and its
        corresponding consensus malignancy score
        """
        nodule = Nodule(
            nodule_record=self.nodule_df.iloc[idx],
            nodule_context_size=self.context_size,
            segmentation_setting=self.segmentation_configuration,
            nodule_dim=self.n_dims,
        )

        return nodule.nodule_roi, nodule.malignancy_consensus


class PrecomputedNoduleROIs(Dataset):
    def __init__(self, preprocessed_dir: str, data_augmentation: bool = True) -> None:
        self.files = [f"{preprocessed_dir}/{f}" for f in os.listdir(preprocessed_dir)]
        self.data_augmentation = data_augmentation

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        data: torch.Tensor = torch.load(self.files[idx], weights_only=True)
        feature, label = data[0], data[1]
        if self.data_augmentation:
            feature = apply_augmentations(feature)
        return feature, label


# %%
if __name__ == "__main__":
    # testing precomputed dataset ----------------
    import matplotlib.pyplot as plt

    pdataset = PrecomputedNoduleROIs(
        "/Users/newuser/Documents/ITU/master_thesis/data/precomputed_rois_70C_2.5D",
        data_augmentation=True,
    )
    loader = DataLoader(pdataset, batch_size=2, shuffle=False)
    for i, (roi, label) in enumerate(loader):
        plt.imshow(roi[0][1], cmap="gray")
        plt.show()
        break
    # -----------------------

    # testing dataloader
    dataset = LIDC_IDRI_DATASET(
        context_size=30, segmentation_configuration="none", n_dims="2.5D"
    )
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    roi, label = next(iter(train_loader))
    print(roi.shape)
    middle_slice = roi.shape[-1] // 2
    plt.imshow(roi[0][1], cmap="gray")
    plt.title(f"Label malignancy score: {label[0]}")
    plt.show()

    # # Testing data set:
    # dataset = LIDC_IDRI_DATASET(img_dim=30)
    # roi_consensus, label = dataset.__getitem__(0)
    # plt.imshow(roi_consensus[:, :, 35], cmap="gray")
    # plt.title(f"Label malignancy score: {label}")
    # plt.show()

    # Test Nodule
    IMG_DIM = 70
    dataset = LIDC_IDRI_DATASET(
        context_size=IMG_DIM, segmentation_configuration="none", n_dims="2.5D"
    )

    test_nodule = Nodule(
        dataset.nodule_df.iloc[1],
        nodule_context_size=IMG_DIM,
        segmentation_setting="none",
        nodule_dim="3D",
        # nodule_dim="2.5D",
    )
    test_nodule.nodule_roi.shape
    middle_slice = test_nodule.nodule_roi.shape[-1] // 2
    plt.imshow(test_nodule.nodule_roi[0, :, :, middle_slice], cmap="gray")
    plt.show()

    # test_nodule.visualise_nodule_bbox()

# %%