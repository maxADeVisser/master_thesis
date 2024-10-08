"""Scrip for creating a pandas dataframe with nodule annotation data"""

# %%
import itertools
import math

from project_config import env_config
from utils.common_imports import *
from utils.logger_setup import logger

# SCIPRT PARAMS:
IMAGE_DIMS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
MAX_IMG_DIM_USED = 70
assert (
    MAX_IMG_DIM_USED in IMAGE_DIMS
), "The maximum image dimensions needs to be in the @image_dims list"
CSV_FILE_NAME = f"nodule_df"
verbose = False
logger.info(
    f"Creating nodule df with image_dims: {IMAGE_DIMS} as {CSV_FILE_NAME}.csv ..."
)


def calculate_consensus(annotation_variable_values: list[int]) -> int:
    """
    Calculate the consensus of discrete annotation variable values
    """
    round_to_nearest = lambda x: math.floor(x + 0.5)
    # return median_high(annotation_variable_values)
    return round_to_nearest(np.mean(annotation_variable_values))


def get_malignancy_label(malignancy_scores: tuple[int]) -> str:
    """
    Return the consensun malignancy label based on the median high of the annotations.
    If the median high is greater than 3, the nodule is labelled as "Malignant".
    If the median high is less than 3, the nodule is labelled as "Benign".
    If the median high is equal to 3, the nodule is labelled as "Ambiguous".
    (source: https://github.com/benkeel/VAE_lung_lesion_BMVC/blob/main/Preprocessing/LIDC_DICOM_to_Numpy.ipynb)
    """
    consensus_malignancy = calculate_consensus(malignancy_scores)
    if consensus_malignancy > 3:
        return "Malignant"
    elif consensus_malignancy < 3:
        return "Benign"
    else:
        return "Ambiguous"


def compute_consensus_centroid(
    nodule_anns: list[pl.Annotation],
) -> tuple[int, int, int]:
    """Return the consensus centroid of the nodule based on the annotations"""
    ann_nodule_centroids = [ann.centroid for ann in nodule_anns]
    consensus_centroid = tuple(
        [
            int(np.mean([centroid[i] for centroid in ann_nodule_centroids]))
            for i in range(3)
        ]
    )
    return consensus_centroid


def compute_consensus_bbox(
    image_dim: int,
    consensus_centroid: tuple[int, int, int],
    scan_dims: tuple[int, int, int],
) -> tuple[tuple[int]]:
    """
    Computes the POSSIBLE consensus bbox with standardise dimensions of size @image_dim. If the bbox along a axis exceeds the edge of the scan, it is cut off at that point. That is, there is not guarantee that the copmuted bbox will have standardised dimensions on all axes.
    """

    def _compute_bbox_dim(
        axis_centroid: int, axis_scan_dim: int, image_dim: int
    ) -> tuple[int]:
        """
        Computes the bounding bbox dimension for a single axis
        """
        bbox_start_dim = axis_centroid - (image_dim // 2)
        bbox_end_dim = axis_centroid + (image_dim // 2)

        if bbox_start_dim < 0:
            bbox_start_dim = 0
        if axis_scan_dim < bbox_end_dim:
            bbox_end_dim = axis_scan_dim
        return (bbox_start_dim, bbox_end_dim)

    x_centroid, y_centroid, z_centroid = consensus_centroid
    x_scan_dim, y_scan_dim, z_scan_dim = scan_dims
    x_dim = _compute_bbox_dim(x_centroid, x_scan_dim, image_dim)
    y_dim = _compute_bbox_dim(y_centroid, y_scan_dim, image_dim)
    z_dim = _compute_bbox_dim(z_centroid, z_scan_dim, image_dim)

    return (x_dim, y_dim, z_dim)


def create_nodule_df(file_name: str = CSV_FILE_NAME, add_bbox: bool = True) -> None:
    dict_df = {}
    for pid in tqdm(env_config.patient_ids, desc="Processing patients"):
        scan: list[pl.Scan] = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()
        if len(scan) > 1:
            logger.debug(f"A patient {pid} has more than one scan: {len(scan)}")
        scan: pl.Scan = scan[0]
        scan_dims = scan.to_volume(verbose=verbose).shape if add_bbox else None

        # Get the annotations for the individual nodules in the scan:
        nodules_annotation: list[list[pl.Annotation]] = scan.cluster_annotations(
            verbose=verbose
        )

        if len(nodules_annotation) >= 1:
            for nodule_idx, nodule_anns in enumerate(nodules_annotation):
                nodule_id = f"{nodule_idx}_{pid}"
                dict_df[nodule_id] = {
                    "pid": pid,
                    "nodule_idx": nodule_idx,
                    "ann_mean_diameter": np.mean([ann.diameter for ann in nodule_anns]),
                    "ann_mean_volume": np.mean([ann.volume for ann in nodule_anns]),
                    "nodule_annotation_ids": tuple(
                        [int(ann.id) for ann in nodule_anns]
                    ),
                    "nodule_annotation_count": len(nodule_anns),
                    "malignancy_scores": tuple([ann.malignancy for ann in nodule_anns]),
                    "subtlety_scores": tuple([ann.subtlety for ann in nodule_anns]),
                }

                if add_bbox:
                    # Compute the consensus centroid and bbox for the nodule:
                    consensus_centroid: tuple[int, int, int] = (
                        compute_consensus_centroid(nodule_anns)
                    )
                    dict_df[nodule_id]["consensus_centroid"] = consensus_centroid

                    # Compute the consensus bbox at different img dimensions:
                    for img_dim in IMAGE_DIMS:
                        (x_dim, y_dim, z_dim) = compute_consensus_bbox(
                            img_dim, consensus_centroid, scan_dims
                        )
                        dict_df[nodule_id][f"consensus_bbox_{img_dim}"] = (
                            x_dim,
                            y_dim,
                            z_dim,
                        )

    nodule_df = pd.DataFrame.from_dict(dict_df, orient="index").reset_index()

    # CALCULATE CONSENSUS OF MALIGNANCY SCORES:
    nodule_df["malignancy_consensus"] = nodule_df["malignancy_scores"].apply(
        calculate_consensus
    )
    nodule_df["subtlety_consensus"] = nodule_df["subtlety_scores"].apply(
        calculate_consensus
    )
    nodule_df["cancer_label"] = nodule_df["malignancy_scores"].apply(
        get_malignancy_label
    )

    # TYPE CASTING
    nodule_df = nodule_df.assign(
        pid=nodule_df["pid"].astype("string"),
        nodule_idx=nodule_df["nodule_idx"].astype("int"),
        malignancy_consensus=nodule_df["malignancy_consensus"].astype("int"),
        cancer_label=nodule_df["cancer_label"].astype("category"),
    )

    # VERIFICATIONS:
    # Check that no annotations id are repeated (i.e. that the annotations are unique to each nodule)
    all_ids_flattened = list(
        itertools.chain.from_iterable(nodule_df["nodule_annotation_ids"])
    )
    if not len(all_ids_flattened) == len(set(all_ids_flattened)):
        logger.debug("Some nodule annotation IDs are repeated")

    if add_bbox:
        # Flag nodules where the dimensions of the bbox along all axis are not standardised (=image_dim):
        for img_dim in IMAGE_DIMS:
            _verify_image_dim = (
                lambda roi_dim: img_dim
                == roi_dim[0][1] - roi_dim[0][0]  # x_dim
                == roi_dim[1][1] - roi_dim[1][0]  # y_dim
                == roi_dim[2][1] - roi_dim[2][0]  # z_dim
            )
            nodule_df[f"bbox_{img_dim}_standardised"] = nodule_df[
                f"consensus_bbox_{img_dim}"
            ].apply(_verify_image_dim)

        # log how many nodules bboxes that do not have standardised img dimensions at max_img:
        logger.debug(
            f"Number of standardised nodule bboxes at {MAX_IMG_DIM_USED} bbox sizes:\n{nodule_df[f'bbox_{MAX_IMG_DIM_USED}_standardised'].value_counts()}"
        )

    # validate that there are at most 4 annotations per nodule:
    if max(nodule_df["nodule_annotation_ids"].apply(len)) > 4:
        logger.debug("There are nodules with more than 4 annotations")

    # WRITE FILE:
    try:
        nodule_df.to_csv(f"{env_config.OUT_DIR}/{file_name}.csv", index=False)
        logger.info(f"nodule_df saved to:\n{env_config.OUT_DIR}/{file_name}.csv")
    except Exception as e:
        logger.error(f"Error saving nodule_df dataframe: {e}")


# %%
if __name__ == "__main__":
    create_nodule_df()

    # For creating the annotations df used for agreement analysis:
    # create_nodule_df(file_name="annotation_df", add_bbox=False)
