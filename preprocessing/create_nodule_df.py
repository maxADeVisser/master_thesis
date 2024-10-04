"""Scrip for creating a pandas dataframe with nodule annotation data"""

# %%
import itertools
import math

from preprocessing.create_cv_df import add_cv_info
from project_config import SEED, env_config
from utils.common_imports import *
from utils.logger_setup import logger

# SCIPRT PARAMS:
# IMAGE_DIMS = [8, 16, 32, 64, 128]
IMAGE_DIMS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
CSV_FILE_NAME = f"nodule_df"
HOLD_OUT_PIDS = env_config.hold_out_pids
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
) -> tuple[tuple[int]]:
    """
    Returns the consensus bbox with standardise dimensions of size @image_dim
    of the nodule based on @consensus_centroid of the annotations
    """
    # calculate the consensus bbox of the nodule
    x_dim = (
        consensus_centroid[0] - (image_dim // 2),
        consensus_centroid[0] + (image_dim // 2),
    )
    y_dim = (
        consensus_centroid[1] - (image_dim // 2),
        consensus_centroid[1] + (image_dim // 2),
    )
    z_dim = (
        consensus_centroid[2] - (image_dim // 2),
        consensus_centroid[2] + (image_dim // 2),
    )
    return (x_dim, y_dim, z_dim)


def verify_bbox_within_scan(
    scan_dims: tuple[int, int, int], consensus_bbox: tuple[tuple[int]]
) -> bool:
    """
    Verify that the bbox (computed using @compute_consensus_bbox()) lies within the scan dimensions
    """
    return all(
        [
            (0 < consensus_bbox[0][1] < scan_dims[0]),
            (0 < consensus_bbox[1][1] < scan_dims[1]),
            (0 < consensus_bbox[2][1] < scan_dims[2]),
        ]
    )


def main() -> None:
    dict_df = {}
    for pid in tqdm(env_config.patient_ids, desc="Processing patients"):
        scan: list[pl.Scan] = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()
        if len(scan) > 1:
            logger.debug(f"A patient {pid} has more than one scan: {len(scan)}")
        scan: pl.Scan = scan[0]
        scan_dims = scan.to_volume(verbose=verbose).shape

        # Get the annotations for the individual nodules in the scan:
        nodules_annotation: list[list[pl.Annotation]] = scan.cluster_annotations(
            verbose=verbose
        )

        # TODO we can also exlude nodules if they only have a single annotation? (there are a significant portion)
        if len(nodules_annotation) >= 1:
            for nodule_idx, nodule_anns in enumerate(nodules_annotation):
                consensus_centroid: tuple[int, int, int] = compute_consensus_centroid(
                    nodule_anns
                )

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
                    "margin_scores": tuple([ann.margin for ann in nodule_anns]),
                    "consensus_centroid": consensus_centroid,
                }

                for img_dim in IMAGE_DIMS:
                    # Compute the bounding box at different dimensions:
                    (x_dim, y_dim, z_dim) = compute_consensus_bbox(
                        img_dim, consensus_centroid
                    )
                    dict_df[nodule_id][f"consensus_bbox_{img_dim}"] = (
                        x_dim,
                        y_dim,
                        z_dim,
                    )

                    # Record whether the nodule bbox is within the scan dimensions:
                    dict_df[nodule_id][f"bbox_within_scan_{img_dim}"] = (
                        verify_bbox_within_scan(scan_dims, (x_dim, y_dim, z_dim))
                    )

                # TODO might need to do some more processing here...
                # i.e. check if the mask is too small to be included

    nodule_df = pd.DataFrame.from_dict(dict_df, orient="index").reset_index()

    # CALCULATE CONSENSUS OF SCORES:
    nodule_df["malignancy_consensus"] = nodule_df["malignancy_scores"].apply(
        calculate_consensus
    )
    nodule_df["subtlety_consensus"] = nodule_df["subtlety_scores"].apply(
        calculate_consensus
    )
    nodule_df["margin_consensus"] = nodule_df["margin_scores"].apply(
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
    # Flatten the list of annotation ids:
    all_ids = list(itertools.chain.from_iterable(nodule_df["nodule_annotation_ids"]))
    if not len(all_ids) == len(set(all_ids)):
        logger.debug("Some nodule annotation IDs are repeated")

    # check that the dimensions of the nodule bbox are all the same (aligned with the image_dim):

    for img_dim in IMAGE_DIMS:
        _verify_image_dim = (
            lambda roi_dim: img_dim
            == roi_dim[0][1] - roi_dim[0][0]  # x_dim
            == roi_dim[1][1] - roi_dim[1][0]  # y_dim
            == roi_dim[2][1] - roi_dim[2][0]  # z_dim
        )
        if not all(
            list(nodule_df[f"consensus_bbox_{img_dim}"].apply(_verify_image_dim))
        ):
            logger.debug(
                "Some nodule bbox dimensions are not aligned with the image_dim"
            )
            raise ValueError(
                "Some nodule bbox dimensions are not aligned with the image_dim"
            )

    # verify that there are at most 4 annotations per nodule:
    if max(nodule_df["nodule_annotation_ids"].apply(len)) > 4:
        logger.debug("There are nodules with more than 4 annotations")

    # CREATE HOLD OUT SET:
    # NOTE: uncomment two following lines to create the hold out set:
    # hold_out_df = nodule_df[nodule_df["pid"].isin(HOLD_OUT_PIDS)].reset_index(drop=True)
    # hold_out_df.to_csv(f"{env_config.OUT_DIR}/hold_out_nodule_df.csv", index=False)
    nodule_df = nodule_df[~nodule_df["pid"].isin(HOLD_OUT_PIDS)].reset_index(drop=True)

    # ADD CROSS VALIDATION FOLDS:
    nodule_df = add_cv_info(nodule_df)

    # WRITE FILE:
    try:
        nodule_df.to_csv(f"{env_config.OUT_DIR}/{CSV_FILE_NAME}.csv", index=False)
    except Exception as e:
        logger.error(f"Error saving nodule_df dataframe: {e}")


# %%
if __name__ == "__main__":
    main()
