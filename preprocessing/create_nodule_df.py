"""Scrip for creating a pandas dataframe with nodule annotation data"""

# %%
import itertools
from statistics import median_high

from utils.common_imports import *
from utils.logger_setup import logger

# SCIPRT PARAMS:
SCRIPT_PARAMS = pipeline_config["preprocessing"]["nodule_df"]
image_dim = SCRIPT_PARAMS["image_dim"]
csv_file_name = "nodule_df_all"
verbose = False


def get_malignancy_label(malignancy_scores: tuple[int]) -> str:
    """
    Return the consensun malignancy label based on the median high of the annotations.
    If the median high is greater than 3, the nodule is labelled as "Malignant".
    If the median high is less than 3, the nodule is labelled as "Benign".
    If the median high is equal to 3, the nodule is labelled as "Ambiguous".
    (source: https://github.com/benkeel/VAE_lung_lesion_BMVC/blob/main/Preprocessing/LIDC_DICOM_to_Numpy.ipynb)
    """
    consensus_malignancy = median_high(malignancy_scores)
    if consensus_malignancy > 3:
        return "Malignant"
    elif consensus_malignancy < 3:
        return "Benign"
    else:
        return "Ambiguous"


def compute_consensus_bbox(
    nodule_anns: list[pl.Annotation], image_dim: int
) -> tuple[tuple[int]]:
    """
    Returns the consensus bbox with standardise dimensions of size @image_dim
    of the nodule based on the consensus centroid of the annotations
    """
    ann_nodule_centroids = [ann.centroid for ann in nodule_anns]
    consensus_centroid = tuple(
        [
            int(np.mean([centroid[i] for centroid in ann_nodule_centroids]))
            for i in range(3)
        ]
    )

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
    return (x_dim, y_dim, z_dim), consensus_centroid


def main() -> None:
    dict_df = {}
    for pid in tqdm(config.patient_ids):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()
        if len(scan) > 1:
            logger.debug(f"A patient {pid} has more than one scan: {len(scan)}")
        scan = scan[0]

        # Get the annotations for the individual nodules in the scan:
        nodules_annotation: list[list[pl.Annotation]] = scan.cluster_annotations(
            verbose=verbose
        )

        # TODO we can also exlude nodules if they only have a single annotation? (there are a significant portion)
        if len(nodules_annotation) >= 1:
            for nodule_idx, nodule_anns in enumerate(nodules_annotation):

                # calculate the consensus centroid and bbox of the nodule:
                consensus_bbox, consensus_centroid = compute_consensus_bbox(
                    nodule_anns, image_dim
                )

                dict_df[f"{nodule_idx}_{pid}"] = {
                    "pid": pid,
                    "nodule_idx": nodule_idx,
                    "consensus_centroid": consensus_centroid,
                    "bbox_dim": image_dim,
                    "consensus_bbox": consensus_bbox,
                    "ann_mean_diameter": np.mean([ann.diameter for ann in nodule_anns]),
                    "ann_mean_volume": np.mean([ann.volume for ann in nodule_anns]),
                    "nodule_annotation_ids": tuple(
                        [int(ann.id) for ann in nodule_anns]
                    ),
                    "nodule_annotation_count": len(nodule_anns),
                    "malignancy_scores": tuple([ann.malignancy for ann in nodule_anns]),
                }

                # TODO might need to do some more processing here...
                # i.e. check if the mask is too small to be included

    nodule_df = pd.DataFrame.from_dict(dict_df, orient="index")

    nodule_df["malignancy_consensus"] = nodule_df["malignancy_scores"].apply(
        median_high
    )
    nodule_df["cancer_label"] = nodule_df["malignancy_scores"].apply(
        get_malignancy_label
    )

    # FILTERING:
    # nodule_df = nodule_df[nodule_df["cancer_label"] != "Ambiguous"]
    # ...

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
        logger.debug("Some nodule annotation ids are repeated")

    # check that the dimensions of the nodule bbox are all the same (aligned with the image_dim):
    verify_dim = (
        lambda roi_dim: image_dim
        == roi_dim[0][1] - roi_dim[0][0]  # x_dim
        == roi_dim[1][1] - roi_dim[1][0]  # y_dim
        == roi_dim[2][1] - roi_dim[2][0]  # z_dim
    )
    if not all(list(nodule_df["consensus_bbox"].apply(verify_dim))):
        logger.debug("Some nodule bbox dimensions are not aligned with the image_dim")
        raise ValueError(
            "Some nodule bbox dimensions are not aligned with the image_dim"
        )

    # verify that there are at most 4 annotations per nodule:
    if max(nodule_df["nodule_annotation_ids"].apply(len)) > 4:
        logger.debug("Some nodules have more than 4 annotations")

    # WRITE FILE:
    try:
        nodule_df.to_csv(f"{config.OUT_DIR}/{csv_file_name}.csv", index=True)
    except Exception as e:
        logger.error(f"Error saving nodule_df dataframe: {e}")


# %%
if __name__ == "__main__":
    main()
    import pandas as pd

    pd.read_csv("out/nodule_df_all.csv")
