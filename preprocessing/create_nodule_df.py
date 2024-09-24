"""Scrip for creating a pandas dataframe with nodule annotation data"""

# %%
import itertools
from statistics import median_high

from utils.common_imports import *
from utils.logger_setup import logger

SCRIPT_PARAMS = pipeline_config["preprocessing"]["nodule_df"]
image_dim = SCRIPT_PARAMS["image_dim"]
csv_file_name = SCRIPT_PARAMS["nodule_df_csv_name"]
verbose = False


def compute_nodule_malignancy(nodule: pl.Annotation) -> str:
    """
    Compute the malignancy of a nodule with the annotations made by 4 doctors.
    Return median high of the annotated cancer, True or False label for cancer,
    (source: https://github.com/benkeel/VAE_lung_lesion_BMVC/blob/main/Preprocessing/LIDC_DICOM_to_Numpy.ipynb)
    """
    malignancy = median_high([ann.malignancy for ann in nodule])
    if malignancy > 3:
        return malignancy, "Malignant"
    elif malignancy < 3:
        return malignancy, "Benign"
    else:
        return malignancy, "Ambiguous"


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

        # TODO we can also exlude nodules if they only have a single annotation?
        if len(nodules_annotation) >= 1:
            for nodule_idx, nodule_anns in enumerate(nodules_annotation):

                # calculate the consensus centroid of the nodule (average of the point cloud)
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

                # Compute consensus diameter of the nodule:
                ann_mean_diameter = np.mean([ann.diameter for ann in nodule_anns])

                # Compute consensus volumne of the nodule:
                ann_mean_volume = np.mean([ann.volume for ann in nodule_anns])

                # Calculate the malignancy of the nodule:
                malignancy_score, cancer_label = compute_nodule_malignancy(nodule_anns)

                dict_df[f"{nodule_idx}_{pid}"] = {
                    "pid": pid,
                    "nodule_idx": nodule_idx,
                    "malignancy_score": malignancy_score,
                    "cancer_label": cancer_label,
                    "consensus_centroid": consensus_centroid,
                    "consensus_bbox": (x_dim, y_dim, z_dim),
                    "ann_mean_diameter": ann_mean_diameter,
                    "ann_mean_volume": ann_mean_volume,
                    "nodule_annotation_ids": tuple(
                        [int(ann.id) for ann in nodule_anns]
                    ),
                    "nodule_annotation_count": len(nodule_anns),
                    "image_dim": image_dim,
                }

                # TODO might need to do some more processing here...
                # i.e. check if the mask is too small to be included (this can maybe be done using the nodules size list from provided as augmentation for the dataset)
                # for nodule in range(consensus_mask.shape[-1]):
                #     if np.sum(consensus_mask[:, :, nodule]) <= mask_threshold:
                #         continue

    nodule_df = pd.DataFrame.from_dict(dict_df, orient="index")

    # FILTERING:
    # nodule_df = nodule_df[nodule_df["cancer_label"] != "Ambiguous"]

    # TYPE CASTING
    nodule_df = nodule_df.assign(
        pid=nodule_df["pid"].astype("string"),
        nodule_idx=nodule_df["nodule_idx"].astype("int"),
        malignancy_score=nodule_df["malignancy_score"].astype("int"),
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
        nodule_df.to_csv(f"{config.OUT_DIR}/{csv_file_name}.csv")
    except Exception as e:
        logger.error(f"Error saving nodule_df dataframe: {e}")


# %%
if __name__ == "__main__":
    main()
