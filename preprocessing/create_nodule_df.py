"""Scrip for creating a pandas dataframe with nodule data and malignancy information"""

# %%
import itertools
from statistics import median_high

from pylidc.utils import consensus

from utils.common_imports import *
from utils.logger_setup import logger

# SCRIPT PARAMS:
# confidence level: A pixel will be considered part of the nodule if it is annotated by at least @c_level of the radiologists
c_level = 0.5
padding = 10
verbose = False
csv_file_name = "nodule_df_all_pad_10"


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
    for pid in tqdm(config.patient_ids[:200]):  # DEBUGGING
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()
        if len(scan) > 1:
            logger.debug(f"A patient {pid} has more than one scan: {len(scan)}")
        scan = scan[0]

        # Get the annotations for the individual nodules in the scan:
        nodules_annotation: list[list[pl.Annotation]] = scan.cluster_annotations(
            verbose=verbose
        )

        # TODO add the size of the nodule from the nodules size list
        # TODO we can also exlude nodules if they only have a single annotation?
        if len(nodules_annotation) >= 1:
            for nodule_idx, nodule_anns in enumerate(nodules_annotation):

                # Get the consensus mask and bbox at @c_level consensus from the 4 radiologists
                # Refer to documentation for more information
                # NOTE: The padding should only be applied to the x-y dimensions, thus remove it from the z dimension
                _, cmbbox = consensus(
                    anns=nodule_anns, clevel=c_level, pad=padding, ret_masks=False
                )
                x = (int(cmbbox[0].start), int(cmbbox[0].stop))
                y = (int(cmbbox[1].start), int(cmbbox[1].stop))
                z = (int(cmbbox[2].start) + padding, int(cmbbox[2].stop) - padding)

                # Calculate the malignancy of the nodule:
                malignancy_score, cancer_label = compute_nodule_malignancy(nodule_anns)

                dict_df[f"{nodule_idx}_{pid}"] = {
                    "pid": pid,
                    "nodule_idx": nodule_idx,
                    "malignancy_score": malignancy_score,
                    "cancer_label": cancer_label,
                    "consensus_bbox": (x, y, z),
                    "nodule_annotation_ids": tuple(
                        [int(ann.id) for ann in nodule_anns]
                    ),
                    "padding": padding,
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

    try:
        nodule_df.to_csv(f"{config.OUT_DIR}/{csv_file_name}.csv")
    except Exception as e:
        logger.error(f"Error saving nodule_df dataframe: {e}")


# %%
if __name__ == "__main__":
    main()

# %%
