"""Scrip for creating a pandas dataframe with nodule data and malignancy information"""

# %%
from statistics import median_high

from pylidc.utils import consensus

from utils.common_imports import *
from utils.logger_setup import logger

# script params
c_level = 0.5  # confidence level: A pixel is considered part of the nodule if it is annotated by at least 50% of the radiologists
padding = None
verbose = False


def compute_nodule_malignancy(nodule: pl.Annotation) -> bool | str:
    """
    Compute the malignancy of a nodule with the annotations made by 4 doctors.
    Return median high of the annotated cancer, True or False label for cancer,
    (source: https://github.com/benkeel/VAE_lung_lesion_BMVC/blob/main/Preprocessing/LIDC_DICOM_to_Numpy.ipynb)
    """
    malignancy = median_high([ann.malignancy for ann in nodule])
    if malignancy > 3:
        return malignancy, True
    elif malignancy < 3:
        return malignancy, False
    else:
        return malignancy, "Ambiguous"


dict_df = {}

# ------------------------------------

for pid in tqdm(config.patient_ids):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()
    if len(scan) > 1:
        logger.debug(f"A patient {pid} has more than one scan: {len(scan)}")
    scan = scan[0]

    # Get the annotations for the different nodules:
    nodules_annotation: list[list[pl.Annotation]] = scan.cluster_annotations(
        verbose=verbose
    )

    # TODO we can also exlude nodules if they only have a single annotation?
    if len(nodules_annotation) >= 1:
        for nodule_idx, nodule_ann in enumerate(nodules_annotation):

            # Get the consensus mask and bbox at @c_level consensus from the 4 radiologists
            # Refer to documentation for more information
            consensus_mask, cmbbox = consensus(
                anns=nodule_ann, clevel=c_level, pad=padding, ret_masks=False
            )
            x = (int(cmbbox[0].start), int(cmbbox[0].stop))
            y = (int(cmbbox[1].start), int(cmbbox[1].stop))
            z = (int(cmbbox[2].start), int(cmbbox[2].stop))

            malignancy, cancer_label = compute_nodule_malignancy(nodule_ann)

            # TODO there is something wrong with the consensus_bbox, it is not returning the bbox in 3D.
            dict_df[f"{nodule_idx}_{pid}"] = {
                "pid": pid,
                "nodule_idx": nodule_idx,
                "malignancy_score": malignancy,
                "cancer_label": cancer_label,
                "consensus_bbox": (x, y, z),
            }

            # TODO might need to do some more processing here...
            # i.e. check if the mask is too small to be included
            # for nodule in range(consensus_mask.shape[-1]):
            #     if np.sum(consensus_mask[:, :, nodule]) <= mask_threshold:
            #         continue

nodule_df = pd.DataFrame.from_dict(dict_df, orient="index")

# Filtering:
nodule_df = nodule_df[nodule_df["cancer_label"] != "Ambiguous"]

# Type casting
nodule_df = nodule_df.assign(
    pid=nodule_df["pid"].astype("string"),
    nodule_idx=nodule_df["nodule_idx"].astype("int"),
    malignancy_score=nodule_df["malignancy_score"].astype("int"),
    cancer_label=nodule_df["cancer_label"].astype("bool"),
)

try:
    nodule_df.to_csv(f"{config.OUT_DIR}/nodule_df.csv")
except Exception as e:
    logger.error(f"Error saving nodule_df dataframe: {e}")


# %%

# DEBUGGING:
nodule_df = pd.read_csv(f"{config.OUT_DIR}/nodule_df.csv")
nodule_df