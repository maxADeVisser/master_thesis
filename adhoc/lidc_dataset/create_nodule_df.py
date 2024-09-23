"""Scrip for creating a pandas dataframe with nodule data and malignancy information"""

# %%
from pylidc.utils import consensus

from logs.logging_setup import logger
from model.dataset import compute_malignancy
from utils.common_imports import *

# script params
c_level = 0.5  # confidence level
padding = None
verbose = False
mask_threshold = 10

dict_df = {}

# ------------------------------------

for pid in tqdm(config.patient_ids):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()
    if len(scan) > 1:
        logger.warning(f"A patient {pid} has more than one scan: {len(scan)}")
    scan = scan[0]

    # Get the annotations for the different nodules:
    nodules_annotation: list[list[pl.Annotation]] = scan.cluster_annotations(
        verbose=verbose
    )

    # TODO we can also exlude nodules if they only have a single annotation?
    if len(nodules_annotation) >= 1:
        for nodule_idx, nodule in enumerate(nodules_annotation):

            # Get the consensus mask and bbox at @c_level consensus from the 4 radiologists
            # Refer to documentation for more information
            consensus_mask, cmbbox = consensus(
                anns=nodule, clevel=c_level, pad=padding, ret_masks=False
            )
            malignancy, cancer_label = compute_malignancy(nodule)

            # TODO is the following something we need to do?
            # if malignancy == "Ambiguous":
            #     continue

            dict_df[f"{nodule_idx}_{pid}"] = {
                "pid": pid,
                "nodule_idx": nodule_idx,
                "malignancy_score": malignancy,
                "cancer_label": cancer_label,
                "cmbbox": cmbbox,
            }

            # TODO might need to do some more processing here...
            # i.e. check if the mask is too small to be included
            # for nodule in range(consensus_mask.shape[-1]):
            #     if np.sum(consensus_mask[:, :, nodule]) <= mask_threshold:
            #         continue

            #     ...

nodule_df = pd.DataFrame.from_dict(dict_df, orient="index")
nodule_df.to_csv(f"{config.OUT_DIR}/nodule_df.csv")

# %%
