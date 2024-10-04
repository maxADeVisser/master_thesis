"""
This script is only meant to be run once to create the hold out set.
It uses the StratifiedGroupKFold to split the data into 10 folds
and saves the first fold as the hold out set.

NOTE: It relies on the nodule_df.csv file created by the create_nodule_df.py script.
That is, it needs to be run after the create_nodule_df.py script has
run once without filtering out the hold out set. And then the create_nodule_df.py
script should be run again with the hold out set filtered out.
"""

import json

from sklearn.model_selection import StratifiedGroupKFold

from project_config import SEED, env_config
from utils.common_imports import *
from utils.logger_setup import logger

nodule_df = pd.read_csv(f"{env_config.OUT_DIR}/nodule_df.csv")
FOLDS = 10  # use ~10 percent of the data in the hold out set

sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idxs, test_idxs) in enumerate(
    sgkf.split(
        X=nodule_df, y=nodule_df["malignancy_consensus"], groups=nodule_df["pid"]
    )
):
    hold_out_pids = {
        "hold_out_pids": nodule_df.iloc[test_idxs]["pid"].unique().tolist()
    }

    with open(f"{env_config.OUT_DIR}/hold_out_pids.json", "w") as f:
        json.dump(hold_out_pids, f)
    logger.info(
        f"Saved {len(nodule_df.query(f'pid in {hold_out_pids}'))} hold out pids to {env_config.OUT_DIR}/hold_out_pids.json"
    )
    break
