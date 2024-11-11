"""File with code for StratifiedGroupKFold cross-validation split"""

from sklearn.model_selection import StratifiedGroupKFold

from project_config import SEED, env_config, pipeline_config
from utils.common_imports import *

# SCRIPT PARAMS:
CV_N_FOLDS = pipeline_config.dataset.cross_validation_folds
np.random.seed(SEED)


def create_cv_df(nodule_df: pd.DataFrame, cv: StratifiedGroupKFold) -> pd.DataFrame:
    fold_info = {}
    for fold, (train_idxs, test_idxs) in enumerate(
        cv.split(
            X=nodule_df, y=nodule_df["malignancy_consensus"], groups=nodule_df["pid"]
        )
    ):
        fold_info[fold] = {
            "train_idxs": train_idxs,
            "test_idxs": test_idxs,
            "train_pids": nodule_df.loc[train_idxs, "pid"].unique(),
            "test_pids": nodule_df.loc[test_idxs, "pid"].unique(),
            "train_malignancy_distribution": nodule_df["malignancy_consensus"][
                train_idxs
            ]
            .value_counts()
            .sort_index()
            .to_list(),
            "test_malignancy_distribution": nodule_df["malignancy_consensus"][test_idxs]
            .value_counts()
            .sort_index()
            .to_list(),
            "train_malignancy_distribution_normalised": nodule_df[
                "malignancy_consensus"
            ][train_idxs]
            .value_counts(normalize=True)
            .sort_index()
            .round(2)
            .to_list(),
            "test_malignancy_distribution_normalised": nodule_df[
                "malignancy_consensus"
            ][test_idxs]
            .value_counts(normalize=True)
            .sort_index()
            .round(2)
            .to_list(),
            "train_size": len(train_idxs),
            "test_size": len(test_idxs),
        }

    cv_df = pd.DataFrame(fold_info).T
    return cv_df


def add_cv_info(nodule_df: pd.DataFrame) -> None:
    """
    Add cross-validation fold info to the nodule dataframe
    created by create_nodule_df.py script. Function uses StratifiedGroupKFold
    to make sure that nodule from the same patient are in the same fold.
    """
    sgkf = StratifiedGroupKFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=SEED)
    cv_df = create_cv_df(nodule_df, sgkf)

    # make n_folds fold columns in the nodule_df that indicates which rows are in the train and test set:
    for fold in range(CV_N_FOLDS):
        nodule_df[f"fold_{fold + 1}"] = "no_fold"  # initialise all rows to "no_fold"
        nodule_df.loc[cv_df.loc[fold, "train_idxs"], f"fold_{fold + 1}"] = "train"
        nodule_df.loc[cv_df.loc[fold, "test_idxs"], f"fold_{fold + 1}"] = "test"

    # VALIDATION
    # Train and test sets should be stratified by malignancy and grouped by pid:
    fold_validations = []
    for fold in range(1, CV_N_FOLDS + 1):
        fold_validations.append(
            all(nodule_df.groupby("pid")[f"fold_{fold}"].nunique() == 1)
        )
    assert all(
        fold_validations
    ), "Train and test sets are not stratified by malignancy and grouped by pid in all folds"

    return nodule_df


if __name__ == "__main__":
    nodule_df = pd.read_csv(f"{env_config.nodule_df_file}")
    nodule_df = add_cv_info(nodule_df)
