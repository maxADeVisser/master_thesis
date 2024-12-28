"""
Annotation data is computed using the create_annotations_df.py script in the preprocessing folder
"""

import pandas as pd

from project_config import env_config


def compute_kendall_w(annotation_data: pd.DataFrame) -> pd.DataFrame:
    """Computes Kendall's W assuming that there are no rank ties."""

    # n: number of items (nodules) and m: number of raters (annotations):
    n, m = annotation_data.shape

    # rank the ratings for each subject:
    ranked_data = annotation_data.rank(axis=0, method="average")

    # sum of ranks for each item:
    sum_of_total_ranks = ranked_data.sum(axis=1)  # R_i

    # mean of the sum of total ranks:
    mean_of_ranks = sum_of_total_ranks.mean()  # \bar{R}

    # sum of squared deviations from the mean rank sum:
    S = ((sum_of_total_ranks - mean_of_ranks) ** 2).sum()

    # kendall's W:
    W = (12 * S) / (m**2 * (n**3 - n))
    return W


if __name__ == "__main__":
    # compute kendall's W for the malignancy annotations:
    anno_2 = compute_kendall_w(
        pd.read_csv(
            f"{env_config.PROJECT_DIR}/analysis/annotation_agreement/annotation_files/malignancy_2.csv"
        )
    )
    anno_3 = compute_kendall_w(
        pd.read_csv(
            f"{env_config.PROJECT_DIR}/analysis/annotation_agreement/annotation_files/malignancy_3.csv"
        )
    )
    anno_4 = compute_kendall_w(
        pd.read_csv(
            f"{env_config.PROJECT_DIR}/analysis/annotation_agreement/annotation_files/malignancy_4.csv"
        )
    )
    print(f"Kendall's W for malignancy annotations:")
    print(f"Annotator 2: {anno_2}")
    print(f"Annotator 3: {anno_3}")
    print(f"Annotator 4: {anno_4}")
