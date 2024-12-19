import pandas as pd


def compute_kendall_w(annotation_data: pd.DataFrame) -> pd.DataFrame:
    """Computes Kendall's W assuming that there are no rank ties."""

    # n: number of items (nodules) and m: number of raters (annotations):
    n, m = annotation_data.shape

    # Rank the ratings for each subject:
    ranked_data = annotation_data.rank(axis=0, method="average")

    # Compute the sum of ranks for each item:
    sum_of_total_ranks = ranked_data.sum(axis=1)  # R_i

    # Compute the mean of the sum of total ranks:
    mean_of_ranks = sum_of_total_ranks.mean()  # \bar{R}

    # Compute the sum of squared deviations from the mean rank sum:
    S = ((sum_of_total_ranks - mean_of_ranks) ** 2).sum()

    #  Calculate Kendall's W:
    W = (12 * S) / (m**2 * (n**3 - n))

    return W


if __name__ == "__main__":
    anno_2 = pd.read_csv("out/annotation_files/malignancy_2.csv")
    anno_3 = pd.read_csv("out/annotation_files/malignancy_3.csv")
    anno_4 = pd.read_csv("out/annotation_files/malignancy_4.csv")

    compute_kendall_w(anno_2)
    compute_kendall_w(anno_3)
    compute_kendall_w(anno_4)
