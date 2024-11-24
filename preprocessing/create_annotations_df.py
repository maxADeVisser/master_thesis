"""This file is for creating grouping the nodules based on the number of annotations they have,
and then explore the tuple into individual columns for easier analysis.
This is used for the agreement analysis"""

from preprocessing.create_nodule_df import create_nodule_df
from utils.common_imports import *

df = pd.read_csv("out/annotation_df.csv")
out_path = "out/annotation_files"


def make_annotation_file(count: int, attribute: str) -> None:
    filt_df = df.query(f"nodule_annotation_count == {count}")
    filt_df = filt_df[f"{attribute}_scores"].apply(lambda x: eval(x)).tolist()
    cols = [f"{attribute}_ann{i}" for i in range(1, count + 1)]
    pd.DataFrame(filt_df, columns=cols).to_csv(
        f"{out_path}/{attribute}_{count}.csv", index=False
    )


if __name__ == "__main__":
    counts = [2, 3, 4]
    for c in counts:
        make_annotation_file(c, "malignancy")
        make_annotation_file(c, "subtlety")
