"""This file is for creating grouping the nodules based on the number of annotations they have,
and then explore the tuple into individual columns for easier analysis.
This is used for the agreement analysis"""

from project_config import env_config
from utils.common_imports import *

df = pd.read_csv(f"{env_config.PROJECT_DIR}/preprocessing/annotation_df.csv")
out_path = "out/annotation_files"


def make_annotation_file(annotator_count: int, lidc_annotation_variable: str) -> None:
    filt_df = df.query(f"nodule_annotation_count == {annotator_count}")
    filt_df = (
        filt_df[f"{lidc_annotation_variable}_scores"].apply(lambda x: eval(x)).tolist()
    )
    cols = [f"{lidc_annotation_variable}_ann{i}" for i in range(1, annotator_count + 1)]
    pd.DataFrame(filt_df, columns=cols).to_csv(
        f"{out_path}/{lidc_annotation_variable}_{annotator_count}.csv", index=False
    )


if __name__ == "__main__":
    counts = [2, 3, 4]
    for c in counts:
        make_annotation_file(c, "malignancy")
        make_annotation_file(c, "subtlety")
