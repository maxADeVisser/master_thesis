# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylidc as pl
from pylidc.utils import volume_viewer

from model.dataset import LIDC_IDRI_DATASET


def plot_slices(
    num_rows: int,
    num_cols: int,
    start_scan_idx: int,
    end_scan_idx: int,
    slices: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Plot a montage of @num_rows * @num_cols CT slices.
    Source: https://keras.io/examples/vision/3D_image_classification/"""
    assert (
        end_scan_idx - start_scan_idx == num_rows * num_cols
    ), "The number of selected slices must equal the number of rows and columns multiplied"

    # the number of selected slices must equal the number of rows and columns multiplied:
    slices = slices[start_scan_idx:end_scan_idx]
    _, width, height = slices.shape

    slices = np.reshape(slices, (num_rows, num_cols, width, height))
    rows_data, columns_data = slices.shape[0], slices.shape[1]
    heights = [slc[0].shape[0] for slc in slices]
    widths = [slc.shape[1] for slc in slices[0]]
    fig_width = 12.0
    fig_height = fig_width * (sum(heights) / sum(widths))

    _, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(slices[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=400)
    plt.show()


if __name__ == "__main__":
    # TESTING
    """Need to Load a scan from a folder and returns a 3D numpy array of all the scans stacked together
    in the shape: (n_slices, width, height):"""
    slices = ...
    plot_slices(
        num_rows=4,
        num_cols=10,
        start_scan_idx=30,
        end_scan_idx=70,
        slices=slices,
        # save_path="out/test.png",
    )

# %%
