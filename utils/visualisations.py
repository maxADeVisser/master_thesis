import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom

# from matplotlib import animation, rc

# rc("animation", html="jshtml")


# see https://www.kaggle.com/code/grolakr/hesaplanabilir-sinirbilim-proje-2
# def create_animation(ims):
#     fig = plt.figure(figsize=(6, 6))
#     plt.axis("off")
#     im = plt.imshow(ims[0], cmap="gray")

#     def animate_func(i):
#         im.set_array(ims[i])
#         return [im]

#     return animation.FuncAnimation(
#         fig, animate_func, frames=len(ims), interval=1000 // 24
#     )


def plot_slices(
    num_rows: int,
    num_cols: int,
    start_scan_idx: int,
    end_scan_idx: int,
    scan_dir: str,
    save_path: str | None = None,
) -> None:
    """Plot a montage of @num_rows * @num_cols CT slices.
    Source: https://keras.io/examples/vision/3D_image_classification/"""
    assert (
        end_scan_idx - start_scan_idx == num_rows * num_cols
    ), "The number of selected slices must equal the number of rows and columns multiplied"

    slices = [pydicom.dcmread(scan_dir + f) for f in os.listdir(scan_dir)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    slices = np.stack([s.pixel_array for s in slices])

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
    plot_slices(
        num_rows=4,
        num_cols=10,
        start_scan_idx=30,
        end_scan_idx=70,
        scan_dir="/Users/newuser/Documents/ITU/master_thesis/data/lung_data/manifest-1725363397135/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/",
        save_path="out/test.png",
    )
