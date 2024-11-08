# %%
import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
from pylidc.utils import consensus, find_contours, volume_viewer
from tqdm import tqdm

from project_config import env_config, pipeline_config
from utils.utils import get_scans_by_patient_id


def plot_val_error_distribution(validation_erros: list[int], out_dir: str) -> None:
    """Plots the distribution of validation errors."""
    plt.hist(
        validation_erros,
        bins=range(min(validation_erros), max(validation_erros) + 2),
        edgecolor="black",
        align="left",
    )
    plt.axvline(
        x=np.mean(validation_erros),
        color="r",
        linestyle="--",
        label=f"Mean Error: {np.mean(validation_erros):.2f}",
    )
    plt.axvline(
        x=np.median(validation_erros),
        color="b",
        linestyle="--",
        label=f"Median Error: {np.median(validation_erros):.2f}",
    )
    plt.legend()
    plt.xlabel("Validation Error")
    plt.savefig(f"{out_dir}/loss_plot.png")


def plot_loss(avg_epoch_losses: list[float], val_loss, out_dir: str) -> None:
    """Plots the training and validation loss across epochs."""
    plt.plot(avg_epoch_losses, label="train_loss", color="red")
    plt.plot(val_loss, label="val_loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{out_dir}/loss_plot.png")


def show_segmentation_consensus(
    scan: pl.Scan, nodule_idx: int, clevel: float = 0.5
) -> None:
    """
    Display the consensus segmentation for a nodule in a scan.

    Args:
        @scan: The pylidc.Scan object.
        @nodule_idx: The index of the nodule in the scan.
        @clevel: The consensus fraction. For example, if clevel=0.5, then a voxel will have value 1 (True) in the returned boolean volume when at least 50% of the annotations have a value of 1 at that voxel, and 0 (False) otherwise.
    Source: https://pylidc.github.io/tuts/consensus.html
    """
    # Query for a scan, and convert it to an array volume.
    vol = scan.to_volume(verbose=False)

    # Cluster the annotations for the scan, and grab one.
    nods = scan.cluster_annotations()
    anns = nods[nodule_idx]

    # Perform a consensus consolidation and 50% agreement level.
    # We pad the slices to add context for viewing.
    cmask, cbbox, masks = consensus(
        anns, clevel=clevel, pad=[(20, 20), (20, 20), (0, 0)]
    )

    # Get the central slice of the computed bounding box.
    k = int(0.5 * (cbbox[2].stop - cbbox[2].start))

    # Set up the plot.
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(vol[cbbox][:, :, k], cmap=plt.cm.gray, alpha=0.5)

    # Plot the annotation contours for the kth slice.
    colors = ["r", "g", "b", "y"]
    for j in range(len(masks)):
        for c in find_contours(masks[j][:, :, k].astype(float), 0.5):
            label = "Annotation %d" % (j + 1)
            plt.plot(c[:, 1], c[:, 0], colors[j], label=label)

    # Plot the 50% consensus contour for the kth slice.
    for c in find_contours(cmask[:, :, k].astype(float), 0.5):
        plt.plot(c[:, 1], c[:, 0], "--k", label="50% Consensus")

    ax.axis("off")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_scan_hounsfield_histogram(
    pids: list[str],
    bounds: tuple[int, int] = (
        pipeline_config.dataset.clipping_bounds[0],
        pipeline_config.dataset.clipping_bounds[1],
    ),
    bins: int = 80,
) -> None:
    """
    Plots the histogram of Hounsfield units in a CT scan
    @scan - a 3D numpy array of the CT scan
    """
    scan = get_scans_by_patient_id(pids[0], to_numpy=True)

    for pid in tqdm(range(1, len(pids)), desc="Loading Scans"):
        next_pid_scan = get_scans_by_patient_id(
            env_config.patient_ids[pid], to_numpy=True
        )
        scan = np.concatenate([scan, next_pid_scan], axis=2)

    print("Plotting...")
    plt.title(f"Hounsfield Units (HU) Distribution for {len(pids)} Scans")
    plt.hist(scan.flatten(), bins=bins, color="c")
    plt.axvline(x=bounds[0], color="r", linestyle="--", label="Lower Bound")
    plt.axvline(x=bounds[1], color="b", linestyle="--", label="Upper Bound")
    plt.legend()
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()


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


def visualise_scan_interactively(pid: str) -> None:
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    volume_viewer(scan.to_volume(verbose=False))


# %%
if __name__ == "__main__":
    # TESTING
    """Need to Load a scan from a folder and returns a 3D numpy array of all the scans stacked together
    in the shape: (n_slices, width, height):"""
    # slices = ...
    # plot_slices(
    #     num_rows=4,
    #     num_cols=10,
    #     start_scan_idx=30,
    #     end_scan_idx=70,
    #     slices=slices,
    #     # save_path="out/test.png",
    # )

    # TESTING
    plot_scan_hounsfield_histogram(env_config.patient_ids[:50], bins=80)

    # TESTING
    pid = "LIDC-IDRI-0010"
    pid_scan = get_scans_by_patient_id(pid, to_numpy=False)
    show_segmentation_consensus(pid_scan, 2)

    # TESTING
    visualise_scan_interactively("LIDC-IDRI-0010")
# %%
