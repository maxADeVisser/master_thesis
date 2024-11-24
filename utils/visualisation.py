# %%
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylidc as pl
import seaborn as sns
from pylidc.utils import consensus, find_contours, volume_viewer

# from project_config import env_config, pipeline_config
from utils.data_models import load_fold_from_json


def plot_val_error_distribution(
    validation_erros: list[int], out_dir: str | None = None, show: bool = False
) -> None:
    """
    Plots the distribution of validation errors.
    """
    res = Counter(validation_erros)
    all_numbers = list(res.keys())
    counts = list(res.values())
    plt.bar(all_numbers, counts)
    for i in range(len(all_numbers)):
        plt.text(all_numbers[i], counts[i], str(counts[i]), ha="center", va="bottom")
    plt.xlabel("Malignancy Error")
    plt.ylabel("Frequency")
    if out_dir:
        plt.savefig(f"{out_dir}/error_distribution.png")
    if not show:
        plt.close()
        return
    else:
        plt.show()


def plot_loss(
    train_losses: list[float], val_losses, out_dir: str | None = None
) -> None:
    """
    Plots the training and validation loss across epochs.
    """
    plt.plot(train_losses, label="train_loss", color="red")
    plt.plot(val_losses, label="val_loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{out_dir}/loss_plot.png")
    plt.close()


def plot_fold_results(
    experiment_id: str, fold_num: int, rolling_window: int = 10, epochs_dampen: int = 2
) -> None:
    """
    Visualises a fold results downloaded from the HPC.
    @experiment_id: The ID of the experiment.
    @fold_num: The fold number.
    @rolling_window: The window size for the rolling average.
    @epochs_dampen: The number of epochs to dampen the start of the loss plot.
    """
    model_info = experiment_id.split("_")[:2]
    model_context_size = model_info[0]
    model_dimensionality = model_info[1]
    exp_path = f"hpc/jobs/{experiment_id}"
    fold_path = f"{exp_path}/fold_{fold_num}"
    fold = load_fold_from_json(f"{fold_path}/fold{fold_num}_{experiment_id}.json")
    fold_num_epochs = len(fold.val_losses)

    epochs = range(fold_num_epochs)

    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[1, 1, 1, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, :])

    if epochs_dampen > 0:
        # set the first @epochs_damped epochs where the loss is very high to the mean of the rest of the epoch losses
        dampen = (
            lambda vals: [float(np.mean(vals[epochs_dampen:]))] * epochs_dampen
            + vals[epochs_dampen:]
        )
        fold.val_maes = dampen(fold.val_maes)
        fold.val_mses = dampen(fold.val_mses)
        fold.val_AUC_filtered = dampen(fold.val_AUC_filtered)
        fold.val_AUC_ovr = dampen(fold.val_AUC_ovr)
        fold.val_accuracies = dampen(fold.val_accuracies)
        fold.val_binary_accuracies = dampen(fold.val_binary_accuracies)
        # TODO plot the binary accuracies
        fold.val_cwces = dampen(fold.val_cwces)
        fold.train_losses = dampen(fold.train_losses)
        fold.val_losses = dampen(fold.val_losses)

    # AX1 -- MAE
    sns.lineplot(x=epochs, y=fold.val_maes, ax=ax1, color="green")
    sns.lineplot(
        x=epochs,
        y=get_rolling_avg(fold.val_maes, rolling_window),
        ax=ax1,
        alpha=0.5,
        color="red",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE")
    ax1.grid()

    # AX2 -- MSE
    sns.lineplot(x=epochs, y=fold.val_mses, ax=ax2, color="green")
    sns.lineplot(
        x=epochs,
        y=get_rolling_avg(fold.val_mses, rolling_window),
        ax=ax2,
        alpha=0.5,
        color="red",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.grid()

    # AX3 -- AUC Filtered
    sns.lineplot(
        x=epochs,
        y=fold.val_AUC_filtered,
        ax=ax3,
        color="green",
    )
    sns.lineplot(
        x=epochs,
        y=get_rolling_avg(fold.val_AUC_filtered, rolling_window),
        ax=ax3,
        alpha=0.5,
        color="red",
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("AUC Filtered")
    ax3.grid()

    # AX4 -- AUC OVR
    sns.lineplot(
        x=epochs,
        y=fold.val_AUC_ovr,
        ax=ax4,
        color="green",
    )
    sns.lineplot(
        x=epochs,
        y=get_rolling_avg(fold.val_AUC_ovr, rolling_window),
        ax=ax4,
        alpha=0.5,
        color="red",
    )
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("AUC OVR")
    ax4.grid()

    # AX5 -- Validation Accuracy
    sns.lineplot(
        x=epochs,
        y=fold.val_accuracies,
        ax=ax5,
        color="green",
    )
    sns.lineplot(
        x=epochs,
        y=get_rolling_avg(fold.val_accuracies, rolling_window),
        ax=ax5,
        alpha=0.5,
        color="red",
    )
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Validation Accuracy")
    ax5.grid()

    # AX6 -- Class-wise Calibration Error
    sns.lineplot(
        x=epochs,
        y=fold.val_cwces,
        ax=ax6,
        color="green",
    )
    sns.lineplot(
        x=epochs,
        y=get_rolling_avg(fold.val_cwces, rolling_window),
        ax=ax6,
        alpha=0.5,
        color="red",
    )
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Class-wise Calibration Error")
    ax6.grid()

    # AX7 -- Loss Plot
    sns.lineplot(
        x=epochs,
        y=fold.train_losses,
        ax=ax7,
        label="Train Loss",
    )
    sns.lineplot(
        x=epochs,
        y=fold.val_losses,
        ax=ax7,
        label="Val Loss",
    )
    ax7.set_title(f"Train vs. Validation Loss")
    ax7.set_xlabel("Epoch")
    ax7.set_ylabel("CORN Loss")
    ax7.grid()

    plt.suptitle(
        f"Model {model_context_size} | {model_dimensionality} - Fold {fold_num} Validation Metrics"
    )
    plt.savefig(f"{fold_path}/fold_metric_results.png")
    plt.show()


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


# def plot_scan_hounsfield_histogram(
#     pids: list[str],
#     bounds: tuple[int, int] = (
#         pipeline_config.dataset.clipping_bounds[0],
#         pipeline_config.dataset.clipping_bounds[1],
#     ),
#     bins: int = 80,
# ) -> None:
#     """
#     Plots the histogram of Hounsfield units in a CT scan
#     @scan - a 3D numpy array of the CT scan
#     """
#     scan = get_scans_by_patient_id(pids[0], to_numpy=True)

#     for pid in tqdm(range(1, len(pids)), desc="Loading Scans"):
#         next_pid_scan = get_scans_by_patient_id(
#             env_config.patient_ids[pid], to_numpy=True
#         )
#         scan = np.concatenate([scan, next_pid_scan], axis=2)

#     print("Plotting...")
#     plt.title(f"Hounsfield Units (HU) Distribution for {len(pids)} Scans")
#     plt.hist(scan.flatten(), bins=bins, color="c")
#     plt.axvline(x=bounds[0], color="r", linestyle="--", label="Lower Bound")
#     plt.axvline(x=bounds[1], color="b", linestyle="--", label="Upper Bound")
#     plt.legend()
#     plt.xlabel("Hounsfield Units (HU)")
#     plt.ylabel("Frequency")
#     plt.show()


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


def get_rolling_avg(values: list[float], w: int = 10) -> list[float]:
    """Util func for plotting"""
    return pd.Series(values).rolling(window=w).mean().to_list()


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

    # # TESTING
    # plot_scan_hounsfield_histogram(env_config.patient_ids[:50], bins=80)

    # # TESTING
    # pid = "LIDC-IDRI-0010"
    # pid_scan = get_scans_by_patient_id(pid, to_numpy=False)
    # show_segmentation_consensus(pid_scan, 2)

    # # TESTING
    # visualise_scan_interactively("LIDC-IDRI-0010")

    from utils.data_models import ExperimentAnalysis

    with open("experiment_analysis_parameters.json", "r") as f:
        config = ExperimentAnalysis.model_validate(json.load(f))

    dampen_epochs = 3
    plot_fold_results(
        config.experiment_id, fold_num=config.analysis.fold, epochs_dampen=dampen_epochs
    )
# %%
