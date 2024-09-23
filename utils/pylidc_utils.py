# %%
"""Refer to learning_pylidc.ipynb in the project for usage examples"""

import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
from pylidc.utils import consensus
from skimage.measure import find_contours

# to avoid error in pylidc due to deprecated types:
np.int = int
np.float = float


def get_scans_by_patient_id(
    patient_id: str, to_numpy: bool = True
) -> list[pl.Scan] | np.ndarray:
    """Returns the first scan for a given patient_id (i think there is only one scan per patient)"""
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    return scan.to_volume(verbose=False) if to_numpy else scan


def get_annotations_by_scan_id(scan_id: int) -> list[pl.Annotation]:
    """Returns all annotations for a given scan_id"""
    annotations = pl.query(pl.Annotation).filter(pl.Annotation.scan_id == scan_id).all()
    return annotations


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


# %%
if __name__ == "__main__":
    pid = "LIDC-IDRI-0010"
    pid_scan = get_scans_by_patient_id(pid, to_numpy=False)
    scan_annotations = get_annotations_by_scan_id(pid_scan.id)
    scan_nodules = get_nodules_by_scan(pid_scan)

    show_segmentation_consensus(pid_scan, 2)
