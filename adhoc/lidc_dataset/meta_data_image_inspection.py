"""This script is used to inspect scans with different meta data attributes"""

# %%
from adhoc.lidc_dataset.meta_data_distribution import (
    collect_meta_fields_pr_scan,
    make_encoding_mapping_file,
)
from project_config import config
from utils.utils import get_ct_scan_slice_paths, load_dicom_images_from_folder
from utils.visualisations import plot_slices

# 1. need method for fetching the dominant value for a given meta data field


# %%
if __name__ == "__main__":
    patient_scan_dir = get_ct_scan_slice_paths(
        config.patient_ids[0], return_parent_dir=True
    )
    cif = collect_meta_fields_pr_scan(
        patient_scan_dir,
        meta_attributes=["Exposure", "KVP", "Focal Spo"],
        return_only_dominant=True,
    )

    attribute = "Patient Position"  # attribute to see different values for
    n_rows = 2
    n_cols = 2

    val1 = "FFS"
    val1_found = False
    start_scan_idx1 = 40

    val2 = "HFP"
    val2_found = False
    start_scan_idx2 = 80

    results = {}
    for pid in config.patient_ids:
        patient_scan_dir = get_ct_scan_slice_paths(pid, return_parent_dir=True)

        cif = collect_meta_fields_pr_scan(
            patient_scan_dir,
            meta_attributes=[attribute],
            return_only_dominant=True,
        )
        if cif[attribute] == val1 and not val1_found:
            results[pid] = load_dicom_images_from_folder(patient_scan_dir)
            val1_found = True
            plot_slices(
                num_rows=n_rows,
                num_cols=n_cols,
                start_scan_idx=start_scan_idx1,
                end_scan_idx=start_scan_idx1 + (n_rows * n_rows),
                slices=results[pid],
            )

        if cif[attribute] == val2 and not val2_found:
            results[pid] = load_dicom_images_from_folder(patient_scan_dir)
            val2_found = True
            plot_slices(
                num_rows=n_rows,
                num_cols=n_cols,
                start_scan_idx=start_scan_idx2,
                end_scan_idx=start_scan_idx2 + (n_rows * n_rows),
                slices=results[pid],
            )

        if val1_found and val2_found:
            break

# TODO just use pylidc....
# %%
