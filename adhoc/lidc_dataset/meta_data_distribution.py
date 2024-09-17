# %%
import pickle
import re
from glob import glob
from typing import Any

from tqdm import tqdm

from utils.common_imports import *
from utils.utils import get_ct_scan_slice_paths


def _get_dicom_vals(dicom_file: pydicom.dataset.FileDataset) -> list[Any]:
    """Returns the values of the dicom image in a clean format.
    Util func for @parse_dicom_to_dict"""
    vals = list(dicom_file.to_json_dict().values())
    return_vals = []
    for v in vals:
        if "Value" not in v:
            # handle empty values
            return_vals.append(None)
            continue

        if len(v["Value"]) == 1:
            # if there is only one value, return the value itself (not as a list):
            return_vals.append(v["Value"][0])
        else:
            # if there are multiple values, return them as a list:
            return_vals.append(v["Value"])

    return return_vals


def _get_dicom_keys(dicom_file: pydicom.dataset.FileDataset):
    """Returns the keys of the dicom image in a clean format.
    Util func for @parse_dicom_to_dict"""
    return_keys = []
    extract_key = lambda k: re.sub(r"\s+", " ", k.split(")")[1][1:].split(":")[0])
    for val in dicom_file.values():
        k = extract_key(str(val))[:-3]
        return_keys.append(k)
    return return_keys


def parse_dicom_to_dict(dicom_file: pydicom.dataset.FileDataset):
    return dict(zip(_get_dicom_keys(dicom_file), _get_dicom_vals(dicom_file)))


def collect_meta_fields(
    patient_scan_dir: str, meta_attribute: str | None = None
) -> dict:
    """
    Fetches all meta data dicom fields from a patient scan dir (a single scan).
    Each meta data value from each slice is appended to a list with the key of the
    attribute as the list name.
    # TODO add a list input of attributes that only fetches those if specified
    """
    patient_scan_paths = sorted(glob(f"{patient_scan_dir}/*.dcm"))
    collected_meta_fields = {}

    for s in patient_scan_paths:
        with pydicom.dcmread(
            fp=s, force=True, specific_tags=[meta_attribute]
        ) as dicom_image:
            # NOTE: the print statement (or pixel_array access) forces the image to be read into memory. Do not remove.
            try:
                dicom_image.pixel_array
            except Exception as e:
                print(dicom_image)

            dicom_dict = parse_dicom_to_dict(dicom_image)

        for k, v in dicom_dict.items():
            if k not in collected_meta_fields:
                collected_meta_fields[k] = []
            collected_meta_fields[k].append(v)

    return collected_meta_fields


# %%

if __name__ == "__main__":
    # MAKE THE DICOM ENCODING MAPPING FILE:
    # read in a random dicom file:
    # dicom_file = pydicom.dcmread(
    #     "/Users/newuser/Documents/ITU/master_thesis/data/lung_data/manifest-1725363397135/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-001.dcm",
    #     force=True,
    # )
    # dicom_dict = dicom_file.to_json_dict()
    # encoding_key_mapping = dict(zip(_get_dicom_keys(dicom_file), dicom_dict.keys()))
    # with open("utils/dicom_encoding_mapping.pkl", "wb") as f:
    #     pickle.dump(encoding_key_mapping, f)

    # ----------------------------------------------------------

    # LOAD THE DICOM ENCODING MAPPING FILE:
    with open("utils/dicom_encoding_mapping.pkl", "rb") as f:
        encoding_key_mapping = pickle.load(f)

    patient_scan_dir = get_ct_scan_slice_paths(
        config.patient_ids[0], return_parent_dir=True
    )
    cif = collect_meta_fields(patient_scan_dir, encoding)
    collect_meta_fields()

    attribute = "Exposure"

    encoding = encoding_key_mapping[attribute]

    vals = []
    for pid in tqdm(config.patient_ids):
        patient_scan_dir: str = get_ct_scan_slice_paths(pid, return_parent_dir=True)
        cif = collect_meta_fields(patient_scan_dir, encoding)
        if attribute in cif.keys():
            vals.extend(cif[attribute])
        else:
            continue

    plt.figure(figsize=(10, 8))
    pd.Series(vals).value_counts(ascending=False).plot(kind="hist", bins=30)
    plt.title(f"{attribute} distribtion for individual images")
    plt.tight_layout()
    plt.savefig(f"out/figures/meta_data_dists/{attribute}.png")
    plt.show()


# %%
