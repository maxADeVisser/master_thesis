# %%
import pickle
import re
from glob import glob
from typing import Any

from tqdm import tqdm

from utils.common_imports import *
from utils.utils import get_scan_directory_path_by_patient_id


def _get_dicom_vals(dicom_file: pydicom.dataset.FileDataset) -> list[Any]:
    """Returns the values of the dicom image in a clean format.
    Util func for @parse_dicom_to_dict"""
    vals = [v for v in dicom_file.to_json_dict().values()]
    return_vals = []
    for v in vals:
        if isinstance(v, dict):
            if "Value" in v:
                if len(v["Value"]) == 1:
                    return_vals.append(v["Value"][0])
                else:
                    return_vals.append(v["Value"])
            else:
                return_vals.append(None)
        else:
            raise ValueError(f"Value is not a dict: {type(v)}")

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
            # print(dicom_image)
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
    attribute = "Focal Spo"

    with open("utils/dicom_encoding_mapping.pkl", "rb") as f:
        encoding_key_mapping = pickle.load(f)

    encoding = encoding_key_mapping[attribute]

    vals = []
    for pid in tqdm(config.patient_ids):
        patient_scan_dir: str = get_scan_directory_path_by_patient_id(pid)
        cif = collect_meta_fields(patient_scan_dir, encoding)
        if attribute in cif.keys():
            vals.extend(cif[attribute])
        else:
            continue

    # plt.figure(figsize=(10, 8))
    pd.Series(vals).value_counts(ascending=False).plot(kind="bar")
    plt.title(f"{attribute} distribtion for individual images")
    plt.tight_layout()
    plt.savefig(f"out/figures/meta_data_dists/{attribute}.png")
    plt.show()


# %%
