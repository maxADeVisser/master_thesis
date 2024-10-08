# %%
import pickle
import re
from glob import glob
from typing import Any, Literal

from tqdm import tqdm  # type: ignore

from project_config import env_config
from utils.common_imports import *
from utils.logger_setup import logger
from utils.utils import get_ct_scan_slice_paths


def _get_dicom_keys(dicom_file: pydicom.dataset.FileDataset):
    """Returns the keys of the dicom image in a clean format.
    Util func for @parse_dicom_to_dict"""
    return_keys = []
    extract_key = lambda k: re.sub(r"\s+", " ", k.split(")")[1][1:].split(":")[0])
    for val in dicom_file.values():
        k = extract_key(str(val))[:-3]
        return_keys.append(k)
    return return_keys


# REPLACE THIS WITH THE NATIVE PYDICOM FUNCTION
# def parse_dicom_to_dict(dicom_file: pydicom.dataset.FileDataset):
#     return dict(zip(_get_dicom_keys(dicom_file), _get_dicom_vals(dicom_file)))


def collect_meta_fields_pr_scan(
    patient_scan_dir: str,
    meta_attributes: list[str] | None = None,
    return_only_dominant: bool = False,
) -> dict:
    """
    Returns all meta data dicom fields from a patient scan dir (a single scan) as a dict.
    Each meta data value from each slice is appended to a list with the key of the
    attribute as the list name.

    @meta_attributes: list of dicom attribute to fetch. Get this from the dicom_encoding_mapping.pkl file.
    """
    pid_slice_paths = sorted(glob(f"{patient_scan_dir}/*.dcm"))
    cif = {}

    if meta_attributes:
        encodings = _map_to_encoding_key(meta_attributes)
    else:
        # get all encoding values
        with open(env_config.dicom_encoding_mapping_file, "rb") as f:
            encoding_mapping = pickle.load(f)
        encodings = list(encoding_mapping.values())

    # load reversed encoding mapping
    with open(
        env_config.dicom_encoding_mapping_file.replace(".pkl", "_reverse.pkl"), "rb"
    ) as f:
        reversed_encoding_mapping = pickle.load(f)

    for slice_path in pid_slice_paths:
        dicom_file = pydicom.dcmread(
            fp=slice_path,
            force=True,
            specific_tags=encodings if encodings else None,
        )

        dicom_dict = {
            reversed_encoding_mapping[k]: v.get("Value")
            for i, (k, v) in enumerate(dicom_file.to_json_dict().items())
        }

        for k, v in dicom_dict.items():
            if k not in cif:
                # if the key is not in the dict, add it and append the value to a list
                cif[k] = []
            cif[k].append(v)

    return cif


def make_encoding_mapping_file(reverse: bool = False) -> None:
    """
    Creates the encoding mapping for the dicom file keys and saves it to a file.
    If @reverse then the mapping is reversed.
    """
    # read in a random dicom file:
    dicom_file_path = f"{env_config.DATA_DIR}/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-001.dcm"
    assert os.path.exists(dicom_file_path), "First Dicom file not found"
    dicom_file = pydicom.dcmread(
        dicom_file_path,
        force=True,
    )
    dicom_dict = dicom_file.to_json_dict()
    if reverse:
        encoding_key_mapping = dict(zip(dicom_dict.keys(), _get_dicom_keys(dicom_file)))
        save_path = env_config.dicom_encoding_mapping_file.replace(
            ".pkl", "_reverse.pkl"
        )
    else:
        encoding_key_mapping = dict(zip(_get_dicom_keys(dicom_file), dicom_dict.keys()))
        save_path = env_config.dicom_encoding_mapping_file

    with open(save_path, "wb") as f:
        pickle.dump(encoding_key_mapping, f)

    if os.path.exists(save_path):
        logger.info("INFO: Dicom encoding mapping file exists.")
    else:
        logger.info("ERROR: Dicom encoding mapping file does not exists.")


def _map_to_encoding_key(attributes: list[str]) -> list[Any]:
    """Util func to map the attributes to the encoding keys."""
    assert os.path.exists(
        env_config.dicom_encoding_mapping_file
    ), "Encoding file not found. Create it first using @make_encoding_mapping_file()"

    with open(env_config.dicom_encoding_mapping_file, "rb") as f:
        encoding_key_mapping = pickle.load(f)

    encoding = [encoding_key_mapping[key] for key in attributes]
    return encoding


def plot_meta_attribute_distribution(
    attribute: str,
    dir_save_path: str | None = None,
    granularity: Literal["scan", "slice"] = "slice",
) -> None:
    """Get values for a SPECIFIC attribute for all scans in the dataset and plots them.

    Attributes:
        @dir_save_path: str, the directory where to save the plot
    """
    encoding = _map_to_encoding_key([attribute])

    vals = []
    for pid in tqdm(env_config.patient_ids):
        patient_scan_dir: str = get_ct_scan_slice_paths(pid, return_parent_dir=True)
        cif = collect_meta_fields_pr_scan(
            patient_scan_dir,
            encoding,
            return_only_dominant=True if granularity == "scan" else False,
        )
        if attribute in cif.keys():
            match granularity:
                case "scan":
                    vals.append(cif[attribute])
                case "slice":
                    vals.extend(cif[attribute])
        else:
            continue

    plt.figure(figsize=(10, 8))
    pd.Series(vals).value_counts(ascending=False).plot(kind="hist", bins=30)
    plt.title(f"{attribute_keys} distribtion for individual images")
    plt.tight_layout()
    if dir_save_path:
        plt.savefig(f"{dir_save_path}/{attribute_keys}.png")
    plt.show()


# %%

if __name__ == "__main__":
    make_encoding_mapping_file()
    make_encoding_mapping_file(reverse=True)

    # How to use plot_meta_attribute_distribution:
    # attribute = "Exposure"
    # granularity = "scan"
    # plot_meta_attribute_distribution(attribute=attribute, granularity=granularity)

    patient_scan_dir = get_ct_scan_slice_paths(
        patient_id_dir=env_config.patient_ids[0], return_parent_dir=True
    )
    cif = collect_meta_fields_pr_scan(patient_scan_dir)


# %%
