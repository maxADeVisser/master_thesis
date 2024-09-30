# %%
import pickle
from collections import Counter

from adhoc.lidc_dataset.meta_data_distribution import collect_meta_fields_pr_scan
from project_config import env_config
from utils.common_imports import *
from utils.utils import get_ct_scan_slice_paths


def main() -> None:
    with open("utils/dicom_encoding_mapping.pkl", "rb") as f:
        dicom_attribrutes = list(pickle.load(f).keys())

    c = 0
    scan_values = {}
    for pid in tqdm(env_config.patient_ids):
        patient_scan_dir = get_ct_scan_slice_paths(pid, return_parent_dir=True)
        cif = collect_meta_fields_pr_scan(patient_scan_dir)
        scan_values[pid] = []

        for attr in dicom_attribrutes:
            if attr in env_config.excluded_dicom_attributes:
                # skip excluded attributes
                continue

            if attr not in cif:
                # if the current attribute does not exists in the cif, append None
                scan_values[pid].append(None)
                continue

            if all([x is None for x in cif[attr]]):
                # if all values for the attribute are None, append None
                scan_values[pid].append(None)
                continue

            if all(x == cif[attr][0] for x in cif[attr]):
                # if all values are the same, append only the first value
                scan_values[pid].append(cif[attr][0][0])
                continue

            # else:
            scan_values[pid].append(cif[attr])

        # if c == 10:
        #     break
        # c += 1

    columns = [
        x for x in dicom_attribrutes if x not in env_config.excluded_dicom_attributes
    ]
    df = pd.DataFrame.from_dict(scan_values, orient="index", columns=columns)

    # DO MORE PROCESSING AND CLEANING OF THE DATAFRAME HERE ...

    df.to_csv("out/lidc_meta_data.csv")


# %%
if __name__ == "__main__":
    main()

    # How to read:
    df = pd.read_csv("out/lidc_meta_data.csv", index_col=0)
    df
    from sklearn.metrics import mutual_info_score

    mutual_info_score(df[""])

    # get the columns without any missing values
    df.dropna(axis="columns", how="all")
# %%
