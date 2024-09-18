# %%
from collections import Counter

from adhoc.lidc_dataset.meta_data_distribution import collect_meta_fields_pr_scan
from project_config import config
from utils.common_imports import *
from utils.utils import get_ct_scan_slice_paths

# %%
if __name__ == "__main__":
    # dicom = pydicom.dcmread(
    #     "/Users/newuser/Documents/ITU/master_thesis/data/lung_data/manifest-1725363397135/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192/1-001.dcm"
    # )

    # c = 0

    # with open("utils/dicom_encoding_mapping.pkl", "rb") as f:
    #     dicom_attribrutes = list(pickle.load(f).keys())

    scan_values = {}
    for pid in tqdm(config.patient_ids):
        patient_scan_dir = get_ct_scan_slice_paths(pid, return_parent_dir=True)
        cif = collect_meta_fields_pr_scan(patient_scan_dir)
        scan_values[pid] = []

        for key, values in cif.items():  # for each attribute in the scan: ...
            if key in config.excluded_dicom_attributes:
                # skip excluded attributes
                continue

            if all([x is None for x in values]):
                # if all values for the attribute are None, append None
                scan_values[pid].append(None)
                continue

            if all(x == values[0] for x in values):
                # if all values are the same, append only one value
                scan_values[pid].append(values[0][0])
                continue

            # else just append all values
            scan_values[pid].append(values)

        # if c == 10:
        #     break
        # c += 1

    columns = [x for x in list(cif.keys()) if x not in config.excluded_dicom_attributes]
    df = pd.DataFrame.from_dict(scan_values, orient="index", columns=columns)
    df
    df.to_csv("out/lidc_meta_data.csv")

    pd.read_csv("out/lidc_meta_data.csv", index_col=0)

    # DEBUGGING:
    # len(scan_values["Contrast/Bolus Agent"])
    # list(scan_values.keys())[18]

    # NOT USED ANYWAY I THINK
    # def get_mode(x: list[list[Any]]) -> Any | None:
    #     counter = Counter(map(tuple, x))
    #     most_common = counter.most_common(1)
    #     return most_common


# %%
