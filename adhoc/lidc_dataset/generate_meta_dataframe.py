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
    # dicom

    excluded_attributes = [
        "Contrast/Bolus Agent",  # TODO THIS MUST NOT BE EXCLUDED !!
        "SOP Class UID",
        "SOP Instance UID",
        "Accession Number",
        "Referring Physician's Name",
        "Referenced SOP Instance UID",
        "Patient's Name",
        "Study Instance UID",
        "Series Instance UID",
        "Study ID",
        "Series Number",
        "Instance Number",
        "Frame of Reference UID",
        "Position Reference Indicator",
        "Slice Location",
        "Rows",
        "Columns",
        "Bits Allocated",
        "Bits Stored",
        "High Bit",
        "Longitudinal Temporal Information M",
        "Admitting Date",
        # "Scheduled Procedure Step Start Date",
        # "Scheduled Procedure Step End Date",
        # "Performed Procedure Step Start Date",
        "Placer Order Number / Imaging Servi",
        "Filler Order Number / Imaging Servi",
        "Verifying Observer Name",
        "Person Name",
        "Content Creator's Name",
        "Storage Media File-set UID",
        "Pixel Data",
    ]

    agg_attributes = [
        "Acquisition DateTime",
        "Study Time",
        "Acquisition Time",
        "Content Time",
        "Study Date",
        "Series Date",
        "Acquisition Date",
        "Content Date",
        "Overlay Date",
        "Curve Date",
        "UID",
    ]

    # DEBUGGING:
    # len([f for f in excluded_attributes if f in cif.keys()]) == len(excluded_attributes)
    # all([type(x) == list for x in cif.values()]) # make sure all values are lists

    def get_mode(x: list[list[Any]]) -> Any | None:
        counter = Counter(map(tuple, x))
        most_common = counter.most_common(1)
        return most_common

    c = 0

    all_values = {}
    for pid in tqdm(config.patient_ids):
        patient_scan_dir = get_ct_scan_slice_paths(pid, return_parent_dir=True)
        cif = collect_meta_fields_pr_scan(patient_scan_dir)

        for key, value in cif.items():
            if key in excluded_attributes:
                # skip excluded attributes
                continue

            if all([x is None for x in value]):
                # if all values are None, append None
                continue

            if key not in all_values:
                # if the attribute does not exists in the return dict yet, add it
                all_values[key] = []

            # if key in agg_attributes:
            #     # aggregate selected attributes
            #     all_values[key].append(tuple(get_mode(value)))
            # else:
            all_values[key].extend(value)

        if c == 10:
            break
        c += 1

    # pd.DataFrame(all_values)
    pd.DataFrame.from_dict(all_values)

    [len(x) for x in all_values.values()]

    len(all_values["Contrast/Bolus Agent"])
    list(all_values.keys())[18]

# %%

# THIS IS HOW THE DICT SHOULD BE CONSTRUCTED:
test = {"ID 1": [1, 2, 3], "ID 2": [4, 5, 6], "ID 3": [7, 8, 9]}
pd.DataFrame.from_dict(test, orient="index")
