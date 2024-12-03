from glob import glob

from tqdm import tqdm

from project_config import env_config
from utils.common_imports import *
from utils.utils import get_ct_scan_slice_paths

results = {
    "scan_id": [],
    "manufacturer": [],
    "manufacturer_model_name": [],
    "x-ray tube current": [],
    "kvp": [],
    "exposure time": [],
    "exposure": [],
    "software_versions": [],
}
for pid in tqdm(env_config.patient_ids):
    patient_scan_dir = get_ct_scan_slice_paths(pid, return_parent_dir=True)
    ds = pydicom.dcmread(glob(f"{patient_scan_dir}/*.dcm")[0])

    results["scan_id"].append(pid)

    results["manufacturer"].append(ds.get("Manufacturer"))
    results["manufacturer_model_name"].append(ds.get("ManufacturerModelName"))
    results["x-ray tube current"].append(ds.get("XRayTubeCurrent"))
    results["kvp"].append(ds.get("KVP"))
    results["exposure time"].append(ds.get("ExposureTime"))
    results["exposure"].append(ds.get("Exposure"))
    results["software_versions"].append(ds.get("SoftwareVersions"))

df = pd.DataFrame(results)

# group together attributes
df["manufacturer_model_name"] = df["manufacturer_model_name"].apply(
    lambda x: "Brilliance" if x.startswith("Brilliance") else x
)

df["manufacturer_model_name"] = df["manufacturer_model_name"].apply(
    lambda x: "LightSpeed" if x.startswith("LightSpeed") else x
)

df["manufacturer_model_name"] = df["manufacturer_model_name"].apply(
    lambda x: "Revolution" if x.startswith("Revolution") else x
)

df["manufacturer_model_name"] = df["manufacturer_model_name"].apply(
    lambda x: "Sensation" if x.startswith("Sensation") else x
)

df["manufacturer_model_name"] = df["manufacturer_model_name"].apply(
    lambda x: "Emotion" if x.startswith("Emotion") else x
)

# SOFTWARE VERSIONS
df["software_versions"] = df["software_versions"].apply(
    lambda x: (
        "Ads Application Package"
        if (x is not None) and (x.startswith("Ads Application Package"))
        else x
    )
)

df["software_versions"] = df["software_versions"].apply(
    lambda x: "LightSpeed" if (x is not None) and (x.startswith("LightSpeed")) else x
)

df.to_csv(f"{env_config.PROJECT_DIR}/preprocessing/dicom_meta_data.csv", index=False)
