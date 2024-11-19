# %%
import json

import fiftyone as fo

# %%
nodule_roi_dir_path = (
    "/Users/newuser/Documents/ITU/master_thesis/data/middle_slice_images_c70"
)

# Only run once:
# dataset = fo.Dataset.from_images_patt(
#     images_patt=f"{nodule_roi_dir_path}/*.jpg", name="LIDC-IDRI_Nodule_ROIs"
# )
dataset = fo.load_dataset("LIDC-IDRI_Nodule_ROIs")
# %%

annotations_path = "data/annotations70.json"
with open(annotations_path, "r") as f:
    annotations = json.load(f)

for sample in dataset:
    # Store classification in a field name of your choice
    instance_data = annotations[sample.filepath]
    sample["malignancy_score"] = fo.Classification(
        label=str(instance_data["malignancy"])
    )
    # sample["prediction"] = fo.Classification(
    #     label=label, confidence=1.0
    # )  # TODO add confidence
    sample["nodule_id"] = instance_data["nodule_id"]
    sample["subtlety"] = instance_data["subtlety"]
    sample["cancer_label"] = instance_data["cancer_label"]
    sample["ann_mean_volume"] = instance_data["ann_mean_volume"]
    sample["ann_mean_diameter"] = instance_data["ann_mean_diameter"]
    sample.save()

session = fo.launch_app(dataset=dataset)
session.wait()
