import torch

from model.dataset import LIDC_IDRI_DATASET
from model.MEDMnist.ResNet import load_resnet_model

experiment_id = "c50_25d_1311_1450"
fold = 0
out_dir = f"hpc/jobs/{experiment_id}/fold_{fold}"
weights_path = f"{out_dir}/model_fold{fold}.pth"
n_dims = "2.5D"

model = load_resnet_model(weights_path, in_channels=3, dims=n_dims)
model.eval()

dataset = LIDC_IDRI_DATASET(
    context_size=50,
    segmentation_configuration="none",
    n_dims=n_dims,
    nodule_df_path="preprocessing/hold_out_nodule_df.csv",
)

# validation data? use the one from the experiment

embedding = model.get_feature_vector()

# QUESTION which data to do the embedding on?
