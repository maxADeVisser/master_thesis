# import torch
# import torchio as tio

# from project_config import SEED

# torch.manual_seed(SEED)


# flip along each axis:
# probability of flipping is computed for each axis independently
# see https://torchio.readthedocs.io/transforms/augmentation.html#randomflip
# transforms = tio.Compose(
#     [
# tio.RandomFlip(
#     axes=["Anterior", "Posterior"],  # flip along the y-axis
#     flip_probability=0.5,
# ),
# tio.RandomFlip(
#     axes=["Left", "Right", "Anterior", "Posterior", "Posteior", "Superior"],
#     flip_probability=0.4,
# ),
# fixed 90 degree rotation along each axis
# tio.RandomAffine(
#     scales=1,  # No scaling
#     degrees=(90, 90, 90),  # 90 degree rotation along each axis
#     isotropic=False,
#     default_pad_value=0,
#     translation=0,
#     center="image",
# )
#     ]
# )


# def apply_augmentations(nodule_roi: torch.Tensor) -> torch.Tensor:
#     return transforms(nodule_roi)


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     from model.dataset import LIDC_IDRI_DATASET

#     dataset = LIDC_IDRI_DATASET(
#         img_dim=70, segmentation_configuration="none", augment_scans=False
#     )
#     feature, label = dataset.__getitem__(0)
#     feature.shape
#     middle_slice = feature.shape[-1] // 2
#     plt.imshow(feature[0][:, :, middle_slice], cmap="gray")
#     plt.imshow(apply_augmentations(feature)[0][:, :, middle_slice], cmap="gray")
