import torch
import torchio as tio

transforms = tio.Compose(
    [
        # tio.RandomElasticDeformation(num_control_points=7, locked_borders=2, p=1.0),
        tio.RandomFlip(
            axes=["Anterior", "Posterior"], flip_probability=0.5
        ),  # flip along the y-axis (get front and back views)
    ]
)


def apply_augmentations(nodule_roi: torch.Tensor) -> torch.Tensor:
    return transforms(nodule_roi)


if __name__ == "__main__":
    # Testing:
    import matplotlib.pyplot as plt

    from model.dataset import LIDC_IDRI_DATASET

    dataset = LIDC_IDRI_DATASET(img_dim=70, segmentation_configuration="none")
    feature, label = dataset.__getitem__(0)
    feature.shape
    middle_slice = feature.shape[-1] // 2
    plt.imshow(apply_augmentations(feature)[0][:, :, middle_slice], cmap="gray")
