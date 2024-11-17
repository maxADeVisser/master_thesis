# %%
import random

import torch
import torchvision.transforms as transforms

from project_config import SEED, pipeline_config

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)

DIMENSIONALITY = pipeline_config.dataset.dimensionality


def random_90_degree_rotation_2D(image: torch.Tensor) -> torch.Tensor:
    """Randomly rotate the 2D input tensor by 90 degrees with a probability"""
    k = random.choice([0, 1, 2, 3])
    return transforms.functional.rotate(image, angle=90 * k)


def random_90_degree_rotation_3D(tensor3D: torch.Tensor) -> torch.Tensor:
    """Randomly rotate the 3D input tensor by 90 degrees along each axis with a independent probability of 0.5"""
    tensor3D = torch.squeeze(tensor3D, dim=0)  # remove the channel dimension
    k_x = random.choice([0, 1, 2, 3])
    k_y = random.choice([0, 1, 2, 3])
    k_z = random.choice([0, 1, 2, 3])
    tensor3D = torch.rot90(tensor3D, k=k_x, dims=(1, 2))  # rotate around x-axis
    tensor3D = torch.rot90(tensor3D, k=k_y, dims=(0, 2))  # rotate around y-axis
    tensor3D = torch.rot90(tensor3D, k=k_z, dims=(0, 1))  # rotate around z-axis
    # add the channel dimension back
    tensor3D = torch.unsqueeze(tensor3D, dim=0)
    return tensor3D


match DIMENSIONALITY:
    case "2.5D":
        t = transforms.Compose([transforms.Lambda(random_90_degree_rotation_2D)])
    case "3D":
        t = transforms.Compose([transforms.Lambda(random_90_degree_rotation_3D)])


def apply_augmentations(nodule_roi: torch.Tensor) -> torch.Tensor:
    return t(nodule_roi)


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from data.dataset import PrecomputedNoduleROIs

    # TODO fix that indices does not match across 2.5D and 3D  precomputed datasets??
    dataset = PrecomputedNoduleROIs(
        "/Users/newuser/Documents/ITU/master_thesis/data/precomputed_rois_70C_3D",
        data_augmentation=True,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    feature, _ = next(iter(loader))
    feature.shape

    # 3D case:
    middle_slice = feature.shape[-1] // 2
    plt.imshow(feature[0, 0, :, :, middle_slice], cmap="gray")