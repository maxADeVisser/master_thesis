# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import shap.maskers
import torch

from data.dataset import PrecomputedNoduleROIs
from model.ResNet import get_unconditional_probas, load_resnet_model


class ResNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ResNetWrapper, self).__init__()
        self.model = model

    def forward(self, x: np.ndarray):
        """Expects input of shape (batch_size, height, width, channels)"""
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float()
        logits = self.model(x)

        # Convert logits to probabilities:
        probs = get_unconditional_probas(logits)
        return probs


dataset = PrecomputedNoduleROIs(
    "/Users/newuser/Documents/ITU/master_thesis/data/precomputed_resampled_rois_50C_2.5D",
    data_augmentation=False,
    dimensionality="2.5D",
    center_mask_size=None,
)
preds = pd.read_csv("model/predictions/c30_25D_2411_1543/pred_nodule_df_fold0.csv")

# get 3 nodule ids with highest and lowest confidences
top3_conf = preds.sort_values("confidence", ascending=False).head(3)
low3_conf = preds.sort_values("confidence", ascending=True).head(3)

# Get 3 nodules with the largest and smallest errors
preds["error"] = abs(preds["pred"] - preds["malignancy_consensus"])
top3_error = preds.sort_values("error", ascending=False).head(3)
low3_error = preds.sort_values("error", ascending=True).head(3)


# Construct test batch for nodules selected by the confidence and error criteria:
def get_test_batch(indices: pd.Series, dataset: PrecomputedNoduleROIs):
    test = []
    nodule_ids = []
    for idx in indices.index:
        nodule, _, nodule_id = dataset.__getitem__(idx)
        test.append(nodule)
        nodule_ids.append(nodule_id)
    return torch.stack(test).permute(0, 2, 3, 1).numpy(), nodule_ids


# 1. Top 3 confidence
top3_conf_batch, top3_conf_ids = get_test_batch(top3_conf, dataset)
low3_conf_batch, low3_conf_ids = get_test_batch(low3_conf, dataset)
top3_error_batch, top3_error_ids = get_test_batch(top3_error, dataset)
low3_error_batch, low3_error_ids = get_test_batch(low3_error, dataset)

# Define the classes for which we want to compute the SHAP values:
classes = ["P(y > 1)", "P(y > 2)", "P(y > 3)", "P(y > 4)"]

# Specifies the shape of individual inputs (ignoring batch size):
input_shape = top3_conf_batch.shape[1:]
masker = shap.maskers.Image(mask_value=0, shape=input_shape)

model = ResNetWrapper(
    load_resnet_model(
        "hpc/jobs/c50_25D_2411_1812/fold_0/model.pth",
        in_channels=3,
        dims="2.5D",
    )
)
model.eval()


def f(x: np.ndarray):
    """
    Takes a batch of inputs and returns the model's predictions in the correct format
    """
    tmp = x.copy()
    pred = model(tmp)
    pred = pred.detach().numpy().astype("float32")
    return pred


# wrap the prediction function f with SHAP's masker:
explainer = shap.Explainer(model=f, masker=masker, output_names=classes)

# generate the SHAP values for the test:
top_n_outputs = 4  # Number of classes to explain (do not change)
shap_values = explainer(
    top3_conf_batch,
    max_evals=1000,
    batch_size=10,
    outputs=shap.Explanation.argsort.flip[:top_n_outputs],
)
# shap_values contains the computed explanations for model predictions, indicating the importance of each pixel in the prediction.


# %%
def plot_shap_overlay(
    test_img_idx: int, shap_values: np.ndarray, predicted_rank: int, test: np.ndarray
):
    r_threshold = predicted_rank - 1
    shap_overlay = shap_values[test_img_idx, :, :, :, r_threshold].values
    original_image = test[test_img_idx]

    _, ax = plt.subplots(1, 2, figsize=(8, 8), constrained_layout=True)

    ax[0].imshow(original_image[:, :, 1], cmap="gray")
    ax[0].axis("off")
    ax[0].set_aspect("equal")

    im = ax[1].imshow(original_image[:, :, 1], cmap="gray")
    shap_im = ax[1].imshow(
        shap_overlay.sum(axis=-1),  # Summing across channels
        cmap="coolwarm",
        alpha=0.5,
        vmin=-np.max(np.abs(shap_overlay)),
        vmax=np.max(np.abs(shap_overlay)),
    )
    ax[1].axis("off")
    ax[1].set_aspect("equal")

    # Add colorbar for the SHAP values to the second axis
    cbar = plt.colorbar(
        shap_im, ax=ax[1], orientation="vertical", fraction=0.046, pad=0.04
    )
    cbar.set_label("SHAP value")

    plt.suptitle(f"SHAP explanation for malignancy rank {predicted_rank}", y=0.75)
    plt.show()


for img_idx in range(3):
    predicted_rank = preds.query(f"nodule_id == '{low3_error_ids[img_idx]}'")[
        "pred"
    ].values[0]
    plot_shap_overlay(img_idx, shap_values, predicted_rank, top3_conf_batch)

# %%
