# %%
import matplotlib.pyplot as plt
import numpy as np
import shap
import shap.maskers
import torch
from torch.utils.data import DataLoader

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
)

loader = DataLoader(dataset, batch_size=110, shuffle=False)
batch, labels, ids = next(iter(loader))
batch = batch.permute(0, 2, 3, 1).numpy()

# select a set of background examples to take an expectation over:
baseline = batch[:100]
baseline_labels = labels[:100]

# select a set of test examples to explain:
test = batch[100:-1]
test_labels = labels[100:-1]
test = np.expand_dims(batch[-1], axis=0)
test_labels = labels[-1]

# Define the classes for which we want to compute the SHAP values:
classes = ["P(y > 1)", "P(y > 2)", "P(y > 3)", "P(y > 4)"]
# Specifies the shape of individual inputs (ignoring batch size):
input_shape = test.shape[1:]
masker = shap.maskers.Image(mask_value=0, shape=input_shape)

# Define a function that takes a batch of inputs and returns the model's predictions in the correct format:
model = ResNetWrapper(
    load_resnet_model(
        "hpc/jobs/c50_25D_2411_1812/fold_0/model.pth",
        in_channels=3,
        dims="2.5D",
    )
)
model.eval()


def f(x):
    tmp = x.copy()
    pred = model(tmp)
    pred = pred.detach().numpy().astype("float32")
    return pred


# wrap the prediction function f with SHAP's masker:
explainer = shap.Explainer(model=f, masker=masker, output_names=classes)

# generate the SHAP values for the test:
top_n_outputs = 4  # Number of classes to explain (do not change)
shap_values = explainer(
    test,
    max_evals=1000,
    batch_size=10,
    outputs=shap.Explanation.argsort.flip[:top_n_outputs],
)
# shap_values contains the computed explanations for model predictions, indicating the importance of each pixel in the prediction.
# %%

# Show explanation plot
shap.image_plot(shap_values, test)

# OR Creating own shap plot:
# overlap shap values with the original image
img_idx = 0
logits = model(test[img_idx : img_idx + 1]).detach().numpy()
predicted_class = np.sum([0.5 < logits])
shap_overlay = shap_values[img_idx, :, :, :, predicted_class].values
original_image = test[img_idx]

plt.figure(figsize=(8, 8))
plt.imshow(original_image[:, :, 1], cmap="gray")
plt.imshow(
    shap_overlay.sum(axis=-1),  # Summing across channels
    cmap="coolwarm",
    alpha=0.4,
    vmin=-np.max(np.abs(shap_overlay)),
    vmax=np.max(np.abs(shap_overlay)),
)
plt.colorbar(label="SHAP value", cmap="coolwarm")
plt.title(f"SHAP Explanation for model prediction: {predicted_class + 1}")
plt.axis("off")
plt.tight_layout()
plt.show()
