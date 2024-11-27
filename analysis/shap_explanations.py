import numpy as np
import shap
import shap.maskers
import torch
import torch.nn.functional as F
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
    "/Users/newuser/Documents/ITU/master_thesis/data/precomputed_rois_50C_2.5D",
    data_augmentation=False,
)

loader = DataLoader(dataset, batch_size=110, shuffle=False)
batch, labels, ids = next(iter(loader))
batch = batch.permute(0, 2, 3, 1).numpy()
batch.shape

model = ResNetWrapper(
    load_resnet_model(
        "/Users/newuser/Documents/ITU/master_thesis/hpc/jobs/c50_25D_1911_1125/fold_0/model.pth",
        in_channels=3,
        dims="2.5D",
    )
)
model.eval()

background = batch[:100]
test = batch[100:102]
test.shape


def f(x):
    tmp = x.copy()
    pred = model(tmp)
    pred = pred.detach().numpy().astype("float32")
    return pred


f(test)

classes = ["Above 1", "Above 2", "Above 3", "Above 4"]
masker = shap.maskers.Image(0, tuple(test.shape[1:]))
explainer = shap.Explainer(f, masker=masker, output_names=classes)

shap_values = explainer(
    test, max_evals=1000, batch_size=10, outputs=shap.Explanation.argsort.flip[:4]
)
shap_values.shape

# Show explanation plot
fig = shap.image_plot(shap_values[0, :, :, :, :], labels=classes, show=False)
