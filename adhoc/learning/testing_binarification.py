# %%
import torch

from utils.common_imports import *


def plot_prediction(rank_predictions: list[float]) -> None:
    plt.bar(np.arange(len(rank_predictions)), rank_predictions)
    plt.ylim(0, 1)
    plt.hlines(
        y=predictions_threshold,
        xmin=0,
        xmax=len(rank_predictions),
        colors="r",
        linestyles="--",
        label="50% threshold",
    )
    plt.legend()
    plt.show()


out1 = [0.8, 0.7, 0.4, 0.3]
out2 = [0.7, 0.6, 0.5, 0.4]
predictions_threshold = 0.5
output = torch.Tensor(
    [
        out1,
        out2,
    ]
)


def binary_inference(model_output):
    # model_output[0] corresponds to P(y > 1)
    # model_output[1] corresponds to P(y > 2)
    # model_output[2] corresponds to P(y > 3)
    # model_output[3] corresponds to P(y > 4)

    threshold = 0.5
    if (
        model_output[2] >= threshold
    ):  # Change 2 to 1 if you want to classify for score >= 2
        return 1  # Malignant
    else:
        return 0  # Benign


binary_inference(out2)

output[:, 2] >= predictions_threshold

plot_prediction(out2)

# %%
