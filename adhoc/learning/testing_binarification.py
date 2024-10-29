# %%
import torch
from coral_pytorch.dataset import corn_label_from_logits

from model.MEDMnist.ResNet import compute_class_probs_from_logits
from utils.common_imports import *


def plot_prediction(rank_predictions: list[float]) -> None:
    plt.bar(x=np.arange(len(rank_predictions)), height=rank_predictions)
    plt.ylim(0, 1)
    plt.hlines(
        y=0.5,
        xmin=0 - 0.5,
        xmax=len(rank_predictions) - 1 + 0.5,
        colors="r",
        linestyles="--",
        label="50% threshold",
    )
    labels = [f"P(X>{i + 1})" for i, x in enumerate(rank_predictions)]
    plt.xticks(np.arange(len(rank_predictions)), labels=labels)
    plt.legend()
    plt.show()


out1 = [0.9, 0.7, 0.4, 0.3]  # should predict rank 3
out2 = [0.7, 0.6, 0.5, 0.4]  # should predict rank 4
out3 = [0.4, 0.3, 0.2, 0.1]  # should predict rank 1
sigmoid_output = torch.Tensor([out1, out2, out3])

compute_class_probs_from_logits(sigmoid_output)

# corn_label_from_logits(logits)
probas = torch.cumprod(sigmoid_output, dim=1)  # cumulative probability
predict_levels = probas > 0.5
predicted_ranks = torch.sum(predict_levels, dim=1) + 1

plot_prediction(out3)


with torch.no_grad():
    probas = torch.cumprod(probas, dim=1)
    predicted_rank = torch.sum(probas > 0.5, dim=1) + 1

# understanding cumprod func
test = torch.Tensor([[2, 3, 4, 5]])
torch.cumprod(test, dim=1)

# generate 100 normally distribution samples between 0 and 1
probas = np.random.normal(0.5, 0.2, 100)

# %%
