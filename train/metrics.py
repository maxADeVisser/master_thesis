# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from project_config import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)

DECIMAL_PLACES = 4


def compute_errors(y_true: torch.Tensor, y_pred: torch.Tensor) -> list[int]:
    """
    Returns the absolute errors between the true and predicted labels. The mean of this is the MAE.
    We do not aggregate them here, as we can plot the distribution of the errors and descriptive statistics.
    """
    return (y_pred - y_true).tolist()


def compute_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the mean squared error (MSE).
    """
    return torch.mean((y_true - y_pred).float() ** 2).item()


def compute_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the mean absolute error (MAE).
    """
    return torch.mean(torch.abs(y_true - y_pred).float()).item()


def compute_ovr_AUC(y_true: np.ndarray, all_class_proba_preds: np.ndarray) -> float:
    """
    Computes the one-vs-rest AUC.
    """
    ovr_AUC = roc_auc_score(
        y_true=y_true,
        y_score=all_class_proba_preds,
        multi_class="ovr",
        average="weighted",
    )
    return float(ovr_AUC)


def compute_filtered_AUC(
    y_true: np.ndarray, all_binary_prob_predictions: np.ndarray
) -> float:
    """Calculate binary AUC for non-ambiguous cases only."""
    # Filter out labels equal to 3:
    non_ambiguous_mask = y_true != 3
    binary_predictions_filtered = all_binary_prob_predictions[non_ambiguous_mask]
    labels_filtered = y_true[non_ambiguous_mask]

    # Create binary labels (1 if label > 3, else 0):
    binary_labels = (labels_filtered > 3).astype(int)

    return float(
        roc_auc_score(y_true=binary_labels, y_score=binary_predictions_filtered)
    )


def compute_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the accuracy of the model.
    """
    return torch.mean((y_true == y_pred).float()).item()


def compute_binary_accuracy(
    y_true: np.ndarray, all_binary_prob_predictions: np.ndarray
) -> float:
    """
    Computes the binary accuracy of the model using only non-ambiguous cases.
    """
    # Filter out labels equal to 3:
    non_ambiguous_mask = y_true != 3
    binary_predictions_filtered = all_binary_prob_predictions[non_ambiguous_mask]
    labels_filtered = y_true[non_ambiguous_mask]

    # Create binary labels and predictions (1 if X > 3, else 0):
    binary_labels = (labels_filtered > 3).astype(int)
    binary_predictions = (0.5 <= binary_predictions_filtered).astype(int)

    # Compute accuracy:
    return float(np.mean(binary_labels == binary_predictions))


def compute_cwce(
    true_labels: torch.Tensor, pred_class_probas: torch.Tensor, M: int
) -> float:
    """
    Calculates the Expected Calibration Error (ECE) for a set of samples and their true labels.
    Works for binary and multi-class classification settings (aggregated ECE).

    Params:
    ---
    @pred_class_probas: torch.Tensor - shape (n_samples, n_classes)
        The predicted probabilities for each class for each sample.
    @true_labels: torch.Tensor - shape (n_samples,)
        The true labels for each sample.
    @M is the number of bins to divide the confidence interval [0, 1] into.

    Returns
    ---
    @ece: torch.Tensor - shape (1,)

    Source:
    https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d
    """
    # uniform binning approach with M number of bins
    bin_boundaries = torch.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i (confidences) and the final predictions from these confidences
    confidences, predicted_label = torch.max(pred_class_probas, 1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label.eq(true_labels)

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].float().mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def reliability_diagram(
    y_true: np.ndarray, pred_probas: np.ndarray, num_bins: int = 5, n_classes: int = 5
):
    """Plots a reliability diagram for each class"""
    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "green", "orange", "purple"]
    markers = ["o", "s", "D", "v", "^"]
    for c in range(n_classes):
        class_probs = pred_probas[:, c]
        binary_class_labels = (y_true == c + 1).astype(int)
        prob_true, prob_pred = calibration_curve(
            binary_class_labels, class_probs, n_bins=num_bins, strategy="uniform"
        )
        plt.plot(
            prob_pred,
            prob_true,
            marker=markers[c],
            label=f"Malignancy {c + 1}",
            color=colors[c],
            alpha=0.7,
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
    plt.legend(loc="right")
    plt.grid()
    plt.show()


def _simulate_binary_calibrated_data():
    """Util function for testing the reliability diagram"""
    num_samples = 100
    sup_sample_size = num_samples // 5
    y_true = np.concatenate(
        [
            np.random.choice([0, 1], size=sup_sample_size, p=[0.9, 0.1]),  # 10% 1s
            np.random.choice([0, 1], size=sup_sample_size, p=[0.7, 0.3]),  # 30% 1s
            np.random.choice([0, 1], size=sup_sample_size, p=[0.5, 0.5]),  # 50% 1s
            np.random.choice([0, 1], size=sup_sample_size, p=[0.3, 0.7]),  # 70% 1s
            np.random.choice([0, 1], size=sup_sample_size, p=[0.1, 0.9]),  # 90% 1s
        ]
    )

    y_probs = np.linspace(0, 1, num=num_samples)
    # add noise to the probabilities:
    y_probs = y_probs + np.random.normal(0, 0.1, size=num_samples)
    y_probs = np.clip(y_probs, 0, 1)  # rescale to [0, 1]
    return y_true, y_probs


# %%
if __name__ == "__main__":
    # cwce: list[float] = classwise_ece(pred_probas, y_true)
    # plt.plot(cwce, linestyle="--", marker="o")
    # avg_cwce = sum(cwce) / len(cwce)
    # std_cwce = np.std(cwce)
    # print(f"Class-wise ECE: {cwce}\nAvg. ECE: {avg_cwce}\nStd. ECE: {std_cwce}")

    # reliability_diagram(y_true, pred_probas, num_bins=5, n_classes=2)

    # --- Simulate binary data for testing the reliability diagram ---
    y_true, pred_probas = _simulate_binary_calibrated_data()
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(
        y_true, pred_probas, n_bins=10, strategy="uniform"
    )
    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        alpha=0.7,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
    plt.legend(loc="right")
    plt.grid()
    plt.show()

    # --- Plot mean predicted probability against counts ---
    # TODO turn this plot into a line plot, so we better can plot the calibration curve for each class
    plt.hist(pred_probas, bins=10)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Counts")
    plt.title("Mean Predicted Probability Distribution")
    plt.show()

    # --- another plot ---
    # plt.bar(np.arange(len(pred_probas)), pred_probas)
