# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from project_config import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_aes(y_true: list[int], y_pred: list[int]) -> list:
    assert len(y_true) == len(y_pred)
    return np.abs(np.array(y_true) - np.array(y_pred)).tolist()


def compute_filtered_AUC(
    all_true_labels: list[int], all_binary_prob_predictions: list[float]
) -> float:
    """Calculate binary AUC for non-ambiguous cases."""
    # Filter out ambiguous cases for binary AUC evaluation:
    non_ambiguous_idxs = [i for i, label in enumerate(all_true_labels) if label != 3]
    binary_predictions_filtered = [
        all_binary_prob_predictions[i] for i in non_ambiguous_idxs
    ]
    labels_filtered = [all_true_labels[i] for i in non_ambiguous_idxs]
    binary_labels = [1 if label > 3 else 0 for label in labels_filtered]
    return roc_auc_score(y_true=binary_labels, y_score=binary_predictions_filtered)


# NOTE: NOT USED
# def expected_calibration_error(
#     true_labels: torch.Tensor, pred_probas: torch.Tensor, bins=10
# ) -> float:
#     """
#     Params
#     ---
#     @pred_probas: torch.Tensor - shape (n_samples, n_classes)
#         The predicted probabilities for each class for each sample.
#     @true_labels: torch.Tensor - shape (n_samples,)
#         The true labels for each sample.

#     Returns
#     ---
#     @ece: float

#     Source:
#         https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d
#     """
#     # uniform binning approach with @bins number of bins
#     bin_boundaries = torch.linspace(0, 1, bins + 1)
#     bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]

#     # get max probability per sample i (confidences) and the final predictions from these confidences
#     confidences, predicted_label = torch.max(pred_probas, 1)
#     # get a boolean list of correct/false predictions
#     accuracies = predicted_label.eq(true_labels)  # predicted_label == true_labels

#     # Compute ECE:
#     ece = torch.zeros(1)
#     for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
#         # determine if sample is in bin m (between bin lower & upper)
#         in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())

#         # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
#         prop_in_bin = in_bin.float().mean()
#         if prop_in_bin.item() > 0:
#             # get the accuracy of bin m: acc(Bm)
#             accuracy_in_bin = accuracies[in_bin].float().mean()
#             # get the average confidence of bin m: conf(Bm)
#             avg_confidence_in_bin = confidences[in_bin].mean()
#             # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
#             ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
#     return ece


def classwise_ece(
    pred_probas: np.ndarray, y_true: np.ndarray, num_classes: int = 5, num_bins: int = 5
):
    """
    Computes the Expected Calibration Error (ECE) per class.

    Params
    ---
    @pred_probas: The predicted probabilities for each class.
        Shape: (N, num_classes)
    @y_true: The true class labels.
        Shape: (N,)
    """
    classwise_ece = []
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    N = len(y_true)

    for c in range(num_classes):
        class_ece = 0.0

        # Subset the class of interest:
        class_probs = pred_probas[:, c]  # predicted probabilities for class c
        binary_class_labels = (y_true == c + 1).astype(int)

        # For each bin, compute the confidence and accuracy:
        for i in range(num_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            bin_mask = (class_probs >= bin_lower) & (class_probs < bin_upper)

            n_bin = bin_mask.sum()  # number of samples in the bin
            if n_bin > 0:
                # average confidence of the predictions in the bin:
                avg_bin_confidence = class_probs[bin_mask].mean()

                # P(y = c | y_pred in bin) - proportion of correct predictions in the bin:
                bin_accuracy = binary_class_labels[bin_mask].sum() / n_bin

                # n_b/N: proportion of samples in the bin:
                bin_weight = n_bin / N
                class_ece += bin_weight * abs(bin_accuracy - avg_bin_confidence)

        classwise_ece.append(round(float(class_ece), 4))

    return classwise_ece


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
