"""
Stealing Cathrine and Trines implementation of CWCE
source: https://github.com/trer2547/LabelReliability_and_PathologyDetection_in_ChestXrays/blob/main/Task-MultiTask/MT_predictions.ipynb
"""

# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve


def compute_aes(y_true: list[int], y_pred: list[int]) -> list:
    """Returns the absolute errors between true and predicted values."""
    assert len(y_true) == len(y_pred)
    return np.abs(np.array(y_true) - np.array(y_pred)).tolist()


# NOTE: NOT USED
def expected_calibration_error(
    true_labels: torch.Tensor, pred_probas: torch.Tensor, bins=10
) -> float:
    """
    Params
    ---
    @pred_probas: torch.Tensor - shape (n_samples, n_classes)
        The predicted probabilities for each class for each sample.
    @true_labels: torch.Tensor - shape (n_samples,)
        The true labels for each sample.

    Returns
    ---
    @ece: float

    Source:
        https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d
    """
    # uniform binning approach with @bins number of bins
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]

    # get max probability per sample i (confidences) and the final predictions from these confidences
    confidences, predicted_label = torch.max(pred_probas, 1)
    # get a boolean list of correct/false predictions
    accuracies = predicted_label.eq(true_labels)  # predicted_label == true_labels

    # Compute ECE:
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
    return ece


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
    """Plots a reliability diagram for a given class ECE."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_avg_confidences = np.zeros(shape=num_bins)  # goes on the x-axis
    bin_accuracies = np.zeros(shape=num_bins)  # goes on the y-axis

    c = 2  # DEBUGGING, using a single class for now
    # QUESTION This is filtering out everything except the class of interest? (correct?)
    class_probs = pred_probas[:, c]  # predicted probabilities for class c
    binary_class_labels = (y_true == c + 1).astype(int)

    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        bin_mask = (class_probs >= bin_lower) & (class_probs < bin_upper)
        n_bin = bin_mask.sum()  # number of samples in the current bin

        if n_bin > 0:
            bin_labels = binary_class_labels[bin_mask]
            bin_probs = class_probs[bin_mask]

            bin_avg_confidences[i] = bin_probs.mean()
            bin_accuracies[i] = bin_labels.sum() / n_bin

    # Plotting the reliability diagram
    # bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bar_width = 1 / num_bins
    plt.figure(figsize=(8, 6))
    plt.bar(
        x=bin_avg_confidences,
        height=bin_accuracies,
        width=bar_width,
        label="Accuracy",
        color="red",
        alpha=0.7,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration")
    plt.xlabel("Confidence\n(Mean Predicted Probability)")
    plt.ylabel("Accuracy\n(Empirical Probability)")
    plt.title("Reliability Diagram")
    plt.show()
    plt.legend()


# %%
if __name__ == "__main__":
    pred_probas = np.array(
        [
            [0.25, 0.2, 0.22, 0.18, 0.15],
            [0.16, 0.06, 0.5, 0.07, 0.21],
            [0.06, 0.03, 0.8, 0.07, 0.04],
            [0.02, 0.03, 0.01, 0.04, 0.9],
            [0.4, 0.15, 0.16, 0.14, 0.15],
            [0.15, 0.28, 0.18, 0.17, 0.22],
            [0.07, 0.8, 0.03, 0.06, 0.04],
            [0.1, 0.05, 0.03, 0.75, 0.07],
            [0.25, 0.22, 0.05, 0.3, 0.18],
            [0.12, 0.09, 0.02, 0.17, 0.6],
            [0.25, 0.2, 0.22, 0.18, 0.15],
            [0.16, 0.06, 0.5, 0.07, 0.21],
            [0.06, 0.03, 0.8, 0.07, 0.04],
            [0.02, 0.03, 0.01, 0.04, 0.9],
            [0.4, 0.15, 0.16, 0.14, 0.15],
            [0.15, 0.28, 0.18, 0.17, 0.22],
            [0.07, 0.8, 0.03, 0.06, 0.04],
            [0.1, 0.05, 0.03, 0.75, 0.07],
            [0.25, 0.22, 0.05, 0.3, 0.18],
            [0.12, 0.09, 0.02, 0.17, 0.6],
            [0.25, 0.2, 0.22, 0.18, 0.15],
            [0.16, 0.06, 0.5, 0.07, 0.21],
            [0.06, 0.03, 0.8, 0.07, 0.04],
            [0.02, 0.03, 0.01, 0.04, 0.9],
            [0.4, 0.15, 0.16, 0.14, 0.15],
            [0.15, 0.28, 0.18, 0.17, 0.22],
            [0.07, 0.8, 0.03, 0.06, 0.04],
            [0.1, 0.05, 0.03, 0.75, 0.07],
            [0.25, 0.22, 0.05, 0.3, 0.18],
            [0.12, 0.09, 0.02, 0.17, 0.6],
        ]
    )

    y_true = (
        np.array(
            [
                0,
                2,
                3,
                4,
                2,
                0,
                1,
                3,
                3,
                2,
                0,
                2,
                3,
                4,
                2,
                0,
                1,
                3,
                3,
                2,
                0,
                2,
                3,
                4,
                2,
                0,
                1,
                3,
                3,
                2,
            ]
        )
        + 1
    )

    cwce: list[float] = classwise_ece(pred_probas, y_true)
    plt.plot(cwce, linestyle="--", marker="o")
    avg_cwce = sum(cwce) / len(cwce)
    std_cwce = np.std(cwce)
    print(f"Class-wise ECE: {cwce}\nAvg. ECE: {avg_cwce}\nStd. ECE: {std_cwce}")

    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "green", "orange", "purple"]
    for c in range(5):
        class_probs = pred_probas[:, c]
        binary_class_labels = (y_true == c + 1).astype(int)
        prob_true, prob_pred = calibration_curve(
            binary_class_labels, class_probs, n_bins=5
        )
        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            label=f"Malignancy {c + 1}",
            color=colors[c],
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
    plt.legend(loc="right")
    plt.show()
