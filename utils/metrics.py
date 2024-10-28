"""
Stealing Cathrine and Trines implementation of CWCE
source: https://github.com/trer2547/LabelReliability_and_PathologyDetection_in_ChestXrays/blob/main/Task-MultiTask/MT_predictions.ipynb
"""

import numpy as np


def binary_ECE(
    y_true: np.ndarray, probas: np.ndarray, power: int = 1, bins: int = 10
) -> float:
    """
    Binary Expected Calibration Error (ECE)

    Parameters
    ----------
    y_true : indicator vector (n_samples, )
        True labels.
    probs : matrix (n_samples, )
        Predicted probabilities for positive class.
    power : int (default=1)
        Power to raise the calibration error. Using a higher power gives more weight to larger errors (?????).

    Returns
    -------
    score : float
    """

    def _bin_ece(
        observed_probas: list[float], pred_probas: list[float], idx_bins: list[bool]
    ) -> float:
        """
        Util func for computing the ECE for a single bin.
        @idx_bins: boolean array indicating which indices belong to the current bin.

        Return value @bin_ece represents the error for the bin, adjusted by its sample size relative to the entire dataset.
        """
        pred_probs_bin_mean = np.mean(pred_probas[idx_bins])
        true_freq_bin_mean = np.mean(observed_probas[idx_bins])

        # Absolute difference between the two bin means and
        # raising the diff according to the L_p calibration error specified, typically power = 1:
        bin_calibration_error = (
            np.abs(pred_probs_bin_mean - true_freq_bin_mean) ** power
        )

        # weight by bin sample count and normalise by total sample count:
        bin_sample_count = np.sum(idx_bins)
        total_sample_count = len(pred_probas)
        bin_ece = bin_calibration_error * (bin_sample_count / total_sample_count)
        return bin_ece

    # Get the indices of the bins to which each value in input array belongs:
    create_bins = np.linspace(start=0, stop=1, num=bins + 1)
    idx_bins = np.digitize(x=probas, bins=create_bins) - 1

    # Cumulate the binary ECE for each specified bin:
    total_bin_ece = 0
    for i in np.unique(idx_bins):
        total_bin_ece += _bin_ece(y_true, probas, idx_bins == i)

    return total_bin_ece


def classwise_ECE(y_true, probs, classes_list, power=1, bins=10, print_ece=False):
    """Class-wise Expected Calibration Error

    Parameters
    ----------
    y_true : label indicator matrix (n_samples, n_classes)
        True labels.
    probs : matrix (n_samples, n_classes)
        Predicted probabilities.

    Returns
    -------
    score : float
    """

    n_classes = len(classes_list)

    # Computing the binary ECE for each class
    class_eces = []
    for c in range(n_classes):  # Looping through the classes
        binary_ece = binary_ECE(y_true[:, c], probs[:, c], power=power, bins=bins)
        if print_ece:
            print("ECE for {}: {}".format(classes_list[c], round(binary_ece, 3)))
        class_eces.append(binary_ece)

    # if print_ece:
    # print()
    # print('Average Class-Wise ECE: ', round(np.mean(class_eces), 3))

    return class_eces
    # Right now, not printing the average class-wise calibration error


def compute_aes(y_true: list[int], y_pred: list[int]) -> list:
    """Returns the absolute errors between true and predicted values."""
    assert len(y_true) == len(y_pred)
    return np.abs(np.array(y_true) - np.array(y_pred)).tolist()


if __name__ == "__main__":
    test1 = [1, 2, 3, 4, 5]
    test2 = [5, 1, 3, 2, 5]

    compute_aes(test2, test1)
