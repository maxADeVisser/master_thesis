"""
Stealing Cathrine and Trines implementation of CWCE
source: https://github.com/trer2547/LabelReliability_and_PathologyDetection_in_ChestXrays/blob/main/Task-MultiTask/MT_predictions.ipynb
"""

import numpy as np


def binary_ECE(y_true, probs, power=1, bins=10):
    r"""
    Binary Expected Calibration Error

    Parameters
    ----------
    y_true : indicator vector (n_samples, )
        True labels.
    probs : matrix (n_samples, )
        Predicted probabilities for positive class.

    Returns
    -------
    score : float
    """

    create_bins = np.linspace(
        start=0, stop=1, num=bins + 1
    )  # Returns 'num' evenly spaced samples, calculated over the interval [start, stop]
    # print('bins created: ', create_bins)
    idx_bins = np.digitize(
        x=probs, bins=create_bins
    )  # Return the indices of the bins to which each value in input array belongs
    idx_bins -= 1  # Need to subtract 1 from the bin indices to start at 0

    # Function for computing the ECE for one bin
    def bin_func(y, p, idx_bins):
        probs_bin_mean = np.mean(p[idx_bins])  # Mean of probs in bin i
        true_bin_mean = np.mean(y[idx_bins])  # Mean of true values in bin i
        diff = np.abs(
            probs_bin_mean - true_bin_mean
        )  # Absolute difference between the two bin means
        diff_power = (
            diff**power
        )  # Raising the diff according to the L_p calibration error specified, typically power = 1
        ece = (
            diff_power * np.sum(idx_bins) / len(p)
        )  # Multiplying by the fraction of probs in that bin
        return ece

    # Computing the binary ECE for each bin and summing them
    ece = 0

    for i in np.unique(idx_bins):  # Looping through the unique bins (len(bins))
        ece += bin_func(y_true, probs, idx_bins == i)  # Summing the error for each bin

    return ece


def classwise_ECE(y_true, probs, classes_list, power=1, bins=10, print_ece=False):
    r"""Classwise Expected Calibration Error

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
