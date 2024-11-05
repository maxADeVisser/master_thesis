import unittest

import numpy as np
import torch

from preprocessing.processing import clip_and_normalise_volume


class TestProcessing(unittest.TestCase):

    def test_clip_and_normalise_volume_basic(self):
        nodule_scan = torch.tensor([[-1000, 400], [0, 1000]])
        expected_output = np.array([[0.0, 1.0], [0.71428573, 1.0]])
        result = clip_and_normalise_volume(nodule_scan)
        np.testing.assert_almost_equal(result.numpy(), expected_output, decimal=5)

    def test_clip_and_normalise_volume_all_within_bounds(self):
        nodule_scan = torch.tensor([[100, 200], [300, 400]])
        expected_output = np.array([[0.0, 0.33333334], [0.6666667, 1.0]])
        result = clip_and_normalise_volume(nodule_scan)
        np.testing.assert_almost_equal(result.numpy(), expected_output, decimal=5)

    def test_clip_and_normalise_volume_all_outside_bounds(self):
        nodule_scan = torch.tensor([[-2000, -1500], [1500, 2000]])
        expected_output = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = clip_and_normalise_volume(nodule_scan)
        np.testing.assert_almost_equal(result.numpy(), expected_output, decimal=5)

    def test_clip_and_normalise_volume_mixed_bounds(self):
        nodule_scan = torch.tensor([[-1000, 0], [400, 1000]])
        expected_output = np.array([[0.0, 0.71428573], [1.0, 1.0]])
        result = clip_and_normalise_volume(nodule_scan)
        np.testing.assert_almost_equal(result.numpy(), expected_output, decimal=5)


if __name__ == "__main__":
    unittest.main()
