import unittest
from unittest.mock import MagicMock, patch

import torch

from model.MEDMnist.ResNet import (
    compute_class_probs_from_logits,
    get_pred_malignancy_score_from_logits,
    predict_binary_from_logits,
)


@patch("model.MEDMnist.ResNet.get_unconditional_probas")
class TestPredictMalignancyFromLogits(unittest.TestCase):
    def setUp(self):
        # text example for 5 classes
        self.logits = torch.tensor(
            [[14.152, -6.1942, 0.47710, 0.96850], [65.667, 0.303, 11.500, -4.524]]
        )

    def test_get_pred_malignancy_score_from_logits(self, mock_get_unconditional_probas):
        mock_get_unconditional_probas.return_value = torch.tensor(
            [
                [0.4, 0.4, 0.4, 0.4],
                [0.5, 0.4, 0.3, 0.2],
                [0.8, 0.7, 0.4, 0.2],
                [0.9, 0.7, 0.6, 0.0],
                [0.95, 0.9, 0.85, 0.5],
            ]
        )
        expected_output = torch.tensor([1, 2, 3, 4, 5])
        output = get_pred_malignancy_score_from_logits(self.logits)
        self.assertTrue(torch.equal(output, expected_output))
        mock_get_unconditional_probas.assert_called_once_with(self.logits)


@patch("model.MEDMnist.ResNet.get_unconditional_probas")
class TestPredictBinaryFromLogits(unittest.TestCase):
    def setUp(self):
        # text example for 5 classes
        self.logits = torch.tensor(
            [[14.152, -6.1942, 0.47710, 0.96850], [65.667, 0.303, 11.500, -4.524]]
        )

    def test_predict_binary_from_logits(self, mock_get_unconditional_probas):
        mock_get_unconditional_probas.return_value = torch.tensor(
            [[0.8, 0.7, 0.4, 0.2], [0.95, 0.9, 0.85, 0.5]]
        )
        expected_output = torch.tensor([0.0, 1.0])
        output = predict_binary_from_logits(self.logits)
        self.assertTrue(torch.equal(output, expected_output))
        mock_get_unconditional_probas.assert_called_once_with(self.logits)

    def test_predict_binary_from_logits_with_probabilities(
        self, mock_get_unconditional_probas
    ):
        mock_get_unconditional_probas.return_value = torch.tensor(
            [[0.8, 0.7, 0.4, 0.2], [0.95, 0.9, 0.85, 0.5]]
        )
        expected_output = torch.tensor([0.4, 0.85])
        output = predict_binary_from_logits(self.logits, return_probs=True)
        self.assertAlmostEqual(output[0], expected_output[0], places=4)
        self.assertAlmostEqual(output[1], expected_output[1], places=4)
        mock_get_unconditional_probas.assert_called_once_with(self.logits)


@patch("model.MEDMnist.ResNet.get_unconditional_probas")
class TestComputeClassProbsFromLogits(unittest.TestCase):
    def setUp(self):
        # text example for 5 classes
        self.logits = torch.tensor(
            [
                [14.152, -6.1942, 0.47710, 0.96850],
                [65.667, 0.303, 11.500, -4.524],
            ]
        )

    def test_compute_class_probs_from_logits(self, mock_get_unconditional_probas):
        mock_get_unconditional_probas.return_value = torch.tensor(
            [
                [0.4, 0.3, 0.2, 0.1],
                [0.9, 0.8, 0.7, 0.6],
            ]
        )
        expected_output = torch.tensor(
            [
                [0.6, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.6],
            ]
        )
        output = compute_class_probs_from_logits(self.logits)
        self.assertTrue(torch.allclose(torch.sum(output, dim=1), torch.ones(2)))
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))
        mock_get_unconditional_probas.assert_called_once_with(self.logits)


if __name__ == "__main__":
    unittest.main()
