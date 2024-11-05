import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from model.train import predict_rank_from_logits, train_epoch, validate_model
from project_config import env_config
from utils.early_stopping import EarlyStopping


class TestTrainEpoch(TestCase):
    def setUp(self):
        global DEVICE
        DEVICE = "cpu"
        self.dummy_model = nn.Linear(10, 1)

        # Set up a simple dataset and dataloader
        inputs = torch.randn(100, 10)  # 100 samples, 10 features
        labels = torch.randn(100, 1)  # 100 labels
        dataset = TensorDataset(inputs, labels)
        self.train_loader = DataLoader(dataset, batch_size=10)

        # Define optimizer and loss criterion
        self.optimizer = optim.SGD(self.dummy_model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

        self.train_func_args = (
            self.dummy_model,
            self.train_loader,
            self.optimizer,
            self.criterion,
        )

    def test_returns_float_loss(self):
        """Test that train_epoch returns a float as the average loss."""
        loss = train_epoch(*self.train_func_args)
        self.assertIsInstance(
            loss, float, "train_epoch should return a float as the average batch loss."
        )

    def test_model_set_to_train(self):
        """Test that the model is set to train mode."""
        # Mock the train method to check if it was called:
        self.dummy_model.train = MagicMock()
        train_epoch(*self.train_func_args)
        self.dummy_model.train.assert_called_once()

    def test_optimizer_step_called(self):
        """Test that optimizer step is called during training."""
        original_step = self.optimizer.step  # Backup original step function
        self.optimizer.step = MagicMock()  # Mock optimizer.step to check if it's called
        train_epoch(*self.train_func_args)
        self.optimizer.step.assert_called()  # Ensure optimizer step was called at least once
        self.optimizer.step = original_step  # Restore original optimizer step

    def test_loss_reduction_over_epochs(self):
        """Test that loss generally reduces over epochs for the same data (only a rough check)."""
        losses = []
        for _ in range(10):
            loss = train_epoch(*self.train_func_args)
            losses.append(loss)

        self.assertTrue(
            all(x >= y for x, y in zip(losses, losses[1:])),
            "Loss should generally decrease over consecutive epochs.",
        )


class TestValidateModel(TestCase):
    @patch("model.train.predict_binary_from_logits")
    @patch("model.train.predict_rank_from_logits")
    @patch("model.train.accuracy_score")
    @patch("model.train.f1_score")
    @patch("model.train.roc_auc_score")
    @patch("model.train.mean_absolute_error")
    @patch("model.train.compute_aes")
    @patch("model.train.root_mean_squared_error")
    def test_validate_model(
        self,
        mock_rmse,
        mock_aes,
        mock_mae,
        mock_auc,
        mock_f1,
        mock_accuracy,
        mock_predict_rank,
        mock_predict_binary,
    ):
        # TODO, this needs to be updated!!!
        # Mock the outputs of dependent functions
        mock_predict_binary.return_value = [0.4, 0.6]  # Sample binary predictions
        mock_predict_rank.return_value = [2, 3]  # Sample rank predictions
        mock_accuracy.return_value = 0.75
        mock_f1.return_value = 0.8
        mock_auc.return_value = 0.85
        mock_mae.return_value = 0.4
        mock_aes.return_value = 0.5
        mock_rmse.return_value = 0.6

        dummy_model = nn.Linear(10, 5)
        inputs = torch.randn(4, 10)  # 4 samples, 10 features each
        y_true = [1, 2, 3, 4]
        y_pred = [2, 3, 2, 3]
        batch_size = 2
        labels = torch.tensor(y_true)  # True labels
        validation_data = [(inputs[i], labels[i]) for i in range(4)]
        validation_loader = DataLoader(validation_data, batch_size=batch_size)

        with patch("model.train.DEVICE", "cpu"):
            results = validate_model(dummy_model, validation_loader)

        # Check the function calls and validate output
        mock_accuracy.assert_called_once_with(y_true=y_true, y_pred=y_pred)
        mock_f1.assert_called_once_with(
            y_true=y_true, y_pred=y_pred, average="weighted"
        )
        mock_auc.assert_called_once_with(y_true=[1, 1, 1, 0], y_score=[0, 1, 0, 1])
        mock_mae.assert_called_once_with(y_true=y_true, y_pred=y_pred)
        mock_aes.assert_called_once_with(y_true=y_true, y_pred=y_pred)
        mock_rmse.assert_called_once_with(y_true=y_true, y_pred=y_pred)

        # Validate results dictionary
        expected_results = {
            "accuracy": 0.75,
            "f1": 0.8,
            "AUC_filtered": 0.85,
            "AUC_n_samples": 4,  # This should be len(binary_labels)
            "mae": 0.4,
            "aes": 0.5,
            "rmse": 0.6,
        }
        self.assertEqual(results, expected_results)


class TestPredictRankFromLogits(TestCase):
    def test_predict_rank_from_logits(self):
        # Test case 1: Simple logits input
        # (2 samples with 3 classes)
        logits = torch.tensor([[2.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
        expected_output = [
            3.0,
            1.0,
        ]  # Expected ranks after applying sigmoid and cumprod
        output = predict_rank_from_logits(logits)
        self.assertEqual(output, expected_output)

    def test_predict_rank_from_logits_all_zero_logits(self):
        # Test case 2: All zero logits
        logits = torch.zeros((3, 4))  # 3 samples, 4 classes
        expected_output = [1.0, 1.0, 1.0]  # All ranks should be 1
        output = predict_rank_from_logits(logits)
        self.assertEqual(output, expected_output)

    def test_predict_rank_from_logits_increasing_logits(self):
        # Test case 3: Increasing logits
        logits = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [1.0, 1.0, 1.0]])
        expected_output = [
            2.0,
            3.0,
            3.0,
        ]  # Expected ranks based on the sigmoid and cumprod
        output = predict_rank_from_logits(logits)
        self.assertEqual(output, expected_output)

    def test_predict_rank_from_logits_decreasing_logits(self):
        # Test case 4: Decreasing logits
        logits = torch.tensor([[1.0, 0.5, 0.0], [0.2, 0.1, 0.0]])
        expected_output = [3.0, 1.0]  # Expected ranks based on the sigmoid and cumprod
        output = predict_rank_from_logits(logits)
        self.assertEqual(output, expected_output)

    def test_predict_rank_from_logits_empty_tensor(self):
        # Test case 5: Empty logits tensor
        logits = torch.empty((0, 3))  # Empty tensor with 0 samples, 3 classes
        expected_output = []  # Expected output should also be empty
        output = predict_rank_from_logits(logits)
        self.assertEqual(output, expected_output)


class TestEarlyStopping(TestCase):
    def setUp(self):
        self.dummy_model = nn.Linear(10, 1)
        self.test_file_name = f"{env_config.OUT_DIR}/test_checkpoint.pth"
        self.early_stopper = EarlyStopping(
            patience=3, min_delta=0.01, checkpoint_path=self.test_file_name
        )

    def tearDown(self):
        # Clean up checkpoint file if it exists
        if os.path.exists(self.test_file_name):
            os.remove(self.test_file_name)

    def test_no_early_stop_with_improving_loss(self):
        """Test that early stopping does not trigger when loss keeps improving."""
        losses = [0.5, 0.4, 0.3, 0.2, 0.1]  # Continually improving loss

        for loss in losses:
            self.early_stopper(loss, self.dummy_model)
            self.assertFalse(
                self.early_stopper.early_stop,
                "Early stopping should not trigger when loss is improving.",
            )

        self.assertTrue(
            os.path.exists(self.test_file_name),
            "Checkpoint file should be saved when there is improvement.",
        )

    def test_early_stop_when_loss_stagnates(self):
        """Test that early stopping triggers when the loss does not improve."""
        losses = [0.5, 0.5, 0.5, 0.5, 0.5]  # No improvement in loss

        for loss in losses:
            self.early_stopper(loss, self.dummy_model)

        self.assertTrue(
            self.early_stopper.early_stop,
            "Early stopping should trigger when loss does not improve.",
        )

    def test_early_stop_with_fluctuating_loss(self):
        """Test that early stopping does not trigger if fluctuations are within min_delta."""
        losses = [0.5, 0.49, 0.48, 0.47, 0.46]  # Minor improvements within min_delta

        for loss in losses:
            self.early_stopper(loss, self.dummy_model)

        self.assertFalse(
            self.early_stopper.early_stop,
            "Early stopping should not trigger with minor improvements.",
        )

    def test_checkpoint_saving(self):
        """Test that the model checkpoint is saved when the loss improves."""
        initial_loss = 0.5
        improved_loss = 0.4
        self.early_stopper(initial_loss, self.dummy_model)

        # Check that checkpoint is saved with the first improvement
        self.assertTrue(
            os.path.exists(self.test_file_name),
            "Checkpoint should be saved when loss improves.",
        )

        # Check the best loss value update
        self.assertEqual(
            self.early_stopper.best_loss,
            initial_loss,
            "Best loss should update to the initial loss.",
        )

        # New improvement
        self.early_stopper(improved_loss, self.dummy_model)

        # Check that best loss is updated
        self.assertEqual(
            self.early_stopper.best_loss,
            improved_loss,
            "Best loss should update to the improved loss.",
        )
