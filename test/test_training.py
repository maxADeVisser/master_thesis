import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from coral_pytorch.losses import CornLoss
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from project_config import env_config
from train.train import get_pred_malignancy_score_from_logits, train_epoch, train_model
from utils.early_stopping import EarlyStopping


class TestTrainEpoch(TestCase):
    def setUp(self):
        global DEVICE
        DEVICE = "cpu"
        self.n_classes = 5
        self.dummy_model = nn.Linear(10, self.n_classes - 1)

        # Set up a simple dataset and dataloader
        inputs = torch.randn(100, 10)  # 100 samples, 10 features
        labels = torch.randint(1, 6, (100,))
        dataset = TensorDataset(inputs, labels)
        self.train_loader = DataLoader(dataset, batch_size=10)
        self.optimizer = optim.SGD(self.dummy_model.parameters(), lr=0.01)
        self.criterion = CornLoss(num_classes=self.n_classes)

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


# TODO finish this test:
# @patch("model.MEDMnist.ResNet.predict_binary_from_logits")
# @patch("model.MEDMnist.ResNet.get_pred_malignancy_score_from_logits")
# @patch("model.MEDMnist.ResNet.compute_class_probs_from_logits")
# class TestValidateModel(TestCase):
#     def setUp(self):
#         global DEVICE

#         # Set up a simple dataset and dataloader
#         inputs = torch.randn(100, 10)  # 100 samples, 10 features
#         labels = torch.randn(100, 1)  # 100 labels
#         dataset = TensorDataset(inputs, labels)
#         self.test_loader = DataLoader(dataset, batch_size=10)
