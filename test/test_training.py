import os
from unittest import TestCase

from torch import nn

from project_config import env_config
from utils.early_stopping import EarlyStopping


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
