import numpy as np
import torch

from project_config import SEED

torch.manual_seed(SEED)


class EarlyStopping:
    def __init__(
        self,
        checkpoint_path: str,
        patience: int = 5,
        min_delta: int = 0.5,
    ):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            path (str): Path for saving the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.path = checkpoint_path

    def __call__(self, val_loss, model):
        # Check if the current validation loss improved
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            torch.save(model.state_dict(), self.path)
        else:
            # No improvement, increment counter
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
