"""Metrics for PyTorch model evaluation."""

import torch


class MAEMetric(torch.nn.Module):
    """
    Mean Absolute Error (MAE) metric.

    Computes the mean absolute error between predictions and targets.
    """

    def __init__(self):
        """
        Initialize MAEMetric.
        """
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Compute the mean absolute error.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True values.

        Returns
        -------
        mae : torch.Tensor
            Mean absolute error.
        """
        return torch.mean(torch.abs(y_pred - y_true))
