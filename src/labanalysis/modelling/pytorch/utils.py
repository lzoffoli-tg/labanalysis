"""pytorch custom utils

This module provides utility classes for PyTorch model training, including a custom
dataset for structured input/output and a trainer class with early stopping, learning
rate decay, and metric tracking.
"""

import time
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ...utils import split_data

__all__ = [
    "CustomDataset",
    "TorchTrainer",
    "TrainingLogger",
    "PinballLoss",
    "StandardizedMSELoss",
    "QuantilicRangeLoss",
    "ComboLoss",
    "MAEMetric",
    "UncertaintyWeighting",
]


class CustomDataset(Dataset):
    """
    Custom dataset for structured input and output tensors.

    Supports input and output as dictionaries of tensors, where each tensor
    must have the same number of samples (first dimension).

    Parameters
    ----------
    x : dict of str to torch.Tensor
        Dictionary of input tensors. Each tensor must have shape (N, ...).
    y : dict of str to torch.Tensor
        Dictionary of target tensors. Each tensor must have shape (N, ...).

    Attributes
    ----------
    x : dict of str to torch.Tensor
        Dictionary of input tensors.
    y : dict of str to torch.Tensor
        Dictionary of target tensors.
    """

    def __init__(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor],
    ):
        """
        Initialize the CustomDataset.

        Parameters
        ----------
        x : dict of str to torch.Tensor
            Dictionary of input tensors.
        y : dict of str to torch.Tensor
            Dictionary of target tensors.

        Raises
        ------
        ValueError
            If x or y are not dictionaries, or if tensors do not have the same number of samples.
        """
        self._x = x
        self._y = y

        # Validate x
        if isinstance(x, dict):
            x_sizes = [v.shape[0] for v in x.values()]
            if not all(size == x_sizes[0] for size in x_sizes):
                raise ValueError("All x tensors must have the same number of samples")
            size_x = x_sizes[0]
        else:
            size_x = x.shape[0]

        # Validate y
        if isinstance(y, dict):
            y_sizes = [v.shape[0] for v in y.values()]
            if not all(size == y_sizes[0] for size in y_sizes):
                raise ValueError("All y tensors must have the same number of samples")
            size_y = y_sizes[0]
        else:
            size_y = y.shape[0]

        if size_x != size_y:
            raise ValueError("x and y must have the same number of samples")

        self._size = size_x

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        size : int
            Number of samples.
        """
        return self._size

    def __getitem__(self, idx: int):
        """
        Retrieve a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        x_sample : dict of str to torch.Tensor
            Dictionary of input tensors for the sample.
        y_sample : dict of str to torch.Tensor
            Dictionary of target tensors for the sample.
        """
        if isinstance(self._x, dict):
            x_sample = {
                k: v[idx].unsqueeze(0) if v[idx].ndim < 1 else v[idx]
                for k, v in self._x.items()
            }
        else:
            x_sample = (
                self._x[idx].unsqueeze(0) if self._x[idx].ndim < 1 else self._x[idx]
            )
        if isinstance(self._y, dict):
            y_sample = {
                k: v[idx].unsqueeze(0) if v[idx].ndim < 1 else v[idx]
                for k, v in self._y.items()
            }
        else:
            y_sample = (
                self._y[idx].unsqueeze(0) if self._y[idx].ndim < 1 else self._y[idx]
            )
        return x_sample, y_sample

    @property
    def x(self):
        """
        Access the full input data.

        Returns
        -------
        x : dict of str to torch.Tensor
            Dictionary of input tensors.
        """
        return self._x

    @property
    def y(self):
        """
        Access the full target data.

        Returns
        -------
        y : dict of str to torch.Tensor
            Dictionary of target tensors.
        """
        return self._y


class UncertaintyWeighting(torch.nn.Module):
    """
    Task uncertainty weighting for multi-output loss balancing.

    Implements the method from:
        Kendall & Gal, "Multi-Task Learning Using Uncertainty", CVPR 2018.

    For each task/output i, a learnable parameter σᵢ² is introduced.
    The final combined loss becomes:

        L = Σ_i ( exp(-sᵢ) * Lᵢ + sᵢ )

    where sᵢ = log(σᵢ²). This ensures:
    - tasks with higher noise get lower weight
    - tasks with lower noise get higher weight
    - the optimization remains stable due to the regularizing +sᵢ term

    Parameters
    ----------
    output_keys : list[str]
        Names of outputs for which losses are computed.

    Attributes
    ----------
    log_vars : torch.nn.Parameter
        Learnable vector of log-variances, one per output.
    """

    def __init__(self, output_keys):
        super().__init__()
        self.output_keys = output_keys
        self.log_vars = torch.nn.Parameter(
            torch.zeros(
                len(output_keys),
                dtype=torch.float32,
            )
        )

    def forward(self, losses: dict[str, torch.Tensor]):
        """
        Combine per-output losses into a single weighted loss.

        Parameters
        ----------
        losses : dict[str, torch.Tensor]
            Dictionary mapping each output name to its scalar loss.

        Returns
        -------
        torch.Tensor
            The uncertainty-weighted combined loss.
        """
        total = torch.tensor(0, dtype=torch.float32)
        for idx, key in enumerate(self.output_keys):
            log_var = self.log_vars[idx]  # sᵢ
            precision = torch.exp(-log_var)  # exp(-sᵢ) = 1/σᵢ²
            total += precision * losses[key] + log_var
        return total


class PinballLoss(torch.nn.Module):
    """
    Standard multi-output pinball loss for quantile regression.

    Parameters
    ----------
    quantile : float
        Quantile level τ in (0, 1).
    """

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be between 0 and 1.")
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        """
        Compute pinball loss for multi-output regression.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted quantiles, shape (batch, n_outputs).
        y_true : torch.Tensor
            Ground truth, shape (batch, n_outputs).

        Returns
        -------
        torch.Tensor
            Scalar pinball loss.
        """
        error = y_true - y_pred

        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)

        return loss.mean()


class StandardizedMSELoss(torch.nn.Module):
    """
    Mean Squared Error loss computed on standardized values with running statistics.

    This loss function standardizes predictions and targets before computing MSE,
    which can improve training stability and make the loss scale-invariant across
    different output dimensions. Supports multi-output regression by maintaining
    per-output statistics (mean and standard deviation).

    Parameters
    ----------
    mean : torch.Tensor or None, optional
        Pre-computed mean tensor of shape (n_outputs,) with mean values per
        output dimension. If provided along with `std`, statistics are fixed
        (no updates during training). Default is None.
    std : torch.Tensor or None, optional
        Pre-computed standard deviation tensor of shape (n_outputs,) with
        std values per output dimension. If provided along with `mean`,
        statistics are fixed. Default is None.
    eps : float, optional
        Numerical stability term to avoid division by zero when standardizing.
        Default is 1e-8.

    Attributes
    ----------
    running_mean : torch.Tensor
        Running mean per output, shape (n_outputs,). Registered as buffer
        (no gradients, no optimization).
    running_std : torch.Tensor
        Running standard deviation per output, shape (n_outputs,). Registered
        as buffer (no gradients, no optimization).
    running_count : torch.Tensor
        Cumulative sample count for incremental statistics update.
    freeze_stats : bool
        If True, statistics are fixed. If False, statistics are updated
        incrementally each forward pass.

    Notes
    -----
    Operating Modes:
    1. Fixed statistics: If both `mean` and `std` are provided, statistics
       remain constant (freeze_stats=True, no updates).
    2. Dynamic statistics: If `mean` or `std` is None, statistics are
       initialized on first batch and updated incrementally with each batch,
       weighted by batch size.
    3. All statistics are stored as buffers (no gradient computation).

    Assumptions:
    - Inputs y_true and y_pred have shape (batch_size, n_outputs, ...)
    - Statistics are computed across the batch dimension
    - Additional dimensions beyond (batch_size, n_outputs) are flattened
      for statistics computation

    The incremental update uses Welford's online algorithm for numerical
    stability when combining batch statistics with running statistics.

    Examples
    --------
    >>> # Fixed statistics mode
    >>> mean = torch.tensor([0.5, 1.0])
    >>> std = torch.tensor([0.2, 0.5])
    >>> loss_fn = StandardizedMSELoss(mean=mean, std=std)

    >>> # Dynamic statistics mode
    >>> loss_fn = StandardizedMSELoss()  # Updates stats automatically
    """

    def __init__(self, mean=None, std=None, eps=1e-8):
        super().__init__()
        self.eps = eps

        if mean is not None and std is not None:
            self.register_buffer("running_mean", mean.clone().detach())
            self.register_buffer("running_std", std.clone().detach())
            self.freeze_stats = True
        else:
            # Initialized on first forward pass
            self.register_buffer("running_mean", torch.tensor([]))
            self.register_buffer("running_std", torch.tensor([]))
            self.freeze_stats = False

        self.register_buffer("running_count", torch.tensor(0.0))

    def update_stats(self, y_true):
        """
        Update running mean and std using incremental batch-weighted algorithm.

        Uses Welford's online algorithm to combine batch statistics with
        running statistics, weighted by the number of samples in each.

        Parameters
        ----------
        y_true : torch.Tensor
            Target tensor of shape (batch_size, n_outputs, ...).
            Additional dimensions beyond n_outputs are flattened.
        """
        # Flatten any extra dimensions beyond (batch_size, n_outputs)
        batch_size = y_true.shape[0]
        y_flat = y_true.view(batch_size, y_true.shape[1], -1)

        batch_mean = y_flat.mean(dim=(0, 2))  # (n_outputs,)
        batch_var = y_flat.var(dim=(0, 2), unbiased=False)

        if self.running_count == 0:
            # First batch: initialize statistics
            self.running_mean = batch_mean
            self.running_std = torch.sqrt(batch_var)
            self.running_count = torch.tensor(float(batch_size), device=y_true.device)
        else:
            # Incremental update with Welford's algorithm
            total_count = self.running_count + batch_size

            delta = batch_mean - self.running_mean

            new_mean = self.running_mean + delta * (batch_size / total_count)

            old_var = self.running_std**2
            new_var = (
                old_var * (self.running_count / total_count)
                + batch_var * (batch_size / total_count)
                + delta**2 * (self.running_count * batch_size / total_count**2)
            )

            self.running_mean = new_mean
            self.running_std = torch.sqrt(new_var)
            self.running_count = total_count

    def forward(self, y_pred, y_true):
        """
        Compute MSE between y_pred and y_true after per-output standardization.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions, shape (batch_size, n_outputs, ...).
        y_true : torch.Tensor
            Target values, same shape as y_pred.

        Returns
        -------
        torch.Tensor
            Scalar standardized MSE loss value.

        Notes
        -----
        Standardization formula per output i:
            z_i = (x_i - mean_i) / (std_i + eps)

        The loss is:
            L = mean((z_pred - z_true)^2)
        """
        if not self.freeze_stats:
            self.update_stats(y_true)

        mean = self.running_mean
        std = self.running_std + self.eps

        # Automatic broadcasting across batch and extra dimensions
        y_pred_std = (y_pred - mean) / std
        y_true_std = (y_true - mean) / std

        return torch.mean((y_pred_std - y_true_std) ** 2)


class QuantilicRangeLoss(torch.nn.Module):
    """
    Multi-output quantile range loss for interval regression.

    Computes the width of a confidence interval of the residual distribution
    using quantiles (q1, q2) for each output dimension.

    Parameters
    ----------
    confidence : float
        Confidence level in (0, 1). Example: 0.99 → interval [0.5%, 99.5%]
    """

    def __init__(self, confidence: float = 0.99):
        super().__init__()
        if not isinstance(confidence, float) or not 0 < confidence < 1:
            raise ValueError("confidence must be a float in (0, 1).")

        diff = (1 - confidence) / 2
        self.q1 = diff
        self.q2 = 1 - diff

    def forward(self, y_pred, y_true):
        """
        Compute multi-output quantile range loss.

        Parameters
        ----------
        y_pred : torch.Tensor (batch, n_outputs)
        y_true : torch.Tensor (batch, n_outputs)

        Returns
        -------
        torch.Tensor (scalar)
            Mean quantile range across outputs.
        """
        error = y_true - y_pred  # (batch, n_outputs)

        # Per‑output quantiles
        q1 = torch.quantile(error, self.q1, dim=0, keepdim=False)  # (n_outputs,)
        q2 = torch.quantile(error, self.q2, dim=0, keepdim=False)  # (n_outputs,)

        range_width = q2 - q1  # (n_outputs,)

        return range_width.mean()


class ComboLoss(torch.nn.Module):
    """
    Combine multiple loss functions by summing their outputs.

    This class allows you to aggregate several loss modules into a single loss,
    which is computed as the sum of the individual losses.

    Parameters
    ----------
    *losses : torch.nn.Module
        Loss modules to combine.
    """

    def __init__(self, *losses):
        """
        Initialize ComboLoss.

        Parameters
        ----------
        *losses : torch.nn.Module
            Loss modules to combine.
        """
        super().__init__()
        self.losses = torch.nn.ModuleList(losses)

    def forward(self, y_pred, y_true):
        """
        Compute the combined loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True values.

        Returns
        -------
        loss : torch.Tensor
            Combined loss value (sum of all individual losses).
        """
        return torch.stack([loss(y_pred, y_true) for loss in self.losses]).sum()


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


class TrainingLogger:
    """
    Training logger for tracking model training progress.

    Handles logging of losses, metrics, learning rates, and early-stopping state
    during model training. Provides methods to update, query, and export training
    history.

    Attributes
    ----------
    _history : dict[str, list]
        Dictionary storing all logged values as lists.
    _best_loss : float
        Best validation loss observed so far.
    _epochs_without_improvement : int
        Number of consecutive epochs without improvement.
    _patience : int
        Total patience for early stopping.
    """

    def __init__(
        self,
        early_stopping_patience: int = 1000,
    ):
        """
        Initialize the TrainingLogger.

        Parameters
        ----------
        early_stopping_patience : int, default=1000
            Maximum number of epochs without improvement before early stopping.
        """
        self._history = {}
        self._best_loss = float("inf")
        self._epochs_without_improvement = 0
        self._patience = early_stopping_patience
        self._last_print_lines = 0  # Track lines printed for in-place updates
        self._last_minimal_length = 0  # Track length of last minimal output
        self._start_time = None  # Training start time

    def update(self, key: str, value: float):
        """
        Append a value to the specified metric key.

        Parameters
        ----------
        key : str
            Metric name (e.g., 'training_loss', 'validation_loss').
        value : float
            Value to append.
        """
        self._history.setdefault(key, []).append(value)

    def update_early_stopping_state(
        self, current_val_loss: float, threshold: float
    ) -> bool:
        """
        Update early stopping state based on validation loss.

        Parameters
        ----------
        current_val_loss : float
            Current epoch's validation loss.
        threshold : float
            Minimum improvement threshold.

        Returns
        -------
        improved : bool
            True if validation loss improved beyond threshold.
        """
        if current_val_loss < self._best_loss - threshold:
            self._best_loss = current_val_loss
            self._epochs_without_improvement = 0
            return True
        else:
            self._epochs_without_improvement += 1
            return False

    def get_early_stopping_gap(self):
        """
        Get the remaining epochs before triggering early stopping.

        Returns
        -------
        gap : int
            Number of epochs remaining before early stopping is triggered.
        """
        return max(0, self._patience - self._epochs_without_improvement)

    def start_timer(self):
        """Start the training timer."""
        self._start_time = time.time()

    def get_elapsed_time(self):
        """
        Get formatted elapsed time since training started.

        Returns
        -------
        elapsed : str
            Formatted elapsed time (e.g., "1h 23m 45s" or "45s").
        """
        if self._start_time is None:
            return "0s"

        elapsed = time.time() - self._start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def log_learning_rate(self, lr: float):
        """
        Log the current learning rate.

        Parameters
        ----------
        lr : float
            Current learning rate.
        """
        self.update("learning_rate", lr)

    def get_current_epoch(self):
        """
        Get the current epoch number.

        Returns
        -------
        epoch : int
            Current epoch number (0 if no epochs logged).
        """
        return len(self._history.get("epoch", []))

    def get_last_value(self, key: str):
        """
        Get the most recent value for a metric.

        Parameters
        ----------
        key : str
            Metric name.

        Returns
        -------
        value : float or None
            Most recent value, or None if key doesn't exist.
        """
        values = self._history.get(key)
        return values[-1] if values else None

    def to_dataframe(self):
        """
        Export training history to a pandas DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with all logged metrics.
        """
        return pd.DataFrame(self._history)

    @property
    def history(self):
        """
        Access the full training history.

        Returns
        -------
        history : dict[str, list]
            Dictionary of all logged metrics.
        """
        return self._history

    @property
    def best_loss(self):
        """Best validation loss observed."""
        return self._best_loss

    @property
    def epochs_without_improvement(self):
        """Number of epochs without improvement."""
        return self._epochs_without_improvement

    def print_epoch_summary(
        self,
        epoch: int,
        val_loss: float,
        lr: float,
        verbose: str = "minimal",
    ):
        """
        Print a summary of the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        val_loss : float
            Validation loss for this epoch.
        lr : float
            Current learning rate.
        verbose : {'full', 'minimal', 'off'}, default='minimal'
            Verbosity level:
            - 'full': Multi-line output with per-output losses and metrics (updates in place)
            - 'minimal': Single-line output with global losses and metrics (updates in place)
            - 'off': No output

        Notes
        -----
        - 'full' mode displays training and validation losses/metrics for each model output
        - 'minimal' mode displays only global training/validation losses and metrics
        - Both modes update in place using ANSI escape codes for clean terminal output
        - 'full' mode shows: epoch, learning rate, best loss, epochs without improvement, gap, and elapsed time
        - 'minimal' mode shows: epoch, training/validation losses, learning rate, gap, and elapsed time
        """
        if verbose == "off":
            return

        gap = self.get_early_stopping_gap()
        elapsed_time = self.get_elapsed_time()

        if verbose == "full":
            # Clear previous output (move cursor up and clear lines)
            if self._last_print_lines > 0:
                for _ in range(self._last_print_lines):
                    print("\033[F\033[K", end="")  # Move up and clear line

            # Extract all training and validation metrics
            lines = [
                f"{'='*80}",
                f"Epoch {epoch} | LR: {lr:.2e} | Best: {self._best_loss:.6f} | "
                f"No Improve: {self._epochs_without_improvement} | Gap: {gap} | Time: {elapsed_time}",
                f"{'-'*80}",
            ]

            # Get training loss
            train_loss = self.get_last_value("training_loss")
            if train_loss is not None:
                lines.append(f"Training Loss:   {train_loss:.6f}")

            # Get validation loss
            lines.append(f"Validation Loss: {val_loss:.6f}")

            # Find all output-specific metrics
            output_keys = set()
            metric_keys = set()
            for key in self._history.keys():
                if "_loss" in key and key not in ["training_loss", "validation_loss"]:
                    # Extract output name (e.g., "training_output1_loss" -> "output1")
                    parts = key.split("_")
                    if len(parts) >= 3:
                        output_name = "_".join(
                            parts[1:-1]
                        )  # everything between step_type and "loss"
                        output_keys.add(output_name)
                elif key.startswith("training_") or key.startswith("validation_"):
                    # Check if it's a metric
                    parts = key.split("_")
                    if len(parts) >= 3 and parts[-1] not in ["loss"]:
                        metric_name = parts[-1]
                        metric_keys.add(metric_name)

            # Display per-output metrics
            if output_keys:
                lines.append(f"{'-'*80}")
                for output_name in sorted(output_keys):
                    lines.append(f"\n{output_name.upper()}:")

                    # Loss
                    train_out_loss = self.get_last_value(f"training_{output_name}_loss")
                    val_out_loss = self.get_last_value(f"validation_{output_name}_loss")
                    if train_out_loss is not None and val_out_loss is not None:
                        lines.append(
                            f"  Loss:    train={train_out_loss:.6f}  val={val_out_loss:.6f}"
                        )

                    # Metrics
                    for metric_name in sorted(metric_keys):
                        train_metric = self.get_last_value(
                            f"training_{output_name}_{metric_name}"
                        )
                        val_metric = self.get_last_value(
                            f"validation_{output_name}_{metric_name}"
                        )
                        if train_metric is not None and val_metric is not None:
                            lines.append(
                                f"  {metric_name.upper()}:  train={train_metric:.6f}  val={val_metric:.6f}"
                            )

            lines.append(f"{'='*80}")

            # Join and print
            output_text = "\n".join(lines)
            print(output_text)

            # Track number of lines printed (count newlines + 1)
            self._last_print_lines = output_text.count("\n") + 1

        elif verbose == "minimal":
            # Get training loss
            train_loss = self.get_last_value("training_loss")
            train_str = f"train={train_loss:.4f}" if train_loss is not None else ""

            # Build metric strings
            metric_strs = []
            metric_keys = set()
            for key in self._history.keys():
                if (
                    key.startswith("training_")
                    and not "_loss" in key
                    and not "learning_rate" in key
                ):
                    parts = key.split("_")
                    if len(parts) >= 2:
                        metric_name = parts[-1]
                        if f"validation_{metric_name}" in self._history or any(
                            f"validation_" in k and k.endswith(f"_{metric_name}")
                            for k in self._history.keys()
                        ):
                            metric_keys.add(metric_name)

            for metric_name in sorted(metric_keys):
                train_metric = self.get_last_value(f"training_{metric_name}")
                val_metric = self.get_last_value(f"validation_{metric_name}")

                # Try to find the metric in any output
                if train_metric is None or val_metric is None:
                    for key in self._history.keys():
                        if key.endswith(f"_{metric_name}"):
                            if key.startswith("training_"):
                                train_metric = self.get_last_value(key)
                            elif key.startswith("validation_"):
                                val_metric = self.get_last_value(key)

                if train_metric is not None and val_metric is not None:
                    metric_strs.append(
                        f"{metric_name}:t={train_metric:.4f}/v={val_metric:.4f}"
                    )

            metrics_part = " | ".join(metric_strs) if metric_strs else ""

            output = f"Epoch {epoch} | {train_str} | val={val_loss:.4f}"
            if metrics_part:
                output += f" | {metrics_part}"
            output += f" | lr={lr:.2e} | gap={gap} | time={elapsed_time}"

            # Clear previous output by overwriting with spaces
            if self._last_minimal_length > 0:
                print(" " * self._last_minimal_length, end="\r")

            # Print new output and track its length
            print(output, end="\r")
            self._last_minimal_length = len(output)


class TorchTrainer:
    """
    Trainer class for PyTorch models with support for multi-output training,
    early stopping, learning rate decay, metric tracking, and optional
    Uncertainty Weighting (Kendall & Gal, CVPR 2018).

    This trainer handles:
    - multi-output predictions (dict-based)
    - internal optimizer initialization
    - optional Uncertainty Weighting for automatic task balancing
    - early stopping with patience
    - learning rate decay after stagnation
    - detailed per-output and global logging of losses/metrics
    - validation split management and best weight restoration

    Parameters
    ----------
    loss : callable
        Loss function taking (y_pred, y_true) → scalar tensor.

    metrics : callable or list or dict
        Metric functions evaluated on predictions each epoch.

    optimizer_class : type(torch.optim.Optimizer), optional
        Class of the optimizer to create internally. Default: torch.optim.Adam

    optimizer_kwargs : dict, optional
        Additional keyword arguments for optimizer construction.
        Example: {"lr": 1e-3, "weight_decay": 1e-5}
        The 'lr' parameter can be a single float or a list of floats.
        If a list is provided, learning rates are applied sequentially:
        when early_stopping_patience is exceeded, the next learning rate
        is applied and weights are restored to the best state found so far.

    epochs : int, default=100000
        Maximum number of epochs.

    batch_size : int or None, default=None
        Batch size. If None, use full batch.

    early_stopping_threshold : float, default=1e-5
        Minimum delta in validation loss to reset patience.

    early_stopping_patience : int, default=2000
        Max number of epochs without improvement.

    validation_split : float, default=0.2
        Fraction of data used for validation.

    restore_best_weights : bool, default=True
        Whether to restore best model after early stopping.

    verbose : {'full', 'minimal', 'off'}, default='minimal'
        Verbosity level during training.

    debug : bool, default=False
        Enables gradient sanity checks.

    use_uncertainty_weighting : bool, default=False
        Whether to apply task uncertainty weighting.

    Attributes
    ----------
    logger : dict[str, list]
        Stores all training logs (losses, metrics, epochs).

    _uw_module : UncertaintyWeighting or None
        Internal module providing learnable loss weights.

    Notes
    -----
    - Models must return either a dict or a tensor.
    - Extra losses must be functions with no parameters.
    - Multi-output handling is native and automatic.
    - Learning rate scheduling: If optimizer_kwargs["lr"] is a list, the trainer will
      automatically switch to the next learning rate when early_stopping_patience is
      exceeded, restoring the best weights found so far. Training stops when all
      learning rates have been tried or max epochs is reached.
    """

    def __init__(
        self,
        loss=ComboLoss(PinballLoss(0.5), QuantilicRangeLoss(0.99)),
        metrics=[MAEMetric()],
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": [1e-3, 1e-4, 1e-5]},
        epochs: int = 100000,
        batch_size: int | None = None,
        early_stopping_threshold: float = 1e-5,
        early_stopping_patience: int = 200,
        validation_split: float = 0.2,
        restore_best_weights: bool = True,
        verbose: Literal["full", "minimal", "off"] = "minimal",
        debug: bool = False,
        use_uncertainty_weighting: bool = True,
    ):

        # ------------ BASIC VALIDATION ------------
        if batch_size is not None and not isinstance(batch_size, int):
            raise ValueError("'batch_size' must be int or None.")

        self._batch_size = batch_size
        self._loss = loss
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._epochs = epochs

        self._early_stopping_threshold = early_stopping_threshold
        self._early_stopping_patience = early_stopping_patience

        self._validation_split = validation_split
        self._restore_best_weights = restore_best_weights
        self._verbose = verbose
        self._debug = debug

        # ------------ LEARNING RATE SCHEDULING ------------
        # Handle both single lr and list of lrs
        lr_param = self._optimizer_kwargs.get("lr", 0.001)
        if isinstance(lr_param, list):
            self._learning_rates = lr_param.copy()
            self._current_lr_index = 0
        else:
            self._learning_rates = [lr_param]
            self._current_lr_index = 0

        # ------------ METRICS ------------
        if isinstance(metrics, Callable):
            self._metrics = {metrics.__class__.__name__.lower(): metrics}
        elif isinstance(metrics, list):
            self._metrics = {m.__class__.__name__.lower(): m for m in metrics}
        elif isinstance(metrics, dict):
            self._metrics = metrics
        else:
            raise ValueError("metrics must be callable, list, or dict.")

        # ------------ UNCERTAINTY WEIGHTING ------------
        self._use_uncertainty_weighting = use_uncertainty_weighting
        self._uw_module = None

        # ------------ LOGGER ------------
        self._logger = TrainingLogger(
            early_stopping_patience=early_stopping_patience,
        )

    # ==================================================================
    # LOGGING HELPERS
    # ==================================================================

    @property
    def logger(self):
        """Return training metrics history."""
        return self._logger.history

    def _update_logger(self, key: str, value: float):
        """Append a value to the given metric key."""
        self._logger.update(key, value)

    # ==================================================================
    # BATCH PROCESSING
    # ==================================================================

    def _process_batch(self, module, x_batch, y_batch, extra_losses=[]):
        """
        Compute predictions, mask invalid samples, compute per-output losses.

        Parameters
        ----------
        module : torch.nn.Module
            Model to evaluate.

        x_batch, y_batch : dict[str, Tensor] or Tensor
            Mini-batch inputs/targets.

        extra_losses : list[callable]
            Additional regularization terms.

        Returns
        -------
        batch_trues, batch_preds : dict[str, Tensor] or Tensor
            Valid samples only.

        batch_losses : dict[str, Tensor] or Tensor
            Per-output loss values.

        batch_samples : dict[str, int] or int
            Counts of valid samples.
        """

        preds = module(x_batch)
        was_dict = True

        if not isinstance(preds, dict):
            preds = {"output": preds}
            was_dict = False
        if not isinstance(y_batch, dict):
            y_batch = {"output": y_batch}
            was_dict = False

        batch_losses = {}
        batch_trues = {}
        batch_preds = {}
        batch_samples = {}

        for key in y_batch:
            mask = torch.isfinite(y_batch[key]) & torch.isfinite(preds[key])
            t = y_batch[key][mask]
            p = preds[key][mask]

            batch_trues[key] = t
            batch_preds[key] = p
            batch_samples[key] = mask.sum()
            batch_losses[key] = self._loss(p, t)

        for extra_loss in extra_losses:
            batch_losses[extra_loss.__name__] = extra_loss()

        if not was_dict:
            return (
                list(batch_trues.values())[0],
                list(batch_preds.values())[0],
                list(batch_losses.values())[0],
                list(batch_samples.values())[0],
            )

        return batch_trues, batch_preds, batch_losses, batch_samples

    # ==================================================================
    # STEP (TRAIN OR VALIDATION)
    # ==================================================================

    def _step(self, module, loader, step_type, extra_losses=[]):
        """
        Execute one full epoch pass (train or validation).

        Parameters
        ----------
        module : torch.nn.Module
        loader : DataLoader
        step_type : {"training", "validation"}
        extra_losses : list

        Notes
        -----
        - Applies UW weighting if active.
        - Logs per-output and global losses.
        - Computes all metrics.
        """

        if isinstance(loader.dataset.y, dict):
            keys = list(loader.dataset.y.keys())
            losses = {k: torch.tensor(0.0) for k in keys}
            samples = {k: torch.tensor(0.0) for k in keys}
            trues = {k: [] for k in keys}
            preds = {k: [] for k in keys}
        else:
            losses, samples, trues, preds = torch.tensor(0.0), torch.tensor(0.0), [], []

        total_loss = 0
        batches = 0

        for xb, yb in loader:
            bt, bp, bl, bs = self._process_batch(module, xb, yb, extra_losses)

            if isinstance(bl, dict):
                if self._use_uncertainty_weighting and self._uw_module:
                    batch_loss = self._uw_module(bl)
                else:
                    batch_loss = sum(bl.values())
            else:
                batch_loss = bl

            if step_type == "training":
                if isinstance(batch_loss, int):
                    raise ValueError("batch_loss is int but it cannot.")
                self._optimizer.zero_grad()
                batch_loss.backward()
                self._optimizer.step()

            with torch.no_grad():
                if isinstance(bl, dict):
                    for k, lv in bl.items():
                        v = lv.item()
                        n = bs[k].item()
                        losses[k] += v * n
                        samples[k] += n
                        trues[k].append(bt[k])
                        preds[k].append(bp[k])
                else:
                    v = bl.item()  # type: ignore
                    n = bs.item()  # type: ignore
                    losses += v * n
                    samples += n
                    trues.append(bt)  # type: ignore
                    preds.append(bp)  # type: ignore

                total_loss += batch_loss.item()  # type: ignore
                batches += 1

        epoch_loss = total_loss / batches
        self._update_logger(f"{step_type}_loss", epoch_loss)

        if not isinstance(losses, dict):
            losses = {"output": losses}
            samples = {"output": samples}
            trues = {"output": trues}
            preds = {"output": preds}

        for k in losses:
            avg = losses[k] / samples[k]  # type: ignore
            self._update_logger(f"{step_type}_{k}_loss", avg)  # type: ignore

            t = torch.cat(trues[k], 0)  # type: ignore
            p = torch.cat(preds[k], 0)  # type: ignore

            for mname, fun in self._metrics.items():
                self._update_logger(f"{step_type}_{k}_{mname}", fun(p, t).item())

    # ==================================================================
    # FIT
    # ==================================================================

    def fit(self, module, x_data, y_data, extra_losses=[]):
        """
        Fit the model to the provided dataset.

        Parameters
        ----------
        module : torch.nn.Module
            Model to train.

        x_data : tensor or dict[str, Tensor]
            Full input dataset.

        y_data : tensor or dict[str, Tensor]
            Full target dataset.

        extra_losses : list[callable]
            Additional zero-argument loss functions.

        Returns
        -------
        module : torch.nn.Module
            Trained model (best weights if restore_best_weights=True)

        history : pandas.DataFrame
            Training history.
        """

        # ---------------- SPLIT DATA ----------------
        with torch.no_grad():
            if isinstance(y_data, dict):
                arr = np.concatenate([v.numpy() for v in y_data.values()], 1)
            else:
                arr = y_data.numpy()

        arr = np.nanmean(arr, 1)
        splits = split_data(
            arr,
            {
                "training": 1 - self._validation_split,
                "validation": self._validation_split,
            },
            5,
        )

        train_idx, val_idx = list(splits.values())

        if isinstance(x_data, dict):
            train_x = {k: v[train_idx] for k, v in x_data.items()}
            val_x = {k: v[val_idx] for k, v in x_data.items()}
        else:
            train_x = x_data[train_idx]
            val_x = x_data[val_idx]

        if isinstance(y_data, dict):
            train_y = {k: v[train_idx] for k, v in y_data.items()}
            val_y = {k: v[val_idx] for k, v in y_data.items()}
        else:
            train_y = y_data[train_idx]
            val_y = y_data[val_idx]

        batch_size = self._batch_size or len(train_idx)

        train_loader = DataLoader(
            CustomDataset(train_x, train_y), batch_size, shuffle=True
        )
        val_loader = DataLoader(CustomDataset(val_x, val_y), batch_size, shuffle=True)

        # ---------------- INITIALIZE OPTIMIZER ----------------
        # Use the first learning rate from the list
        optimizer_kwargs_copy = self._optimizer_kwargs.copy()
        optimizer_kwargs_copy["lr"] = self._learning_rates[self._current_lr_index]

        self._optimizer = self._optimizer_class(
            module.parameters(), **optimizer_kwargs_copy
        )

        # ---------------- IF UW → ADD UW PARAMS ----------------
        if isinstance(train_y, dict) and self._use_uncertainty_weighting:
            keys = list(train_y.keys())
            device = next(module.parameters()).device
            self._uw_module = UncertaintyWeighting(keys).to(device)

            self._optimizer.add_param_group({"params": self._uw_module.parameters()})

        # ---------------- TRAINING LOOP ----------------
        best_weights = None

        # Start training timer
        self._logger.start_timer()

        for epoch in range(self._epochs):

            self._update_logger("epoch", epoch + 1)

            module.train()
            self._step(module, train_loader, "training", extra_losses)

            module.eval()
            self._step(module, val_loader, "validation")

            val_loss = self._logger.get_last_value("validation_loss")
            if val_loss is None:
                raise RuntimeError("Validation loss not found in logger")

            lr = self._optimizer.param_groups[0]["lr"]

            # Log learning rate
            self._logger.log_learning_rate(lr)

            # Update early stopping state
            improved = self._logger.update_early_stopping_state(
                val_loss, self._early_stopping_threshold
            )

            if improved:
                if self._restore_best_weights:
                    best_weights = module.state_dict().copy()

            # Print epoch summary
            self._logger.print_epoch_summary(epoch + 1, val_loss, lr, self._verbose)

            # Early stopping check with learning rate scheduling
            if self._logger.epochs_without_improvement >= self._early_stopping_patience:
                # Check if there are more learning rates to try
                if self._current_lr_index < len(self._learning_rates) - 1:
                    # Move to next learning rate
                    self._current_lr_index += 1
                    next_lr = self._learning_rates[self._current_lr_index]

                    # Restore best weights found so far
                    if best_weights is not None:
                        module.load_state_dict(best_weights)

                    # Update optimizer learning rate
                    for pg in self._optimizer.param_groups:
                        pg["lr"] = next_lr

                    # Reset early stopping counters
                    self._logger._epochs_without_improvement = 0

                    if self._verbose == "full":
                        print(
                            f"\n[LR Schedule] Switching to learning rate {next_lr:.2e} "
                            f"({self._current_lr_index + 1}/{len(self._learning_rates)})"
                        )
                else:
                    # No more learning rates to try, stop training
                    if self._restore_best_weights and best_weights is not None:
                        module.load_state_dict(best_weights)
                    break

        print("")
        module.eval()

        return module, self._logger.to_dataframe()
