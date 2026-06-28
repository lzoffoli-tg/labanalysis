"""Loss functions for PyTorch training."""

import torch


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
