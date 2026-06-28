"""Dataset utilities for PyTorch training."""

import torch
from torch.utils.data import Dataset


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
        # Simplified indexing - DataLoader handles batching, no need for unsqueeze
        if isinstance(self._x, dict):
            x_sample = {k: v[idx] for k, v in self._x.items()}
        else:
            x_sample = self._x[idx]

        if isinstance(self._y, dict):
            y_sample = {k: v[idx] for k, v in self._y.items()}
        else:
            y_sample = self._y[idx]

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
