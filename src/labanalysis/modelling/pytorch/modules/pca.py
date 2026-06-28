"""PCA module."""

import torch


class PCA(torch.nn.Module):
    """
    PCA-like dimensionality reduction layer with learnable orthogonality.

    Linear projection layer that can be trained with orthogonality constraints
    via regularization loss, similar to Principal Component Analysis.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    output_dim : int
        Dimensionality of output features (number of components).

    Attributes
    ----------
    linear : torch.nn.Linear
        Linear transformation layer without bias.

    Notes
    -----
    Use `orthogonality_loss()` as a regularization term during training to
    encourage orthonormal rows in the weight matrix.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )

    def forward(self, x):
        """
        Apply linear projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Projected tensor of shape (..., output_dim).
        """
        return self.linear(x)

    def orthogonality_loss(self):
        """
        Compute orthogonality regularization loss.

        Encourages rows of the weight matrix to be orthonormal by penalizing
        deviation from identity in W @ W^T.

        Returns
        -------
        torch.Tensor
            Frobenius norm of (W @ W^T - I), where W is the weight matrix.
        """
        W = self.linear.weight
        WT_W = torch.matmul(W, W.t())
        I = torch.eye(WT_W.size(0), device=W.device)
        return torch.linalg.norm(WT_W - I, "fro")
