"""Lasso module."""

import torch


class Lasso(torch.nn.Module):
    """
    Linear regression layer with learnable L1 (Lasso) penalization.

    Linear regression model where the L1 regularization coefficient is learned
    as a parameter during training, allowing adaptive feature selection.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output targets.
    bias : bool, default=True
        Whether to include bias term in the regression.

    Attributes
    ----------
    linear : torch.nn.Linear
        Linear transformation layer.
    alpha_raw : torch.nn.Parameter
        Raw learnable L1 penalty coefficients of shape (out_features, in_features).
        Transformed to positive values via softplus during loss computation.

    Notes
    -----
    Use `lasso_loss()` as a regularization term during training to apply
    adaptive L1 penalization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features,
            out_features,
            bias=bias,
        )
        self.alpha_raw = torch.nn.Parameter(torch.ones(out_features, in_features))

    def forward(self, x):
        """
        Apply linear transformation to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Output of linear regression, shape (batch_size, out_features).
        """
        return self.linear(x)

    def lasso_loss(self):
        """
        Compute adaptive L1 penalization with learned weights.

        Returns
        -------
        torch.Tensor
            Scalar L1 penalty value (sum of alpha * |weights|).
        """
        alpha = torch.log1p(torch.exp(self.alpha_raw))
        l1_penalty = torch.sum(alpha * torch.abs(self.linear.weight))
        return l1_penalty
