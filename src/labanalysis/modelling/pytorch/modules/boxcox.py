"""BoxCoxTransform module."""

import torch


class BoxCoxTransform(torch.nn.Module):
    """
    Learnable Box-Cox transformation layer.

    Applies a parametric Box-Cox transformation with learnable lambda parameters:
        - If λ = 0: y = log(x)
        - If λ ≠ 0: y = (x^λ - 1) / λ

    Parameters
    ----------
    n_features : int
        Number of input features. Each feature has its own learnable lambda parameter.

    Attributes
    ----------
    n_features : int
        Number of features.
    lambda_param : torch.nn.Parameter
        Learnable lambda parameters of shape (n_features,), constrained to be
        positive via softplus activation.

    Notes
    -----
    - The lambda parameters are initialized to 1.0
    - Softplus activation ensures lambda values remain positive during training
    - Provides inverse transformation for reconstructing original values
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.lambda_param = torch.nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor):
        """
        Apply Box-Cox transformation to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size,) or (batch_size, n_features).

        Returns
        -------
        out : torch.Tensor
            Transformed tensor of the same shape as input.
        """
        if x.ndim == 1:
            x = x.unsqueeze(1)

        lambda_param = torch.nn.functional.softplus(self.lambda_param)
        lambda_param = lambda_param.unsqueeze(0)

        zero_mask = lambda_param == 0.0
        nonzero_mask = ~zero_mask

        out = torch.empty_like(x)

        out = torch.where(zero_mask, torch.log(torch.clamp(x, min=1e-8)), out)

        boxcox = (torch.pow(x, lambda_param) - 1) / torch.clamp(lambda_param, min=1e-8)
        out = torch.where(nonzero_mask, boxcox, out)

        return out

    def inverse(self, y: torch.Tensor):
        """
        Apply inverse Box-Cox transformation to reconstruct original values.

        Parameters
        ----------
        y : torch.Tensor
            Transformed tensor of shape (batch_size,) or (batch_size, n_features).

        Returns
        -------
        out : torch.Tensor
            Reconstructed tensor of the same shape as input.
        """
        if y.ndim == 1:
            y = y.unsqueeze(1)

        lambda_param = torch.nn.functional.softplus(self.lambda_param)
        lambda_param = lambda_param.unsqueeze(0)

        zero_mask = lambda_param == 0.0
        nonzero_mask = ~zero_mask

        out = torch.empty_like(y)
        out = torch.where(zero_mask, torch.exp(y), out)

        inv_boxcox = torch.pow(
            lambda_param * y + 1, 1 / torch.clamp(lambda_param, min=1e-8)
        )
        out = torch.where(nonzero_mask, inv_boxcox, out)

        return out
