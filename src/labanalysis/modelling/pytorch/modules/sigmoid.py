"""SigmoidTransformer module."""

import torch


class SigmoidTransformer(torch.nn.Module):
    """
    Sigmoid transformation layer with learnable parameters.

    Applies the transformation:
        Y = 1 / (1 + exp(-((X - J) @ Q)))

    Parameters
    ----------
    input_dim : int
        Dimension of the input features (K).
    output_dim : int
        Dimension of the output features (M).
    transform_dim : int
        Axis along which to apply the transformation.
    """

    def __init__(self, input_dim: int, output_dim: int, transform_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform_dim = transform_dim

        self.J = torch.nn.Parameter(torch.zeros(1, input_dim))
        self.Q = torch.nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, x: torch.Tensor):
        """
        Apply sigmoid transformation to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with transform_dim dimension of size input_dim.

        Returns
        -------
        y : torch.Tensor
            Transformed tensor with transform_dim dimension of size output_dim.
        """
        x = x.transpose(self.transform_dim, -1)

        projected = (x - self.J) @ self.Q
        activated = 1 / (1 + torch.exp(-projected))

        y = activated.transpose(-1, self.transform_dim)
        return y
