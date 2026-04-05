"""pytorch custom modules"""

import itertools

import torch

__all__ = ["FeaturesGenerator", "BoxCoxTransform", "SigmoidTransformer", "PCA", "Lasso"]


class FeaturesGenerator(torch.nn.Module):
    """
    Polynomial feature generator with transformations and interactions.

    Generates polynomial features from input tensors by applying various
    transformations (powers, logarithms, inverses) and optionally computing
    interaction terms between features.

    Parameters
    ----------
    order : int, default=2
        Maximum polynomial order for power transformations.
    apply_log_transform : bool, default=True
        Whether to apply logarithmic transformations (log1p).
    apply_inverse_transform : bool, default=True
        Whether to apply inverse transformations (1/x).
    include_interactions : bool, default=True
        Whether to generate interaction terms between different variables.
    input_keys : list of str, optional
        Specific input keys to process. If None, processes all inputs.

    Attributes
    ----------
    order : int
        Maximum polynomial order.
    apply_log_transform : bool
        Flag for logarithmic transformations.
    apply_inverse_transform : bool
        Flag for inverse transformations.
    include_interactions : bool
        Flag for interaction terms.
    input_keys : list of str or None
        Keys to process from input dictionary.

    Notes
    -----
    - Boolean tensors are treated differently (no log/inverse transformations)
    - NaN, inf, and -inf values are replaced with 0.0 in the output
    - Output keys are constructed by joining transformation names with underscores
    """

    def __init__(
        self,
        order: int = 2,
        apply_log_transform: bool = True,
        apply_inverse_transform: bool = True,
        include_interactions: bool = True,
        input_keys: list[str] | None = None,
    ):
        super().__init__()
        self.order = order
        self.apply_log_transform = apply_log_transform
        self.apply_inverse_transform = apply_inverse_transform
        self.include_interactions = include_interactions
        self.input_keys = input_keys if input_keys is not None else None

    def forward(self, inputs: dict[str, torch.Tensor]):
        """
        Generate polynomial features with transformations.

        Parameters
        ----------
        inputs : dict of str to torch.Tensor
            Dictionary of input tensors to transform.

        Returns
        -------
        outputs : dict of str to torch.Tensor
            Dictionary containing original inputs plus all generated features.
            Keys are constructed from transformation names (e.g., "x_pow2", "x_log",
            "x_inv", "x_x_y" for interactions).
        """
        outputs: dict[str, torch.Tensor] = {}
        transformed_by_var: dict[str, list[str]] = {}
        keys_to_use = (
            self.input_keys if self.input_keys is not None else list(inputs.keys())
        )
        epsilon = 1e-8

        for name in keys_to_use:
            tensor = inputs.get(name)
            if tensor is None:
                continue
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)

            outputs[name] = tensor
            transformed_by_var[name] = [name]

            # Apply transformations only if not boolean
            if tensor.dtype == torch.bool:
                if self.apply_inverse_transform:
                    inv_name = name + "_inv"
                    outputs[inv_name] = torch.sign(tensor) / torch.clamp(
                        torch.abs(tensor), min=epsilon
                    )
                    transformed_by_var[name].append(inv_name)

                if self.apply_log_transform:
                    log_name = name + "_log"
                    outputs[log_name] = torch.log1p(torch.clamp(tensor, min=0))
                    transformed_by_var[name].append(log_name)

                if self.apply_log_transform and self.apply_inverse_transform:
                    invlog_name = name + "_invlog"
                    log_tensor = torch.log1p(torch.clamp(tensor, min=0))
                    outputs[invlog_name] = 1 / torch.clamp(log_tensor, min=epsilon)
                    transformed_by_var[name].append(invlog_name)

                for p in range(2, self.order + 1):
                    pow_name = name + f"_pow{p}"
                    invpow_name = name + f"_invpow{p}"
                    outputs[pow_name] = tensor**p
                    outputs[invpow_name] = tensor ** (-p)
                    transformed_by_var[name].extend([pow_name, invpow_name])

        if self.include_interactions:
            feature_to_var = {
                feat: orig
                for orig, feats in transformed_by_var.items()
                for feat in feats
            }
            all_features = list(outputs.keys())
            for r in range(2, self.order + 1):
                for comb in itertools.combinations(all_features, r):
                    orig_vars = {feature_to_var[c] for c in comb}
                    if len(orig_vars) == r:
                        name = "_x_".join(comb)
                        prod = outputs[comb[0]]
                        for c in comb[1:]:
                            prod = prod * outputs[c]
                        outputs[name] = prod

        for key in outputs:
            if outputs[key].ndim == 1:
                outputs[key] = outputs[key].squeeze(1)
            outputs[key] = torch.nan_to_num(
                outputs[key],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        return outputs


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
        lambda_param = lambda_param.unsqueeze(0)  # shape: (1, n_features)

        # Costruisci maschera tensoriale per lambda == 0
        zero_mask = lambda_param == 0.0
        nonzero_mask = ~zero_mask

        # Prepara output tensor
        out = torch.empty_like(x)

        # Calcola log(x) dove lambda == 0
        out = torch.where(zero_mask, torch.log(torch.clamp(x, min=1e-8)), out)

        # Calcola Box-Cox dove lambda != 0
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
        # Move transform_dim to last
        x = x.transpose(self.transform_dim, -1)

        # Apply transformation
        projected = (x - self.J) @ self.Q
        activated = 1 / (1 + torch.exp(-projected))

        # Move back to original dimension order
        y = activated.transpose(-1, self.transform_dim)
        return y


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
        W = self.linear.weight  # Shape: [output_dim, input_dim]
        WT_W = torch.matmul(W, W.t())  # Shape: [output_dim, output_dim]
        I = torch.eye(WT_W.size(0), device=W.device)
        return torch.linalg.norm(WT_W - I, "fro")


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
        # Parametro raw che verrà trasformato in alpha tramite softplus
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
        alpha = torch.log1p(torch.exp(self.alpha_raw))  # softplus
        l1_penalty = torch.sum(alpha * torch.abs(self.linear.weight))
        return l1_penalty
