"""pytorch custom modules"""

import itertools
from typing import Dict, List, Optional, Union

import torch

__all__ = ["FeaturesGenerator", "BoxCoxTransform", "SigmoidTransformer", "PCA", "Lasso"]


class FeaturesGenerator(torch.nn.Module):
    def __init__(
        self,
        order: int = 2,
        apply_log_transform: bool = True,
        apply_inverse_transform: bool = True,
        include_interactions: bool = True,
        input_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.order = order
        self.apply_log_transform = apply_log_transform
        self.apply_inverse_transform = apply_inverse_transform
        self.include_interactions = include_interactions
        self.input_keys = input_keys if input_keys is not None else None

    def forward(self, inputs: Dict[str, torch.Tensor]):
        outputs: Dict[str, torch.Tensor] = {}
        transformed_by_var: Dict[str, List[str]] = {}
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
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.lambda_param = torch.nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor):
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

        # Move transform_dim to last
        x = x.transpose(self.transform_dim, -1)

        # Apply transformation
        projected = (x - self.J) @ self.Q
        activated = 1 / (1 + torch.exp(-projected))

        # Move back to original dimension order
        y = activated.transpose(-1, self.transform_dim)
        return y


class PCA(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        PCA-like layer with learnable orthogonality via regularization.

        Args:
            input_dim (int): Dimensionality of input features.
            output_dim (int): Dimensionality of output features.
        """
        super().__init__()
        self.linear = torch.nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )

    def forward(self, x):
        return self.linear(x)

    def orthogonality_loss(self):
        """
        Computes the orthogonality loss for the weight matrix.
        Encourages rows of the weight matrix to be orthonormal.
        """
        W = self.linear.weight  # Shape: [output_dim, input_dim]
        WT_W = torch.matmul(W, W.t())  # Shape: [output_dim, output_dim]
        I = torch.eye(WT_W.size(0), device=W.device)
        return torch.linalg.norm(WT_W - I, "fro")


class Lasso(torch.nn.Module):
    """
    Modello di regressione lineare con penalizzazione L1 personalizzata
    (tipo Lasso), dove il coefficiente di penalizzazione è appreso come
    parametro.

    Args:
        in_features (int): Numero di feature in input.
        out_features (int): Numero di output.
        bias (bool, optional): Se includere il termine di bias nella regressione.
        Default: True.
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
        Applica la trasformazione lineare all'input.

        Args:
            x (Tensor): Input tensor di forma (batch_size, in_features).

        Returns:
            Tensor: Output della regressione lineare.
        """
        return self.linear(x)

    def lasso_loss(self):
        """
        Calcola la penalizzazione L1 con pesi adattivi appresi.

        Returns:
            Tensor: Valore della penalizzazione L1.
        """
        alpha = torch.log1p(torch.exp(self.alpha_raw))  # softplus
        l1_penalty = torch.sum(alpha * torch.abs(self.linear.weight))
        return l1_penalty
