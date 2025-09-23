"""pytorch custom modules"""

import itertools
from typing import Dict, List, Optional, Union

import torch


class FeaturesGenerator(torch.nn.Module):
    """
    Feature expansion module for tabular data.

    Applies polynomial, logarithmic, inverse, and interaction transformations
    to input features for use in machine learning models.

    Parameters
    ----------
    order : int, optional
        Maximum degree of polynomial and interaction terms (default is 2).
    apply_log_transform : bool, optional
        Whether to apply log(x + 1) transformation to each feature
        (default is True).
    apply_inverse_transform : bool, optional
        Whether to apply inverse transformation 1 / max(|x|, ε) to each feature
        (default is True).
    include_interactions : bool, optional
        Whether to include interaction terms between features (default is True).
    input_keys : list of str or None, optional
        List of expected input feature names. If None, all keys in the input
        dict are used (default is None).
    """

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
        """
        Apply feature transformations and interactions.

        Parameters
        ----------
        inputs : dict of str to torch.Tensor
            Dictionary of input features. Each tensor must have shape
            (batch_size,) or (batch_size, 1).

        Returns
        -------
        outputs : dict of str to torch.Tensor
            Dictionary containing selected features, their transformations,
            and valid interactions.
            All output tensors have shape (batch_size, 1).
        """
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

            is_boolean = torch.all(torch.eq(tensor, 0) | torch.eq(tensor, 1))
            if not is_boolean:
                if self.apply_inverse_transform:
                    inv_name = name + "_inv"
                    outputs[inv_name] = (
                        1
                        * torch.sign(tensor)
                        / torch.clamp(torch.abs(tensor), min=epsilon)
                    )

                    transformed_by_var[name].append(inv_name)

                if self.apply_log_transform:
                    log_name = name + "_log"
                    outputs[log_name] = torch.log1p(torch.clamp(tensor, min=0))
                    transformed_by_var[name].append(log_name)

                if self.apply_log_transform and self.apply_inverse_transform:
                    invlog_name = name + "_invlog"
                    outputs[invlog_name] = 1 / torch.max(
                        torch.tensor(epsilon), torch.log1p(torch.clamp(tensor, min=0))
                    )
                    transformed_by_var[name].append(invlog_name)

                for p in range(1, self.order + 1):
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
                        tensors = [outputs[c] for c in comb]
                        prod = tensors[0]
                        for t in tensors[1:]:
                            prod = prod * t
                        outputs[name] = prod

        return outputs

    def get_config(self):
        """
        Return the configuration of the module.

        Returns
        -------
        config : dict
            Dictionary containing the values of initialization parameters.
        """
        return {
            "order": self.order,
            "apply_log_transform": self.apply_log_transform,
            "apply_inverse_transform": self.apply_inverse_transform,
            "include_interactions": self.include_interactions,
            "input_keys": self.input_keys,
        }

    @staticmethod
    def from_config(config: dict):
        """
        Create a FeaturesGenerator instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing keys: 'order', 'apply_log_transform',
            'apply_inverse_transform', 'include_interactions', 'input_keys'.

        Returns
        -------
        generator : FeaturesGenerator
            A new instance of FeaturesGenerator initialized with the given config.
        """
        return FeaturesGenerator(
            order=config.get("order", 2),
            apply_log_transform=config.get("apply_log_transform", True),
            apply_inverse_transform=config.get("apply_inverse_transform", True),
            include_interactions=config.get("include_interactions", True),
            input_keys=config.get("input_keys", None),
        )


class BoxCoxTransform(torch.nn.Module):
    """
    Box-Cox transformation layer with learnable lambda parameters.

    Parameters
    ----------
    n_features : int
        Number of input features to transform.
    """

    def __init__(self, n_features: int):
        super(BoxCoxTransform, self).__init__()
        self.n_features = n_features
        self.lambda_param = torch.nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor):
        """
        Apply the Box-Cox transformation to input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        transformed : torch.Tensor
            Transformed tensor of shape (batch_size, n_features).
        """
        if x.ndim == 1:
            x = x.unsqueeze(1)

        lambda_param = torch.nn.functional.softplus(self.lambda_param)
        lambda_param = lambda_param.unsqueeze(0)  # shape: (1, n_features)

        return torch.where(
            lambda_param == 0,
            torch.log(x),
            (torch.pow(x, lambda_param) - 1) / lambda_param,
        )

    def inverse(self, y: torch.Tensor):
        """
        Apply the inverse Box-Cox transformation.

        Parameters
        ----------
        y : torch.Tensor
            Transformed tensor of shape (batch_size, n_features).

        Returns
        -------
        original : torch.Tensor
            Original tensor before transformation.
        """
        if y.ndim == 1:
            y = y.unsqueeze(1)

        lambda_param = torch.nn.functional.softplus(self.lambda_param)
        lambda_param = lambda_param.unsqueeze(0)  # shape: (1, n_features)

        return torch.where(
            lambda_param == 0,
            torch.exp(y),
            torch.pow(lambda_param * y + 1, 1 / lambda_param),
        )

    def get_config(self):
        """
        Return the configuration of the module.

        Returns
        -------
        config : dict
            Dictionary containing the number of features.
        """
        return {"n_features": self.n_features}

    @staticmethod
    def from_config(config: dict):
        """
        Create a BoxCoxTransform instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing key 'n_features'.

        Returns
        -------
        transform : BoxCoxTransform
            A new instance of BoxCoxTransform initialized with the given config.
        """
        return BoxCoxTransform(n_features=config["n_features"])


class MinMaxScaler(torch.nn.Module):
    """
    Min-Max scaling layer with optional learnable parameters.

    Parameters
    ----------
    min_value : list or torch.Tensor or None, optional
        Minimum values for each input feature. If None, initialized as learnable
        parameters.
    max_value : list or torch.Tensor or None, optional
        Maximum values for each input feature. If None, initialized as learnable
        parameters.
    input_dim : int or None, optional
        Number of input features. Required if min_value or max_value is None.
    """

    def __init__(
        self,
        min_value: Optional[Union[List[float], torch.Tensor]] = None,
        max_value: Optional[Union[List[float], torch.Tensor]] = None,
        input_dim: Optional[int] = None,
    ):
        super().__init__()

        if (min_value is None or max_value is None) and input_dim is None:
            msg = "'input_dim' must be provided if either 'min_value' or "
            msg += "'max_value' is None."
            raise ValueError(msg)

        self.input_dim = input_dim

        if min_value is None:
            if input_dim is None:
                raise ValueError("min_value and input_dim cannot be both None.")
            self.min_value = torch.nn.Parameter(torch.zeros(1, input_dim))
        else:
            min_tensor = torch.tensor(min_value, dtype=torch.float32).view(1, -1)
            self.register_buffer("min_value", min_tensor)

        if max_value is None:
            if input_dim is None:
                raise ValueError("max_value and input_dim cannot be both None.")
            self.max_value = torch.nn.Parameter(torch.ones(1, input_dim))
        else:
            max_tensor = torch.tensor(max_value, dtype=torch.float32).view(1, -1)
            self.register_buffer("max_value", max_tensor)

    def forward(self, x: torch.Tensor):
        """
        Apply min-max scaling to input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        scaled : torch.Tensor
            Scaled tensor of shape (batch_size, input_dim).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        slope = self.max_value - self.min_value
        intercept = self.min_value
        return torch.nn.functional.sigmoid(x) * slope + intercept

    def inverse(self, y: torch.Tensor):
        """
        Invert the min-max scaling transformation.

        Parameters
        ----------
        y : torch.Tensor
            Scaled tensor of shape (batch_size, input_dim).

        Returns
        -------
        original : torch.Tensor
            Original tensor before scaling.
        """
        if y.ndim == 1:
            y = y.unsqueeze(0)

        slope = self.max_value - self.min_value
        intercept = self.min_value
        # Undo min-max scaling
        sig_x = (y - intercept) / slope
        # Clamp to avoid logit instability at 0 or 1
        sig_x = torch.clamp(sig_x, min=1e-6, max=1 - 1e-6)
        # Apply inverse sigmoid (logit)
        original = torch.log(sig_x / (1 - sig_x))
        return original

    def get_config(self):
        """
        Return the configuration of the module.

        Returns
        -------
        config : dict
            Dictionary containing the values of initialization parameters.
        """
        return {
            "input_dim": self.input_dim,
            "min_value": (
                self.min_value.detach().cpu().numpy().tolist()
                if isinstance(self.min_value, torch.Tensor)
                and not self.min_value.requires_grad
                else None
            ),
            "max_value": (
                self.max_value.detach().cpu().numpy().tolist()
                if isinstance(self.max_value, torch.Tensor)
                and not self.max_value.requires_grad
                else None
            ),
        }

    @staticmethod
    def from_config(config: dict):
        """
        Create a MinMaxScaler instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing keys: 'input_dim', 'min_value', 'max_value'.

        Returns
        -------
        scaler : MinMaxScaler
            A new instance of MinMaxScaler initialized with the given config.
        """
        return MinMaxScaler(
            min_value=config.get("min_value", None),
            max_value=config.get("max_value", None),
            input_dim=config.get("input_dim", None),
        )
