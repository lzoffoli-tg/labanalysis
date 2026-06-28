"""FeaturesGenerator module."""

import itertools

import torch


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
