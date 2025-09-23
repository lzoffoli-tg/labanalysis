"""pytorch custom utils

This module provides utility classes for PyTorch model training, including a custom
dataset for structured input/output and a trainer class with early stopping, learning
rate decay, and metric tracking.
"""

import warnings
from datetime import datetime
from os import makedirs
from os.path import dirname
from typing import Callable, Dict, Literal

import numpy as np
import pandas as pd
import torch
from onnxmodels import OnnxModel
from torch.utils.data import DataLoader, Dataset

from ...utils import split_data


__all__ = [
    "CustomDataset",
    "TorchTrainer",
    "PinballLoss",
    "QuantilicRangeLoss",
    "ComboLoss",
    "MAEMetric",
    "ModelWrapper",
    "to_onnx",
]


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

    def __init__(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]):
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
        if not isinstance(x, dict):
            raise ValueError("x must be a dict of torch.Tensors")
        x_sizes = [v.shape[0] for v in x.values()]
        if not all(size == x_sizes[0] for size in x_sizes):
            raise ValueError("All x tensors must have the same number of samples")
        size_x = x_sizes[0]

        # Validate y
        if not isinstance(y, dict):
            raise ValueError("y must be a dict of torch.Tensors")
        y_sizes = [v.shape[0] for v in y.values()]
        if not all(size == y_sizes[0] for size in y_sizes):
            raise ValueError("All y tensors must have the same number of samples")
        size_y = y_sizes[0]

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
        x_sample = {
            k: v[idx].unsqueeze(0) if v[idx].ndim < 1 else v[idx]
            for k, v in self._x.items()
        }
        y_sample = {
            k: v[idx].unsqueeze(0) if v[idx].ndim < 1 else v[idx]
            for k, v in self._y.items()
        }
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


class TorchTrainer:
    """
    Trainer class for PyTorch models with support for early stopping,
    learning rate decay, and metric tracking.

    Parameters
    ----------
    loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function to optimize.
    metrics : Callable or list or dict
        Metric(s) to evaluate model performance. Can be a single callable,
        a list of callables, or a dictionary of named callables.
    optimizer : torch.optim.Optimizer
        Optimizer instance from PyTorch.
    epochs : int, optional
        Number of training epochs (default is 100000).
    batch_size : int or None, optional
        Size of training batches. If None, full-batch training is used
        (default is None).
    early_stopping_threshold : float, optional
        Minimum change in loss to continue training (default is 1e-5).
    early_stopping_patience : int, optional
        Number of epochs to wait before stopping if no improvement
        (default is 2000).
    learning_rate : float, optional
        Initial learning rate (default is 1e-3).
    decaying_rate : float or None, optional
        Learning rate decay factor (default is 1e-6).
    minimum_learning_rate : float, optional
        Minimum learning rate allowed (default is 1e-6).
    variance_adjusted_loss : bool, optional
        Whether to use variance-adjusted loss (default is True).
    restore_best_weights : bool, optional
        Whether to restore best weights after early stopping (default is True).
    verbose : {'full', 'minimal', 'off'}, optional
        Verbosity level of training output (default is 'minimal').
    debug : bool, optional
        Enable debug mode (default is False).

    Attributes
    ----------
    batch_size : int or None
        Batch size used for training.
    learning_rate : float
        Learning rate for optimizer.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    loss : Callable
        Loss function.
    variance_adjusted_loss : bool
        Whether variance-adjusted loss is used.
    metrics : dict
        Dictionary of metric functions.
    epochs : int
        Number of epochs.
    early_stopping_threshold : float
        Early stopping threshold.
    early_stopping_patience : int
        Early stopping patience.
    decaying_rate : float or None
        Learning rate decay factor.
    minimum_learning_rate : float
        Minimum learning rate.
    restore_best_weights : bool
        Whether to restore best weights.
    verbose : str
        Verbosity level.
    debug : bool
        Debug mode.
    logger : dict
        Training history.
    loss_weights : dict
        Loss weights for each output.
    """

    def __init__(
        self,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            | list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            | dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
        ),
        optimizer: torch.optim.Optimizer,
        epochs: int = 100000,
        batch_size: int | None = None,
        early_stopping_threshold: int | float = 1e-5,
        early_stopping_patience: int = 2000,
        learning_rate: float = 1e-3,
        decaying_rate: float | None = 1e-6,
        minimum_learning_rate: float = 1e-6,
        variance_adjusted_loss: bool = True,
        restore_best_weights: bool = True,
        verbose: Literal["full", "minimal", "off"] = "minimal",
        debug: bool = False,
    ):
        """
        Initialize the TorchTrainer.

        See class docstring for parameter details.
        """
        if batch_size is not None and not isinstance(batch_size, int):
            raise ValueError("'batch_size' must be a positive int or None.")
        self._batch_size = batch_size

        if not isinstance(learning_rate, float):
            raise ValueError("'learning_rate' must be a float.")
        self._learning_rate = learning_rate

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("'optimizer' must be a subclass of torch.optim.Optimizer")
        self._optimizer = optimizer

        if not isinstance(loss, Callable):
            raise ValueError("'loss' must be a callable")
        self._loss = loss

        if not isinstance(variance_adjusted_loss, bool):
            raise ValueError("variance_adjusted_loss must be True or False")
        self._variance_adjusted_loss = variance_adjusted_loss

        if isinstance(metrics, Callable):
            self._metrics = {metrics.__class__.__name__.lower(): metrics}
        elif isinstance(metrics, list):
            self._metrics = {}
            for i in metrics:
                if not isinstance(i, Callable):
                    raise ValueError("metrics elements must all be callable objects.")
                self._metrics[i.__class__.__name__.lower()] = i
        elif isinstance(metrics, dict):
            for i, v in metrics.items():
                if not isinstance(v, Callable):
                    raise ValueError("metrics elements must all be callable objects.")
            self._metrics = metrics
        else:
            msg = "metrics must be a callable, a list of callable objects or a "
            msg += "dict of callable objects."
            raise ValueError(msg)

        if not isinstance(epochs, int):
            raise ValueError("'epochs' must be a positive int.")
        self._epochs = epochs

        if not isinstance(early_stopping_threshold, (int, float)):
            raise ValueError("'early_stopping_threshold' must be an int or float.")
        self._early_stopping_threshold = early_stopping_threshold

        if not isinstance(early_stopping_patience, int):
            raise ValueError("'early_stopping_patience' must be a positive int.")
        self._early_stopping_patience = early_stopping_patience

        if decaying_rate is not None and not isinstance(decaying_rate, float):
            raise ValueError("'decaying_rate' must be a float.")
        self._decaying_rate = decaying_rate

        if not isinstance(minimum_learning_rate, float):
            raise ValueError("'minimum_learning_rate' must be a float.")
        self._minimum_learning_rate = minimum_learning_rate

        if not isinstance(restore_best_weights, bool):
            raise ValueError("'restore_best_weights' must be True or False.")
        self._restore_best_weights = restore_best_weights

        if verbose not in ["full", "minimal", "off"]:
            raise ValueError("'verbose' must be one of 'full', 'minimal', or 'off'.")
        self._verbose = verbose

        if not isinstance(debug, bool):
            raise ValueError("'debug' must be True or False.")
        self._debug = debug

        self._logger = {}
        self._loss_weights: dict[str, float | int | torch.Tensor] = {}

    @property
    def batch_size(self):
        """int or None: Batch size used for training."""
        return self._batch_size

    @property
    def learning_rate(self):
        """float: Learning rate for optimizer."""
        return self._learning_rate

    @property
    def optimizer(self):
        """torch.optim.Optimizer: Optimizer instance."""
        return self._optimizer

    @property
    def loss(self):
        """callable: Loss function."""
        return self._loss

    @property
    def variance_adjusted_loss(self):
        """bool: Whether variance-adjusted loss is used."""
        return self._variance_adjusted_loss

    @property
    def metrics(self):
        """dict: Dictionary of metric functions."""
        return self._metrics

    @property
    def epochs(self):
        """int: Number of epochs."""
        return self._epochs

    @property
    def early_stopping_threshold(self):
        """float: Early stopping threshold."""
        return self._early_stopping_threshold

    @property
    def early_stopping_patience(self):
        """int: Early stopping patience."""
        return self._early_stopping_patience

    @property
    def decaying_rate(self):
        """float or None: Learning rate decay factor."""
        return self._decaying_rate

    @property
    def minimum_learning_rate(self):
        """float: Minimum learning rate."""
        return self._minimum_learning_rate

    @property
    def restore_best_weights(self):
        """bool: Whether to restore best weights."""
        return self._restore_best_weights

    @property
    def verbose(self):
        """str: Verbosity level."""
        return self._verbose

    @property
    def debug(self):
        """bool: Debug mode."""
        return self._debug

    @property
    def logger(self):
        """dict: Training history."""
        return self._logger

    @property
    def loss_weights(self):
        """dict: Loss weights for each output."""
        return self._loss_weights

    def _update_logger(self, key: str, value: float):
        """
        Update the logger with a new value for a given key.

        Parameters
        ----------
        key : str
            The key for the logger entry.
        value : float
            The value to append.
        """
        if key not in list(self._logger.keys()):
            self._logger[key] = []
        self._logger[key].append(value)

    def _process_batch(
        self,
        module: torch.nn.Module,
        x_batch: dict[str, torch.Tensor],
        y_batch: dict[str, torch.Tensor],
    ):
        """
        Process a single batch for forward and backward pass.

        Parameters
        ----------
        module : torch.nn.Module
            The model to train/evaluate.
        x_batch : dict of str to torch.Tensor
            Batch of input tensors.
        y_batch : dict of str to torch.Tensor
            Batch of target tensors.

        Returns
        -------
        batch_trues : dict
            True values for each output.
        batch_preds : dict
            Predicted values for each output.
        batch_losses : dict
            Loss for each output.
        batch_samples : dict
            Number of valid samples for each output.
        """
        z_batch = module(x_batch)

        batch_losses = {}
        batch_samples = {}
        batch_trues = {}
        batch_preds = {}
        for key in y_batch:
            true_samples = torch.isfinite(y_batch[key])
            pred_samples = torch.isfinite(z_batch[key])
            valid_samples = true_samples & pred_samples
            valid_trues = y_batch[key][valid_samples]
            valid_preds = z_batch[key][valid_samples]

            batch_samples[key] = torch.sum(valid_samples)
            batch_losses[key] = self.loss(valid_trues, valid_preds)
            batch_losses[key] = batch_losses[key] * self.loss_weights[key]
            batch_trues[key] = valid_trues
            batch_preds[key] = valid_preds

        return batch_trues, batch_preds, batch_losses, batch_samples

    def _step(
        self,
        module: torch.nn.Module,
        loader: DataLoader,
        step_type: Literal["training", "validation"],
        verbose: Literal["full", "minimal", "off"] = "off",
    ):
        """
        Perform a training or validation step over all batches.

        Parameters
        ----------
        module : torch.nn.Module
            The model to train/evaluate.
        loader : DataLoader
            DataLoader for the step.
        step_type : {'training', 'validation'}
            Type of step.
        verbose : {'full', 'minimal', 'off'}, optional
            Verbosity level.
        """
        loss_keys = list(loader.dataset.y.keys())  # type: ignore
        losses = {key: 0.0 for key in loss_keys}
        samples = {key: 0.0 for key in loss_keys}
        trues: dict[str, list[torch.Tensor]] = {key: [] for key in loss_keys}
        preds: dict[str, list[torch.Tensor]] = {key: [] for key in loss_keys}
        loss = 0
        batches = 0
        for xbatch, ybatch in loader:
            batch_outs = self._process_batch(module, xbatch, ybatch)
            batch_trues, batch_preds, batch_losses, batch_samples = batch_outs

            batch_loss = torch.sum(torch.stack(list(batch_losses.values())))

            if step_type == "training":
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                if self.debug:
                    for name, param in module.named_parameters():
                        if param.grad is None:
                            raise ValueError(f"gradient of {name} is None")
                        elif torch.isnan(param.grad).any():
                            raise ValueError(f"NaN in gradient of {name}")
                        elif torch.sum(param.grad) == 0:
                            warnings.warn(f"gradient of {name} is zero.")

            with torch.no_grad():
                for key in loss_keys:
                    loss_val = batch_losses[key].item()
                    n_samp = batch_samples[key].item()
                    losses[key] += loss_val * n_samp
                    samples[key] += n_samp
                    true_values = batch_trues[key]
                    pred_values = batch_preds[key]
                    trues[key].append(true_values)
                    preds[key].append(pred_values)
                loss += batch_loss.item()
                batches += 1

        with torch.no_grad():
            loss /= batches
            model_name = module.__class__.__name__
            self._update_logger(f"{step_type}_{model_name}_loss", loss)
            for key in loss_keys:
                key_loss = losses[key] / samples[key]
                self._update_logger(f"{step_type}_{key}_loss", key_loss)
                trues_ten = torch.cat(trues[key], dim=0)
                preds_ten = torch.cat(preds[key], dim=0)
                for metric, fun in self.metrics.items():
                    lbl = f"{step_type}_{key}_{metric}"
                    val = fun(preds_ten, trues_ten).item()
                    self._update_logger(lbl, val)

    def _train_step(
        self,
        module: torch.nn.Module,
        train_loader: DataLoader,
        verbose: Literal["full", "minimal", "off"] = "minimal",
    ):
        """
        Perform a training step.

        Parameters
        ----------
        module : torch.nn.Module
            The model to train.
        train_loader : DataLoader
            DataLoader for training data.
        verbose : {'full', 'minimal', 'off'}, optional
            Verbosity level.
        """
        module.train()
        self._step(module, train_loader, "training", verbose)

    @torch.no_grad()
    def _validation_step(
        self,
        module: torch.nn.Module,
        val_loader: DataLoader,
        verbose: Literal["full", "minimal", "off"] = "minimal",
    ):
        """
        Perform a validation step.

        Parameters
        ----------
        module : torch.nn.Module
            The model to evaluate.
        val_loader : DataLoader
            DataLoader for validation data.
        verbose : {'full', 'minimal', 'off'}, optional
            Verbosity level.
        """
        module.eval()
        self._step(module, val_loader, "validation", verbose)

    def fit(
        self,
        module: torch.nn.Module,
        x_data: torch.Tensor | dict[str, torch.Tensor],
        y_data: torch.Tensor | dict[str, torch.Tensor],
    ):
        """
        Fit the model to the data.

        Parameters
        ----------
        module : torch.nn.Module
            The model to train.
        x_data : torch.Tensor or dict of str to torch.Tensor
            Input data.
        y_data : torch.Tensor or dict of str to torch.Tensor
            Target data.

        Returns
        -------
        module : torch.nn.Module
            The trained model.
        history : pd.DataFrame
            Training history as a DataFrame.
        """
        invalid_type_msg = "{} must be a torch.Tensor with shape "
        invalid_type_msg += "[batch_size, n_features] or a "
        invalid_type_msg += "dict[str, torch.Tensor] with each tensor having"
        invalid_type_msg += " shape [batch_size, 1]."
        if isinstance(x_data, dict):
            if self.debug and not (
                all(isinstance(i, torch.Tensor) for i in x_data.values())
                and all(i.ndim == 2 for i in x_data.values())
                and all(i.shape[1] == 1 for i in x_data.values())
            ):
                raise TypeError(invalid_type_msg.format("x_data"))
            x = x_data

        elif isinstance(x_data, torch.Tensor):
            if self.debug and not x_data.ndim == 2:
                raise TypeError(invalid_type_msg.format("x_data"))
            x = {"input": x_data}

        else:
            raise TypeError(invalid_type_msg.format("x_data"))

        if isinstance(y_data, dict):
            if self.debug and not (
                all(isinstance(i, torch.Tensor) for i in y_data.values())
                and all(i.ndim == 2 for i in y_data.values())
                and all(i.shape[1] == 1 for i in y_data.values())
            ):
                raise TypeError(invalid_type_msg.format("y_data"))
            y = y_data

        elif isinstance(y_data, torch.Tensor):
            if self.debug and not y_data.ndim == 2:
                raise TypeError(invalid_type_msg.format("y_data"))
            y = {"output": y_data}

        else:
            raise TypeError(invalid_type_msg.format("y_data"))

        with torch.no_grad():
            arr = [i.detach().numpy() for i in y.values()]
            arr = np.concatenate(arr, axis=1)
        arr = np.nanmean(arr, axis=1)
        splits = split_data(
            data=arr,
            proportion={"training": 0.8, "validation": 0.2},
            groups=5,
        )
        train_idx, val_idx = list(splits.values())
        train_x = {i: v[train_idx] for i, v in x.items()}
        val_x = {i: v[val_idx] for i, v in x.items()}
        train_y = {i: v[train_idx] for i, v in y.items()}
        val_y = {i: v[val_idx] for i, v in y.items()}

        if self.batch_size is None:
            batch_size = len(train_idx)
        else:
            batch_size = self.batch_size

        train_loader = DataLoader(
            dataset=CustomDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            dataset=CustomDataset(val_x, val_y),
            batch_size=batch_size,
            shuffle=True,
        )

        loss_keys = list(y.keys())

        if self.variance_adjusted_loss:
            self._loss_weights = {}
            for i in loss_keys:
                valid_values = train_y[i][torch.isfinite(train_y[i])]
                self._loss_weights[i] = 1 / torch.std(valid_values)
        else:
            self._loss_weights = {i: 1 for i in train_y.keys()}

        best_loss = float("inf")
        epochs_no_improve = 0
        no_improve_from_last_lr_update = 0
        best_weights = None
        model_name = module.__class__.__name__

        # start training
        training_start_time = datetime.now()
        for epoch in range(self.epochs):
            self._update_logger("epoch", epoch + 1)
            self._train_step(module, train_loader)
            self._validation_step(module, val_loader)

            val_loss = self._logger[f"validation_{model_name}_loss"][-1]
            if val_loss < best_loss - self.early_stopping_threshold:
                best_loss = val_loss
                epochs_no_improve = 0
                no_improve_from_last_lr_update = 0
                if self.restore_best_weights:
                    best_weights = module.state_dict().copy()
            else:
                epochs_no_improve += 1
                no_improve_from_last_lr_update += 1

            if self.verbose != "off":
                es_prc = epochs_no_improve / self.early_stopping_patience * 100
                msg = f"\tearly_stopping_patience: {es_prc:0.1f}%\t"
                for key, val in self._logger.items():
                    if self.verbose == "full" or (
                        self.verbose == "minimal"
                        and ((key == "epoch") or (f"{model_name}_loss" in key))
                    ):
                        num = val[-1]
                        msg += f"\t{key}: "
                        msg += f"{num}" if key == "epoch" else f"{num:0.4f}"
                        msg += "\t"
                lr = self.optimizer.param_groups[0]["lr"]
                msg += f"\tlearning rate: {lr:0.2e}\t"
                time_lapse = str(datetime.now() - training_start_time)
                time_lapse = time_lapse.split(".")[0]
                msg += f"time_lapse: {time_lapse}\t"
                print(msg, end="\r")

            if (
                no_improve_from_last_lr_update >= self.early_stopping_patience * 0.33
            ) and (self.decaying_rate is not None):
                for param_group in self._optimizer.param_groups:
                    param_group["lr"] = max(
                        self.minimum_learning_rate,
                        param_group["lr"] * self.decaying_rate,
                    )
                no_improve_from_last_lr_update = 0

            if epochs_no_improve >= self.early_stopping_patience:
                if self.restore_best_weights and best_weights is not None:
                    module.load_state_dict(best_weights)
                break
        if self.verbose != "off":
            print("")

        module.eval()
        return module, pd.DataFrame(self._logger)


class PinballLoss(torch.nn.Module):
    """
    Pinball loss for quantile regression.

    Parameters
    ----------
    quantile : float, optional
        Quantile to estimate, must be in (0, 1). Default is 0.5.
    """

    def __init__(self, quantile: float = 0.5):
        """
        Initialize PinballLoss.

        Parameters
        ----------
        quantile : float, optional
            Quantile to estimate, must be in (0, 1). Default is 0.5.

        Raises
        ------
        ValueError
            If quantile is not in the (0, 1) range.
        """
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be between 0 and 1.")
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        """
        Compute the pinball loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True values.

        Returns
        -------
        loss : torch.Tensor
            Pinball loss value.
        """
        error = y_true - y_pred
        qnt = torch.quantile(error, self.quantile, dim=0, keepdim=True)
        return torch.mean(torch.abs(qnt))


class QuantilicRangeLoss(torch.nn.Module):
    """
    Quantilic range loss for interval regression.

    This loss computes the difference between two quantiles of the error distribution,
    providing a measure of the prediction interval width at a given confidence level.

    Parameters
    ----------
    confidence : float, optional
        Confidence level for the interval, must be in (0, 1). Default is 0.99.

    Raises
    ------
    ValueError
        If confidence is not a float in the (0, 1) range.
    """

    def __init__(self, confidence: float = 0.99):
        """
        Initialize QuantilicRangeLoss.

        Parameters
        ----------
        confidence : float, optional
            Confidence level for the interval, must be in (0, 1). Default is 0.99.

        Raises
        ------
        ValueError
            If confidence is not a float in the (0, 1) range.
        """
        super().__init__()
        if not isinstance(confidence, float) or not 0 < confidence < 1:
            msg = "q1 and q2 must be float values within the [0, 1] range, "
            msg += "with q1 strictly higher than q2."
            raise ValueError(msg)
        diff = (1 - confidence) / 2
        self.q1 = diff
        self.q2 = 1 - diff

    def forward(self, y_pred, y_true):
        """
        Compute the quantilic range loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True values.

        Returns
        -------
        loss : torch.Tensor
            Quantilic range loss value (difference between upper and lower quantiles).
        """
        error = y_true - y_pred
        q1 = torch.quantile(error, self.q1, dim=0, keepdim=True)
        q2 = torch.quantile(error, self.q2, dim=0, keepdim=True)
        return torch.mean(q2 - q1)


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


class MAEMetric(torch.nn.Module):
    """
    Mean Absolute Error (MAE) metric.

    Computes the mean absolute error between predictions and targets.
    """

    def __init__(self):
        """
        Initialize MAEMetric.
        """
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Compute the mean absolute error.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True values.

        Returns
        -------
        mae : torch.Tensor
            Mean absolute error.
        """
        return torch.mean(torch.abs(y_pred - y_true))


class ModelWrapper(torch.nn.Module):
    """
    Wraps a PyTorch model to freeze its parameters and concatenate its outputs.

    This class is useful for exporting models to ONNX or for inference pipelines
    where you want to ensure the model is not trainable and outputs are concatenated.

    Parameters
    ----------
    model : torch.nn.Module
        The model to wrap.
    inputs : list of str
        List of input feature names.
    """

    def __init__(self, model: torch.nn.Module, inputs: list[str]):
        """
        Initialize ModelWrapper.

        Parameters
        ----------
        model : torch.nn.Module
            The model to wrap.
        inputs : list of str
            List of input feature names.
        """
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.inputs = inputs

    def forward(self, x):
        """
        Forward pass through the wrapped model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, n_features).

        Returns
        -------
        output : torch.Tensor
            Concatenated model outputs.
        """
        inputs = {key: x[:, i : i + 1] for i, key in enumerate(self.inputs)}
        outputs = self.model(inputs)
        # Assicurati che ogni output sia di shape [N, 1]
        out_list = [v if v.ndim == 2 else v.unsqueeze(1) for v in outputs.values()]
        return torch.cat(out_list, dim=1)


def to_onnx(
    model: torch.nn.Module,
    inputs: list[str],
    outputs: list[str],
    onnx_file: str,
):
    """
    Export a PyTorch model to ONNX and create an OnnxModel instance.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to export.
    inputs : list of str
        List of input feature names.
    outputs : list of str
        List of output feature names.
    onnx_file : str
        File path to save the exported ONNX model.

    Returns
    -------
    onnx_model : OnnxModel
        An instance of OnnxModel wrapping the exported ONNX model.

    Raises
    ------
    TypeError
        If model is not a torch.nn.Module or onnx_file is not a string.
    """
    # check the inputs
    if not isinstance(model, torch.nn.Module):
        raise TypeError("'model' must be a DefaultRegressionModel subclass.")
    if not isinstance(onnx_file, str):
        msg = "'onnx_model' must be the file path where to store the "
        msg += "converted model."
        raise TypeError(msg)
    wrapped_model = ModelWrapper(model, inputs)
    dummy_list = [
        (
            torch.rand([2, 1], dtype=torch.float32)
            if i.lower() not in ["sex", "gender"]
            else torch.tensor([[0], [1]], dtype=torch.float32)
        )
        for i in inputs
    ]
    dummy_input = torch.cat(dummy_list, dim=1).detach()
    wrapped_model.eval()
    makedirs(dirname(onnx_file), exist_ok=True)
    torch.onnx.export(
        wrapped_model,
        dummy_input,  # type: ignore
        onnx_file,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=11,
    )
    return OnnxModel(onnx_file, inputs, outputs)
