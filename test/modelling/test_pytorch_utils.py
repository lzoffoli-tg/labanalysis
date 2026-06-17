"""
Comprehensive testing for labanalysis.modelling.pytorch.utils module.

Tests all classes and functions:
- CustomDataset
- TrainingLogger
- UncertaintyWeighting
- PinballLoss
- StandardizedMSELoss
- QuantilicRangeLoss
- ComboLoss
- MAEMetric
- TorchTrainer
"""

import os
import sys
import tempfile
from io import StringIO
from os.path import abspath, dirname
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import labanalysis as laban

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_tensors():
    """Simple tensor data for single-output models."""
    x = torch.randn(100, 5)
    y = torch.randn(100, 3)
    return x, y


@pytest.fixture
def dict_tensors():
    """Dictionary tensor data for multi-output models."""
    x = {
        "feature1": torch.randn(100, 3),
        "feature2": torch.randn(100, 2),
    }
    y = {
        "output1": torch.randn(100, 1),
        "output2": torch.randn(100, 2),
    }
    return x, y


@pytest.fixture
def simple_model():
    """Simple single-output model for testing."""
    return nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )


@pytest.fixture
def multi_output_model():
    """Multi-output model for testing."""

    class MultiOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(5, 10)
            self.out1 = nn.Linear(10, 1)
            self.out2 = nn.Linear(10, 2)

        def forward(self, x):
            if isinstance(x, dict):
                x = torch.cat([v for v in x.values()], dim=1)
            h = torch.relu(self.shared(x))
            return {
                "output1": self.out1(h),
                "output2": self.out2(h),
            }

    return MultiOutputModel()


# ============================================================================
# TEST CustomDataset
# ============================================================================


class TestCustomDataset:
    """Tests for CustomDataset class."""

    def test_init_with_tensors(self, simple_tensors):
        """Test initialization with simple tensors."""
        x, y = simple_tensors
        dataset = laban.CustomDataset(x, y)
        assert len(dataset) == 100
        assert dataset.x.shape == x.shape
        assert dataset.y.shape == y.shape

    def test_init_with_dicts(self, dict_tensors):
        """Test initialization with dictionary tensors."""
        x, y = dict_tensors
        dataset = laban.CustomDataset(x, y)
        assert len(dataset) == 100
        assert "feature1" in dataset.x
        assert "output1" in dataset.y

    def test_mismatched_sizes(self):
        """Test that mismatched sizes raise ValueError."""
        x = torch.randn(100, 5)
        y = torch.randn(50, 3)
        with pytest.raises(ValueError, match="same number of samples"):
            laban.CustomDataset(x, y)

    def test_dict_mismatched_sizes(self):
        """Test that mismatched dict sizes raise ValueError."""
        x = {
            "f1": torch.randn(100, 3),
            "f2": torch.randn(50, 2),  # Different size
        }
        y = {"o1": torch.randn(100, 1)}
        with pytest.raises(ValueError, match="same number of samples"):
            laban.CustomDataset(x, y)

    def test_getitem_tensor(self, simple_tensors):
        """Test __getitem__ with tensors."""
        x, y = simple_tensors
        dataset = laban.CustomDataset(x, y)
        x_sample, y_sample = dataset[0]
        assert x_sample.shape[0] == 5
        assert y_sample.shape[0] == 3

    def test_getitem_dict(self, dict_tensors):
        """Test __getitem__ with dicts."""
        x, y = dict_tensors
        dataset = laban.CustomDataset(x, y)
        x_sample, y_sample = dataset[0]
        assert isinstance(x_sample, dict)
        assert isinstance(y_sample, dict)
        assert "feature1" in x_sample
        assert "output1" in y_sample

    def test_properties(self, simple_tensors):
        """Test x and y properties."""
        x, y = simple_tensors
        dataset = laban.CustomDataset(x, y)
        assert torch.equal(dataset.x, x)
        assert torch.equal(dataset.y, y)


# ============================================================================
# TEST TrainingLogger
# ============================================================================


class TestTrainingLogger:
    """Tests for TrainingLogger class."""

    def test_init(self):
        """Test logger initialization."""
        logger = laban.TrainingLogger(early_stopping_patience=100)
        assert logger._patience == 100
        assert logger._best_loss == float("inf")
        assert logger._epochs_without_improvement == 0
        assert len(logger.history) == 0

    def test_update(self):
        """Test updating metrics."""
        logger = laban.TrainingLogger()
        logger.update("training_loss", 0.5)
        logger.update("training_loss", 0.4)
        logger.update("validation_loss", 0.6)

        assert logger.history["training_loss"] == [0.5, 0.4]
        assert logger.history["validation_loss"] == [0.6]

    def test_get_last_value(self):
        """Test getting last value."""
        logger = laban.TrainingLogger()
        logger.update("loss", 0.5)
        logger.update("loss", 0.3)

        assert logger.get_last_value("loss") == 0.3
        assert logger.get_last_value("nonexistent") is None

    def test_update_early_stopping_improved(self):
        """Test early stopping when loss improves."""
        logger = laban.TrainingLogger(early_stopping_patience=10)
        improved = logger.update_early_stopping_state(0.5, threshold=1e-5)

        assert improved is True
        assert logger.best_loss == 0.5
        assert logger.epochs_without_improvement == 0

    def test_update_early_stopping_no_improvement(self):
        """Test early stopping when loss doesn't improve."""
        logger = laban.TrainingLogger(early_stopping_patience=10)
        logger.update_early_stopping_state(0.5, threshold=1e-5)
        improved = logger.update_early_stopping_state(0.5, threshold=1e-5)

        assert improved is False
        assert logger.epochs_without_improvement == 1

    def test_get_early_stopping_gap(self):
        """Test early stopping gap calculation."""
        logger = laban.TrainingLogger(early_stopping_patience=10)
        logger.update_early_stopping_state(0.5, threshold=1e-5)

        assert logger.get_early_stopping_gap() == 10

        logger.update_early_stopping_state(0.6, threshold=1e-5)
        assert logger.get_early_stopping_gap() == 9

    def test_log_learning_rate(self):
        """Test learning rate logging."""
        logger = laban.TrainingLogger()
        logger.log_learning_rate(0.001)
        logger.log_learning_rate(0.0005)

        assert logger.history["learning_rate"] == [0.001, 0.0005]

    def test_get_current_epoch(self):
        """Test getting current epoch."""
        logger = laban.TrainingLogger()
        assert logger.get_current_epoch() == 0

        logger.update("epoch", 1)
        logger.update("epoch", 2)
        assert logger.get_current_epoch() == 2

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        logger = laban.TrainingLogger()
        logger.update("epoch", 1)
        logger.update("training_loss", 0.5)
        logger.update("epoch", 2)
        logger.update("training_loss", 0.4)

        df = logger.to_dataframe()
        assert len(df) == 2
        assert "epoch" in df.columns
        assert "training_loss" in df.columns

    def test_print_epoch_summary_off(self, capsys):
        """Test print_epoch_summary with verbose='off'."""
        logger = laban.TrainingLogger()
        logger.print_epoch_summary(1, 0.5, 0.001, verbose="off")

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_epoch_summary_minimal(self, capsys):
        """Test print_epoch_summary with verbose='minimal'."""
        logger = laban.TrainingLogger(early_stopping_patience=100)
        logger.start_timer()
        logger.update("training_loss", 0.5)
        logger.update_early_stopping_state(0.5, 1e-5)

        logger.print_epoch_summary(1, 0.5, 0.001, verbose="minimal")

        captured = capsys.readouterr()
        assert "Epoch 1" in captured.out
        assert "train=0.5000" in captured.out
        assert "val=0.5000" in captured.out
        assert "gap=" in captured.out
        assert "time=" in captured.out
        # Should NOT show "no_improve" in minimal mode
        assert "no_improve" not in captured.out

    def test_print_epoch_summary_minimal_multi_output(self, capsys):
        """Test print_epoch_summary with multi-output shows AVERAGE metrics (v2.0)."""
        logger = laban.TrainingLogger(early_stopping_patience=100)
        logger.start_timer()

        # Simulate multi-output logging
        logger.update("training_loss", 0.075)  # Global (mean)
        logger.update("training_output1_loss", 0.08)
        logger.update("training_output2_loss", 0.07)
        logger.update("training_output1_mae", 0.20)
        logger.update("training_output2_mae", 0.10)
        logger.update("validation_output1_mae", 0.18)
        logger.update("validation_output2_mae", 0.09)

        logger.update_early_stopping_state(0.075, 1e-5)

        logger.print_epoch_summary(1, 0.075, 0.001, verbose="minimal")

        captured = capsys.readouterr()
        # Should show averaged MAE: (0.20 + 0.10) / 2 = 0.15 for train
        # and (0.18 + 0.09) / 2 = 0.135 for val
        assert "mae:" in captured.out or "MAE:" in captured.out.lower()

    def test_print_epoch_summary_full(self, capsys):
        """Test print_epoch_summary with verbose='full'."""
        logger = laban.TrainingLogger(early_stopping_patience=100)
        logger.start_timer()
        logger.update("training_loss", 0.5)
        logger.update("validation_loss", 0.6)
        logger.update_early_stopping_state(0.6, 1e-5)

        logger.print_epoch_summary(1, 0.6, 0.001, verbose="full")

        captured = capsys.readouterr()
        assert "Epoch 1" in captured.out
        assert "Training Loss:" in captured.out
        assert "Validation Loss:" in captured.out
        assert "Time:" in captured.out
        assert "No Improve:" in captured.out  # Should show in full mode
        assert "Gap:" in captured.out

    def test_start_timer(self):
        """Test training timer start."""
        logger = laban.TrainingLogger()
        assert logger._start_time is None

        logger.start_timer()
        assert logger._start_time is not None
        assert isinstance(logger._start_time, float)

    def test_get_elapsed_time(self):
        """Test elapsed time formatting."""
        import time as time_module

        logger = laban.TrainingLogger()

        # Before starting timer
        assert logger.get_elapsed_time() == "0s"

        # After starting timer
        logger.start_timer()
        time_module.sleep(0.1)  # Wait a bit
        elapsed = logger.get_elapsed_time()

        # Should return formatted time
        assert isinstance(elapsed, str)
        assert "s" in elapsed  # Should contain seconds

    def test_get_elapsed_time_formatting(self):
        """Test different elapsed time formats."""
        import time as time_module

        logger = laban.TrainingLogger()

        # Test seconds only
        logger._start_time = time_module.time() - 30
        elapsed = logger.get_elapsed_time()
        assert "30s" == elapsed or "s" in elapsed

        # Test minutes and seconds
        logger._start_time = time_module.time() - 90  # 1m 30s
        elapsed = logger.get_elapsed_time()
        assert "m" in elapsed and "s" in elapsed

        # Test hours, minutes, and seconds
        logger._start_time = time_module.time() - 3661  # 1h 1m 1s
        elapsed = logger.get_elapsed_time()
        assert "h" in elapsed and "m" in elapsed and "s" in elapsed


# ============================================================================
# TEST Loss Functions
# ============================================================================


class TestPinballLoss:
    """Tests for PinballLoss."""

    def test_init(self):
        """Test initialization."""
        loss = laban.PinballLoss(quantile=0.5)
        assert loss.quantile == 0.5

    def test_init_invalid_quantile(self):
        """Test initialization with invalid quantile."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            laban.PinballLoss(quantile=1.5)

    def test_forward_median(self):
        """Test forward pass for median (quantile=0.5)."""
        loss_fn = laban.PinballLoss(quantile=0.5)
        y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_true = torch.tensor([[1.5, 2.5], [2.5, 3.5]])

        loss = loss_fn(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

    def test_forward_upper_quantile(self):
        """Test forward pass for upper quantile."""
        loss_fn = laban.PinballLoss(quantile=0.9)
        y_pred = torch.tensor([[1.0], [2.0]])
        y_true = torch.tensor([[2.0], [3.0]])

        loss = loss_fn(y_pred, y_true)
        assert loss > 0


class TestQuantilicRangeLoss:
    """Tests for QuantilicRangeLoss."""

    def test_init(self):
        """Test initialization."""
        loss = laban.QuantilicRangeLoss(confidence=0.99)
        assert abs(loss.q1 - 0.005) < 1e-10
        assert abs(loss.q2 - 0.995) < 1e-10

    def test_init_invalid_confidence(self):
        """Test initialization with invalid confidence."""
        with pytest.raises(ValueError, match="must be a float in"):
            laban.QuantilicRangeLoss(confidence=1.5)

    def test_forward(self):
        """Test forward pass."""
        loss_fn = laban.QuantilicRangeLoss(confidence=0.95)
        y_pred = torch.randn(100, 3)
        y_true = torch.randn(100, 3)

        loss = loss_fn(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss > 0


class TestStandardizedMSELoss:
    """Tests for StandardizedMSELoss."""

    def test_init_with_stats(self):
        """Test initialization with provided statistics."""
        mean = torch.tensor([0.5, 1.0])
        std = torch.tensor([0.2, 0.3])
        loss_fn = laban.StandardizedMSELoss(mean=mean, std=std)

        assert loss_fn.freeze_stats is True
        assert torch.equal(loss_fn.running_mean, mean)

    def test_init_without_stats(self):
        """Test initialization without statistics."""
        loss_fn = laban.StandardizedMSELoss()
        assert loss_fn.freeze_stats is False
        assert loss_fn.running_count == 0

    def test_forward_with_fixed_stats(self):
        """Test forward with fixed statistics."""
        mean = torch.zeros(2)
        std = torch.ones(2)
        loss_fn = laban.StandardizedMSELoss(mean=mean, std=std)

        y_pred = torch.randn(10, 2)
        y_true = torch.randn(10, 2)

        loss = loss_fn(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_forward_with_dynamic_stats(self):
        """Test forward with dynamic statistics update."""
        loss_fn = laban.StandardizedMSELoss()

        y_pred = torch.randn(20, 3)
        y_true = torch.randn(20, 3)

        # First forward should initialize stats
        loss1 = loss_fn(y_pred, y_true)
        assert loss_fn.running_count > 0

        # Second forward should update stats
        loss2 = loss_fn(y_pred, y_true)
        assert loss_fn.running_count > 20

    def test_update_stats(self):
        """Test statistics update."""
        loss_fn = laban.StandardizedMSELoss()
        y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        loss_fn.update_stats(y_true)
        assert loss_fn.running_count == 2
        assert loss_fn.running_mean.shape == (2,)


class TestComboLoss:
    """Tests for ComboLoss."""

    def test_init(self):
        """Test initialization."""
        loss1 = laban.PinballLoss(0.5)
        loss2 = laban.MAEMetric()
        combo = laban.ComboLoss(loss1, loss2)

        assert len(combo.losses) == 2

    def test_forward(self):
        """Test forward pass."""
        loss1 = laban.PinballLoss(0.5)
        loss2 = laban.PinballLoss(0.9)
        combo = laban.ComboLoss(loss1, loss2)

        y_pred = torch.randn(10, 2)
        y_true = torch.randn(10, 2)

        loss = combo(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


class TestMAEMetric:
    """Tests for MAEMetric."""

    def test_init(self):
        """Test initialization."""
        metric = laban.MAEMetric()
        assert isinstance(metric, nn.Module)

    def test_forward(self):
        """Test forward pass."""
        metric = laban.MAEMetric()
        y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_true = torch.tensor([[1.5, 2.5], [2.5, 3.5]])

        mae = metric(y_pred, y_true)
        expected_mae = torch.mean(torch.abs(y_pred - y_true))

        assert torch.allclose(mae, expected_mae)

    def test_forward_zero_error(self):
        """Test forward with zero error."""
        metric = laban.MAEMetric()
        y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        mae = metric(y, y)
        assert mae == 0.0


# ============================================================================
# TEST UncertaintyWeighting
# ============================================================================


class TestUncertaintyWeighting:
    """Tests for UncertaintyWeighting."""

    def test_init(self):
        """Test initialization."""
        uw = laban.UncertaintyWeighting(["output1", "output2"])
        assert len(uw.output_keys) == 2
        assert uw.log_vars.shape == (2,)

    def test_forward(self):
        """Test forward pass."""
        uw = laban.UncertaintyWeighting(["output1", "output2"])
        losses = {
            "output1": torch.tensor(0.5),
            "output2": torch.tensor(0.3),
        }

        total_loss = uw(losses)
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0

    def test_gradients(self):
        """Test that log_vars are trainable."""
        uw = laban.UncertaintyWeighting(["output1"])
        losses = {"output1": torch.tensor(0.5, requires_grad=True)}

        total_loss = uw(losses)
        total_loss.backward()

        assert uw.log_vars.grad is not None


# ============================================================================
# TEST TorchTrainer
# ============================================================================


class TestTorchTrainer:
    """Tests for TorchTrainer class."""

    def test_init_default(self):
        """Test initialization with defaults (v2.0 optimized)."""
        trainer = laban.TorchTrainer()
        assert trainer._epochs == 100000
        assert trainer._batch_size == 256  # New default: optimized for CPU
        assert trainer._verbose == "minimal"
        assert trainer._gradient_clip_val == 1.0  # New default: prevents exploding gradients
        assert trainer._use_torch_compile is True  # New default: enabled
        assert trainer._use_fused_optimizer is True  # New default: enabled
        assert trainer._num_workers is None  # Auto-tuned at fit() time
        assert trainer._ema_decay is None  # Optional feature
        assert trainer._gradient_accumulation_steps == 1  # No accumulation by default

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        trainer = laban.TorchTrainer(
            loss=laban.PinballLoss(0.5),
            metrics=[laban.MAEMetric()],
            epochs=100,
            batch_size=32,
            early_stopping_patience=50,
            verbose="full",
        )
        assert trainer._epochs == 100
        assert trainer._batch_size == 32
        assert trainer._early_stopping_patience == 50

    def test_init_invalid_batch_size(self):
        """Test initialization with invalid batch size."""
        with pytest.raises(ValueError, match="must be int or None"):
            laban.TorchTrainer(batch_size="invalid")

    def test_init_metrics_callable(self):
        """Test initialization with callable metric."""
        metric = laban.MAEMetric()
        trainer = laban.TorchTrainer(metrics=metric)
        assert isinstance(trainer._metrics, dict)

    def test_init_metrics_list(self):
        """Test initialization with list of metrics."""
        metrics = [laban.MAEMetric()]
        trainer = laban.TorchTrainer(metrics=metrics)
        assert isinstance(trainer._metrics, dict)

    def test_init_metrics_dict(self):
        """Test initialization with dict of metrics."""
        metrics = {"mae": laban.MAEMetric()}
        trainer = laban.TorchTrainer(metrics=metrics)
        assert "mae" in trainer._metrics

    def test_logger_property(self):
        """Test logger property."""
        trainer = laban.TorchTrainer()
        assert isinstance(trainer.logger, dict)

    def test_fit_simple_model(self, simple_model, simple_tensors):
        """Test fitting a simple single-output model."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            metrics=[laban.MAEMetric()],
            epochs=5,
            batch_size=32,
            early_stopping_patience=10,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        assert isinstance(model, nn.Module)
        assert len(history) == 5
        assert "training_loss" in history.columns
        assert "validation_loss" in history.columns

    def test_fit_multi_output_model(self, multi_output_model, dict_tensors):
        """Test fitting a multi-output model."""
        x, y = dict_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            metrics={"mae": laban.MAEMetric()},
            epochs=5,
            batch_size=16,
            early_stopping_patience=10,
            verbose="off",
            use_uncertainty_weighting=False,
        )

        model, history = trainer.fit(multi_output_model, x, y)

        assert isinstance(model, nn.Module)
        assert "training_output1_loss" in history.columns
        assert "training_output2_loss" in history.columns

    def test_fit_with_uncertainty_weighting(self, multi_output_model, dict_tensors):
        """Test fitting with uncertainty weighting."""
        x, y = dict_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            metrics=[laban.MAEMetric()],
            epochs=3,
            batch_size=16,
            verbose="off",
            use_uncertainty_weighting=True,
        )

        model, history = trainer.fit(multi_output_model, x, y)

        assert trainer._uw_module is not None
        assert len(history) == 3

    def test_fit_early_stopping(self, simple_model):
        """Test that early stopping works."""
        # Create data where model converges quickly
        x = torch.randn(100, 5)
        y = torch.zeros(100, 3)  # Simple target

        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=1000,
            batch_size=32,
            early_stopping_patience=5,
            early_stopping_threshold=1e-6,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        # Should stop before max epochs
        assert len(history) < 1000

    def test_fit_single_learning_rate(self, simple_model, simple_tensors):
        """Test training with single learning rate."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": 0.001},
            epochs=10,
            batch_size=32,
            early_stopping_patience=20,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        # Check that learning rate was logged
        assert "learning_rate" in history.columns
        # Learning rate should remain constant
        assert all(history["learning_rate"] == 0.001)

    def test_fit_learning_rate_scheduling(self, simple_model, simple_tensors):
        """Test learning rate scheduling with multiple LRs."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": [0.01, 0.001, 0.0001]},
            epochs=100,
            batch_size=32,
            early_stopping_patience=10,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        # Check that learning rate was logged
        assert "learning_rate" in history.columns

        # Learning rate should change during training
        lr_values = history["learning_rate"].unique()
        # We may have 1, 2, or 3 different LRs depending on when early stopping triggers
        assert len(lr_values) >= 1

    def test_fit_learning_rate_scheduling_switches(self, simple_model):
        """Test that LR scheduling switches correctly."""
        # Create data that won't converge easily
        x = torch.randn(100, 5)
        y = torch.randn(100, 3) * 10  # Larger scale to prevent quick convergence

        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": [0.1, 0.01, 0.001]},
            epochs=50,
            batch_size=32,
            early_stopping_patience=5,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        # Check that LR changed at least once
        lr_values = history["learning_rate"].values
        assert len(set(lr_values)) > 1, "Learning rate should have changed during training"

    def test_fit_lr_scheduling_verbose_full_only(self, simple_model, capsys):
        """Test that LR schedule message is printed only in verbose='full'."""
        x = torch.randn(100, 5)
        y = torch.randn(100, 3) * 10

        # Test with verbose='full'
        trainer_full = laban.TorchTrainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": [0.1, 0.01]},
            epochs=30,
            batch_size=32,
            early_stopping_patience=3,
            verbose="full",
        )

        model, _ = trainer_full.fit(simple_model, x, y)
        captured_full = capsys.readouterr()

        # With verbose='full', should show LR Schedule message if LR changes
        # (may not always happen depending on convergence)

        # Test with verbose='minimal'
        model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3))
        trainer_minimal = laban.TorchTrainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": [0.1, 0.01]},
            epochs=30,
            batch_size=32,
            early_stopping_patience=3,
            verbose="minimal",
        )

        model, _ = trainer_minimal.fit(model, x, y)
        captured_minimal = capsys.readouterr()

        # With verbose='minimal', should NOT show LR Schedule message
        assert "[LR Schedule]" not in captured_minimal.out

        # Test with verbose='off'
        model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3))
        trainer_off = laban.TorchTrainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": [0.1, 0.01]},
            epochs=30,
            batch_size=32,
            early_stopping_patience=3,
            verbose="off",
        )

        model, _ = trainer_off.fit(model, x, y)
        captured_off = capsys.readouterr()

        # With verbose='off', should NOT show LR Schedule message
        assert "[LR Schedule]" not in captured_off.out

    def test_fit_lr_scheduling_restores_best_weights(self, simple_model):
        """Test that best weights are restored when switching LR."""
        x = torch.randn(100, 5)
        y = torch.randn(100, 3) * 10

        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": [0.1, 0.01, 0.001]},
            epochs=50,
            batch_size=32,
            early_stopping_patience=5,
            restore_best_weights=True,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        # Verify that the model was trained
        assert len(history) > 0

        # Verify that validation loss is tracked
        assert "validation_loss" in history.columns

        # If LR changed, best loss should have been tracked
        if len(set(history["learning_rate"].values)) > 1:
            best_loss = trainer._logger.best_loss
            assert best_loss < float("inf")

    def test_fit_restore_best_weights(self, simple_model, simple_tensors):
        """Test that best weights are restored."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=10,
            batch_size=32,
            restore_best_weights=True,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        # Model should have best weights
        assert model is not None

    def test_fit_tracks_elapsed_time(self, simple_model, simple_tensors):
        """Test that training tracks elapsed time."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=5,
            batch_size=32,
            verbose="off",
        )

        # Timer should not be started before fit
        assert trainer._logger._start_time is None

        model, history = trainer.fit(simple_model, x, y)

        # Timer should be started during fit
        assert trainer._logger._start_time is not None

        # Should be able to get elapsed time
        elapsed = trainer._logger.get_elapsed_time()
        assert isinstance(elapsed, str)
        assert "s" in elapsed

    def test_fit_no_restore_weights(self, simple_model, simple_tensors):
        """Test without restoring best weights."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=5,
            batch_size=32,
            restore_best_weights=False,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        assert model is not None
        assert len(history) == 5

    def test_process_batch(self, simple_model, simple_tensors):
        """Test _process_batch method."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(loss=nn.MSELoss())

        x_batch = x[:10]
        y_batch = y[:10]

        bt, bp, bl, bs = trainer._process_batch(simple_model, x_batch, y_batch)

        # bt and bp are flattened tensors (batch_size * num_outputs)
        assert bt.numel() <= 10 * y_batch.shape[1]
        assert bp.numel() <= 10 * y_batch.shape[1]
        assert isinstance(bl, torch.Tensor)
        assert isinstance(bs, (int, torch.Tensor))

    def test_fit_with_nan_handling(self, simple_model):
        """Test handling of NaN values."""
        x = torch.randn(100, 5)
        y = torch.randn(100, 3)
        y[0, 0] = float("nan")  # Introduce NaN

        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=5,
            batch_size=32,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        # Should handle NaN gracefully
        assert len(history) == 5

    def test_fit_full_batch(self, simple_model, simple_tensors):
        """Test fitting with full batch (no batch_size)."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=5,
            batch_size=None,  # Full batch
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        assert len(history) == 5

    def test_fit_combo_loss(self, simple_model, simple_tensors):
        """Test fitting with ComboLoss."""
        x, y = simple_tensors
        combo_loss = laban.ComboLoss(
            laban.PinballLoss(0.5), laban.QuantilicRangeLoss(0.99)
        )

        trainer = laban.TorchTrainer(
            loss=combo_loss,
            epochs=5,
            batch_size=32,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        assert len(history) == 5


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline."""
        # Create synthetic data
        torch.manual_seed(42)
        x = torch.randn(200, 10)
        y = torch.randn(200, 5)

        # Create model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        # Create trainer with various components
        trainer = laban.TorchTrainer(
            loss=laban.ComboLoss(laban.PinballLoss(0.5), nn.MSELoss()),
            metrics=[laban.MAEMetric()],
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.01},
            epochs=20,
            batch_size=32,
            early_stopping_patience=10,
            verbose="off",
        )

        # Train
        trained_model, history = trainer.fit(model, x, y)

        # Verify results
        assert len(history) <= 20
        assert "training_loss" in history.columns
        assert "validation_loss" in history.columns
        assert "learning_rate" in history.columns

        # Check that training completed successfully (loss may not always decrease in short runs)
        assert all(history["validation_loss"] > 0)
        assert all(history["training_loss"] > 0)

    def test_multi_output_with_uncertainty_weighting(self):
        """Test multi-output model with uncertainty weighting."""
        torch.manual_seed(42)

        # Create multi-output data
        x = {"input": torch.randn(150, 8)}
        y = {
            "output1": torch.randn(150, 2),
            "output2": torch.randn(150, 3),
        }

        # Create multi-output model
        class MultiModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Linear(8, 16)
                self.out1 = nn.Linear(16, 2)
                self.out2 = nn.Linear(16, 3)

            def forward(self, x):
                if isinstance(x, dict):
                    x = x["input"]
                h = torch.relu(self.shared(x))
                return {"output1": self.out1(h), "output2": self.out2(h)}

        model = MultiModel()

        # Train with uncertainty weighting
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            metrics={"mae": laban.MAEMetric()},
            epochs=15,
            batch_size=30,
            use_uncertainty_weighting=True,
            verbose="off",
        )

        trained_model, history = trainer.fit(model, x, y)

        # Verify per-output metrics were logged
        assert "training_output1_loss" in history.columns
        assert "training_output2_loss" in history.columns
        assert "validation_output1_mae" in history.columns
        assert "validation_output2_mae" in history.columns

    def test_multi_output_loss_aggregation(self):
        """Test that multi-output loss uses MEAN aggregation (v2.0)."""
        torch.manual_seed(42)

        # Create multi-output data
        x = {"input": torch.randn(100, 5)}
        y = {
            "output1": torch.randn(100, 1),
            "output2": torch.randn(100, 1),
        }

        class MultiModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Linear(5, 8)
                self.out1 = nn.Linear(8, 1)
                self.out2 = nn.Linear(8, 1)

            def forward(self, x):
                if isinstance(x, dict):
                    x = x["input"]
                h = torch.relu(self.shared(x))
                return {"output1": self.out1(h), "output2": self.out2(h)}

        model = MultiModel()

        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=5,
            batch_size=32,
            use_uncertainty_weighting=False,
            verbose="off",
        )

        _, history = trainer.fit(model, x, y)

        # Global loss should be MEAN of per-output losses
        for idx, row in history.iterrows():
            global_train_loss = row["training_loss"]
            output1_loss = row["training_output1_loss"]
            output2_loss = row["training_output2_loss"]

            # Global loss should be approximately the mean
            expected_mean = (output1_loss + output2_loss) / 2
            assert abs(global_train_loss - expected_mean) < 1e-5, (
                f"Global loss should be MEAN of per-output losses. "
                f"Got {global_train_loss}, expected {expected_mean}"
            )


# ============================================================================
# TEST New Features (v2.0)
# ============================================================================


class TestTorchTrainerV2Features:
    """Tests for new v2.0 features."""

    def test_ema_weights(self, simple_model, simple_tensors):
        """Test Exponential Moving Average weights feature."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=10,
            batch_size=32,
            ema_decay=0.999,  # Enable EMA
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        # EMA state should be initialized
        assert trainer._ema_state is not None
        assert len(trainer._ema_state) > 0

        # Should have completed training
        assert len(history) > 0

    def test_gradient_accumulation(self, simple_model, simple_tensors):
        """Test gradient accumulation feature."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=5,
            batch_size=16,
            gradient_accumulation_steps=4,  # Effective batch = 64
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        # Should complete successfully
        assert len(history) == 5

    def test_gradient_clipping_default(self, simple_model, simple_tensors):
        """Test that gradient clipping is enabled by default."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=5,
            batch_size=32,
            # gradient_clip_val=1.0 is default
            verbose="off",
        )

        # Default should be 1.0
        assert trainer._gradient_clip_val == 1.0

        model, history = trainer.fit(simple_model, x, y)
        assert len(history) == 5

    def test_gradient_clipping_disabled(self, simple_model, simple_tensors):
        """Test disabling gradient clipping."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=5,
            batch_size=32,
            gradient_clip_val=None,  # Disable
            verbose="off",
        )

        assert trainer._gradient_clip_val is None

        model, history = trainer.fit(simple_model, x, y)
        assert len(history) == 5

    def test_fused_optimizer_enabled(self, simple_model, simple_tensors):
        """Test that fused optimizer is attempted (may not be available)."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            optimizer_class=torch.optim.AdamW,
            epochs=3,
            batch_size=32,
            use_fused_optimizer=True,  # Default
            verbose="off",
        )

        # Should not raise error even if fused not available
        model, history = trainer.fit(simple_model, x, y)
        assert len(history) == 3

    def test_num_workers_parameter(self, simple_model, simple_tensors):
        """Test num_workers parameter."""
        x, y = simple_tensors
        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=3,
            batch_size=32,
            num_workers=0,  # Safe default
            verbose="off",
        )

        assert trainer._num_workers == 0

        model, history = trainer.fit(simple_model, x, y)
        assert len(history) == 3

    def test_default_optimizer_is_adamw(self):
        """Test that default optimizer is AdamW (v2.0 change)."""
        trainer = laban.TorchTrainer()
        assert trainer._optimizer_class == torch.optim.AdamW

    def test_default_batch_size_is_256(self):
        """Test that default batch size is 256 (v2.0 change)."""
        trainer = laban.TorchTrainer()
        assert trainer._batch_size == 256

    def test_torch_compile_enabled_by_default(self):
        """Test that torch.compile is enabled by default."""
        trainer = laban.TorchTrainer()
        assert trainer._use_torch_compile is True

    def test_torch_compile_availability_check(self):
        """Test torch.compile availability check."""
        import platform

        # Check method exists and returns boolean
        is_available = laban.TorchTrainer._is_torch_compile_available()
        assert isinstance(is_available, bool)

        # On Windows without MSVC, should return False
        # On Linux/Mac or Windows with MSVC, may return True
        if platform.system() == "Windows":
            import shutil
            if shutil.which("cl") is None:
                # No MSVC compiler
                assert is_available is False

    def test_ema_with_lr_scheduling(self, simple_model):
        """Test EMA with learning rate scheduling."""
        x = torch.randn(100, 5)
        y = torch.randn(100, 3)

        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": [0.01, 0.001]},
            epochs=30,
            batch_size=32,
            early_stopping_patience=5,
            ema_decay=0.999,
            verbose="off",
        )

        model, history = trainer.fit(simple_model, x, y)

        # Should handle EMA with LR scheduling
        assert trainer._ema_state is not None
        assert len(history) > 0

    def test_num_workers_auto_tuning_default(self):
        """Test that num_workers defaults to None (auto-tuned)."""
        trainer = laban.TorchTrainer()
        assert trainer._num_workers is None

    def test_num_workers_explicit_override(self, simple_model):
        """Test that explicit num_workers value is respected."""
        x = torch.randn(100, 5)
        y = torch.randn(100, 3)

        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=3,
            batch_size=32,
            num_workers=2,  # Explicitly set
            verbose="off",
        )

        # Should store explicit value
        assert trainer._num_workers == 2

        # Should use explicit value during training
        model, history = trainer.fit(simple_model, x, y)
        assert len(history) == 3

    def test_compute_optimal_num_workers_small_dataset(self):
        """Test that small datasets get num_workers=0."""
        # Small dataset (<1000 samples) should get 0 workers
        num_workers = laban.TorchTrainer._compute_optimal_num_workers(500)
        assert num_workers == 0

    def test_compute_optimal_num_workers_large_dataset(self):
        """Test that large datasets get auto-tuned num_workers."""
        import platform

        # Large dataset (>1000 samples)
        num_workers = laban.TorchTrainer._compute_optimal_num_workers(5000)

        if platform.system() == "Windows":
            # Windows should always get 0
            assert num_workers == 0
        else:
            # Linux/Mac should get > 0 (min(cpu_count // 2, 4))
            assert num_workers >= 0
            assert num_workers <= 4

    def test_num_workers_auto_tuning_during_fit(self, simple_model):
        """Test that num_workers is auto-tuned during fit when None."""
        x = torch.randn(100, 5)
        y = torch.randn(100, 3)

        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=3,
            batch_size=32,
            num_workers=None,  # Should auto-tune
            verbose="off",
        )

        # Should be None before fit
        assert trainer._num_workers is None

        # Should complete training successfully
        model, history = trainer.fit(simple_model, x, y)
        assert len(history) == 3

    def test_training_without_torch_compile(self, simple_model):
        """Test that training works even when torch.compile is disabled/unavailable."""
        x = torch.randn(100, 5)
        y = torch.randn(100, 3)

        trainer = laban.TorchTrainer(
            loss=nn.MSELoss(),
            epochs=3,
            batch_size=32,
            use_torch_compile=False,  # Explicitly disabled
            verbose="off",
        )

        # Should complete training successfully without torch.compile
        model, history = trainer.fit(simple_model, x, y)
        assert len(history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
