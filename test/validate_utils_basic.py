"""
Quick validation script for pytorch utils (no pytest required).
Run this to verify basic functionality without installing pytest.
"""

import sys
from os.path import abspath, dirname

import torch
import torch.nn as nn

sys.path.append(dirname(dirname(abspath(__file__))))
import src.labanalysis as laban


def test_custom_dataset():
    """Test CustomDataset."""
    print("Testing CustomDataset...", end=" ")
    x = torch.randn(100, 5)
    y = torch.randn(100, 3)
    dataset = laban.CustomDataset(x, y)
    assert len(dataset) == 100
    x_sample, y_sample = dataset[0]
    assert x_sample.shape[0] == 5
    print("✓ PASSED")


def test_training_logger():
    """Test TrainingLogger."""
    print("Testing TrainingLogger...", end=" ")
    logger = laban.TrainingLogger(early_stopping_patience=100)

    # Test update
    logger.update("training_loss", 0.5)
    logger.update("validation_loss", 0.6)
    assert logger.get_last_value("training_loss") == 0.5

    # Test early stopping
    improved = logger.update_early_stopping_state(0.5, 1e-5)
    assert improved is True
    assert logger.get_early_stopping_gap() == 100

    # Test DataFrame export
    df = logger.to_dataframe()
    assert len(df) == 1
    print("✓ PASSED")


def test_pinball_loss():
    """Test PinballLoss."""
    print("Testing PinballLoss...", end=" ")
    loss_fn = laban.PinballLoss(quantile=0.5)
    y_pred = torch.tensor([[1.0, 2.0]])
    y_true = torch.tensor([[1.5, 2.5]])
    loss = loss_fn(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    print("✓ PASSED")


def test_quantilic_range_loss():
    """Test QuantilicRangeLoss."""
    print("Testing QuantilicRangeLoss...", end=" ")
    loss_fn = laban.QuantilicRangeLoss(confidence=0.95)
    y_pred = torch.randn(100, 3)
    y_true = torch.randn(100, 3)
    loss = loss_fn(y_pred, y_true)
    assert loss > 0
    print("✓ PASSED")


def test_standardized_mse_loss():
    """Test StandardizedMSELoss."""
    print("Testing StandardizedMSELoss...", end=" ")
    loss_fn = laban.StandardizedMSELoss()
    y_pred = torch.randn(20, 3)
    y_true = torch.randn(20, 3)
    loss = loss_fn(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss_fn.running_count > 0
    print("✓ PASSED")


def test_combo_loss():
    """Test ComboLoss."""
    print("Testing ComboLoss...", end=" ")
    combo = laban.ComboLoss(laban.PinballLoss(0.5), nn.MSELoss())
    y_pred = torch.randn(10, 2)
    y_true = torch.randn(10, 2)
    loss = combo(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    print("✓ PASSED")


def test_mae_metric():
    """Test MAEMetric."""
    print("Testing MAEMetric...", end=" ")
    metric = laban.MAEMetric()
    y_pred = torch.tensor([[1.0, 2.0]])
    y_true = torch.tensor([[1.5, 2.5]])
    mae = metric(y_pred, y_true)
    assert torch.isclose(mae, torch.tensor(0.5))
    print("✓ PASSED")


def test_uncertainty_weighting():
    """Test UncertaintyWeighting."""
    print("Testing UncertaintyWeighting...", end=" ")
    uw = laban.UncertaintyWeighting(["output1", "output2"])
    losses = {
        "output1": torch.tensor(0.5),
        "output2": torch.tensor(0.3),
    }
    total = uw(losses)
    assert isinstance(total, torch.Tensor)
    print("✓ PASSED")


def test_torch_trainer_simple():
    """Test TorchTrainer with simple model."""
    print("Testing TorchTrainer (simple)...", end=" ")

    # Create simple data and model
    torch.manual_seed(42)
    x = torch.randn(50, 5)
    y = torch.randn(50, 3)
    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3))

    # Train
    trainer = laban.TorchTrainer(
        loss=nn.MSELoss(),
        metrics=[laban.MAEMetric()],
        epochs=3,
        batch_size=16,
        verbose="off",
    )

    trained_model, history = trainer.fit(model, x, y)
    assert len(history) == 3
    assert "training_loss" in history.columns
    assert "validation_loss" in history.columns
    print("✓ PASSED")


def test_torch_trainer_multi_output():
    """Test TorchTrainer with multi-output model."""
    print("Testing TorchTrainer (multi-output)...", end=" ")

    # Create multi-output data
    torch.manual_seed(42)
    x = {"input": torch.randn(50, 5)}
    y = {"output1": torch.randn(50, 1), "output2": torch.randn(50, 2)}

    # Create multi-output model
    class MultiModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(5, 10)
            self.out1 = nn.Linear(10, 1)
            self.out2 = nn.Linear(10, 2)

        def forward(self, x):
            if isinstance(x, dict):
                x = x["input"]
            h = torch.relu(self.shared(x))
            return {"output1": self.out1(h), "output2": self.out2(h)}

    model = MultiModel()

    # Train
    trainer = laban.TorchTrainer(
        loss=nn.MSELoss(),
        epochs=3,
        batch_size=16,
        use_uncertainty_weighting=False,
        verbose="off",
    )

    trained_model, history = trainer.fit(model, x, y)
    assert "training_output1_loss" in history.columns
    assert "training_output2_loss" in history.columns
    print("✓ PASSED")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("PyTorch Utils - Basic Validation (no pytest required)")
    print("=" * 60)
    print()

    tests = [
        test_custom_dataset,
        test_training_logger,
        test_pinball_loss,
        test_quantilic_range_loss,
        test_standardized_mse_loss,
        test_combo_loss,
        test_mae_metric,
        test_uncertainty_weighting,
        test_torch_trainer_simple,
        test_torch_trainer_multi_output,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ All basic validation tests passed!")
        print("\nFor comprehensive testing, install pytest and run:")
        print("  pytest test/test_pytorch_utils.py -v")
    else:
        print(f"\n✗ {failed} test(s) failed. Please investigate.")
        sys.exit(1)


if __name__ == "__main__":
    main()
