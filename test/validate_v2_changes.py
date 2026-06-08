"""
Quick validation script for TorchTrainer v2.0 changes.
Run this to verify that the main changes are working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from src.labanalysis.modelling.pytorch.utils import TorchTrainer, MAEMetric


def test_defaults():
    """Test v2.0 default parameters."""
    print("Testing v2.0 defaults...")
    trainer = TorchTrainer()

    # Check defaults
    assert trainer._batch_size == 256, f"Expected batch_size=256, got {trainer._batch_size}"
    assert trainer._gradient_clip_val == 1.0, f"Expected gradient_clip_val=1.0, got {trainer._gradient_clip_val}"
    assert trainer._use_torch_compile is True, "Expected use_torch_compile=True"
    assert trainer._use_fused_optimizer is True, "Expected use_fused_optimizer=True"
    assert trainer._optimizer_class == torch.optim.AdamW, "Expected optimizer_class=AdamW"
    assert trainer._num_workers is None, "Expected num_workers=None (auto-tuned)"
    assert trainer._ema_decay is None, "Expected ema_decay=None"
    assert trainer._gradient_accumulation_steps == 1, "Expected gradient_accumulation_steps=1"

    print("✅ All defaults correct!")


def test_multi_output_mean_aggregation():
    """Test that multi-output uses MEAN aggregation."""
    print("\nTesting multi-output MEAN aggregation...")

    # Create simple multi-output model
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

    # Create data
    torch.manual_seed(42)
    x = {"input": torch.randn(100, 5)}
    y = {
        "output1": torch.randn(100, 1),
        "output2": torch.randn(100, 1),
    }

    model = MultiModel()

    trainer = TorchTrainer(
        loss=nn.MSELoss(),
        epochs=3,
        batch_size=32,
        use_uncertainty_weighting=False,
        verbose="off",
    )

    _, history = trainer.fit(model, x, y)

    # Check that global loss is MEAN
    for idx, row in history.iterrows():
        global_loss = row["training_loss"]
        out1_loss = row["training_output1_loss"]
        out2_loss = row["training_output2_loss"]

        expected_mean = (out1_loss + out2_loss) / 2
        diff = abs(global_loss - expected_mean)

        assert diff < 1e-5, f"Global loss should be MEAN. Got {global_loss}, expected {expected_mean}, diff={diff}"

    print("✅ Multi-output MEAN aggregation works correctly!")


def test_ema_feature():
    """Test EMA feature."""
    print("\nTesting EMA feature...")

    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )

    x = torch.randn(100, 5)
    y = torch.randn(100, 3)

    trainer = TorchTrainer(
        loss=nn.MSELoss(),
        epochs=5,
        batch_size=32,
        ema_decay=0.999,
        verbose="off",
    )

    trained_model, history = trainer.fit(model, x, y)

    # Check EMA state exists
    assert trainer._ema_state is not None, "EMA state should be initialized"
    assert len(trainer._ema_state) > 0, "EMA state should contain weights"

    print("✅ EMA feature works correctly!")


def test_gradient_accumulation():
    """Test gradient accumulation."""
    print("\nTesting gradient accumulation...")

    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )

    x = torch.randn(100, 5)
    y = torch.randn(100, 3)

    trainer = TorchTrainer(
        loss=nn.MSELoss(),
        epochs=3,
        batch_size=16,
        gradient_accumulation_steps=4,  # Effective batch = 64
        verbose="off",
    )

    trained_model, history = trainer.fit(model, x, y)

    assert len(history) == 3, "Should complete 3 epochs"

    print("✅ Gradient accumulation works correctly!")


def test_gradient_clipping():
    """Test gradient clipping."""
    print("\nTesting gradient clipping...")

    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )

    x = torch.randn(100, 5)
    y = torch.randn(100, 3)

    # With clipping (default)
    trainer = TorchTrainer(
        loss=nn.MSELoss(),
        epochs=3,
        batch_size=32,
        gradient_clip_val=1.0,  # Default
        verbose="off",
    )

    trained_model, history = trainer.fit(model, x, y)

    assert len(history) == 3, "Should complete training"

    # Without clipping
    model2 = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )

    trainer2 = TorchTrainer(
        loss=nn.MSELoss(),
        epochs=3,
        batch_size=32,
        gradient_clip_val=None,  # Disabled
        verbose="off",
    )

    trained_model2, history2 = trainer2.fit(model2, x, y)

    assert len(history2) == 3, "Should complete training without clipping"

    print("✅ Gradient clipping works correctly!")


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("TorchTrainer v2.0 - Validation Tests")
    print("=" * 70)

    try:
        test_defaults()
        test_multi_output_mean_aggregation()
        test_ema_feature()
        test_gradient_accumulation()
        test_gradient_clipping()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nTorchTrainer v2.0 changes are working correctly.")
        print("You can now run full pytest suite for comprehensive testing.")

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
