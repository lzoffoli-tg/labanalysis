"""
Quick test to verify torch.compile availability check.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import platform
import shutil


def test_torch_compile_availability():
    """Test the torch.compile availability check."""
    print("=" * 70)
    print("Testing torch.compile availability check")
    print("=" * 70)

    # Check platform
    print(f"\nPlatform: {platform.system()}")

    # Check for compiler
    if platform.system() == "Windows":
        compiler = shutil.which("cl")
        print(f"MSVC compiler (cl.exe): {'Found' if compiler else 'Not found'}")
        if compiler:
            print(f"  Location: {compiler}")

    # Try to import TorchTrainer
    try:
        from labanalysis.modelling.pytorch.utils import TorchTrainer
        print("\n✅ TorchTrainer imported successfully")

        # Check torch.compile availability
        is_available = TorchTrainer._is_torch_compile_available()
        print(f"\ntorch.compile available: {is_available}")

        # Create trainer with default settings
        trainer = TorchTrainer(verbose="off")
        print(f"use_torch_compile setting: {trainer._use_torch_compile}")

        # Try a simple fit
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
        )

        x = torch.randn(100, 5)
        y = torch.randn(100, 3)

        print("\nTesting training with default settings...")
        trainer_test = TorchTrainer(
            loss=nn.MSELoss(),
            epochs=2,
            batch_size=32,
            verbose="minimal",
        )

        model_trained, history = trainer_test.fit(model, x, y)
        print(f"✅ Training completed successfully! ({len(history)} epochs)")

        print("\n" + "=" * 70)
        print("✅ ALL CHECKS PASSED")
        print("=" * 70)

        if not is_available and platform.system() == "Windows":
            print("\n⚠️  NOTE: torch.compile is disabled (MSVC compiler not found)")
            print("   This is expected and training will work without it.")
            print("   For ~50-100% speedup, install Visual Studio Build Tools:")
            print("   https://visualstudio.microsoft.com/downloads/")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(test_torch_compile_availability())
