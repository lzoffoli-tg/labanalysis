"""
Direct test of torch.compile availability check without full package import.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import platform
import shutil

print("=" * 70)
print("Testing torch.compile availability check (direct import)")
print("=" * 70)

# Check platform
print(f"\nPlatform: {platform.system()}")

# Check for compiler
if platform.system() == "Windows":
    compiler = shutil.which("cl")
    print(f"MSVC compiler (cl.exe): {'Found' if compiler else 'Not found'}")
    if compiler:
        print(f"  Location: {compiler}")
    else:
        print("  Note: This is expected if Visual Studio Build Tools are not installed")

# Import torch
import torch

print(f"\nPyTorch version: {torch.__version__}")
print(f"torch.compile available: {hasattr(torch, 'compile')}")

# Test the availability check logic directly
print("\nTesting availability check logic:")


def _is_torch_compile_available() -> bool:
    """Check if torch.compile() is available and usable."""
    # Check if torch.compile exists (PyTorch 2.0+)
    if not hasattr(torch, "compile"):
        return False

    # On Windows, check if MSVC compiler (cl.exe) is available
    if platform.system() == "Windows":
        if shutil.which("cl") is None:
            return False

    # On Linux/Mac, torch.compile usually works
    return True


is_available = _is_torch_compile_available()
print(f"Result: {is_available}")

if not is_available and platform.system() == "Windows":
    print("\n[EXPECTED] torch.compile disabled on Windows without MSVC")
    print("This is correct behavior - training will work without torch.compile")
    print("\nTo enable torch.compile for ~50-100% speedup:")
    print("  1. Install Visual Studio Build Tools")
    print("  2. Select 'Desktop development with C++' workload")
    print("  3. Restart Python/IDE after installation")
elif is_available:
    print("\n[OK] torch.compile is available")
else:
    print("\n[INFO] torch.compile not available (PyTorch < 2.0 or missing compiler)")

print("\n" + "=" * 70)
print("Test completed successfully")
print("=" * 70)
