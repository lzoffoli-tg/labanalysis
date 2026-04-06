"""
Test script to verify if torch.compile works on this system.
"""
import shutil
import sys

import torch
import torch.nn as nn

print("=" * 70)
print("TORCH.COMPILE VERIFICATION TEST")
print("=" * 70)
print()

# System info
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print()

# Check if torch.compile exists
if not hasattr(torch, "compile"):
    print("[FAIL] torch.compile NOT AVAILABLE (PyTorch < 2.0)")
    sys.exit(1)
else:
    print("[OK] torch.compile is available (PyTorch >= 2.0)")

# Check if compiler is in PATH
compiler_available = shutil.which("cl") is not None
status = "[OK]" if compiler_available else "[FAIL]"
print(f"{status} MSVC compiler (cl.exe) in PATH: {compiler_available}")
print()

# Create a simple model
print("Creating a simple test model...")
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)
print("[OK] Model created")
print()

# Try to compile the model
print("Attempting to compile the model with torch.compile...")
try:
    compiled_model = torch.compile(model, mode="reduce-overhead")
    print("[OK] Model compilation wrapper created (lazy compilation)")
except Exception as e:
    print(f"[FAIL] FAILED to create compilation wrapper: {type(e).__name__}: {e}")
    sys.exit(1)
print()

# Try to run a forward pass (this triggers actual compilation)
print("Running forward pass (this triggers actual C++ compilation)...")
test_input = torch.randn(4, 10)

try:
    with torch.no_grad():
        output = compiled_model(test_input)
    print("[OK] SUCCESS! torch.compile works correctly!")
    print(f"   Output shape: {output.shape}")
    print()
    print("=" * 70)
    print("RESULT: torch.compile is FULLY FUNCTIONAL on your system")
    print("=" * 70)
    sys.exit(0)

except Exception as e:
    print(f"[FAIL] FAILED during execution: {type(e).__name__}")
    print()
    print("Error details:")
    print("-" * 70)

    error_str = str(e)

    # Check for OpenMP issue
    if "omp.h" in error_str or "OpenMP" in error_str:
        print("Issue: Missing OpenMP headers")
        print()
        print("The MSVC compiler cannot find 'omp.h'. This typically means:")
        print("1. OpenMP is not included in the Visual Studio installation, OR")
        print("2. The INCLUDE environment variable is not set correctly")
        print()
        print("Solutions:")
        print("  • Run code in 'Developer Command Prompt for VS 2026'")
        print("  • Use the setup_vs_env.ps1 script before running Python")
        print("  • Install standalone OpenMP: conda install -c conda-forge llvm-openmp")
        print("  • Disable torch.compile: use_torch_compile=False")
    else:
        # Print first few lines of error
        lines = error_str.split('\n')
        for line in lines[:10]:
            print(line)
        if len(lines) > 10:
            print("... (error truncated)")

    print()
    print("=" * 70)
    print("RESULT: torch.compile FAILED - fallback to eager mode will be used")
    print("=" * 70)
    sys.exit(1)
