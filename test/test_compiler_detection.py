"""Test improved compiler detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("Testing Improved Compiler Detection")
print("=" * 70)
print()

# Test 1: Check PATH
import shutil

cl_in_path = shutil.which("cl")
print(f"[1/3] cl.exe in PATH: {cl_in_path is not None}")
if cl_in_path:
    print(f"      Location: {cl_in_path}")
print()

# Test 2: Check Visual Studio locations
print("[2/3] Searching Visual Studio installations...")
vs_patterns = [
    "Program Files/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe",
    "Program Files (x86)/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe",
]

found_in_vs = False
for pattern in vs_patterns:
    matches = list(Path("C:/").glob(pattern))
    if matches:
        print(f"      Found: YES")
        print(f"      Location: {matches[0]}")
        found_in_vs = True
        break

if not found_in_vs:
    print(f"      Found: NO")
print()

# Test 3: Test our _is_torch_compile_available() function
print("[3/3] Testing _is_torch_compile_available()...")
try:
    from labanalysis.modelling.pytorch.utils import TorchTrainer

    available = TorchTrainer._is_torch_compile_available()
    print(f"      Result: {available}")

    if available:
        print()
        print("=" * 70)
        print("SUCCESS: Compiler detected by TorchTrainer!")
        print("=" * 70)
        print()
        print("torch.compile() will be ENABLED automatically")
        print("Expected speedup: +50-100% (2.5-3.5x total)")
        print()

        if not cl_in_path:
            print("NOTE: Compiler found but not in PATH")
            print("      Recommendation: Add to PATH for better compatibility")
            print("      Run: scripts\\add_compiler_to_path.ps1 (as admin)")
    else:
        print()
        print("=" * 70)
        print("Compiler NOT detected")
        print("=" * 70)
        print()
        print("torch.compile() will be DISABLED")
        print("Speedup: ~1.5-2x (without torch.compile)")

except ImportError as e:
    print(f"      Error: Cannot import TorchTrainer")
    print(f"      Reason: {e}")
    print()
    print("This is expected if dependencies are missing in global Python")
except Exception as e:
    print(f"      Error: {e}")
    import traceback

    traceback.print_exc()

print()
print("=" * 70)
