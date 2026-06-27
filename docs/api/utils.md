# labanalysis.utils

Utility functions and type annotations.

**Source**: `src/labanalysis/utils.py`

## Overview

General utilities for units, file operations, and type hints.

## Unit Registry

### ureg

Pint unit registry for physical units.

```python
from labanalysis.utils import ureg

# Create quantities
force = 100 * ureg.N
distance = 1.5 * ureg.m
velocity = 3.5 * ureg.m / ureg.s

# Convert units
force_lbf = force.to(ureg.lbf)
```

### Quantity Helpers

```python
from labanalysis.utils import bpm_quantity, au_quantity, Q_

# Beats per minute
hr = bpm_quantity(150)

# Arbitrary units
signal = au_quantity(1.5)

# General quantity
length = Q_(10, 'cm')
```

---

## File Operations

### get_files()

Find files matching pattern.

```python
def get_files(directory: str, pattern: str = '*') -> list[str]
```

**Example:**
```python
from labanalysis.utils import get_files

# Find all TDF files
tdf_files = get_files('data/', '*.tdf')
```

---

## Data Utilities

### split_data()

Split data into train/test sets.

```python
def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2
) -> tuple
```

---

## Type Annotations

```python
from labanalysis.utils import (
    FloatArray1D,  # 1D numpy array
    FloatArray2D,  # 2D numpy array
    IntArray1D,    # 1D integer array
    TextArray1D,   # 1D string array
)
```

---

## See Also

- [Constants](constants.md) - Physical constants

---

**Utility functions for units, files, and type annotations.**
