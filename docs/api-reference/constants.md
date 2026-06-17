# labanalysis.constants

Physical constants and default thresholds.

**Source**: `src/labanalysis/constants.py`

## Overview

Predefined constants used throughout labanalysis.

## Constants

### MINIMUM_CONTACT_FORCE_N

```python
MINIMUM_CONTACT_FORCE_N = 20  # Newtons
```

Threshold for detecting ground contact in jumps and gait.

**Usage:**
```python
from labanalysis.constants import MINIMUM_CONTACT_FORCE_N

# Used internally by jump/gait classes
# Override if needed for specific populations
```

---

### G

```python
G = 9.81  # m/s²
```

Gravitational acceleration.

**Usage:**
```python
from labanalysis.constants import G

# Convert force to body weights
force_N = 800
bodymass_kg = 75
force_BW = force_N / (bodymass_kg * G)
```

---

## See Also

- [Utils](utils.md) - Utility functions

---

**Physical constants and default thresholds for biomechanical analysis.**
