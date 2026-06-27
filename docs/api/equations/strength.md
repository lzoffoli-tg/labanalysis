# labanalysis.equations.strength

Strength prediction equations (1RM, load-velocity).

**Source**: `src/labanalysis/equations/strength.py`

## Overview

The `equations.strength` module provides equations for predicting maximal strength from submaximal loads.

## Functions

### Brzycki1RM()

Predict 1-repetition maximum (1RM) using Brzycki equation.

```python
def Brzycki1RM(
    load_kg: float,
    reps: int
) -> float
```

**Parameters:**
- `load_kg` (float): Load lifted (kg)
- `reps` (int): Number of repetitions performed

**Returns:**
- `float`: Predicted 1RM (kg)

**Formula:**
```
1RM = load / (1.0278 - 0.0278 * reps)
```

**Valid Range:** 1-10 repetitions (most accurate at 2-6 reps)

**Example:**
```python
from labanalysis.equations.strength import Brzycki1RM

# Athlete lifts 100 kg for 5 reps
load = 100  # kg
reps = 5

predicted_1rm = Brzycki1RM(load, reps)
print(f"Predicted 1RM: {predicted_1rm:.1f} kg")
# Output: Predicted 1RM: 112.5 kg
```

**Example - Multiple Loads:**
```python
from labanalysis.equations.strength import Brzycki1RM
import pandas as pd

# Test data
tests = [
    (100, 5),
    (90, 8),
    (80, 10),
]

results = []
for load, reps in tests:
    pred_1rm = Brzycki1RM(load, reps)
    results.append({
        'load_kg': load,
        'reps': reps,
        'predicted_1rm_kg': pred_1rm
    })

df = pd.DataFrame(results)
print(df)
print(f"\nMean predicted 1RM: {df['predicted_1rm_kg'].mean():.1f} kg")
```

---

## See Also

- [Strength Tests](../protocols/strength-tests.md) - Strength test protocols
- [Cardio Equations](cardio.md) - VO2 prediction equations

---

**Predict maximal strength from submaximal loads.**
