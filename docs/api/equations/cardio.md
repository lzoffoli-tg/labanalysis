# labanalysis.equations.cardio

Cardio/metabolic prediction equations (VO2, energy expenditure).

**Source**: `src/labanalysis/equations/cardio/`

## Overview

The `equations.cardio` module provides ACSM equations for predicting oxygen consumption (VO2) from exercise intensity.

## Classes

### Run

ACSM running/walking VO2 equations.

```python
class Run:
    @staticmethod
    def vo2_ml_kg_min(speed_m_s: float, grade_pct: float = 0.0) -> float
```

**Parameters:**
- `speed_m_s` (float): Running/walking speed (m/s)
- `grade_pct` (float): Treadmill grade (%, default: 0.0)

**Returns:**
- `float`: Predicted VO2 (ml/kg/min)

**Formula (ACSM):**
```
VO2 = Horizontal + Vertical + Resting
    = (0.2 * speed_m/min) + (0.9 * speed_m/min * grade_decimal) + 3.5
```

**Example:**
```python
from labanalysis.equations.cardio import Run

# Running at 3.5 m/s (12.6 km/h) on flat
vo2 = Run.vo2_ml_kg_min(speed_m_s=3.5, grade_pct=0.0)
print(f"VO2: {vo2:.1f} ml/kg/min")

# Running uphill (5% grade)
vo2_uphill = Run.vo2_ml_kg_min(speed_m_s=3.5, grade_pct=5.0)
print(f"VO2 uphill: {vo2_uphill:.1f} ml/kg/min")
```

---

### Bike

ACSM cycling VO2 equations.

```python
class Bike:
    @staticmethod
    def vo2_ml_kg_min(power_watts: float, bodymass_kg: float) -> float
```

**Parameters:**
- `power_watts` (float): Cycling power output (W)
- `bodymass_kg` (float): Body mass (kg)

**Returns:**
- `float`: Predicted VO2 (ml/kg/min)

**Example:**
```python
from labanalysis.equations.cardio import Bike

# Cycling at 200 W, 75 kg athlete
vo2 = Bike.vo2_ml_kg_min(power_watts=200, bodymass_kg=75)
print(f"VO2: {vo2:.1f} ml/kg/min")
```

---

## See Also

- [VO2max Tests](../protocols/vo2max.md) - VO2max test protocols
- [Strength Equations](strength.md) - Strength prediction

---

**Predict oxygen consumption from exercise intensity using ACSM equations.**
