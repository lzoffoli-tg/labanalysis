# Unit Handling with Pint

Guide for working with physical units using Pint integration in labanalysis.

## Overview

`labanalysis` uses [Pint](https://pint.readthedocs.io/) for unit management, providing:

- **Automatic unit conversion** between compatible units
- **Dimensional analysis** preventing incompatible operations
- **Unit-aware arithmetic** maintaining units through calculations
- **International System (SI)** support and customary units

All `Signal1D`, `Signal3D`, and `Point3D` objects carry a `unit` attribute that tracks physical dimensions.

## Quick Reference

```python
import labanalysis as laban

# Signals have units
marker = body.get_point("left_ankle")
print(marker.unit)  # 'm' (meters)

# Convert units
marker_mm = marker.to_unit('mm')  # Convert to millimeters
print(marker_mm.unit)  # 'mm'

# Arithmetic preserves units
velocity = laban.derivative(marker, order=1)  # m/s
acceleration = laban.derivative(velocity, order=1)  # m/s²
```

## Understanding Units in labanalysis

### Default Units

**Spatial measurements**:
- Position (markers): `m` (meters)
- Velocity: `m/s`
- Acceleration: `m/s²`

**Forces and power**:
- Force: `N` (newtons)
- Power: `W` (watts)
- Torque: `N·m`

**Angles**:
- Angular position: `deg` (degrees) or `rad` (radians)
- Angular velocity: `deg/s` or `rad/s`

**Time**:
- All time series use seconds (`s`) for index

### Accessing Units

```python
import labanalysis as laban

# Load data
body = laban.WholeBody.from_tdf_file("gait.tdf", labels="LABEL")
ankle = body.left_ankle

# Check unit
print(f"Ankle position unit: {ankle.unit}")  # 'm'

# All axes share the same unit
print(f"X unit: {ankle['X'].unit}")  # 'm'
print(f"Y unit: {ankle['Y'].unit}")  # 'm'
print(f"Z unit: {ankle['Z'].unit}")  # 'm'
```

## Unit Conversion

### Basic Conversion

```python
# Convert marker from meters to millimeters
marker_m = body.left_ankle  # Default: meters
marker_mm = marker_m.to_unit('mm')

print(marker_m["Z"].to_numpy()[0])   # e.g., 0.085 (meters)
print(marker_mm["Z"].to_numpy()[0])  # 85.0 (millimeters)
```

### Available Conversions

**Length**:
```python
# Metric
marker.to_unit('m')    # meters
marker.to_unit('cm')   # centimeters
marker.to_unit('mm')   # millimeters
marker.to_unit('km')   # kilometers

# Imperial
marker.to_unit('in')   # inches
marker.to_unit('ft')   # feet
marker.to_unit('yd')   # yards
marker.to_unit('mi')   # miles
```

**Force**:
```python
force_N = fp["FORCE", "Z"]  # Newtons
force_kN = force_N.to_unit('kN')  # kilonewtons
force_lbf = force_N.to_unit('lbf')  # pound-force
```

**Angles**:
```python
angle_deg = body.left_knee_flexionextension  # degrees
angle_rad = angle_deg.to_unit('rad')  # radians

print(angle_deg.to_numpy()[100])  # e.g., 45.0 degrees
print(angle_rad.to_numpy()[100])  # 0.785 radians
```

**Power**:
```python
power_W = jump.peak_power  # Watts
power_kW = power_W.to_unit('kW')  # kilowatts
power_hp = power_W.to_unit('hp')  # horsepower
```

## Unit-Aware Operations

### Arithmetic Operations

```python
import labanalysis as laban
from labanalysis.signalprocessing import derivative

# Load marker (meters)
ankle = body.left_ankle  # unit: 'm'

# Derivative maintains unit consistency
velocity = derivative(ankle, order=1, method='winter')  
print(velocity.unit)  # 'm/s'

acceleration = derivative(velocity, order=1, method='winter')
print(acceleration.unit)  # 'm/s²'

# Arithmetic with scalars
doubled = ankle * 2  # unit: 'm' (preserved)
offset = ankle + 0.1  # unit: 'm' (addition requires same units)
```

### Integration

```python
import numpy as np

# Velocity signal (m/s)
velocity = derivative(marker, order=1)

# Integrate to get displacement
dt = np.mean(np.diff(velocity.index))
displacement = np.cumsum(velocity.to_numpy()) * dt

# Create signal with correct unit
displacement_signal = laban.Signal1D(
    data=displacement,
    index=velocity.index,
    columns=['displacement'],
    unit='m'  # Integral of m/s over seconds = meters
)
```

### Force Platform Operations

```python
# Ground reaction force (Newtons)
force_z = fp["FORCE", "Z"]  # unit: 'N'

# Calculate impulse (N·s)
dt = np.mean(np.diff(force_z.index))
impulse = np.trapz(force_z.to_numpy(), force_z.index)

print(f"Impulse: {impulse:.1f} N·s")

# Power from force and velocity
# Power = Force × Velocity
power = force_z.to_numpy() * velocity.to_numpy()  # N × m/s = W

power_signal = laban.Signal1D(
    data=power,
    index=force_z.index,
    columns=['power'],
    unit='W'
)
```

## Common Patterns

### Converting Force Platform to kN

```python
import labanalysis as laban

# Load force platform (default: Newtons)
fp = laban.ForcePlatform.from_tdf_file("jump.tdf", fp_label="FP1")

# Convert all force channels to kilonewtons
fp_kN = laban.ForcePlatform(
    data=fp.to_numpy() / 1000,  # Convert N to kN
    index=fp.index,
    columns=fp.columns,
    unit='kN'
)

print(fp["FORCE", "Z"].to_numpy().max())     # e.g., 2450.5 N
print(fp_kN["FORCE", "Z"].to_numpy().max())  # 2.45 kN
```

### Normalizing to Body Weight

```python
# Force in Newtons
peak_force_N = jump_results.peak_force  # e.g., 2450 N

# Body weight
body_mass_kg = 75  # kg
body_weight_N = body_mass_kg * 9.81  # Convert kg to N

# Normalize
force_BW = peak_force_N / body_weight_N

print(f"Peak force: {peak_force_N:.0f} N ({force_BW:.2f} BW)")
# Output: Peak force: 2450 N (3.33 BW)
```

### Creating Dimensionless Signals

```python
import labanalysis as laban

# Normalized gait cycle (0-100%)
gait_phase = laban.Signal1D(
    data=np.linspace(0, 100, len(knee_angle)),
    index=knee_angle.index,
    columns=['phase'],
    unit='%'  # Percentage (dimensionless)
)

# Symmetry index (dimensionless)
left_force = 1200  # N
right_force = 1100  # N

symmetry_index = (left_force - right_force) / ((left_force + right_force) / 2) * 100

si_signal = laban.Signal1D(
    data=[symmetry_index],
    index=[0],
    columns=['SI'],
    unit=''  # Dimensionless (unitless)
)
```

## Custom Unit Systems

### Using Pint Directly

```python
from pint import UnitRegistry

ureg = UnitRegistry()

# Create quantity with unit
distance = 5 * ureg.meter
time = 2 * ureg.second

# Automatic unit inference
velocity = distance / time
print(velocity)  # 2.5 meter / second

# Convert to different units
velocity_kmh = velocity.to('km/h')
print(velocity_kmh)  # 9.0 kilometer / hour
```

### Custom Unit Definitions

```python
from pint import UnitRegistry

ureg = UnitRegistry()

# Define custom unit (body weight)
ureg.define('BW = bodyweight')
ureg.define('bodyweight = 75 * kg = BW')  # Define 1 BW = 75 kg

# Use custom unit
force = 2450 * ureg.newton
force_BW = force.to('BW * g')  # Convert to body weights

print(f"Force: {force_BW:.2f}")  # 3.33 BW
```

## Unit Validation

### Preventing Invalid Operations

```python
import labanalysis as laban

# Load signals with different units
marker = body.left_ankle  # meters
force = fp["FORCE", "Z"]  # newtons

# Invalid: Cannot add incompatible units
try:
    invalid = marker + force  # Raises error
except Exception as e:
    print(f"Error: {e}")
    # "Cannot add signals with different units: 'm' and 'N'"
```

### Type Checking

```python
def calculate_velocity(position_signal):
    """
    Calculate velocity from position.
    
    Parameters
    ----------
    position_signal : Signal1D or Signal3D
        Position signal (must have length units)
    
    Raises
    ------
    ValueError
        If signal doesn't have length units
    """
    from pint import UnitRegistry
    ureg = UnitRegistry()
    
    # Check if unit is length
    try:
        ureg.Quantity(1, position_signal.unit).to('m')
    except Exception:
        raise ValueError(
            f"Position signal must have length units, got '{position_signal.unit}'"
        )
    
    # Proceed with calculation
    from labanalysis.signalprocessing import derivative
    velocity = derivative(position_signal, order=1)
    
    return velocity
```

## Best Practices

### 1. Always Specify Units Explicitly

```python
# ✅ GOOD: Explicit unit
signal = laban.Signal1D(
    data=force_data,
    index=time,
    columns=['force'],
    unit='N'  # Clear and explicit
)

# ❌ BAD: Implicit/missing unit
signal = laban.Signal1D(
    data=force_data,
    index=time,
    columns=['force']
    # Missing unit attribute
)
```

### 2. Convert Early, Compute Once

```python
# ✅ GOOD: Convert once at the start
marker_mm = body.left_ankle.to_unit('mm')

# Perform all calculations on converted signal
height_mm = marker_mm["Z"].to_numpy().max()
range_mm = marker_mm["Z"].to_numpy().ptp()

# ❌ BAD: Convert repeatedly
height_mm = body.left_ankle.to_unit('mm')["Z"].to_numpy().max()
range_mm = body.left_ankle.to_unit('mm')["Z"].to_numpy().ptp()  # Redundant conversion
```

### 3. Document Expected Units

```python
def calculate_power(force, velocity):
    """
    Calculate mechanical power.
    
    Parameters
    ----------
    force : Signal1D
        Force signal in Newtons (N)
    velocity : Signal1D
        Velocity signal in meters per second (m/s)
    
    Returns
    -------
    Signal1D
        Power in Watts (W)
    
    Notes
    -----
    Power = Force × Velocity
    Expected units: N × m/s = W
    """
    power_data = force.to_numpy() * velocity.to_numpy()
    
    return laban.Signal1D(
        data=power_data,
        index=force.index,
        columns=['power'],
        unit='W'
    )
```

### 4. Use Pint for Complex Conversions

```python
from pint import UnitRegistry

ureg = UnitRegistry()

# Complex unit conversion
force_N = 2450
area_cm2 = 25

# Pressure = Force / Area
pressure = (force_N * ureg.newton) / (area_cm2 * ureg.cm**2)

# Convert to kPa
pressure_kPa = pressure.to('kPa')
print(f"Pressure: {pressure_kPa:.1f}")  # 980.0 kPa
```

## Troubleshooting

### Unit Mismatch Errors

**Problem**: `ValueError: Cannot perform operation on signals with different units`

**Solution**: Convert signals to compatible units before operations.

```python
# Convert both to same unit
marker_cm = marker.to_unit('cm')
reference_cm = reference.to_unit('cm')

# Now operations work
difference = marker_cm - reference_cm
```

### Missing Units

**Problem**: Signal created without unit attribute.

**Solution**: Always provide `unit` parameter when creating signals.

```python
# Fix: Add unit explicitly
signal = laban.Signal1D(
    data=data,
    index=time,
    columns=['value'],
    unit='m'  # Add this
)
```

### Incorrect Unit Conversions

**Problem**: Conversion factor applied manually instead of using `.to_unit()`.

**Solution**: Use built-in conversion method.

```python
# ❌ BAD: Manual conversion (error-prone)
marker_mm_data = marker.to_numpy() * 1000

# ✅ GOOD: Built-in conversion
marker_mm = marker.to_unit('mm')
```

## See Also

- [Signal Processing](../user-guide/signal-processing/filtering.md) - Unit-preserving operations
- [Custom Signals](custom-signals.md) - Creating signals with custom units
- [Performance Tips](performance-tips.md) - Efficient unit handling
- [Pint Documentation](https://pint.readthedocs.io/) - Official Pint guide

---

**Use Pint integration** for automatic unit conversion and dimensional analysis. Always specify units explicitly when creating signals and use `.to_unit()` for conversions.
