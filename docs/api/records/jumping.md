# labanalysis.records.jumping

Jump analysis classes for single jumps, drop jumps, and repeated jumps.

**Source**: `src/labanalysis/records/jumping.py`

## Overview

The `jumping` module provides specialized classes for analyzing different types of vertical jump tests:

- **SingleJump**: Standard countermovement jump (CMJ) or squat jump (SJ)
- **DropJump**: Drop jump with landing and takeoff phases
- **RepeatedJumps**: Multiple consecutive jumps (hopping test)

All classes extend WholeBody and provide:
- Automatic phase detection (eccentric, concentric, flight)
- Jump height calculation from flight time
- Force-time curve analysis
- Performance metrics extraction
- Bilateral and unilateral jump support

## Classes

### SingleJump

Standard single jump analysis (CMJ, SJ).

```python
class SingleJump(WholeBody):
    """
    Represents a single jump trial with phase detection and performance metrics.
    
    Supports countermovement jump (CMJ) and squat jump (SJ) analysis with
    automatic detection of eccentric, concentric, and flight phases from
    ground reaction force data.
    
    Parameters
    ----------
    bodymass_kg : float
        Subject's body mass in kilograms (required)
    left_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for left foot
    right_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for right foot
    left_acromion : Point3D, optional
        Left shoulder marker (for arm swing analysis)
    right_acromion : Point3D, optional
        Right shoulder marker (for arm swing analysis)
    straight_legs : bool, optional
        Whether jump was performed with straight legs (default: False)
    free_hands : bool, optional
        Whether hands were free during jump (default: False)
    **signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals (markers, EMG, joint angles, etc.)
    
    Attributes
    ----------
    bodymass_kg : float
        Subject's body mass (kg)
    side : str
        'bilateral', 'left', or 'right' depending on available force data
    vertical_axis : str
        Name of vertical axis ('X', 'Y', or 'Z')
    anteroposterior_axis : str
        Name of anteroposterior axis
    lateral_axis : str
        Name of lateral axis
    
    Properties
    ----------
    eccentric_phase : WholeBody
        Data during eccentric (lowering) phase
    concentric_phase : WholeBody
        Data during concentric (propulsion) phase
    flight_phase : WholeBody
        Data during flight phase
    contact_phase : WholeBody
        Combined eccentric + concentric phases
    contact_time_s : float
        Duration of contact phase (seconds)
    flight_time_s : float
        Duration of flight phase (seconds)
    takeoff_velocity_ms : float
        Vertical velocity at takeoff (m/s)
    elevation_cm : float
        Jump height calculated from flight time (cm)
    output_metrics : pd.DataFrame
        Summary DataFrame with all jump metrics
    
    Notes
    -----
    At least one foot force platform (left or right) must be provided.
    
    For bilateral jumps, forces are averaged across both platforms.
    
    Phase Detection Algorithm:
    1. Flight phase: Vertical GRF < MINIMUM_CONTACT_FORCE_N (default: 20 N)
    2. Contact phase: All non-flight time before flight
    3. Eccentric phase: Contact start to minimum vertical GRF
    4. Concentric phase: Minimum vertical GRF to flight start
    
    Examples
    --------
    >>> import labanalysis as laban
    >>> 
    >>> # Load jump data
    >>> data = laban.read_tdf(
    ...     "cmj.tdf",
    ...     forceplatform_keys=[".*"]
    ... )
    >>> 
    >>> # Create jump object
    >>> jump = laban.SingleJump(
    ...     bodymass_kg=75.0,
    ...     **data
    ... )
    >>> 
    >>> # Get jump metrics
    >>> print(f"Jump height: {jump.elevation_cm:.1f} cm")
    >>> print(f"Flight time: {jump.flight_time_s:.3f} s")
    >>> print(f"Takeoff velocity: {jump.takeoff_velocity_ms:.2f} m/s")
    >>> 
    >>> # Extract phases
    >>> eccentric = jump.eccentric_phase
    >>> concentric = jump.concentric_phase
    >>> flight = jump.flight_phase
    >>> 
    >>> # Get full metrics DataFrame
    >>> metrics = jump.output_metrics
    >>> print(metrics)
    """
```

#### Key Properties

##### eccentric_phase

Extract data during the eccentric (lowering) phase.

```python
@property
def eccentric_phase(self) -> WholeBody
```

**Returns:**
- `WholeBody`: All signals sliced from contact start to minimum vertical GRF

**Example:**
```python
eccentric = jump.eccentric_phase
ecc_force = eccentric.resultant_force.force['Z']
ecc_duration = eccentric.index[-1] - eccentric.index[0]
print(f"Eccentric phase: {ecc_duration:.3f} s")
```

##### concentric_phase

Extract data during the concentric (propulsion) phase.

```python
@property
def concentric_phase(self) -> WholeBody
```

**Returns:**
- `WholeBody`: All signals sliced from minimum vertical GRF to flight start

**Example:**
```python
concentric = jump.concentric_phase
con_force = concentric.resultant_force.force['Z']
peak_force = con_force.to_numpy().max()
print(f"Peak concentric force: {peak_force:.1f} N")
```

##### flight_phase

Extract data during the flight (airborne) phase.

```python
@property
def flight_phase(self) -> WholeBody
```

**Returns:**
- `WholeBody`: All signals sliced during flight (GRF < threshold)

**Example:**
```python
flight = jump.flight_phase
flight_duration = flight.index[-1] - flight.index[0]
print(f"Flight time: {flight_duration:.3f} s")
```

##### elevation_cm

Calculate jump height from flight time.

```python
@property
def elevation_cm(self) -> float
```

**Returns:**
- `float`: Jump height in centimeters

**Calculation:**
```
h = (g * t²) / 8
where:
  h = jump height (m)
  g = 9.81 m/s²
  t = flight time (s)
```

**Example:**
```python
height_cm = jump.elevation_cm
height_m = height_cm / 100
print(f"Jump height: {height_cm:.1f} cm ({height_m:.3f} m)")
```

##### output_metrics

Get comprehensive jump metrics as DataFrame.

```python
@property
def output_metrics(self) -> pd.DataFrame
```

**Returns:**
- `pd.DataFrame`: Summary metrics including:
  - `jump_height_m`: Jump height (meters)
  - `flight_time_s`: Flight duration (seconds)
  - `contact_time_s`: Contact duration (seconds)
  - `eccentric_duration_s`: Eccentric phase duration
  - `concentric_duration_s`: Concentric phase duration
  - `peak_force_N`: Maximum vertical GRF
  - `eccentric_peak_velocity_m_s`: Peak downward velocity
  - `concentric_peak_velocity_m_s`: Peak upward velocity (takeoff)
  - `rfd_N_s`: Rate of force development
  - Additional metrics depending on available signals

**Example:**
```python
metrics = jump.output_metrics
metrics.to_excel("jump_report.xlsx", index=False)

print("Jump Summary:")
for col in metrics.columns:
    print(f"  {col}: {metrics[col].values[0]}")
```

#### Complete Example

```python
import labanalysis as laban
import matplotlib.pyplot as plt

# Load data
data = laban.read_tdf("cmj.tdf", forceplatform_keys=[".*"])

# Create jump
jump = laban.SingleJump(bodymass_kg=75.0, **data)

# Extract metrics
print(f"Jump height: {jump.elevation_cm:.1f} cm")
print(f"Flight time: {jump.flight_time_s:.3f} s")
print(f"Contact time: {jump.contact_time_s:.3f} s")

# Get phases
eccentric = jump.eccentric_phase
concentric = jump.concentric_phase
flight = jump.flight_phase

# Plot force-time curve with phase markers
force = jump.resultant_force.force['Z'].to_numpy()
time = jump.resultant_force.index

plt.figure(figsize=(12, 6))
plt.plot(time, force, 'k-', linewidth=1.5)
plt.axvline(eccentric.index[0], color='blue', linestyle='--', label='Contact start')
plt.axvline(concentric.index[0], color='green', linestyle='--', label='Concentric start')
plt.axvline(flight.index[0], color='red', linestyle='--', label='Flight start')
plt.xlabel('Time (s)')
plt.ylabel('Vertical GRF (N)')
plt.title(f'CMJ Force-Time Curve (Jump height: {jump.elevation_cm:.1f} cm)')
plt.legend()
plt.grid(True)
plt.show()
```

---

### DropJump

Drop jump with landing and rebound phases.

```python
class DropJump(SingleJump):
    """
    Represents a drop jump trial with landing and takeoff analysis.
    
    Drop jump extends SingleJump to include a landing phase before the
    propulsive takeoff. Useful for analyzing reactive strength and
    stretch-shortening cycle performance.
    
    Parameters
    ----------
    Inherits all parameters from SingleJump
    
    Additional Properties
    ---------------------
    landing_phase : WholeBody
        Data during landing phase (before contact phase)
    landing_time_s : float
        Duration of landing phase (seconds)
    reactive_strength_index : float
        RSI = jump_height_m / contact_time_s
    
    Notes
    -----
    Phase sequence for drop jump:
    1. Landing phase: Initial ground contact after drop
    2. Contact phase: Combined eccentric-concentric
       - Eccentric: Deceleration after landing
       - Concentric: Propulsion for takeoff
    3. Flight phase: Airborne after takeoff
    
    The drop jump is characterized by:
    - Short contact time (<= 250 ms optimal)
    - High reactive strength index (RSI)
    - Minimal eccentric depth (fast stretch-shortening cycle)
    
    Examples
    --------
    >>> import labanalysis as laban
    >>> 
    >>> # Load drop jump data
    >>> data = laban.read_tdf(
    ...     "drop_jump_30cm.tdf",
    ...     forceplatform_keys=[".*"]
    ... )
    >>> 
    >>> # Create drop jump object
    >>> dj = laban.DropJump(
    ...     bodymass_kg=75.0,
    ...     **data
    ... )
    >>> 
    >>> # Get drop jump metrics
    >>> print(f"Jump height: {dj.elevation_cm:.1f} cm")
    >>> print(f"Contact time: {dj.contact_time_s*1000:.0f} ms")
    >>> print(f"RSI: {dj.reactive_strength_index:.2f}")
    >>> 
    >>> # Analyze landing
    >>> landing = dj.landing_phase
    >>> landing_force = landing.resultant_force.force['Z']
    >>> peak_landing_force = landing_force.to_numpy().max()
    >>> print(f"Peak landing force: {peak_landing_force:.0f} N ({peak_landing_force/(dj.bodymass_kg*9.81):.1f} BW)")
    """
```

**Example - Drop Jump Assessment:**

```python
import labanalysis as laban
import pandas as pd

# Analyze multiple drop heights
drop_heights = [20, 30, 40, 50]  # cm
results = []

for height in drop_heights:
    # Load data
    data = laban.read_tdf(f"drop_jump_{height}cm.tdf", forceplatform_keys=[".*"])
    
    # Analyze
    dj = laban.DropJump(bodymass_kg=75.0, **data)
    
    # Metrics
    results.append({
        'drop_height_cm': height,
        'jump_height_cm': dj.elevation_cm,
        'contact_time_ms': dj.contact_time_s * 1000,
        'rsi': dj.reactive_strength_index,
        'peak_landing_force_N': dj.landing_phase.resultant_force.force['Z'].to_numpy().max()
    })

# Find optimal drop height (max RSI)
df = pd.DataFrame(results)
optimal_idx = df['rsi'].idxmax()
optimal_height = df.loc[optimal_idx, 'drop_height_cm']

print(df)
print(f"\nOptimal drop height: {optimal_height} cm (RSI = {df.loc[optimal_idx, 'rsi']:.2f})")
```

---

### RepeatedJumps

Multiple consecutive jumps (hopping test).

```python
class RepeatedJumps(WholeBody):
    """
    Represents a repeated jumps trial with multiple consecutive jumps.
    
    Used for analyzing hopping tests, repeated CMJ protocols, or
    fatigue assessments. Automatically detects individual jumps and
    extracts metrics for each.
    
    Parameters
    ----------
    bodymass_kg : float
        Subject's body mass in kilograms
    left_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for left foot
    right_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for right foot
    exclude_jumps : list of int, optional
        Indices of jumps to exclude from analysis (default: [0, -1])
        First and last jumps excluded by default
    straight_legs : bool, optional
        Whether jumps performed with straight legs (default: False)
    free_hands : bool, optional
        Whether hands were free (default: False)
    **signals : Signal1D, Signal3D, EMGSignal, Point3D
        Additional signals
    
    Properties
    ----------
    jumps : list of SingleJump
        Detected individual jumps (excluding excluded_jumps)
    output_metrics : pd.DataFrame
        Metrics for all jumps combined
    
    Notes
    -----
    Jump Detection Algorithm:
    1. Identify flight phases (GRF < MINIMUM_CONTACT_FORCE_N)
    2. Each flight must be >= MINIMUM_FLIGHT_TIME_S (default: 0.1 s)
    3. Extract contact+flight as individual SingleJump object
    
    First and last jumps are excluded by default because:
    - First jump: Often starts from standing (not representative)
    - Last jump: May be incomplete or influenced by anticipation
    
    Examples
    --------
    >>> import labanalysis as laban
    >>> 
    >>> # Load repeated jumps data
    >>> data = laban.read_tdf(
    ...     "hopping_test.tdf",
    ...     forceplatform_keys=[".*"]
    ... )
    >>> 
    >>> # Create repeated jumps object
    >>> rj = laban.RepeatedJumps(
    ...     bodymass_kg=75.0,
    ...     exclude_jumps=[0, -1],  # Exclude first and last
    ...     **data
    ... )
    >>> 
    >>> # Get individual jumps
    >>> jumps = rj.jumps
    >>> print(f"Detected {len(jumps)} valid jumps")
    >>> 
    >>> # Analyze each jump
    >>> for i, jump in enumerate(jumps):
    ...     print(f"Jump {i+1}: {jump.elevation_cm:.1f} cm, {jump.contact_time_s*1000:.0f} ms")
    >>> 
    >>> # Get combined metrics
    >>> metrics = rj.output_metrics
    >>> print(metrics.describe())
    """
```

**Example - Fatigue Analysis:**

```python
import labanalysis as laban
import pandas as pd
import matplotlib.pyplot as plt

# Load repeated jumps
data = laban.read_tdf("hopping_30s.tdf", forceplatform_keys=[".*"])
rj = laban.RepeatedJumps(bodymass_kg=75.0, exclude_jumps=[0, -1], **data)

# Extract metrics for each jump
jumps_data = []
for i, jump in enumerate(rj.jumps):
    jumps_data.append({
        'jump_number': i + 1,
        'height_cm': jump.elevation_cm,
        'contact_time_ms': jump.contact_time_s * 1000,
        'rsi': jump.elevation_cm / 100 / jump.contact_time_s
    })

df = pd.DataFrame(jumps_data)

# Calculate fatigue index
initial_height = df['height_cm'].iloc[:3].mean()  # First 3 jumps
final_height = df['height_cm'].iloc[-3:].mean()   # Last 3 jumps
fatigue_index = ((initial_height - final_height) / initial_height) * 100

print(f"Fatigue Index: {fatigue_index:.1f}%")
print(f"Initial height: {initial_height:.1f} cm")
print(f"Final height: {final_height:.1f} cm")

# Plot height decline
plt.figure(figsize=(10, 6))
plt.plot(df['jump_number'], df['height_cm'], 'o-', linewidth=2)
plt.axhline(initial_height, color='green', linestyle='--', label='Initial average')
plt.axhline(final_height, color='red', linestyle='--', label='Final average')
plt.xlabel('Jump Number')
plt.ylabel('Jump Height (cm)')
plt.title(f'Jump Height Decline (Fatigue Index: {fatigue_index:.1f}%)')
plt.legend()
plt.grid(True)
plt.show()
```

**Example - Consistency Analysis:**

```python
import labanalysis as laban
import pandas as pd
import numpy as np

# Load data
data = laban.read_tdf("repeated_cmj.tdf", forceplatform_keys=[".*"])
rj = laban.RepeatedJumps(bodymass_kg=75.0, **data)

# Extract heights
heights = [jump.elevation_cm for jump in rj.jumps]

# Calculate consistency metrics
mean_height = np.mean(heights)
std_height = np.std(heights)
cv_percent = (std_height / mean_height) * 100

print(f"Mean jump height: {mean_height:.1f} ± {std_height:.1f} cm")
print(f"Coefficient of variation: {cv_percent:.1f}%")
print(f"Range: {min(heights):.1f} - {max(heights):.1f} cm")

# Identify outliers (> 2 SD from mean)
outliers = [i for i, h in enumerate(heights) if abs(h - mean_height) > 2*std_height]
if outliers:
    print(f"Outlier jumps: {outliers}")
```

---

## Common Workflows

### 1. Single Jump Analysis

```python
import labanalysis as laban

# Load CMJ data
data = laban.read_tdf("cmj.tdf", forceplatform_keys=[".*"])
jump = laban.SingleJump(bodymass_kg=75.0, **data)

# Get metrics
metrics = jump.output_metrics
print(metrics)

# Export
metrics.to_excel("cmj_report.xlsx", index=False)
```

### 2. Bilateral vs Unilateral Comparison

```python
import labanalysis as laban

# Load bilateral jump
bilateral_data = laban.read_tdf("bilateral_cmj.tdf", forceplatform_keys=[".*"])
bilateral = laban.SingleJump(bodymass_kg=75.0, **bilateral_data)

# Load unilateral jumps
left_data = laban.read_tdf("left_leg_cmj.tdf", forceplatform_keys=["left.*"])
left = laban.SingleJump(bodymass_kg=75.0, **left_data)

right_data = laban.read_tdf("right_leg_cmj.tdf", forceplatform_keys=["right.*"])
right = laban.SingleJump(bodymass_kg=75.0, **right_data)

# Compare
print(f"Bilateral: {bilateral.elevation_cm:.1f} cm")
print(f"Left leg: {left.elevation_cm:.1f} cm ({left.elevation_cm/bilateral.elevation_cm*100:.0f}%)")
print(f"Right leg: {right.elevation_cm:.1f} cm ({right.elevation_cm/bilateral.elevation_cm*100:.0f}%)")

# Asymmetry index
asymmetry = abs(left.elevation_cm - right.elevation_cm) / ((left.elevation_cm + right.elevation_cm) / 2) * 100
print(f"Asymmetry: {asymmetry:.1f}%")
```

### 3. Phase-Specific Analysis

```python
import labanalysis as laban
import numpy as np

# Load jump
data = laban.read_tdf("cmj.tdf", forceplatform_keys=[".*"])
jump = laban.SingleJump(bodymass_kg=75.0, **data)

# Eccentric phase analysis
ecc = jump.eccentric_phase
ecc_force = ecc.resultant_force.force['Z'].to_numpy()
ecc_duration = ecc.index[-1] - ecc.index[0]
ecc_impulse = np.trapz(ecc_force, ecc.index)

print(f"Eccentric duration: {ecc_duration:.3f} s")
print(f"Eccentric impulse: {ecc_impulse:.1f} N·s")

# Concentric phase analysis
con = jump.concentric_phase
con_force = con.resultant_force.force['Z'].to_numpy()
con_duration = con.index[-1] - con.index[0]
con_impulse = np.trapz(con_force, con.index)
peak_force = con_force.max()
rfd = np.gradient(con_force, con.index).max()

print(f"Concentric duration: {con_duration:.3f} s")
print(f"Concentric impulse: {con_impulse:.1f} N·s")
print(f"Peak force: {peak_force:.0f} N ({peak_force/(jump.bodymass_kg*9.81):.1f} BW)")
print(f"Peak RFD: {rfd:.0f} N/s")
```

---

## Troubleshooting

### Issue: "No flight phase detected"

**Cause**: Insufficient force data or threshold too high

**Solution**:
```python
# Check raw force data
force = data['left_foot_ground_reaction_force'].force['Z'].to_numpy()
print(f"Force range: {force.min():.1f} - {force.max():.1f} N")

# Verify threshold
from labanalysis.constants import MINIMUM_CONTACT_FORCE_N
print(f"Threshold: {MINIMUM_CONTACT_FORCE_N} N")
```

### Issue: "Jump height seems incorrect"

**Cause**: Body mass not provided or vertical axis incorrect

**Solution**:
```python
# Verify body mass
print(f"Body mass: {jump.bodymass_kg} kg")

# Check vertical axis
print(f"Vertical axis: {jump.vertical_axis}")

# Manually calculate from flight time
flight_time = jump.flight_time_s
height_m = (9.81 * flight_time**2) / 8
print(f"Calculated height: {height_m*100:.1f} cm")
```

### Issue: "Phase durations don't make sense"

**Cause**: Automatic phase detection may fail with noisy data

**Solution**: Filter force data before analysis
```python
import labanalysis as laban

# Filter force platform data
fp = data['left_foot_ground_reaction_force']
filtered_force = laban.butterworth_filt(fp.force['Z'], fs=1000, fc=10, order=4)
fp.force['Z'] = laban.Signal1D(data=filtered_force, index=fp.index, unit='N')

# Re-analyze
jump = laban.SingleJump(bodymass_kg=75.0, **data)
```

---

## See Also

- [WholeBody](bodies.md) - Full body biomechanical model
- [ForcePlatform](records.md#forceplatform) - Force platform data structure
- [Signal Processing](../signalprocessing.md) - Filtering and analysis
- [Jump Tests](../protocols/jump-tests.md) - Jump test protocols
- [Jump Tutorial](../../tutorials/01-jump-analysis.md) - Complete workflow

---

**Analyze vertical jumps with automatic phase detection and comprehensive metrics.**
