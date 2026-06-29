# labanalysis.exercises.gait

Gait analysis classes for running and walking locomotion.

**Source**: `src/labanalysis/exercises/gait/`

## Overview

The `gait` module provides a comprehensive framework for gait analysis, supporting both running and walking movements with automatic cycle detection using kinematic (marker-based) or kinetic (force platform-based) algorithms.

**Module Structure**:

```
labanalysis.exercises.gait/
├── _base.py            - GaitObject (base class)
├── _cycle.py           - GaitCycle (single cycle)
├── _exercise.py        - GaitExercise (multi-cycle container)
├── running_step.py     - RunningStep
├── running_exercise.py - RunningExercise
├── walking_stride.py   - WalkingStride
└── walking_exercise.py - WalkingExercise
```

**Key Features**:
- **Dual algorithm support**: Kinematic (marker-based) or kinetic (force platform)
- **Automatic cycle detection**: Extract individual gait cycles from continuous data
- **Phase extraction**: Separate flight/contact (running) or swing/stance (walking)
- **Gait metrics**: Temporal, spatial, and kinematic/kinetic parameters
- **Visualization**: Interactive Plotly visualizations

## Classes

### Class Hierarchy

```
WholeBody (from bodies.py)
    └── GaitObject (base for all gait classes)
            ├── GaitCycle (single cycle base)
            │       ├── RunningStep
            │       └── WalkingStride
            └── GaitExercise (multi-cycle base)
                    ├── RunningExercise
                    └── WalkingExercise
```

---

## GaitObject

Base class for gait analysis with dual-algorithm cycle detection support.

```python
class GaitObject(WholeBody):
    """
    Base class for gait analysis with kinetic and kinematic cycle detection.
    
    Extends WholeBody to provide specialized functionality for gait analysis,
    including support for multiple cycle detection algorithms, ground reaction
    force tracking, and gait-specific anatomical landmarks.
    
    Parameters
    ----------
    algorithm : {'kinematics', 'kinetics'}
        Cycle detection algorithm:
        - 'kinetics': Uses force platform data (ground reaction forces)
        - 'kinematics': Uses marker trajectories (heel/toe positions)
    ground_reaction_force_threshold : float, optional
        Minimum vertical GRF (N) for contact detection (kinetics algorithm).
        Default: 20 N
    height_threshold : float, optional
        Maximum vertical height (% of max) for contact detection (kinematics).
        Default: 0.1 (10%)
    left_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for left foot
    right_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for right foot
    left_heel : Point3D, optional
        Left heel marker trajectory
    right_heel : Point3D, optional
        Right heel marker trajectory
    left_toe : Point3D, optional
        Left toe marker trajectory
    right_toe : Point3D, optional
        Right toe marker trajectory
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D
        Additional signals (joint angles, EMG, other markers)
    
    Attributes
    ----------
    algorithm : str
        Selected cycle detection algorithm
    ground_reaction_force_threshold : float
        GRF threshold for contact detection (N)
    height_threshold : float
        Height threshold for contact detection (%)
    
    Notes
    -----
    Algorithm selection includes automatic fallback:
    - If 'kinetics' requested but no force data → fallback to 'kinematics'
    - If 'kinematics' requested but incomplete markers → fallback to 'kinetics'
    
    Inherits all 42 anatomical markers from WholeBody. See WholeBody
    documentation for complete marker list.
    """
```

**Example:**

```python
import labanalysis as laban

# Load data
data = laban.read_tdf(
    "running.tdf",
    marker_keys=[".*"],
    forceplatform_keys=[".*"]
)

# Create GaitObject (base class, rarely used directly)
gait = laban.GaitObject(algorithm='kinetics', **data)
```

---

## GaitCycle

Base class for individual gait cycles.

```python
class GaitCycle(GaitObject):
    """
    Represents a single gait cycle.
    
    Parameters
    ----------
    side : {'left', 'right'}
        Side of the cycle
    algorithm : {'kinematics', 'kinetics'}
        Cycle detection algorithm
    ground_reaction_force : ForcePlatform, optional
        Force platform data for this cycle
    **kwargs
        Additional parameters from GaitObject
    
    Attributes
    ----------
    side : str
        'left' or 'right'
    init_s : float
        Cycle start time (toeoff) in seconds
    end_s : float
        Cycle end time (next toeoff) in seconds
    footstrike_s : float
        Footstrike event time in seconds
    midstance_s : float
        Midstance event time in seconds
    
    Notes
    -----
    Cycle timing convention:
    init_s (toeoff) → footstrike_s → midstance_s → end_s (next toeoff)
    
    This is a base class. Use RunningStep or WalkingStride for specific
    locomotion types.
    """
```

**Properties:**

- `cycle_time_s` - Total cycle duration (seconds)
- `output_metrics` - Summary DataFrame with all metrics

---

## GaitExercise

Base class for multi-cycle gait exercises.

```python
class GaitExercise(GaitObject):
    """
    Represents a complete gait exercise with multiple cycles.
    
    Automatically detects and extracts individual gait cycles from
    continuous locomotion data. Subclasses implement specific detection
    algorithms for running vs. walking.
    
    Parameters
    ----------
    Inherits all parameters from GaitObject
    
    Attributes
    ----------
    cycles : list of GaitCycle
        Detected gait cycles
    
    Notes
    -----
    This is an abstract base class. Subclasses must implement:
    - _find_cycles_kinetics() : Detect cycles using force platform data
    - _find_cycles_kinematics() : Detect cycles using marker trajectories
    
    Use RunningExercise or WalkingExercise for specific locomotion types.
    """
```

**Properties:**

- `cycles` - List of detected GaitCycle objects

---

## RunningStep

Single running step (gait cycle during running).

```python
class RunningStep(GaitCycle):
    """
    Represents a single running step with flight and contact phases.
    
    Running is characterized by a flight phase (no ground contact) followed
    by a contact phase (ground contact). Contact phase is subdivided into
    loading response and propulsion.
    
    Parameters
    ----------
    Inherits all parameters from GaitCycle
    
    Attributes
    ----------
    flight_phase : WholeBody
        Data during flight phase (toeoff to footstrike)
    contact_phase : WholeBody
        Data during contact phase (footstrike to next toeoff)
    loading_response_phase : WholeBody
        Data during loading response (footstrike to midstance)
    propulsion_phase : WholeBody
        Data during propulsion (midstance to toeoff)
    flight_time_s : float
        Flight phase duration (seconds)
    contact_time_s : float
        Contact phase duration (seconds)
    loadingresponse_time_s : float
        Loading response duration (seconds)
    propulsion_time_s : float
        Propulsion phase duration (seconds)
    
    Notes
    -----
    Cycle timing pattern:
    init_s (toeoff) → FLIGHT → footstrike_s → LOADING → midstance_s → 
    PROPULSION → end_s (next toeoff)
    
    See Also
    --------
    WalkingStride : Gait cycle for walking
    RunningExercise : Multi-cycle running analysis
    """
```

**Example:**

```python
import labanalysis as laban

# Load running trial
data = laban.read_tdf("running.tdf", marker_keys=[".*"], forceplatform_keys=[".*"])
running = laban.RunningExercise(algorithm='kinetics', **data)

# Get first step
step = running.cycles[0]

# Analyze phases
print(f"Flight time: {step.flight_time_s:.3f} s")
print(f"Contact time: {step.contact_time_s:.3f} s")

# Extract phase data
flight_data = step.flight_phase
contact_data = step.contact_phase

# Get metrics
metrics = step.output_metrics
print(metrics)
```

---

## RunningExercise

Multi-cycle running analysis.

```python
class RunningExercise(GaitExercise):
    """
    Complete running exercise with automatic step detection.
    
    Detects and extracts individual RunningStep cycles from continuous
    running data using either kinematic or kinetic algorithms.
    
    Parameters
    ----------
    algorithm : {'kinematics', 'kinetics'}
        Cycle detection algorithm
    ground_reaction_force_threshold : float, optional
        Minimum vertical GRF (N) for contact detection
        Default: 20 N
    height_threshold : float, optional
        Height threshold for kinematic detection (%)
        Default: 0.1 (10% of max height)
    **kwargs
        Data signals (markers, force platforms, EMG, etc.)
    
    Attributes
    ----------
    cycles : list of RunningStep
        Detected running steps
    
    Examples
    --------
    >>> import labanalysis as laban
    >>> 
    >>> # Load running data
    >>> data = laban.read_tdf(
    ...     "running.tdf",
    ...     marker_keys=[".*"],
    ...     forceplatform_keys=[".*"]
    ... )
    >>> 
    >>> # Analyze with kinetics
    >>> running = laban.RunningExercise(algorithm='kinetics', **data)
    >>> 
    >>> # Get cycles
    >>> print(f"Detected {len(running.cycles)} steps")
    >>> 
    >>> # Analyze each step
    >>> for step in running.cycles:
    ...     metrics = step.output_metrics
    ...     print(f"Flight: {step.flight_time_s:.3f} s, Contact: {step.contact_time_s:.3f} s")
    """
```

**Detection Algorithms:**

### Kinetics Algorithm

Uses vertical ground reaction force to detect foot contact:

1. **Threshold crossing**: Vertical GRF > `ground_reaction_force_threshold`
2. **Event detection**:
   - Footstrike: First frame above threshold
   - Toeoff: Last frame above threshold
   - Midstance: Frame with maximum vertical GRF
3. **Cycle extraction**: From toeoff to next toeoff (same foot)

### Kinematics Algorithm

Uses heel/toe marker vertical position:

1. **Height normalization**: Calculate % of max vertical position
2. **Contact detection**: Heel/toe height < `height_threshold`
3. **Event detection**:
   - Footstrike: First frame below threshold
   - Toeoff: Last frame below threshold
   - Midstance: Frame with minimum vertical velocity
4. **Cycle extraction**: From toeoff to next toeoff

**Example - Batch Analysis:**

```python
import labanalysis as laban
import pandas as pd

# Load data
data = laban.read_tdf("running.tdf", marker_keys=[".*"], forceplatform_keys=[".*"])
running = laban.RunningExercise(algorithm='kinetics', **data)

# Extract metrics for all steps
all_metrics = []
for i, step in enumerate(running.cycles):
    metrics = step.output_metrics
    metrics['step_number'] = i + 1
    all_metrics.append(metrics)

# Combine into DataFrame
results = pd.concat(all_metrics, ignore_index=True)

# Calculate summary statistics
summary = results.describe()
print(summary)

# Export
results.to_csv("running_analysis.csv", index=False)
```

---

## WalkingStride

Single walking stride (gait cycle during walking).

```python
class WalkingStride(GaitCycle):
    """
    Represents a single walking stride with swing and stance phases.
    
    Walking is characterized by stance phase (foot on ground) and swing phase
    (foot in air), with double support periods when both feet are on ground.
    
    Parameters
    ----------
    Inherits all parameters from GaitCycle
    
    Attributes
    ----------
    swing_phase : WholeBody
        Data during swing phase (toeoff to footstrike)
    stance_phase : WholeBody
        Data during stance phase (footstrike to next toeoff)
    swing_time_s : float
        Swing phase duration (seconds)
    stance_time_s : float
        Stance phase duration (seconds)
    opposite_footstrike_s : float
        Time of opposite foot's footstrike (seconds)
    double_support_time_s : float
        Double support duration (seconds)
    
    Notes
    -----
    Cycle timing pattern:
    init_s (toeoff) → SWING → footstrike_s → STANCE → end_s (next toeoff)
    
    Double support occurs when both feet are on ground:
    - Initial double support: opposite_footstrike_s to footstrike_s
    - Terminal double support: footstrike_s to opposite toeoff
    
    See Also
    --------
    RunningStep : Gait cycle for running
    WalkingExercise : Multi-cycle walking analysis
    """
```

**Example:**

```python
import labanalysis as laban

# Load walking trial
data = laban.read_tdf("walking.tdf", marker_keys=[".*"], forceplatform_keys=[".*"])
walking = laban.WalkingExercise(algorithm='kinetics', **data)

# Get first stride
stride = walking.cycles[0]

# Analyze phases
print(f"Swing time: {stride.swing_time_s:.3f} s ({stride.swing_time_s/stride.cycle_time_s*100:.1f}%)")
print(f"Stance time: {stride.stance_time_s:.3f} s ({stride.stance_time_s/stride.cycle_time_s*100:.1f}%)")
print(f"Double support: {stride.double_support_time_s:.3f} s")

# Extract phase data
swing_data = stride.swing_phase
stance_data = stride.stance_phase
```

---

## WalkingExercise

Multi-cycle walking analysis.

```python
class WalkingExercise(GaitExercise):
    """
    Complete walking exercise with automatic stride detection.
    
    Detects and extracts individual WalkingStride cycles from continuous
    walking data using either kinematic or kinetic algorithms.
    
    Parameters
    ----------
    algorithm : {'kinematics', 'kinetics'}
        Cycle detection algorithm
    ground_reaction_force_threshold : float, optional
        Minimum vertical GRF (N) for contact detection
        Default: 20 N
    height_threshold : float, optional
        Height threshold for kinematic detection (%)
        Default: 0.1 (10% of max height)
    **kwargs
        Data signals (markers, force platforms, EMG, etc.)
    
    Attributes
    ----------
    cycles : list of WalkingStride
        Detected walking strides
    
    Examples
    --------
    >>> import labanalysis as laban
    >>> 
    >>> # Load walking data
    >>> data = laban.read_tdf(
    ...     "walking.tdf",
    ...     marker_keys=[".*"],
    ...     forceplatform_keys=[".*"]
    ... )
    >>> 
    >>> # Analyze with kinematics
    >>> walking = laban.WalkingExercise(algorithm='kinematics', **data)
    >>> 
    >>> # Get cycles
    >>> print(f"Detected {len(walking.cycles)} strides")
    >>> 
    >>> # Analyze each stride
    >>> for stride in walking.cycles:
    ...     print(f"Cycle time: {stride.cycle_time_s:.3f} s")
    ...     print(f"Swing: {stride.swing_time_s:.3f} s, Stance: {stride.stance_time_s:.3f} s")
    """
```

**Example - Temporal-Spatial Parameters:**

```python
import labanalysis as laban
import pandas as pd
import numpy as np

# Load walking data
data = laban.read_tdf("walking.tdf", marker_keys=[".*"], forceplatform_keys=[".*"])
walking = laban.WalkingExercise(algorithm='kinetics', **data)

# Calculate temporal-spatial parameters
results = []
for stride in walking.cycles:
    # Temporal parameters
    cycle_time = stride.cycle_time_s
    stance_pct = (stride.stance_time_s / cycle_time) * 100
    swing_pct = (stride.swing_time_s / cycle_time) * 100
    
    # Spatial parameters (if pelvis marker available)
    if hasattr(stride, 'pelvis_center'):
        pelvis = stride.pelvis_center
        stride_length = pelvis['Y'].to_numpy()[-1] - pelvis['Y'].to_numpy()[0]
        velocity = stride_length / cycle_time
    else:
        stride_length = np.nan
        velocity = np.nan
    
    results.append({
        'cycle_time_s': cycle_time,
        'stance_pct': stance_pct,
        'swing_pct': swing_pct,
        'stride_length_m': stride_length,
        'velocity_m_s': velocity,
        'cadence_steps_min': 60 / cycle_time
    })

# Create DataFrame
df = pd.DataFrame(results)
print(df.describe())
```

---

## Comparison: Running vs. Walking

| Feature | RunningStep | WalkingStride |
|---------|-------------|---------------|
| **Flight phase** | Yes (characteristic) | No |
| **Double support** | No | Yes (characteristic) |
| **Phases** | Flight, Contact (Loading, Propulsion) | Swing, Stance |
| **Contact %** | ~30-40% of cycle | ~60% of cycle |
| **Swing %** | N/A | ~40% of cycle |
| **Cycle definition** | Toeoff to next toeoff (same foot) | Toeoff to next toeoff (same foot) |
| **Key events** | Toeoff, Footstrike, Midstance | Toeoff, Footstrike, Opposite Footstrike |

---

## Common Workflows

### 1. Basic Gait Analysis

```python
import labanalysis as laban

# Load data
data = laban.read_tdf("trial.tdf", marker_keys=[".*"], forceplatform_keys=[".*"])

# Analyze (auto-detects locomotion type)
exercise = laban.RunningExercise(algorithm='kinetics', **data)

# Get all cycles
cycles = exercise.cycles
print(f"Detected {len(cycles)} cycles")

# Extract metrics
for i, cycle in enumerate(cycles):
    metrics = cycle.output_metrics
    print(f"Cycle {i+1}: {metrics}")
```

### 2. Phase-Specific Analysis

```python
# Running: analyze flight vs contact
step = running.cycles[0]
flight = step.flight_phase
contact = step.contact_phase

# Calculate peak ankle angle during contact
ankle_angle = contact.left_ankle_flexionextension
peak_dorsiflexion = ankle_angle.to_numpy().max()

# Walking: analyze swing vs stance
stride = walking.cycles[0]
swing = stride.swing_phase
stance = stride.stance_phase

# Calculate peak hip flexion during swing
hip_angle = swing.left_hip_flexionextension
peak_hip_flexion = hip_angle.to_numpy().max()
```

### 3. Algorithm Comparison

```python
# Analyze same data with both algorithms
data = laban.read_tdf("trial.tdf", marker_keys=[".*"], forceplatform_keys=[".*"])

kinetic_exercise = laban.RunningExercise(algorithm='kinetics', **data)
kinematic_exercise = laban.RunningExercise(algorithm='kinematics', **data)

print(f"Kinetics: {len(kinetic_exercise.cycles)} cycles")
print(f"Kinematics: {len(kinematic_exercise.cycles)} cycles")

# Compare cycle times
kinetic_times = [c.cycle_time_s for c in kinetic_exercise.cycles]
kinematic_times = [c.cycle_time_s for c in kinematic_exercise.cycles]
```

---

## Troubleshooting

### Issue: "No cycles detected"

**Cause**: Threshold too strict or incomplete data

**Solution**:
```python
# Adjust thresholds
running = laban.RunningExercise(
    algorithm='kinetics',
    ground_reaction_force_threshold=10,  # Lower threshold (default: 20 N)
    **data
)

# Or try alternative algorithm
running = laban.RunningExercise(algorithm='kinematics', **data)
```

### Issue: "Algorithm fallback warning"

**Cause**: Requested algorithm not applicable with available data

**Solution**: This is automatic. Verify you have the required data:
- Kinetics: Requires force platform data (`left_foot_ground_reaction_force`, etc.)
- Kinematics: Requires heel/toe markers (`left_heel`, `left_toe`, etc.)

### Issue: "Inconsistent cycle count between left/right"

**Cause**: Asymmetric gait or data quality issues

**Solution**:
```python
# Filter cycles by side
left_cycles = [c for c in exercise.cycles if c.side == 'left']
right_cycles = [c for c in exercise.cycles if c.side == 'right']

print(f"Left: {len(left_cycles)}, Right: {len(right_cycles)}")
```

---

## See Also

- [WholeBody](bodies.md) - Full body biomechanical model
- [ForcePlatform](records.md#forceplatform) - Force platform data structure
- [Signal Processing](../signalprocessing.md) - Filtering and analysis functions
- [Gait Analysis Tutorial](../../tutorials/02-gait-analysis.md) - Complete workflow
- [Running Test Protocol](../../guides/test-protocols/gait.md) - Running analysis guide
- [Walking Test Protocol](../../guides/test-protocols/gait.md) - Walking analysis guide

---

**Module refactored**: 2026-06-17 (from monolithic 3052-line file to 6 modular files)  
**Backward compatibility**: 100% maintained via `__init__.py` re-exports
