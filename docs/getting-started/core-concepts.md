# Core Concepts

Understanding the fundamental building blocks of labanalysis will help you use the library effectively. This guide introduces the three core patterns: **Signals**, **Records**, and **Protocols**.

## Architecture Overview

labanalysis is built on three main concepts:

```
Signal (time-series data) → Record (container) → Protocol (analysis)
```

1. **Signals** - Individual time-series measurements (force, position, EMG)
2. **Records** - Containers grouping related signals
3. **Protocols** - Standardized test procedures that produce results

## 1. Signals: Time-Series Data

Signals represent time-indexed data with physical units.

### Signal Types

#### Signal1D - Single-Column Data

For scalar measurements over time (force, voltage, heart rate):

```python
import labanalysis as laban
import numpy as np

# Create a 1D signal
time = np.linspace(0, 10, 1000)
force_z = 500 + 100 * np.sin(time)

signal = laban.Signal1D(
    data=force_z,
    index=time,
    label='Fz',
    unit='N'  # Newtons
)

print(f"Signal: {signal.label}")
print(f"Unit: {signal.unit}")
print(f"Sampling frequency: {signal.sampling_frequency:.1f} Hz")
# Output: Signal: Fz
# Output: Unit: newton
# Output: Sampling frequency: 100.0 Hz
```

#### Signal3D - Three-Component Data

For vector quantities (3D force, acceleration, position):

```python
# Create a 3D signal (e.g., acceleration)
acc_data = np.random.randn(1000, 3)  # X, Y, Z components

signal_3d = laban.Signal3D(
    data=acc_data,
    index=time,
    labels=['AccX', 'AccY', 'AccZ'],
    units=['m/s^2', 'm/s^2', 'm/s^2']
)

print(f"Shape: {signal_3d.data.shape}")
# Output: Shape: (1000, 3)
```

#### Specialized Signals

- **EMGSignal** - Electromyography data (extends Signal1D, auto-converts to µV)
- **Point3D** - 3D marker positions (extends Signal3D, auto-converts to meters)

```python
# EMG signal (automatically handles microvolts)
emg = laban.EMGSignal(
    data=np.random.randn(1000) * 100,
    index=time,
    label='Biceps'
)

# 3D marker position
marker = laban.Point3D(
    data=np.random.randn(1000, 3),
    index=time,
    labels=['X', 'Y', 'Z']
)
```

### Key Signal Operations

```python
# Indexing (like pandas)
signal_slice = signal[0.5:2.0]  # Time-based slicing

# Arithmetic
doubled = signal * 2
sum_signal = signal + 100

# Unit-aware operations (uses Pint)
print(signal.unit)  # newton

# Apply functions
filtered = signal.apply(lambda x: laban.butterworth_filt(x, freq=100, cut=10))
```

## 2. Records: Signal Containers

Records group related signals together.

### Record - Basic Container

Dictionary-like container for signals:

```python
# Create individual signals
force_x = laban.Signal1D(data=np.random.randn(100), index=time[:100], label='Fx', unit='N')
force_y = laban.Signal1D(data=np.random.randn(100), index=time[:100], label='Fy', unit='N')
force_z = laban.Signal1D(data=np.random.randn(100), index=time[:100], label='Fz', unit='N')

# Create a Record
record = laban.Record(
    Fx=force_x,
    Fy=force_y,
    Fz=force_z
)

# Access signals
print(record.Fx.label)  # 'Fx'
print(record['Fy'].unit)  # newton

# Convert to DataFrame
df = record.to_dataframe()
print(df.columns)  # ['Fx', 'Fy', 'Fz']
```

### TimeseriesRecord - Extended Container

Supports ForcePlatform and MetabolicRecord:

```python
# Load from file
record = laban.TimeseriesRecord.from_tdf("data.tdf")

# Filter by type
force_platforms = record.filter(laban.ForcePlatform)
signals_1d = record.filter(laban.Signal1D)
```

### Specialized Records

#### ForcePlatform - Ground Reaction Forces

```python
fp = laban.ForcePlatform(
    origin=np.array([0.0, 0.0, 0.0]),
    force={
        'Fx': force_x,
        'Fy': force_y,
        'Fz': force_z
    },
    torque={
        'Mx': moment_x,
        'My': moment_y,
        'Mz': moment_z
    }
)

print(f"Force platform at: {fp.origin}")
# Access components
vertical_force = fp.force['Fz']
```

#### MetabolicRecord - Metabolic Data

```python
metabolic = laban.MetabolicRecord(
    VO2=vo2_signal,
    VCO2=vco2_signal,
    VE=ve_signal,
    HR=hr_signal
)

# Computed properties
print(f"RQ: {metabolic.RQ}")  # Respiratory quotient
print(f"Fat oxidation: {metabolic.fat_oxidation}")
```

### Record Operations

```python
# Apply processing to all signals
filtered_record = record.apply(
    lambda sig: laban.butterworth_filt(sig.data, freq=1000, cut=10)
)

# Handle missing data
filled_record = record.fillna(method='spline')

# Reset time to start at 0
normalized_record = record.reset_time()
```

## 3. Protocols: Standardized Tests

Protocols combine participant information with test procedures to produce standardized results.

### Participant - Test Subject Information

```python
participant = laban.Participant(
    name="John",
    surname="Doe",
    gender="M",
    height=1.80,  # meters
    weight=75,    # kg
    age=25
)

# Computed properties
print(f"BMI: {participant.bmi:.1f}")  # Body mass index
print(f"Max HR: {participant.max_heart_rate:.0f} bpm")  # Age-predicted max HR
```

### TestProtocol - Standardized Analysis

Test protocols inherit from `TestProtocol` and implement standardized analysis:

```python
# Example: Jump Test Protocol
from labanalysis.protocols import JumpTest

test = JumpTest.from_tdf(
    "jump_data.tdf",
    participant=participant
)

# Access results
results = test.results
print(f"Jump height: {results['jump_height']:.2f} cm")
print(f"Peak power: {results['peak_power']:.0f} W")

# Generate report
report = test.report()

# Visualize
fig = test.plot()
fig.show()
```

### Available Protocol Types

- **JumpTest** - CMJ, SJ, DJ analysis
- **RunningTest**, **WalkingTest** - Gait analysis
- **UprightBalanceTest**, **PlankBalanceTest** - Balance assessment
- **Isokinetic1RMTest**, **IsometricTest** - Strength testing
- **ShuttleTest** - Agility assessment
- **SubmaximalVO2MaxTest** - Aerobic capacity

## Putting It All Together

Typical workflow combining all concepts:

```python
import labanalysis as laban

# 1. Create participant
participant = laban.Participant(
    name="Jane",
    surname="Smith",
    height=1.65,
    weight=60,
    age=28
)

# 2. Load data (creates TimeseriesRecord with Signals)
record = laban.TimeseriesRecord.from_tdf("test_data.tdf")

# 3. Process signals
filtered_record = record.apply(
    lambda sig: laban.butterworth_filt(sig.data, freq=1000, cut=10)
)

# 4. Extract specific signal for analysis
force_signal = filtered_record['FP1'].force['Fz']

# 5. Find events
peaks = laban.find_peaks(force_signal.data, height=500, distance=100)

# 6. Or use a protocol for standardized analysis
from labanalysis.protocols import JumpTest

test = JumpTest.from_tdf("jump_data.tdf", participant=participant)
results = test.results
print(f"Standardized results: {results}")
```

## Design Principles

Understanding the design helps you use the library effectively:

### 1. Unit-Aware Computing

All signals carry physical units (using Pint):

```python
signal = laban.Signal1D(data=[1, 2, 3], index=[0, 1, 2], label='F', unit='N')
print(signal.unit)  # newton

# Unit conversions are automatic
```

### 2. Immutability Options

Most operations support `inplace` parameter:

```python
# Create new object (default)
filtered = signal.fillna()

# Modify in place
signal.fillna(inplace=True)
```

### 3. Method Chaining

Operations can be chained:

```python
processed = signal.strip().reset_time().fillna()
```

### 4. Consistent API

Similar operations work the same across different classes:

```python
# All have from_tdf()
record = laban.TimeseriesRecord.from_tdf("file.tdf")
signal = laban.Signal1D.from_tdf("file.tdf", column="Fz")

# All have to_dataframe()
df1 = record.to_dataframe()
df2 = signal.to_dataframe()
```

## Next Steps

Now that you understand the core concepts:

1. **[Your First Analysis](first-analysis.md)** - Complete walkthrough with real data
2. **[Signal Processing Guide](../user-guide/signal-processing/README.md)** - Deep dive into signal operations
3. **[Test Protocols Guide](../user-guide/test-protocols/README.md)** - Learn all available protocols

## Summary

- **Signals** (`Signal1D`, `Signal3D`, `EMGSignal`, `Point3D`) - Individual time-series with units
- **Records** (`Record`, `TimeseriesRecord`, `ForcePlatform`) - Group related signals
- **Protocols** (`Participant`, `TestProtocol`) - Standardized test analysis
- All support `.from_tdf()` loading and `.to_dataframe()` conversion
- Operations are unit-aware and support both immutable and in-place modes

---

**Questions?** Check the [API Reference](../api-reference/README.md) or contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
