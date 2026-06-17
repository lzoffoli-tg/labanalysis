# labanalysis.records.records

High-level record classes for organizing related signals.

**Source**: `src/labanalysis/records/records.py`

## Overview

The `records` module provides container classes for organizing related timeseries data:

- **Record**: Base container for any collection of signals
- **TimeseriesRecord**: Container for time-synchronized signals
- **ForcePlatform**: Force platform with forces, moments, and COP
- **MetabolicRecord**: Metabolic measurements (VO2, VCO2, HR, etc.)

These classes simplify data management by grouping related signals together.

## Classes

### Record

Base dictionary-like container for signals.

```python
class Record(dict):
    """
    Base container for storing related signals.
    
    Behaves like a dictionary with additional convenience methods.
    
    Examples
    --------
    >>> import labanalysis as laban
    >>> 
    >>> # Create record
    >>> record = laban.Record()
    >>> 
    >>> # Add signals
    >>> record['force'] = laban.Signal1D(...)
    >>> record['emg'] = laban.EMGSignal(...)
    >>> 
    >>> # Access signals
    >>> force = record['force']
    >>> 
    >>> # Iterate
    >>> for name, signal in record.items():
    ...     print(f"{name}: {len(signal.data)} samples")
    """
```

**Methods:**
- All standard dictionary methods (`keys()`, `values()`, `items()`, etc.)
- `copy()`: Create shallow copy
- `update()`: Update with another record

### TimeseriesRecord

Container for time-synchronized timeseries data.

```python
class TimeseriesRecord(Record):
    """
    Container for time-synchronized signals.
    
    Automatically groups signals by type (force platforms, markers, EMG).
    
    Attributes
    ----------
    forceplatforms : Record
        All force platform objects
    markers : Record
        All Point3D marker objects  
    emgsignals : Record
        All EMG signal objects
    metabolic : MetabolicRecord or None
        Metabolic data if present
    
    Examples
    --------
    >>> # Load from TDF file
    >>> record = laban.TimeseriesRecord.from_tdf("trial.tdf")
    >>> 
    >>> # Access by group
    >>> fp1 = record.forceplatforms['FP1']
    >>> c7 = record.markers['C7']
    >>> biceps = record.emgsignals['biceps_R']
    >>> 
    >>> # Or access directly
    >>> fp1 = record['FP1']
    """
```

#### Class Methods

##### from_tdf()

Load from BTS TDF file with automatic signal grouping.

```python
@classmethod
def from_tdf(cls, file_path: str) -> TimeseriesRecord
```

**Parameters:**
- `file_path` (str): Path to TDF file

**Returns:**
- `TimeseriesRecord`: Loaded record with grouped signals

**Example:**
```python
# Load complete file
record = laban.TimeseriesRecord.from_tdf("trial.tdf")

# Check what was loaded
print(f"Force platforms: {list(record.forceplatforms.keys())}")
print(f"Markers: {list(record.markers.keys())}")
print(f"EMG channels: {list(record.emgsignals.keys())}")
```

**Output:**
```
Force platforms: ['FP1', 'FP2']
Markers: ['C7', 'LASI', 'RASI', 'LPSI', 'RPSI', ...]
EMG channels: ['biceps_R', 'biceps_L', 'vastus_R', ...]
```

#### Properties

##### forceplatforms

Access all force platforms.

```python
@property
def forceplatforms(self) -> Record
```

**Returns:**
- `Record`: Dictionary of ForcePlatform objects

**Example:**
```python
# Get all platforms
platforms = record.forceplatforms

# Iterate
for name, fp in platforms.items():
    print(f"{name}: max Fz = {fp.force['Fz'].data.max():.1f} N")
```

##### markers

Access all markers.

```python
@property
def markers(self) -> Record
```

**Returns:**
- `Record`: Dictionary of Point3D objects

##### emgsignals

Access all EMG signals.

```python
@property
def emgsignals(self) -> Record
```

**Returns:**
- `Record`: Dictionary of EMGSignal objects

##### vertical_axis

Automatically detect vertical axis (usually 'Fz' or 'Fy').

```python
@property
def vertical_axis(self) -> str
```

**Returns:**
- `str`: Name of vertical axis ('Fz' or 'Fy')

##### resultant_force

Calculate resultant force from all platforms.

```python
@property
def resultant_force(self) -> ForcePlatform
```

**Returns:**
- `ForcePlatform`: Sum of all force platforms

**Example:**
```python
# Get total force from dual platforms
total_force = record.resultant_force
fz_total = total_force.force['Fz']

print(f"Total vertical force: {fz_total.data.max():.1f} N")
```

#### Instance Methods

##### strip()

Remove leading/trailing NaN from all signals.

```python
def strip(self, axis: int = 0, inplace: bool = False) -> TimeseriesRecord
```

**Parameters:**
- `axis` (int): Axis to strip (0=time)
- `inplace` (bool): Modify in place

**Returns:**
- `TimeseriesRecord`: Stripped record

**Example:**
```python
# Remove quiet periods at start/end
record_trimmed = record.strip(axis=0)
```

### ForcePlatform

Force platform with forces, moments, and COP.

```python
class ForcePlatform:
    """
    Force platform data container.
    
    Contains forces, moments, and automatically calculates COP.
    
    Parameters
    ----------
    force : Signal3D
        3D force signal [Fx, Fy, Fz]
    torque : Signal3D
        3D moment signal [Mx, My, Mz]
    platform_dimensions : tuple, optional
        (length, width) in meters for COP calculation
    
    Attributes
    ----------
    force : Signal3D
        Force signals
    torque : Signal3D
        Moment signals
    cop : Signal3D
        Center of pressure [COPx, COPy, COPz]
    sampling_frequency : float
        Sampling frequency in Hz
    
    Examples
    --------
    >>> # Load force platform
    >>> record = laban.TimeseriesRecord.from_tdf("trial.tdf")
    >>> fp = record['FP1']
    >>> 
    >>> # Access components
    >>> fz = fp.force['Fz']
    >>> mx = fp.torque['Mx']
    >>> cop_x = fp.cop['COPx']
    >>> 
    >>> # Calculate metrics
    >>> peak_force = fz.data.max()
    >>> mean_cop_x = cop_x.data.mean()
    """
```

#### Properties

##### force

3D force signal.

```python
@property
def force(self) -> Signal3D
```

**Component access:**
```python
fx = fp.force['Fx']  # Anteroposterior
fy = fp.force['Fy']  # Mediolateral (or vertical)
fz = fp.force['Fz']  # Vertical (or mediolateral)
```

##### torque

3D moment signal.

```python
@property
def torque(self) -> Signal3D
```

**Component access:**
```python
mx = fp.torque['Mx']
my = fp.torque['My']
mz = fp.torque['Mz']
```

##### cop

Center of pressure (automatically calculated).

```python
@property
def cop(self) -> Signal3D
```

**Calculation:**
```
COPx = -My / Fz
COPy = Mx / Fz
COPz = 0  (on platform surface)
```

**Example:**
```python
# Get COP sway
cop = fp.cop
cop_x = cop['COPx'].data
cop_y = cop['COPy'].data

# Calculate sway metrics
sway_area = np.pi * np.std(cop_x) * np.std(cop_y)
sway_velocity = np.sqrt(np.diff(cop_x)**2 + np.diff(cop_y)**2).sum() / duration

print(f"Sway area: {sway_area:.2f} mm²")
print(f"Sway velocity: {sway_velocity:.2f} mm/s")
```

#### Class Methods

##### from_tdf()

Load force platform from TDF file.

```python
@classmethod
def from_tdf(
    cls,
    file_path: str,
    force_x: str = 'Fx',
    force_y: str = 'Fy',
    force_z: str = 'Fz',
    torque_x: str = 'Mx',
    torque_y: str = 'My',
    torque_z: str = 'Mz'
) -> ForcePlatform
```

**Parameters:**
- `file_path` (str): Path to TDF file
- `force_x/y/z` (str): Column names for forces
- `torque_x/y/z` (str): Column names for moments

**Returns:**
- `ForcePlatform`: Loaded force platform

**Example:**
```python
# Load specific platform
fp1 = laban.ForcePlatform.from_tdf(
    "trial.tdf",
    force_x='FP1_Fx',
    force_y='FP1_Fy',
    force_z='FP1_Fz',
    torque_x='FP1_Mx',
    torque_y='FP1_My',
    torque_z='FP1_Mz'
)
```

### MetabolicRecord

Metabolic measurements container.

```python
class MetabolicRecord(Record):
    """
    Container for metabolic measurements.
    
    Stores VO2, VCO2, heart rate, and derived metrics.
    
    Attributes
    ----------
    vo2 : Signal1D
        Oxygen consumption (ml/min or ml/kg/min)
    vco2 : Signal1D
        Carbon dioxide production (ml/min)
    heart_rate : Signal1D
        Heart rate (bpm)
    respiratory_rate : Signal1D, optional
        Respiratory rate (breaths/min)
    rer : Signal1D
        Respiratory exchange ratio (VCO2/VO2)
    
    Examples
    --------
    >>> # Load metabolic data
    >>> metabolic = laban.MetabolicRecord.from_cosmed("vo2max.txt")
    >>> 
    >>> # Access signals
    >>> vo2 = metabolic.vo2
    >>> hr = metabolic.heart_rate
    >>> 
    >>> # Calculate metrics
    >>> vo2_max = vo2.data.max()
    >>> hr_max = hr.data.max()
    >>> 
    >>> print(f"VO2max: {vo2_max:.1f} ml/kg/min")
    >>> print(f"HRmax: {hr_max:.0f} bpm")
    """
```

#### Properties

##### vo2

Oxygen consumption signal.

```python
@property
def vo2(self) -> Signal1D
```

##### vco2

Carbon dioxide production signal.

```python
@property
def vco2(self) -> Signal1D
```

##### heart_rate

Heart rate signal.

```python
@property
def heart_rate(self) -> Signal1D
```

##### rer

Respiratory exchange ratio (VCO2/VO2).

```python
@property
def rer(self) -> Signal1D
```

**Automatically calculated from VO2 and VCO2.**

#### Class Methods

##### from_cosmed()

Load from Cosmed device file.

```python
@classmethod
def from_cosmed(cls, file_path: str) -> MetabolicRecord
```

**Example:**
```python
metabolic = laban.MetabolicRecord.from_cosmed("test.txt")

# Find VO2max
vo2_max = metabolic.vo2.data.max()
time_at_max = metabolic.vo2.index[np.argmax(metabolic.vo2.data)]

print(f"VO2max: {vo2_max:.1f} ml/kg/min at {time_at_max:.1f} s")
```

## Usage Examples

### Loading Complete Trial

```python
import labanalysis as laban

# Load TDF file
record = laban.TimeseriesRecord.from_tdf("trial.tdf")

# Check contents
print("=== Trial Contents ===")
print(f"Force Platforms: {len(record.forceplatforms)}")
print(f"Markers: {len(record.markers)}")
print(f"EMG Channels: {len(record.emgsignals)}")

# Access specific devices
fp1 = record['FP1']
c7 = record['C7']
biceps = record['biceps_R']

# Get sampling frequencies
print(f"\nFP1 frequency: {fp1.sampling_frequency} Hz")
print(f"C7 frequency: {c7.sampling_frequency} Hz")
print(f"EMG frequency: {biceps.sampling_frequency} Hz")
```

### Working with Force Platforms

```python
# Get vertical force
fz = fp1.force['Fz']

# Filter
fz_filt = laban.butterworth_filt(
    fz.data,
    freq=fz.sampling_frequency,
    cut=10,
    order=4
)

# Calculate metrics
peak_force = fz_filt.max()
mean_force = fz_filt.mean()
impulse = np.trapz(fz_filt - mean_force) / fz.sampling_frequency

print(f"Peak force: {peak_force:.1f} N")
print(f"Impulse: {impulse:.1f} N·s")
```

### Dual Platform Analysis

```python
# Load record with two platforms
record = laban.TimeseriesRecord.from_tdf("gait.tdf")

# Get individual platforms
fp1 = record['FP1']
fp2 = record['FP2']

# Get resultant (sum of both)
total = record.resultant_force

# Compare
fz1 = fp1.force['Fz'].data
fz2 = fp2.force['Fz'].data
fz_total = total.force['Fz'].data

print(f"FP1 max: {fz1.max():.1f} N")
print(f"FP2 max: {fz2.max():.1f} N")
print(f"Total max: {fz_total.max():.1f} N")
print(f"Sum check: {(fz1 + fz2).max():.1f} N")  # Should equal total
```

### COP Analysis

```python
# Get COP from force platform
cop = fp1.cop

# Extract components
cop_x = cop['COPx'].data  # Anteroposterior
cop_y = cop['COPy'].data  # Mediolateral

# Calculate sway metrics
import numpy as np

# Sway area (95% confidence ellipse)
sway_area_95 = np.pi * 2.447 * np.std(cop_x) * np.std(cop_y)

# Sway path length
dx = np.diff(cop_x)
dy = np.diff(cop_y)
sway_path = np.sqrt(dx**2 + dy**2).sum()

# Sway velocity
duration = len(cop_x) / fp1.sampling_frequency
sway_velocity = sway_path / duration

print(f"Sway area (95%): {sway_area_95:.2f} mm²")
print(f"Sway path: {sway_path:.1f} mm")
print(f"Sway velocity: {sway_velocity:.2f} mm/s")
```

### Stripping Quiet Periods

```python
# Load trial with quiet periods at start/end
record = laban.TimeseriesRecord.from_tdf("trial.tdf")

# Check original length
fp = record['FP1']
original_length = len(fp.force['Fz'].data)
print(f"Original length: {original_length} samples")

# Strip NaN and zero-force periods
record_trimmed = record.strip(axis=0)

# Check new length
fp_trimmed = record_trimmed['FP1']
new_length = len(fp_trimmed.force['Fz'].data)
print(f"Trimmed length: {new_length} samples")
print(f"Removed: {original_length - new_length} samples")
```

### Export to Different Formats

```python
# Get force platform
fp = record['FP1']

# Export to DataFrame
df = fp.force.to_dataframe()
df.to_excel("forces.xlsx", index=True)

# Export specific signal
fz = fp.force['Fz']
df_fz = fz.to_dataframe()
df_fz.to_csv("fz_only.csv")

# Export as NumPy arrays
import numpy as np
np.save("fz_data.npy", fz.data)
np.save("time.npy", fz.index)
```

## See Also

- **[Timeseries](timeseries.md)** - Low-level signal classes
- **[Bodies](bodies.md)** - WholeBody model
- **[User Guide: Data Loading](../../user-guide/data-loading/README.md)** - Loading records
- **[User Guide: Force Platforms](../../user-guide/biomechanics/force-platforms.md)** - Force platform analysis

---

**Module**: `src/labanalysis/records/records.py`
