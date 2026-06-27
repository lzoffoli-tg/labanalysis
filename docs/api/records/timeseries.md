# labanalysis.records.timeseries

Time series data structures for representing temporal signals.

**Source**: `src/labanalysis/records/timeseries.py`

## Overview

The `timeseries` module provides the fundamental data structures for handling time-indexed data with physical units:

- **Timeseries**: Base class for multi-column time-indexed data
- **Signal1D**: Single-channel signal (e.g., force component, joint angle)
- **Signal3D**: Three-channel signal (e.g., 3D force, 3D position)
- **EMGSignal**: Specialized signal for electromyography
- **Point3D**: 3D spatial coordinates

All classes support:
- Automatic unit handling via Pint
- Missing data interpolation
- Time axis manipulation
- Pandas DataFrame export

## Classes

### Timeseries

Base class for time-indexed multi-column data with unit support.

```python
class Timeseries:
    """
    Time-indexed multi-column data container with unit support.
    
    Parameters
    ----------
    data : ndarray
        Data array with shape (n_samples, n_columns)
    index : ndarray, optional
        Time index array with shape (n_samples,). If None, uses sample indices.
    labels : list of str, optional
        Column labels. If None, uses ['col_0', 'col_1', ...]
    units : list of str or Quantity, optional
        Physical units for each column. Can be Pint quantities or strings.
    sampling_frequency : float, optional
        Sampling frequency in Hz. Auto-calculated from index if not provided.
    
    Attributes
    ----------
    data : ndarray
        Data array (n_samples, n_columns)
    index : ndarray
        Time index (n_samples,)
    labels : list
        Column labels
    units : list
        Physical units (Pint Quantity objects)
    sampling_frequency : float
        Sampling rate in Hz
    
    Examples
    --------
    >>> import numpy as np
    >>> import labanalysis as laban
    >>> 
    >>> # Create 3-column timeseries
    >>> data = np.random.randn(1000, 3)
    >>> time = np.linspace(0, 10, 1000)
    >>> ts = laban.Timeseries(
    ...     data=data,
    ...     index=time,
    ...     labels=['x', 'y', 'z'],
    ...     units=['m/s', 'm/s', 'm/s']
    ... )
    >>> 
    >>> print(f"Shape: {ts.data.shape}")
    Shape: (1000, 3)
    >>> print(f"Sampling frequency: {ts.sampling_frequency} Hz")
    Sampling frequency: 100.0 Hz
    """
```

#### Class Methods

##### from_tdf()

Load from BTS TDF file.

```python
@classmethod
def from_tdf(
    cls,
    file_path: str,
    columns: list = None
) -> Timeseries
```

**Parameters:**
- `file_path` (str): Path to TDF file
- `columns` (list, optional): Column names to load. If None, loads all.

**Returns:**
- `Timeseries`: Loaded data

**Example:**
```python
ts = laban.Timeseries.from_tdf("data.tdf", columns=['FP1_Fz', 'FP1_Mx'])
```

#### Instance Methods

##### to_dataframe()

Convert to pandas DataFrame.

```python
def to_dataframe(self) -> pd.DataFrame
```

**Returns:**
- `pd.DataFrame`: DataFrame with time index and labeled columns

**Example:**
```python
df = ts.to_dataframe()
df.to_excel("output.xlsx")
```

##### apply()

Apply function to data.

```python
def apply(
    self,
    func: callable,
    axis: int = 0,
    inplace: bool = False
) -> Timeseries
```

**Parameters:**
- `func` (callable): Function to apply
- `axis` (int): Axis along which to apply (0=rows, 1=columns)
- `inplace` (bool): Modify in place or return copy

**Returns:**
- `Timeseries`: Modified timeseries (or None if inplace)

**Example:**
```python
# Apply absolute value
ts_abs = ts.apply(np.abs)

# Apply custom function
ts_scaled = ts.apply(lambda x: x * 2)
```

##### fillna()

Fill missing values (NaN).

```python
def fillna(
    self,
    value: float = None,
    method: str = 'linear',
    regressors: np.ndarray = None,
    inplace: bool = False
) -> Timeseries
```

**Parameters:**
- `value` (float): Fill value (if method is None)
- `method` (str): Interpolation method - 'linear', 'cubic', 'pchip', or 'regression'
- `regressors` (ndarray): For regression method, predictor columns
- `inplace` (bool): Modify in place

**Returns:**
- `Timeseries`: Filled timeseries

**Example:**
```python
# Linear interpolation
ts_filled = ts.fillna(method='linear')

# Cubic spline
ts_filled = ts.fillna(method='cubic')
```

##### strip()

Remove leading/trailing NaN rows.

```python
def strip(self, inplace: bool = False) -> Timeseries
```

##### reset_time()

Reset time index to start at 0.

```python
def reset_time(self, inplace: bool = False) -> Timeseries
```

### Signal1D

Single-channel signal with enhanced functionality.

```python
class Signal1D(Timeseries):
    """
    Single-channel time series signal.
    
    Specialized Timeseries for 1D signals like force components,
    joint angles, EMG channels, etc.
    
    Parameters
    ----------
    data : ndarray
        1D data array (n_samples,)
    sampling_frequency : float
        Sampling frequency in Hz
    label : str, optional
        Signal label
    unit : str or Quantity, optional
        Physical unit
    
    Examples
    --------
    >>> # Create force signal
    >>> fz = laban.Signal1D(
    ...     data=np.random.randn(1000),
    ...     sampling_frequency=1000,
    ...     label='Fz',
    ...     unit='N'
    ... )
    >>> 
    >>> # Access properties
    >>> print(f"Mean: {fz.data.mean():.2f} {fz.unit}")
    >>> print(f"Duration: {len(fz) / fz.sampling_frequency:.2f} s")
    """
```

**Additional Properties:**
- `label` (str): Signal label
- `unit` (Quantity): Physical unit

**Example:**
```python
# Load single column from TDF
fz = laban.Signal1D.from_tdf("trial.tdf", column="FP1_Fz")

# Filter
fz_filt = laban.butterworth_filt(fz.data, freq=fz.sampling_frequency, cut=10)

# Update data
fz_filtered = fz.copy()
fz_filtered.data = fz_filt
```

### Signal3D

Three-channel signal (e.g., 3D force, velocity, position).

```python
class Signal3D(Timeseries):
    """
    Three-channel time series signal.
    
    Specialized Timeseries for 3D signals like forces, moments,
    velocities, etc.
    
    Parameters
    ----------
    data : ndarray
        3D data array with shape (n_samples, 3)
    sampling_frequency : float
        Sampling frequency in Hz
    labels : list of str, optional
        Labels for [x, y, z] components
    units : list of str or Quantity, optional
        Physical units for each component
    
    Examples
    --------
    >>> # Create 3D force signal
    >>> force = laban.Signal3D(
    ...     data=np.random.randn(1000, 3),
    ...     sampling_frequency=1000,
    ...     labels=['Fx', 'Fy', 'Fz'],
    ...     units=['N', 'N', 'N']
    ... )
    >>> 
    >>> # Access components
    >>> fx = force['Fx']  # Returns Signal1D
    >>> fz = force['Fz']
    >>> 
    >>> # Calculate magnitude
    >>> magnitude = force.magnitude()
    """
```

#### Additional Methods

##### Component Access

```python
# Access by label
fx = force['Fx']  # Returns Signal1D

# Access by index
fx = force[0]     # Returns Signal1D
```

##### magnitude()

Calculate resultant magnitude.

```python
def magnitude(self) -> Signal1D
```

**Returns:**
- `Signal1D`: Magnitude signal sqrt(x² + y² + z²)

**Example:**
```python
force_magnitude = force.magnitude()
print(f"Peak force: {force_magnitude.data.max():.1f} N")
```

### EMGSignal

Specialized signal for electromyography data.

```python
class EMGSignal(Signal1D):
    """
    Electromyography (EMG) signal.
    
    Extends Signal1D with muscle-specific metadata.
    
    Parameters
    ----------
    data : ndarray
        EMG data (n_samples,)
    sampling_frequency : float
        Sampling frequency in Hz (typically 1000-2000 Hz)
    muscle_name : str
        Muscle name (e.g., 'biceps femoris', 'vastus lateralis')
    side : str
        'left' or 'right'
    unit : str, optional
        Unit (default: 'mV')
    
    Examples
    --------
    >>> emg = laban.EMGSignal(
    ...     data=np.random.randn(2000),
    ...     sampling_frequency=2000,
    ...     muscle_name='biceps femoris',
    ...     side='right',
    ...     unit='mV'
    ... )
    >>> 
    >>> print(f"Muscle: {emg.muscle_name} ({emg.side})")
    >>> print(f"RMS: {np.sqrt(np.mean(emg.data**2)):.3f} {emg.unit}")
    """
```

**Additional Properties:**
- `muscle_name` (str): Muscle name
- `side` (str): 'left' or 'right'

**Typical Processing:**
```python
# Load EMG
emg = laban.EMGSignal.from_tdf("trial.tdf", column="EMG_biceps_R")

# Standard EMG processing
# 1. Band-pass filter (20-450 Hz)
emg_bp = laban.butterworth_filt(
    emg.data,
    freq=emg.sampling_frequency,
    cut=(20, 450),
    filt_type='band'
)

# 2. Full-wave rectification
emg_rect = np.abs(emg_bp)

# 3. Linear envelope (low-pass at 3 Hz)
emg_env = laban.butterworth_filt(
    emg_rect,
    freq=emg.sampling_frequency,
    cut=3,
    filt_type='low'
)

# Create processed signal
emg_processed = emg.copy()
emg_processed.data = emg_env
```

### Point3D

3D spatial coordinates (marker position).

```python
class Point3D(Signal3D):
    """
    3D spatial coordinates.
    
    Extends Signal3D for 3D positions (e.g., motion capture markers).
    
    Parameters
    ----------
    data : ndarray
        3D position data with shape (n_samples, 3) - [x, y, z]
    sampling_frequency : float
        Sampling frequency in Hz
    labels : list of str, optional
        Coordinate labels (default: ['x', 'y', 'z'])
    units : list of str, optional
        Units for each coordinate (default: ['mm', 'mm', 'mm'])
    
    Examples
    --------
    >>> # Create marker position
    >>> marker = laban.Point3D(
    ...     data=np.random.randn(100, 3) + [0, 1000, 0],
    ...     sampling_frequency=100,
    ...     labels=['x', 'y', 'z'],
    ...     units=['mm', 'mm', 'mm']
    ... )
    >>> 
    >>> # Calculate displacement
    >>> displacement = marker.data - marker.data[0, :]
    >>> 
    >>> # Calculate distance traveled
    >>> distances = np.sqrt(np.sum(np.diff(marker.data, axis=0)**2, axis=1))
    >>> total_distance = distances.sum()
    """
```

**Common Operations:**

```python
# Load marker
c7 = body.c7_vertebra  # Returns Point3D

# Get vertical position
y_position = c7.data[:, 1]  # Y-axis (mm)

# Calculate velocity
y_velocity = laban.winter_derivative1(y_position / 1000, freq=c7.sampling_frequency)

# Calculate resultant displacement
start_pos = c7.data[0, :]
end_pos = c7.data[-1, :]
displacement = np.sqrt(np.sum((end_pos - start_pos)**2))
print(f"Total displacement: {displacement:.1f} mm")
```

## Usage Examples

### Loading Data

```python
import labanalysis as laban

# Load complete record
record = laban.TimeseriesRecord.from_tdf("trial.tdf")

# Access force platform (returns Signal3D for forces)
fp = record['FP1']
fz = fp.force['Fz']  # Returns Signal1D

print(f"Vertical force: {fz.label}")
print(f"Unit: {fz.unit}")
print(f"Sampling frequency: {fz.sampling_frequency} Hz")
print(f"Duration: {len(fz) / fz.sampling_frequency:.2f} s")
```

### Working with Units

```python
from pint import UnitRegistry
ureg = UnitRegistry()

# Signal automatically handles units
fz = laban.Signal1D(
    data=np.array([100, 200, 300]),
    sampling_frequency=100,
    label='Force',
    unit='N'
)

# Convert units
fz_kn = fz.data * ureg.N
fz_kn = fz_kn.to('kN').magnitude

print(f"Force in kN: {fz_kn}")
# Output: [0.1 0.2 0.3]
```

### Handling Missing Data

```python
# Create signal with missing data
data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0])
signal = laban.Signal1D(data=data, sampling_frequency=100)

# Fill with linear interpolation
signal_filled = signal.fillna(method='linear')
print(signal_filled.data)
# Output: [1. 2. 3. 4. 5. 6. 7.]

# Fill with cubic spline
signal_cubic = signal.fillna(method='cubic')
```

### Exporting Data

```python
# Convert to DataFrame
df = signal.to_dataframe()

# Export to Excel
df.to_excel("signal_data.xlsx", index=True)

# Export to CSV
df.to_csv("signal_data.csv")

# Access as NumPy array
data_array = signal.data
time_array = signal.index
```

### Resampling

```python
# Downsample from 1000 Hz to 100 Hz
from scipy import signal as sp_signal

original = laban.Signal1D(data=np.random.randn(10000), sampling_frequency=1000)

# Downsample by factor of 10
downsampled_data = sp_signal.decimate(original.data, q=10)

downsampled = laban.Signal1D(
    data=downsampled_data,
    sampling_frequency=100,
    label=original.label,
    unit=original.unit
)

print(f"Original: {len(original.data)} samples at {original.sampling_frequency} Hz")
print(f"Downsampled: {len(downsampled.data)} samples at {downsampled.sampling_frequency} Hz")
```

## See Also

- **[User Guide: Data Loading](../../guides/data-loading/overview.md)** - Loading timeseries data
- **[User Guide: Signal Processing](../../guides/signal-processing/overview.md)** - Processing signals
- **[Records](records.md)** - Higher-level record classes
- **[Signal Processing API](../signalprocessing.md)** - Processing functions

---

**Module**: `src/labanalysis/records/timeseries.py`
