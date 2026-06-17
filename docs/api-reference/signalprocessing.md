# labanalysis.signalprocessing

Signal processing functions for filtering, peak detection, derivatives, and transformations.

**Source**: `src/labanalysis/signalprocessing.py`

## Filtering Functions

### butterworth_filt()

Apply Butterworth filter (low-pass, high-pass, band-pass, or band-stop).

```python
def butterworth_filt(
    signal: np.ndarray,
    freq: float,
    cut: Union[float, Tuple[float, float]],
    order: int = 4,
    filt_type: str = 'low'
) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal data (1D array)
- `freq` (float): Sampling frequency in Hz
- `cut` (float or tuple): Cut-off frequency/frequencies in Hz
  - Single float for low-pass/high-pass
  - Tuple (low, high) for band-pass/band-stop
- `order` (int, default=4): Filter order
- `filt_type` (str, default='low'): Filter type - 'low', 'high', 'band', or 'stop'

**Returns:**
- `ndarray`: Filtered signal (same shape as input)

**Notes:**
- Uses forward-backward filtering (filtfilt) for zero phase distortion
- Cut-off frequency must be less than Nyquist frequency (freq/2)

**Examples:**
```python
import labanalysis as laban
import numpy as np

# Low-pass filter at 10 Hz
signal = np.random.randn(1000)
filtered = laban.butterworth_filt(signal, freq=1000, cut=10, order=4, filt_type='low')

# High-pass filter at 20 Hz
highpass = laban.butterworth_filt(signal, freq=1000, cut=20, filt_type='high')

# Band-pass filter 20-450 Hz (typical for EMG)
bandpass = laban.butterworth_filt(signal, freq=2000, cut=(20, 450), filt_type='band')

# Band-stop (notch) filter at 50 Hz (powerline noise)
notch = laban.butterworth_filt(signal, freq=1000, cut=(48, 52), filt_type='stop')
```

### fir_filt()

Apply FIR filter with linear phase response.

```python
def fir_filt(
    signal: np.ndarray,
    freq: float,
    cut: float,
    numtaps: int = 101
) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal
- `freq` (float): Sampling frequency in Hz
- `cut` (float): Cut-off frequency in Hz
- `numtaps` (int, default=101): Filter length (odd number recommended)

**Returns:**
- `ndarray`: Filtered signal

**Examples:**
```python
# FIR low-pass filter
filtered = laban.fir_filt(signal, freq=1000, cut=10, numtaps=101)
```

### running_mean()

Apply moving average filter.

```python
def running_mean(signal: np.ndarray, window_size: int) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal
- `window_size` (int): Window size in samples (must be odd)

**Returns:**
- `ndarray`: Smoothed signal

**Examples:**
```python
# 21-sample moving average
smoothed = laban.running_mean(signal, window_size=21)
```

### median_filt()

Apply median filter (robust to outliers).

```python
def median_filt(signal: np.ndarray, window_size: int = 5) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal
- `window_size` (int, default=5): Window size in samples

**Returns:**
- `ndarray`: Filtered signal

**Examples:**
```python
# Remove spike artifacts
denoised = laban.median_filt(signal, window_size=5)
```

### rms_filt()

Calculate RMS over moving window.

```python
def rms_filt(signal: np.ndarray, window_size: int) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal (typically rectified)
- `window_size` (int): Window size in samples

**Returns:**
- `ndarray`: RMS envelope

**Examples:**
```python
# EMG envelope
emg_rect = np.abs(emg_signal)
emg_rms = laban.rms_filt(emg_rect, window_size=50)
```

## Peak Detection

### find_peaks()

Find peaks in signal.

```python
def find_peaks(
    signal: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    width: Optional[int] = None
) -> dict
```

**Parameters:**
- `signal` (ndarray): Input signal
- `height` (float, optional): Minimum peak height
- `distance` (int, optional): Minimum distance between peaks (samples)
- `prominence` (float, optional): Minimum peak prominence
- `width` (int, optional): Minimum peak width (samples)

**Returns:**
- `dict`: Dictionary containing:
  - `'peak_indices'`: Peak locations (indices)
  - `'peak_heights'`: Peak values
  - `'prominences'`: Peak prominences
  - `'widths'`: Peak widths

**Examples:**
```python
# Find peaks taller than 500 N, at least 100 samples apart
peaks = laban.find_peaks(force_signal, height=500, distance=100)

print(f"Found {len(peaks['peak_indices'])} peaks")
print(f"Peak heights: {peaks['peak_heights']}")
```

### find_valleys()

Find valleys (negative peaks) in signal.

```python
def find_valleys(
    signal: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None
) -> dict
```

**Parameters:**
- Same as `find_peaks()` but finds minima

**Returns:**
- `dict`: Valley indices and properties

**Examples:**
```python
# Find valleys
valleys = laban.find_valleys(signal, depth=100, distance=50)
```

## Derivatives

### winter_derivative1()

First derivative using Winter (2009) method.

```python
def winter_derivative1(signal: np.ndarray, freq: float) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal (position)
- `freq` (float): Sampling frequency in Hz

**Returns:**
- `ndarray`: First derivative (velocity)

**Notes:**
- Uses 5-point finite difference with optimal coefficients
- Automatically pads edges

**Examples:**
```python
# Calculate velocity from position
position = marker.data[:, 1]  # Y-axis
velocity = laban.winter_derivative1(position, freq=100)
```

### winter_derivative2()

Second derivative using Winter (2009) method.

```python
def winter_derivative2(signal: np.ndarray, freq: float) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal (position)
- `freq` (float): Sampling frequency in Hz

**Returns:**
- `ndarray`: Second derivative (acceleration)

**Examples:**
```python
# Calculate acceleration from position
acceleration = laban.winter_derivative2(position, freq=100)
```

## Missing Data

### fillna()

Fill missing values (NaN) in signal.

```python
def fillna(
    signal: np.ndarray,
    method: str = 'linear',
    regressors: Optional[np.ndarray] = None
) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal with NaN values
- `method` (str, default='linear'): Interpolation method
  - `'linear'`: Linear interpolation
  - `'cubic'`: Cubic spline interpolation
  - `'pchip'`: Piecewise cubic Hermite
  - `'regression'`: Multiple regression (requires regressors)
- `regressors` (ndarray, optional): Predictor signals for regression method

**Returns:**
- `ndarray`: Signal with NaN filled

**Examples:**
```python
# Linear interpolation
filled = laban.fillna(signal, method='linear')

# Cubic spline
filled = laban.fillna(signal, method='cubic')

# Multiple regression using other markers
filled = laban.fillna(
    signal,
    method='regression',
    regressors=np.column_stack([marker1, marker2, marker3])
)
```

## Frequency Analysis

### psd()

Calculate power spectral density.

```python
def psd(
    signal: np.ndarray,
    freq: float,
    nperseg: int = 1024
) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters:**
- `signal` (ndarray): Input signal
- `freq` (float): Sampling frequency in Hz
- `nperseg` (int, default=1024): Segment length for Welch's method

**Returns:**
- `tuple`: (frequencies, power) arrays

**Examples:**
```python
# Calculate PSD
frequencies, power = laban.psd(signal, freq=1000, nperseg=2048)

# Plot
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=frequencies, y=10*np.log10(power)))
fig.update_xaxes(type='log', title='Frequency (Hz)')
fig.update_yaxes(title='PSD (dB)')
fig.show()
```

### residual_analysis()

Perform residual analysis to determine optimal filter cut-off.

```python
def residual_analysis(
    signal: np.ndarray,
    freq: float,
    cutoffs: np.ndarray = None
) -> dict
```

**Parameters:**
- `signal` (ndarray): Input signal
- `freq` (float): Sampling frequency in Hz
- `cutoffs` (ndarray, optional): Cut-off frequencies to test (Hz)

**Returns:**
- `dict`: Dictionary with:
  - `'cutoffs'`: Tested cut-off frequencies
  - `'residuals'`: RMS residuals for each cut-off
  - `'optimal_cutoff'`: Recommended cut-off frequency

**Examples:**
```python
# Find optimal cut-off
result = laban.residual_analysis(signal, freq=1000)
print(f"Optimal cut-off: {result['optimal_cutoff']:.1f} Hz")
```

## Transformations

### change_reference_frame()

Transform signal between coordinate systems.

```python
def change_reference_frame(
    signal: Point3D,
    from_frame: np.ndarray,
    to_frame: np.ndarray
) -> Point3D
```

**Parameters:**
- `signal` (Point3D): 3D signal to transform
- `from_frame` (ndarray): Source reference frame (3x3 rotation matrix or axes)
- `to_frame` (ndarray): Target reference frame (3x3 rotation matrix or axes)

**Returns:**
- `Point3D`: Transformed signal

**Examples:**
```python
# Transform marker from global to pelvis frame
pelvis_frame = body.pelvis.reference_frame
marker_pelvis = laban.change_reference_frame(
    marker,
    from_frame='global',
    to_frame=pelvis_frame
)
```

### gram_schmidt()

Orthogonalize reference frame using Gram-Schmidt process.

```python
def gram_schmidt(vectors: np.ndarray) -> np.ndarray
```

**Parameters:**
- `vectors` (ndarray): Input vectors (3x3 array, row vectors)

**Returns:**
- `ndarray`: Orthonormalized vectors (3x3 array)

**Examples:**
```python
# Create orthogonal reference frame from markers
x_axis = (marker2.data - marker1.data)
y_axis = (marker3.data - marker1.data)
z_axis = np.cross(x_axis, y_axis)

frame = laban.gram_schmidt(np.array([x_axis, y_axis, z_axis]))
```

## Utility Functions

### normalize()

Normalize signal to 0-1 range.

```python
def normalize(signal: np.ndarray) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal

**Returns:**
- `ndarray`: Normalized signal

**Examples:**
```python
normalized = laban.normalize(signal)
```

### resample()

Resample signal to new sampling frequency.

```python
def resample(
    signal: np.ndarray,
    original_freq: float,
    target_freq: float
) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal
- `original_freq` (float): Original sampling frequency (Hz)
- `target_freq` (float): Target sampling frequency (Hz)

**Returns:**
- `ndarray`: Resampled signal

**Examples:**
```python
# Downsample from 1000 Hz to 100 Hz
downsampled = laban.resample(signal, original_freq=1000, target_freq=100)
```

### zero_phase_shift()

Remove DC offset from signal.

```python
def zero_phase_shift(signal: np.ndarray) -> np.ndarray
```

**Parameters:**
- `signal` (ndarray): Input signal

**Returns:**
- `ndarray`: Zero-centered signal

**Examples:**
```python
centered = laban.zero_phase_shift(signal)
```

## Complete Processing Pipelines

### Force Platform Signal

```python
# Standard force platform processing
fz_raw = fp.force['Fz'].data
freq = fp.sampling_frequency

# 1. Remove outliers
fz_clean = laban.median_filt(fz_raw, window_size=5)

# 2. Low-pass filter
fz_filtered = laban.butterworth_filt(fz_clean, freq=freq, cut=10, order=4)

# 3. Calculate derivatives
velocity = laban.winter_derivative1(fz_filtered, freq=freq)
acceleration = laban.winter_derivative2(fz_filtered, freq=freq)
```

### EMG Signal

```python
# Standard EMG processing
emg_raw = record['EMG']['biceps'].data
freq = 2000

# 1. Band-pass filter (20-450 Hz)
emg_bp = laban.butterworth_filt(emg_raw, freq=freq, cut=(20, 450), filt_type='band')

# 2. Full-wave rectification
emg_rect = np.abs(emg_bp)

# 3. Linear envelope
emg_env = laban.butterworth_filt(emg_rect, freq=freq, cut=3, order=4)
```

### Marker Position Signal

```python
# Standard marker processing
marker_raw = body.c7_vertebra
freq = marker_raw.sampling_frequency

# Process each axis
marker_filtered = np.zeros_like(marker_raw.data)
for i in range(3):
    # 1. Fill missing data
    filled = laban.fillna(marker_raw.data[:, i], method='cubic')
    
    # 2. Low-pass filter at 6 Hz
    marker_filtered[:, i] = laban.butterworth_filt(filled, freq=freq, cut=6, order=4)
```

## See Also

- **[User Guide: Signal Processing](../user-guide/signal-processing/README.md)** - Complete guide
- **[User Guide: Filtering](../user-guide/signal-processing/filtering.md)** - Detailed filtering guide
- **[Tutorial: Signal Processing](../tutorials/05-signal-processing.md)** - Complete workflow

---

**Reference**: Winter DA (2009). Biomechanics and Motor Control of Human Movement. 4th ed.
