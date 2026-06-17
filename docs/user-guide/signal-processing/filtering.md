# Signal Filtering

Guide to filtering signals in labanalysis using Butterworth, FIR, moving average, median, and RMS filters.

## Overview

Filtering is essential for removing noise and extracting meaningful information from biomechanical signals. labanalysis provides multiple filtering approaches optimized for different signal types and noise characteristics.

**Available Filters:**
- **Butterworth** - IIR filter for frequency-domain filtering (low-pass, high-pass, band-pass, band-stop)
- **FIR** - Finite impulse response filter with linear phase
- **Moving Average** - Simple smoothing filter
- **Median Filter** - Robust to outliers
- **RMS Filter** - Root mean square for signal envelope

## Butterworth Filter

Low-pass, high-pass, band-pass, and band-stop filtering using Butterworth design.

### Low-Pass Filter

Remove high-frequency noise while preserving low-frequency signal content.

```python
import labanalysis as laban
import numpy as np

# Load signal
record = laban.TimeseriesRecord.from_tdf("force_data.tdf")
fp = record['FP1']
fz = fp.force['Fz']

# Apply 4th order Butterworth low-pass filter at 10 Hz
filtered = laban.butterworth_filt(
    signal=fz.data,
    freq=fz.sampling_frequency,  # Hz
    cut=10,                       # Cut-off frequency (Hz)
    order=4,                      # Filter order
    filt_type='low'               # Low-pass filter
)

# Create filtered Signal1D
fz_filtered = laban.Signal1D(
    data=filtered,
    index=fz.index,
    label='Fz_filtered',
    unit=fz.unit
)

print(f"Original signal: {fz.data.std():.2f} N std")
print(f"Filtered signal: {fz_filtered.data.std():.2f} N std")
```

**Use cases:**
- Force platform data (cut-off: 10-15 Hz)
- Marker positions (cut-off: 6-10 Hz)
- EMG envelope (cut-off: 3-5 Hz)

### High-Pass Filter

Remove baseline drift and low-frequency artifacts.

```python
# Remove baseline drift from EMG
emg = record['EMG']['biceps']

# High-pass filter at 20 Hz to remove motion artifacts
filtered_emg = laban.butterworth_filt(
    signal=emg.data,
    freq=emg.sampling_frequency,
    cut=20,              # Cut-off frequency
    order=4,
    filt_type='high'     # High-pass filter
)

# EMG is now centered around zero without DC component
print(f"Original mean: {emg.data.mean():.4f}")
print(f"Filtered mean: {filtered_emg.mean():.4f}")
```

**Use cases:**
- EMG signals (cut-off: 20-30 Hz)
- Accelerometer data (cut-off: 0.5-1 Hz for removing gravity)
- Removing baseline drift

### Band-Pass Filter

Isolate a specific frequency band.

```python
# Extract 20-450 Hz band from EMG (typical muscle activity)
emg_bandpass = laban.butterworth_filt(
    signal=emg.data,
    freq=emg.sampling_frequency,
    cut=(20, 450),       # Low and high cut-off frequencies
    order=4,
    filt_type='band'     # Band-pass filter
)

print(f"Retained frequencies: 20-450 Hz")
```

**Use cases:**
- EMG signals (20-450 Hz for muscle activity)
- Isolating specific frequency components
- Removing both low and high frequency noise

### Band-Stop (Notch) Filter

Remove a specific frequency band (e.g., powerline interference).

```python
# Remove 50 Hz powerline interference
notch_filtered = laban.butterworth_filt(
    signal=emg.data,
    freq=emg.sampling_frequency,
    cut=(48, 52),        # Narrow band around 50 Hz
    order=4,
    filt_type='stop'     # Band-stop filter
)

print("Removed 50 Hz powerline noise")
```

**Use cases:**
- Removing 50/60 Hz powerline interference
- Eliminating specific frequency artifacts

### Choosing Filter Order

Filter order affects sharpness of cut-off and computational cost:

```python
import plotly.graph_objects as go

# Compare different filter orders
orders = [2, 4, 6, 8]
fig = go.Figure()

for order in orders:
    filtered = laban.butterworth_filt(
        signal=fz.data,
        freq=1000,
        cut=10,
        order=order,
        filt_type='low'
    )
    fig.add_trace(go.Scatter(y=filtered, name=f'Order {order}'))

fig.update_layout(
    title='Effect of Filter Order',
    yaxis_title='Force (N)',
    xaxis_title='Sample'
)
fig.show()
```

**Guidelines:**
- **Order 2**: Gentle filtering, minimal phase distortion
- **Order 4**: Standard choice, good balance (recommended)
- **Order 6**: Sharper cut-off, more attenuation
- **Order 8+**: Very sharp cut-off, risk of ringing artifacts

## FIR Filter

Finite impulse response filter with guaranteed linear phase (no phase distortion).

```python
# FIR low-pass filter
fir_filtered = laban.fir_filt(
    signal=fz.data,
    freq=fz.sampling_frequency,
    cut=10,
    numtaps=101          # Filter length (odd number)
)

print(f"FIR filter length: 101 taps")
```

**Advantages:**
- Linear phase (no phase distortion)
- Always stable
- Better for signals where phase is critical

**Disadvantages:**
- Longer computational time
- Edge effects at signal boundaries

**When to use FIR vs Butterworth:**
- Use FIR when phase preservation is critical (e.g., precise timing analysis)
- Use Butterworth for most biomechanical signals (faster, standard in literature)

## Moving Average Filter

Simple smoothing filter that replaces each point with the average of surrounding points.

```python
# Smooth signal with 21-sample moving average
smoothed = laban.running_mean(
    signal=fz.data,
    window_size=21       # Must be odd number
)

print(f"Window size: 21 samples = {21/fz.sampling_frequency*1000:.1f} ms")
```

**Use cases:**
- Quick smoothing without frequency-domain considerations
- Real-time processing (causal filter)
- Simple noise reduction

**Choosing window size:**
```python
# Convert time window to samples
desired_window_ms = 50  # milliseconds
window_samples = int(desired_window_ms * fz.sampling_frequency / 1000)

# Ensure odd number
if window_samples % 2 == 0:
    window_samples += 1

smoothed = laban.running_mean(fz.data, window_size=window_samples)
```

## Median Filter

Robust filter that replaces each point with the median of surrounding points. Excellent for removing outliers.

```python
# Remove outliers with median filter
denoised = laban.median_filt(
    signal=fz.data,
    window_size=5        # Small window to preserve features
)

# Detect outliers by comparing original and filtered
outliers = np.abs(fz.data - denoised) > 100  # Threshold in N

print(f"Detected {outliers.sum()} outlier samples")
```

**Use cases:**
- Removing spike artifacts
- Preprocessing before other filters
- Robust to non-Gaussian noise

**Advantages:**
- Preserves edges better than moving average
- Robust to outliers
- Non-linear (handles spikes well)

## RMS Filter

Calculate root mean square over moving window - useful for signal envelope.

```python
# Calculate EMG envelope
emg = record['EMG']['biceps']

# 1. High-pass filter to remove DC
emg_hp = laban.butterworth_filt(emg.data, freq=emg.sampling_frequency, cut=20, filt_type='high')

# 2. Full-wave rectification
emg_rect = np.abs(emg_hp)

# 3. RMS envelope (50 ms window)
window_samples = int(0.050 * emg.sampling_frequency)
emg_rms = laban.rms_filt(emg_rect, window_size=window_samples)

# 4. Optional: smooth envelope
emg_envelope = laban.butterworth_filt(emg_rms, freq=emg.sampling_frequency, cut=5, filt_type='low')

print(f"EMG envelope calculated with {window_samples}-sample RMS window")
```

**Use cases:**
- EMG envelope detection
- Signal amplitude tracking
- Detecting activity periods

## Complete Filtering Pipeline

### Force Platform Signal

Standard pipeline for ground reaction force:

```python
import labanalysis as laban
import numpy as np

# Load force data
record = laban.TimeseriesRecord.from_tdf("jump.tdf")
fp = record['FP1']
fz_raw = fp.force['Fz']

# Step 1: Remove outliers with median filter
fz_denoised = laban.median_filt(fz_raw.data, window_size=5)

# Step 2: Low-pass filter at 10 Hz
fz_filtered = laban.butterworth_filt(
    signal=fz_denoised,
    freq=fz_raw.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# Create clean signal
fz_clean = laban.Signal1D(
    data=fz_filtered,
    index=fz_raw.index,
    label='Fz_clean',
    unit='N'
)

# Visualize
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=fz_raw.index, y=fz_raw.data, name='Raw', opacity=0.5))
fig.add_trace(go.Scatter(x=fz_clean.index, y=fz_clean.data, name='Filtered'))
fig.update_layout(title='Force Signal Processing', yaxis_title='Force (N)', xaxis_title='Time (s)')
fig.show()
```

### EMG Signal

Standard pipeline for electromyography:

```python
# Load EMG
emg_raw = record['EMG']['biceps']

# Step 1: Band-pass filter (20-450 Hz)
emg_bp = laban.butterworth_filt(
    signal=emg_raw.data,
    freq=emg_raw.sampling_frequency,
    cut=(20, 450),
    order=4,
    filt_type='band'
)

# Step 2: Full-wave rectification
emg_rect = np.abs(emg_bp)

# Step 3: Linear envelope (low-pass at 3 Hz)
emg_envelope = laban.butterworth_filt(
    signal=emg_rect,
    freq=emg_raw.sampling_frequency,
    cut=3,
    order=4,
    filt_type='low'
)

# Create processed signals
emg_filtered = laban.EMGSignal(
    data=emg_bp,
    index=emg_raw.index,
    label='biceps_filtered',
    unit='mV'
)

emg_env = laban.Signal1D(
    data=emg_envelope,
    index=emg_raw.index,
    label='biceps_envelope',
    unit='mV'
)

print("EMG processing complete")
```

### Marker Position Signal

Standard pipeline for motion capture markers:

```python
# Load marker
body = laban.WholeBody.from_tdf("mocap.tdf", ...)
c7_raw = body.c7_vertebra

# Filter each axis separately
c7_filtered_data = np.zeros_like(c7_raw.data)

for i in range(3):  # x, y, z
    c7_filtered_data[:, i] = laban.butterworth_filt(
        signal=c7_raw.data[:, i],
        freq=c7_raw.sampling_frequency,
        cut=6,           # 6 Hz typical for human movement
        order=4,
        filt_type='low'
    )

# Create filtered marker
c7_filtered = laban.Point3D(
    data=c7_filtered_data,
    index=c7_raw.index,
    label='C7_filtered',
    unit='m'
)

print("Marker filtered at 6 Hz")
```

## Frequency Selection Guidelines

### Recommended Cut-Off Frequencies

| Signal Type | Typical Cut-Off | Rationale |
|-------------|-----------------|-----------|
| Ground reaction force | 10-15 Hz | Human movement <10 Hz, noise >15 Hz |
| Center of pressure | 5-8 Hz | COP fluctuates at <5 Hz |
| Marker positions | 6-10 Hz | Human limb movement <8 Hz |
| Joint angles | 6-10 Hz | Derived from markers |
| EMG (band-pass) | 20-450 Hz | Muscle activity frequency range |
| EMG (envelope) | 3-5 Hz | Muscle activation dynamics |
| Accelerometer | 10-20 Hz | Movement frequencies |
| Gyroscope | 10-20 Hz | Angular velocity |

### Determining Cut-Off from Data

Use power spectral density (PSD) to find appropriate cut-off:

```python
from scipy import signal as scipy_signal

# Compute PSD
frequencies, psd = scipy_signal.welch(
    fz_raw.data,
    fs=fz_raw.sampling_frequency,
    nperseg=1024
)

# Find frequency where 95% of power is contained
cumulative_power = np.cumsum(psd) / np.sum(psd)
cutoff_95 = frequencies[np.where(cumulative_power >= 0.95)[0][0]]

print(f"95% power contained below {cutoff_95:.1f} Hz")

# Use this as cut-off
fz_filtered = laban.butterworth_filt(
    fz_raw.data,
    freq=fz_raw.sampling_frequency,
    cut=cutoff_95,
    order=4,
    filt_type='low'
)
```

## Common Pitfalls

### 1. Filtering Twice (Forward-Backward)

`butterworth_filt()` already applies **forward-backward filtering** (filtfilt) internally. Do not apply twice:

```python
# ❌ WRONG - applies filter 4 times total
filtered_once = laban.butterworth_filt(signal, freq, cut=10, order=4)
filtered_twice = laban.butterworth_filt(filtered_once, freq, cut=10, order=4)  # Too much!

# ✓ CORRECT - applies once (forward-backward internally)
filtered = laban.butterworth_filt(signal, freq, cut=10, order=4)
```

### 2. Cut-Off Too Close to Nyquist Frequency

Cut-off must be well below Nyquist frequency (fs/2):

```python
# ❌ WRONG - cut-off too close to Nyquist
filtered = laban.butterworth_filt(signal, freq=100, cut=45)  # Nyquist = 50 Hz

# ✓ CORRECT - cut-off should be <40% of Nyquist
filtered = laban.butterworth_filt(signal, freq=100, cut=20)  # Safe
```

### 3. Filtering Before Removing Outliers

Remove outliers before filtering to avoid spreading artifacts:

```python
# ✓ CORRECT order
denoised = laban.median_filt(signal, window_size=5)    # Remove outliers first
filtered = laban.butterworth_filt(denoised, freq, cut=10)  # Then smooth

# ❌ WRONG order - outliers will smear during filtering
filtered = laban.butterworth_filt(signal, freq, cut=10)    # Smears outliers
denoised = laban.median_filt(filtered, window_size=5)      # Too late
```

### 4. Window Size Too Large

Large windows over-smooth and lose features:

```python
# Compare window sizes
window_sizes = [11, 51, 101, 201]
for ws in window_sizes:
    smoothed = laban.running_mean(signal, window_size=ws)
    print(f"Window {ws}: std = {smoothed.std():.2f}")
    
# Window 11: std = 45.23  ← Preserves features
# Window 51: std = 38.14
# Window 101: std = 29.87
# Window 201: std = 18.42  ← Over-smoothed!
```

## Validation

### Visual Inspection

```python
import plotly.graph_objects as go

# Plot original and filtered
fig = go.Figure()
fig.add_trace(go.Scatter(x=fz_raw.index, y=fz_raw.data, name='Raw', opacity=0.4))
fig.add_trace(go.Scatter(x=fz_raw.index, y=fz_filtered, name='Filtered'))
fig.update_layout(title='Visual Filter Check', yaxis_title='Force (N)', xaxis_title='Time (s)')
fig.show()
```

### Residual Analysis

Check that removed content is truly noise:

```python
# Calculate residuals
residuals = fz_raw.data - fz_filtered

# Check residuals are small and random
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(y=residuals, mode='markers', marker=dict(size=2), name='Residuals'))
fig.add_hline(y=0, line_dash='dash')
fig.update_layout(title='Filtering Residuals', yaxis_title='Residual (N)')
fig.show()

# Statistical check
print(f"Residual mean: {residuals.mean():.4f} (should be ~0)")
print(f"Residual std: {residuals.std():.2f}")
```

### Frequency Domain Check

```python
# Compare frequency content
from scipy import signal as scipy_signal

# PSD before filtering
freqs_raw, psd_raw = scipy_signal.welch(fz_raw.data, fs=fz_raw.sampling_frequency)

# PSD after filtering  
freqs_filt, psd_filt = scipy_signal.welch(fz_filtered, fs=fz_raw.sampling_frequency)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=freqs_raw, y=10*np.log10(psd_raw), name='Raw'))
fig.add_trace(go.Scatter(x=freqs_filt, y=10*np.log10(psd_filt), name='Filtered'))
fig.add_vline(x=10, line_dash='dash', annotation_text='Cut-off')
fig.update_layout(
    title='Power Spectral Density',
    xaxis_title='Frequency (Hz)',
    yaxis_title='PSD (dB)',
    xaxis_type='log'
)
fig.show()
```

## See Also

- **[Peak Detection](peak-detection.md)** - Find peaks in filtered signals
- **[Derivatives](derivatives.md)** - Calculate velocity and acceleration
- **[Frequency Analysis](frequency-analysis.md)** - Analyze signal frequency content
- **[API Reference: Signal Processing](../../api-reference/signalprocessing.md)** - Complete function reference

---

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
