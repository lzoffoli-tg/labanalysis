# EMG Signals

Guide to working with electromyography (EMG) data using the EMGSignal class.

## Overview

Electromyography (EMG) measures electrical activity produced by skeletal muscles. The `EMGSignal` class in labanalysis provides a specialized container for EMG data with:

- Automatic unit conversion to microvolts (μV)
- Muscle name and side (left/right/bilateral) tracking
- Integration with signal processing tools (filtering, RMS, rectification)
- Support for normalized EMG (% of maximum voluntary contraction)

**Common EMG Processing Steps:**
1. High-pass filtering (remove DC offset and motion artifacts)
2. Full-wave rectification (absolute value)
3. Low-pass filtering or RMS smoothing (extract envelope)
4. Normalization (to MVC or peak activation)

## Quick Reference

```python
import labanalysis as laban

# Create EMG signal
emg = laban.EMGSignal(
    data=raw_voltage,           # 1D array
    index=time,                 # Time vector
    muscle_name="gastrocnemius",
    side="left",
    unit="mV"                   # Automatically converted to μV
)

# Access metadata
print(f"Muscle: {emg.muscle_name}")
print(f"Side: {emg.side}")
print(f"Unit: {emg.unit}")  # Output: μV

# Process EMG
freq = 2000  # Hz

# 1. High-pass filter (remove artifacts)
emg_hp = laban.butterworth_filt(emg.data, freq=freq, cut=20, order=4, filt_type='high')

# 2. Rectify
emg_rect = np.abs(emg_hp)

# 3. RMS envelope (50ms window)
window_size = int(0.05 * freq)  # 50ms
emg_rms = laban.rms(emg_rect, window_size)

# Update signal
emg.data = emg_rms
```

## Creating EMG Signals

### From Raw Data

```python
import labanalysis as laban
import numpy as np

# Raw EMG data (simulate)
freq = 2000  # Hz
duration = 10  # seconds
n_samples = freq * duration

time = np.linspace(0, duration, n_samples)
raw_emg = np.random.randn(n_samples) * 0.05  # mV (simulate noise)

# Create EMGSignal
emg = laban.EMGSignal(
    data=raw_emg,
    index=time,
    muscle_name="vastus_lateralis",
    side="right",
    unit="mV"  # Will be converted to μV automatically
)

print(f"EMG unit: {emg.unit}")  # Output: μV
print(f"EMG range: {emg.data.min():.1f} to {emg.data.max():.1f} μV")
```

### From TDF Files

```python
# Load TDF file
record = laban.TimeseriesRecord.from_tdf("emg_trial.tdf")

# Check available signals
print(record.keys())
# Output: ['EMG_VastusLateralis_L', 'EMG_Gastrocnemius_R', ...]

# Access EMG signal
emg_vl = record['EMG_VastusLateralis_L']

# If not automatically detected as EMGSignal, create manually
if not isinstance(emg_vl, laban.EMGSignal):
    emg_vl = laban.EMGSignal(
        data=emg_vl.data[:, 0],  # First column
        index=emg_vl.index,
        muscle_name="vastus_lateralis",
        side="left",
        unit="mV"
    )
```

## EMG Processing Pipeline

### Complete Processing Workflow

```python
import labanalysis as laban
import numpy as np

# Load raw EMG
emg_raw = laban.EMGSignal(
    data=raw_data,
    index=time,
    muscle_name="biceps_brachii",
    side="left",
    unit="mV"
)

freq = 2000  # Hz (typical EMG sampling rate)

# Step 1: High-pass filter (remove DC offset and motion artifacts)
# Cutoff: 20-30 Hz (removes movement artifacts while preserving EMG)
emg_hp = laban.butterworth_filt(
    emg_raw.data,
    freq=freq,
    cut=20,  # Hz
    order=4,
    filt_type='high'
)

# Step 2: Band-pass filter (optional, isolate EMG frequency range)
# EMG power is typically 20-500 Hz
emg_bp = laban.butterworth_filt(
    emg_hp,
    freq=freq,
    cut=(20, 450),  # Hz (low, high)
    order=4,
    filt_type='band'
)

# Step 3: Full-wave rectification
emg_rect = np.abs(emg_bp)

# Step 4: Extract envelope (RMS or low-pass filter)
# Option A: RMS with moving window (50-100ms typical)
window_ms = 50  # ms
window_samples = int(window_ms / 1000 * freq)
emg_rms = laban.rms(emg_rect, window_samples)

# Option B: Low-pass filter (6-10 Hz cutoff)
# emg_env = laban.butterworth_filt(emg_rect, freq=freq, cut=6, order=4, filt_type='low')

# Update EMG signal with processed data
emg_processed = emg_raw.copy()
emg_processed.data = emg_rms

print(f"Processed EMG: {emg_processed.muscle_name} ({emg_processed.side})")
print(f"Peak activation: {emg_processed.data.max():.1f} μV")
```

### Filtering Recommendations

| Filter Type | Cutoff | Purpose |
|-------------|--------|---------|
| High-pass | 20-30 Hz | Remove DC offset, motion artifacts |
| Band-pass | 20-450 Hz | Isolate EMG frequency content |
| Low-pass (envelope) | 6-10 Hz | Extract activation envelope |
| Notch | 50/60 Hz | Remove powerline interference (if needed) |

**Order**: 4th order Butterworth is standard for EMG processing.

## Amplitude Analysis

### Peak and Mean Activation

```python
# Get peak activation
peak_activation = emg_processed.data.max()

# Get mean activation during task
# (Assuming task starts at t=2s and ends at t=8s)
task_start = 2.0
task_end = 8.0

task_mask = (emg_processed.index >= task_start) & (emg_processed.index <= task_end)
mean_activation = emg_processed.data[task_mask].mean()

print(f"Peak: {peak_activation:.1f} μV")
print(f"Mean (task): {mean_activation:.1f} μV")
```

### Maximum Voluntary Contraction (MVC) Normalization

```python
# Collect MVC trial
emg_mvc = laban.EMGSignal(
    data=mvc_data,
    index=mvc_time,
    muscle_name="biceps_brachii",
    side="left",
    unit="mV"
)

# Process MVC the same way as task data
emg_mvc_processed = process_emg(emg_mvc, freq=2000)  # Apply same pipeline

# Get MVC value (peak during MVC trial)
mvc_value = emg_mvc_processed.data.max()

# Normalize task EMG to %MVC
emg_normalized = emg_processed.copy()
emg_normalized.data = (emg_processed.data / mvc_value) * 100
emg_normalized.unit = "%"  # Now in percentage of MVC

print(f"Peak activation: {emg_normalized.data.max():.1f} %MVC")
```

### Activation Threshold Detection

```python
# Detect muscle activation onset
# (When EMG exceeds baseline + threshold)

# Calculate baseline (quiet period)
baseline_mask = emg_processed.index < 1.0  # First second
baseline_mean = emg_processed.data[baseline_mask].mean()
baseline_std = emg_processed.data[baseline_mask].std()

# Threshold: baseline + 3 SD
threshold = baseline_mean + 3 * baseline_std

# Detect onset (first sample above threshold)
above_threshold = emg_processed.data > threshold
onset_idx = np.where(above_threshold)[0][0]
onset_time = emg_processed.index[onset_idx]

print(f"Activation onset at t = {onset_time:.3f} s")
print(f"Threshold: {threshold:.1f} μV")
```

## Frequency Analysis

### Power Spectral Density (PSD)

```python
# Calculate PSD of raw EMG (before rectification)
frequencies, power = laban.psd(
    emg_hp,  # High-pass filtered, not rectified
    freq=freq,
    nperseg=1024  # Window size for FFT
)

# Find dominant frequency
peak_freq_idx = np.argmax(power)
peak_freq = frequencies[peak_freq_idx]

print(f"Peak frequency: {peak_freq:.1f} Hz")
print(f"Power at peak: {power[peak_freq_idx]:.2e}")

# Typical EMG: peak around 50-150 Hz
```

### Median Frequency (Fatigue Indicator)

```python
# Median frequency decreases with muscle fatigue
def median_frequency(signal, freq, nperseg=1024):
    """Calculate median frequency of EMG signal."""
    frequencies, power = laban.psd(signal, freq=freq, nperseg=nperseg)
    
    # Cumulative power
    cumulative_power = np.cumsum(power)
    total_power = cumulative_power[-1]
    
    # Find frequency where cumulative power = 50%
    median_idx = np.where(cumulative_power >= total_power / 2)[0][0]
    return frequencies[median_idx]

# Calculate median frequency over time (windowed)
window_duration = 1.0  # seconds
window_samples = int(window_duration * freq)

median_freqs = []
window_times = []

for i in range(0, len(emg_hp) - window_samples, window_samples // 2):
    window = emg_hp[i:i + window_samples]
    mf = median_frequency(window, freq)
    median_freqs.append(mf)
    window_times.append(emg_raw.index[i + window_samples // 2])

# Plot median frequency vs time to observe fatigue
import matplotlib.pyplot as plt
plt.plot(window_times, median_freqs)
plt.xlabel('Time (s)')
plt.ylabel('Median Frequency (Hz)')
plt.title('EMG Median Frequency (Fatigue Indicator)')
plt.show()

# Decreasing median frequency indicates fatigue
```

## Multi-Muscle Analysis

### Bilateral Comparison

```python
# Load bilateral EMG
emg_left = laban.EMGSignal(
    data=left_data, index=time,
    muscle_name="quadriceps", side="left", unit="mV"
)

emg_right = laban.EMGSignal(
    data=right_data, index=time,
    muscle_name="quadriceps", side="right", unit="mV"
)

# Process both
emg_left_proc = process_emg(emg_left, freq=2000)
emg_right_proc = process_emg(emg_right, freq=2000)

# Compare peak activations
peak_left = emg_left_proc.data.max()
peak_right = emg_right_proc.data.max()

asymmetry = abs(peak_left - peak_right) / ((peak_left + peak_right) / 2) * 100

print(f"Left peak: {peak_left:.1f} μV")
print(f"Right peak: {peak_right:.1f} μV")
print(f"Asymmetry: {asymmetry:.1f}%")

# Asymmetry > 15% may indicate imbalance
```

### Co-Contraction Analysis

```python
# Calculate co-contraction index between agonist and antagonist

# Normalize to MVC first
quad_norm = (quad_emg / quad_mvc) * 100  # %MVC
ham_norm = (ham_emg / ham_mvc) * 100     # %MVC

# Co-contraction index (area of overlap)
# Method: sum of minimum activations
co_contraction = np.sum(np.minimum(quad_norm, ham_norm))

# Or as percentage of total activation
total_activation = np.sum(quad_norm + ham_norm)
co_contraction_pct = (2 * co_contraction / total_activation) * 100

print(f"Co-contraction index: {co_contraction_pct:.1f}%")
```

## Troubleshooting

### Issue: "unit must represent voltage"

EMGSignal only accepts voltage units or percentages:

```python
# WRONG: Trying to use force units
# emg = laban.EMGSignal(..., unit="N")  # Error!

# RIGHT: Use voltage units
emg = laban.EMGSignal(..., unit="mV")   # Converted to μV
emg = laban.EMGSignal(..., unit="μV")   # Native unit
emg = laban.EMGSignal(..., unit="%")    # Normalized (%MVC)
```

### Issue: Noisy EMG Signal

Apply more aggressive filtering:

```python
# Increase high-pass cutoff (remove more low-frequency noise)
emg_hp = laban.butterworth_filt(emg.data, freq=2000, cut=30, order=4, filt_type='high')

# Use band-pass to limit high-frequency noise
emg_bp = laban.butterworth_filt(emg_hp, freq=2000, cut=(30, 400), order=4, filt_type='band')

# Increase smoothing window
window_samples = int(0.1 * freq)  # 100ms window instead of 50ms
emg_smooth = laban.rms(emg_rect, window_samples)
```

### Issue: Powerline Interference (50/60 Hz Hum)

Apply notch filter:

```python
# Create notch filter function (not built-in, requires scipy)
from scipy.signal import iirnotch, filtfilt

def notch_filter(signal, freq, notch_freq=50, quality=30):
    """Apply notch filter to remove powerline interference."""
    b, a = iirnotch(notch_freq, quality, freq)
    return filtfilt(b, a, signal)

# Apply 50 Hz notch (Europe) or 60 Hz (USA)
emg_notched = notch_filter(emg.data, freq=2000, notch_freq=50)
emg.data = emg_notched
```

### Issue: Crosstalk from Adjacent Muscles

Reduce electrode spacing or use more selective electrodes. In processing:

```python
# Use higher high-pass cutoff to reduce low-frequency crosstalk
emg_hp = laban.butterworth_filt(emg.data, freq=2000, cut=50, order=4, filt_type='high')

# Document electrode placement carefully
# Note: Software cannot fully remove crosstalk
```

## Export EMG Data

```python
# Convert to DataFrame
df = emg.to_dataframe()

print(df.columns)
# Output: ['biceps_brachii_left_μV']

# Export to CSV
df.to_csv("emg_data.csv")

# Export to Excel
df.to_excel("emg_data.xlsx")

# Include metadata in export
metadata = {
    'muscle': emg.muscle_name,
    'side': emg.side,
    'unit': str(emg.unit),
    'sampling_frequency_Hz': freq,
    'duration_s': emg.index[-1] - emg.index[0]
}

import json
with open("emg_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

## Processing Function Template

```python
def process_emg_pipeline(emg, freq, high_cut=20, low_cut=450, 
                         envelope_window_ms=50, method='rms'):
    """
    Standard EMG processing pipeline.
    
    Parameters
    ----------
    emg : EMGSignal
        Raw EMG signal
    freq : float
        Sampling frequency (Hz)
    high_cut : float
        High-pass filter cutoff (Hz)
    low_cut : float
        Low-pass filter cutoff for band-pass (Hz)
    envelope_window_ms : float
        RMS window size (ms)
    method : {'rms', 'lowpass'}
        Envelope extraction method
    
    Returns
    -------
    EMGSignal
        Processed EMG with envelope
    """
    import labanalysis as laban
    import numpy as np
    
    # 1. Band-pass filter
    emg_filtered = laban.butterworth_filt(
        emg.data,
        freq=freq,
        cut=(high_cut, low_cut),
        order=4,
        filt_type='band'
    )
    
    # 2. Rectify
    emg_rect = np.abs(emg_filtered)
    
    # 3. Extract envelope
    if method == 'rms':
        window_samples = int(envelope_window_ms / 1000 * freq)
        emg_env = laban.rms(emg_rect, window_samples)
    elif method == 'lowpass':
        emg_env = laban.butterworth_filt(
            emg_rect, freq=freq, cut=6, order=4, filt_type='low'
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 4. Create output signal
    emg_out = emg.copy()
    emg_out.data = emg_env
    
    return emg_out

# Usage
emg_processed = process_emg_pipeline(emg_raw, freq=2000)
```

## See Also

- [Signal Processing: Filtering](../signal-processing/filtering.md) - Butterworth and FIR filtering
- [Signal Processing: Peak Detection](../signal-processing/peak-detection.md) - Detecting activation bursts
- [API Reference: EMGSignal](../../api-reference/records/timeseries.md#emgsignal) - Complete EMGSignal API
- [Test Protocols: Strength Tests](../test-protocols/strength-tests.md) - EMG in strength assessment

---

**EMG Signals**: Specialized container for electromyography data with integrated processing tools for activation analysis.
