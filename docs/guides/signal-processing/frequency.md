# Frequency Analysis

Guide to frequency domain analysis in labanalysis using power spectral density (PSD) and residual analysis for optimal filter cut-off selection.

## Overview

Frequency analysis helps you:
- **Understand signal content**: What frequencies are present?
- **Choose filter cut-offs**: Where is the noise?
- **Validate filtering**: Did you preserve the signal?

labanalysis provides two key functions:
- `psd()` - Power Spectral Density estimation
- `residual_analysis()` - Systematic cut-off selection for low-pass filters

## Quick Reference

| Task | Function | Output |
|------|----------|--------|
| Visualize frequency content | `psd()` | Frequency vs. Power |
| Find optimal cut-off | `residual_analysis()` | Cut-off vs. RMS difference |
| Validate filter choice | Compare PSDs before/after | Check preserved vs. removed |

## Power Spectral Density (PSD)

### Basic Usage

Estimate the power distribution across frequencies in your signal.

```python
import labanalysis as laban
import numpy as np
import matplotlib.pyplot as plt

# Load force platform signal
fp = laban.ForcePlatform.from_tdf("jump.tdf", platform_name="FP1")
fz = fp.force['Fz'].data
freq = fp.sampling_frequency

# Calculate PSD
frequencies, power = laban.psd(
    signal=fz,
    freq=freq,
    window='hann',
    nperseg=1024
)

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(frequencies, power)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (N²/Hz)')
plt.title('Force Platform Fz - Power Spectral Density')
plt.grid(True, alpha=0.3)
plt.xlim(0, 50)  # Focus on 0-50 Hz
plt.show()
```

**Expected output:**
```
frequencies: array([0.0, 0.98, 1.95, ..., 500.0]) Hz
power: array([1.2e6, 8.3e5, 4.1e5, ..., 2.3e-2]) N²/Hz

Plot shows:
- High power at low frequencies (0-10 Hz) - signal content
- Decreasing power above 15 Hz - noise dominates
```

### Parameters

```python
frequencies, power = laban.psd(
    signal,          # 1D array
    freq,            # Sampling frequency (Hz)
    window='hann',   # Window function: 'hann', 'hamming', 'blackman'
    nperseg=1024     # Segment length (affects frequency resolution)
)
```

**window**: Windowing reduces spectral leakage
- `'hann'` - Good general purpose (default)
- `'hamming'` - Slightly better frequency resolution
- `'blackman'` - Better sidelobe suppression

**nperseg**: Segment length for Welch's method
- Larger = better frequency resolution, worse variance
- Smaller = worse resolution, better variance
- Typical: 1024-2048 for biomechanical signals

### Interpreting PSDs

```python
# Load marker trajectory
marker = laban.Point3D.from_tdf("walking.tdf", marker_name="ankle_L")
y_pos = marker['y'].data / 1000  # Convert mm → m
freq = marker.sampling_frequency

# Calculate PSD
freqs, psd_y = laban.psd(y_pos, freq=freq, nperseg=2048)

# Find frequency at which 95% of power is below
cumulative_power = np.cumsum(psd_y)
total_power = cumulative_power[-1]
freq_95 = freqs[np.where(cumulative_power >= 0.95 * total_power)[0][0]]

print(f"95% of signal power below: {freq_95:.1f} Hz")
# Output: 95% of signal power below: 6.2 Hz
# → Use cut-off = 6-8 Hz for this marker

# Find dominant frequency
dominant_idx = np.argmax(psd_y[1:]) + 1  # Skip DC component
dominant_freq = freqs[dominant_idx]

print(f"Dominant frequency: {dominant_freq:.2f} Hz")
# Output: Dominant frequency: 1.83 Hz (gait cadence)
```

### Comparing Signals

```python
# Compare raw vs. filtered signal
fz_raw = fp.force['Fz'].data
fz_filt = laban.butterworth_filt(fz_raw, freq=freq, cut=10, order=4)

# Calculate PSDs
freqs, psd_raw = laban.psd(fz_raw, freq=freq)
_, psd_filt = laban.psd(fz_filt, freq=freq)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(freqs, psd_raw, label='Raw', alpha=0.7)
plt.semilogy(freqs, psd_filt, label='Filtered (10 Hz)', alpha=0.7)
plt.axvline(10, color='red', linestyle='--', label='Cut-off')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (N²/Hz)')
plt.legend()
plt.xlim(0, 50)
plt.grid(True, alpha=0.3)
plt.title('PSD Comparison')

plt.subplot(1, 2, 2)
plt.semilogy(freqs, psd_raw - psd_filt, color='red')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Removed Power (N²/Hz)')
plt.xlim(0, 50)
plt.grid(True, alpha=0.3)
plt.title('Removed by Filter')

plt.tight_layout()
plt.show()

# Quantify preserved vs. removed power
power_below_cutoff = np.trapz(psd_raw[freqs <= 10], freqs[freqs <= 10])
power_above_cutoff = np.trapz(psd_raw[freqs > 10], freqs[freqs > 10])
total = power_below_cutoff + power_above_cutoff

print(f"Power preserved: {100 * power_below_cutoff / total:.1f}%")
print(f"Power removed: {100 * power_above_cutoff / total:.1f}%")
# Output:
# Power preserved: 94.3%
# Power removed: 5.7%
```

## Residual Analysis

Systematic method to find optimal low-pass filter cut-off by analyzing RMS difference between filtered versions.

### Basic Usage

```python
# Load signal
marker = laban.Point3D.from_tdf("walking.tdf", marker_name="C7")
y_pos = marker['y'].data
freq = marker.sampling_frequency

# Run residual analysis
cutoffs, rms_diffs = laban.residual_analysis(
    signal=y_pos,
    freq=freq,
    cutoff_range=(3, 20),  # Test cut-offs from 3 to 20 Hz
    step=0.5,              # 0.5 Hz increments
    order=4                # 4th order Butterworth
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(cutoffs, rms_diffs, 'o-', markersize=4)
plt.xlabel('Cut-off Frequency (Hz)')
plt.ylabel('RMS Difference (mm)')
plt.title('Residual Analysis - C7 Vertical Position')
plt.grid(True, alpha=0.3)
plt.show()

# Find elbow (optimal cut-off)
# Look for where curve flattens
diffs_2nd = np.diff(rms_diffs, n=2)
elbow_idx = np.argmin(diffs_2nd) + 2
optimal_cutoff = cutoffs[elbow_idx]

print(f"Suggested cut-off: {optimal_cutoff:.1f} Hz")
# Output: Suggested cut-off: 6.5 Hz
```

**Expected plot shape:**
```
RMS Difference
     │     
 1.5 │    ╱────────  ← Plateau (noise only)
     │   ╱
 1.0 │  ╱
     │ ╱              ← Elbow (optimal cut-off)
 0.5 │╱
     │
 0.0 └─────────────
     3    10    20  Cut-off (Hz)
```

### How It Works

Residual analysis compares signals filtered at adjacent cut-off frequencies:

1. Filter signal at cut-off `fc`
2. Filter signal at cut-off `fc + Δf`
3. Calculate RMS difference between the two
4. Repeat for range of cut-offs

**Interpretation:**
- **Low cut-offs**: Large RMS difference (removing signal content)
- **Optimal range**: Moderate RMS difference (transition from signal to noise)
- **High cut-offs**: Small RMS difference (removing only noise)

The "elbow" indicates the cut-off above which you're only removing noise.

### Parameters

```python
cutoffs, rms_diffs = laban.residual_analysis(
    signal,                 # 1D array
    freq,                   # Sampling frequency (Hz)
    cutoff_range=(3, 20),   # (min, max) cut-off to test
    step=0.5,               # Increment between cut-offs
    order=4,                # Butterworth filter order
    filt_type='low'         # Filter type ('low', 'high', 'band')
)
```

### Finding the Elbow Automatically

```python
def find_elbow(cutoffs, rms_diffs):
    """
    Find elbow point using second derivative.
    """
    # Normalize
    rms_norm = (rms_diffs - rms_diffs.min()) / (rms_diffs.max() - rms_diffs.min())
    
    # Second derivative
    d2 = np.diff(rms_norm, n=2)
    
    # Find minimum (maximum curvature)
    elbow_idx = np.argmin(d2) + 2
    
    return cutoffs[elbow_idx]

# Use it
optimal = find_elbow(cutoffs, rms_diffs)
print(f"Optimal cut-off: {optimal:.1f} Hz")

# Plot with annotation
plt.figure(figsize=(10, 6))
plt.plot(cutoffs, rms_diffs, 'o-', markersize=4)
plt.axvline(optimal, color='red', linestyle='--', 
            label=f'Optimal: {optimal:.1f} Hz')
plt.xlabel('Cut-off Frequency (Hz)')
plt.ylabel('RMS Difference (mm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Residual Analysis with Optimal Cut-off')
plt.show()
```

## Complete Workflow Example

### Determine Cut-off for Force Platform Data

```python
import labanalysis as laban
import numpy as np
import matplotlib.pyplot as plt

# Load force platform
fp = laban.ForcePlatform.from_tdf("cmj.tdf", platform_name="FP1")
fz = fp.force['Fz'].data
freq = fp.sampling_frequency

print("=== Force Platform Cut-off Selection ===\n")

# Step 1: Visual inspection with PSD
print("Step 1: Power Spectral Density")
freqs, psd = laban.psd(fz, freq=freq, nperseg=2048)

# Find 99% power threshold
cumulative = np.cumsum(psd)
freq_99 = freqs[np.where(cumulative >= 0.99 * cumulative[-1])[0][0]]
print(f"  99% power below: {freq_99:.1f} Hz")

# Step 2: Residual analysis
print("\nStep 2: Residual Analysis")
cutoffs, rms = laban.residual_analysis(
    fz,
    freq=freq,
    cutoff_range=(5, 25),
    step=0.5,
    order=4
)

optimal = find_elbow(cutoffs, rms)
print(f"  Optimal cut-off: {optimal:.1f} Hz")

# Step 3: Compare candidates
print("\nStep 3: Compare Candidate Cut-offs")
candidates = [8, 10, 12, 15]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, cut in enumerate(candidates):
    ax = axes[idx // 2, idx % 2]
    
    # Filter
    fz_filt = laban.butterworth_filt(fz, freq=freq, cut=cut, order=4)
    
    # Calculate difference
    diff = fz - fz_filt
    rms_val = np.sqrt(np.mean(diff**2))
    
    # Plot segment
    t = np.arange(len(fz)) / freq
    segment = slice(1000, 3000)  # 2-second window
    
    ax.plot(t[segment], fz[segment], 'gray', alpha=0.5, label='Raw')
    ax.plot(t[segment], fz_filt[segment], 'blue', linewidth=2, label='Filtered')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title(f'Cut-off: {cut} Hz (RMS diff: {rms_val:.2f} N)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Step 4: Final recommendation
print("\n=== Recommendation ===")
print(f"Based on analysis:")
print(f"  PSD suggests: <{freq_99:.1f} Hz")
print(f"  Residual analysis suggests: {optimal:.1f} Hz")
print(f"\nRecommended cut-off: {optimal:.1f} Hz")
print(f"  (preserves signal, removes high-frequency noise)")

# Output:
# === Force Platform Cut-off Selection ===
# 
# Step 1: Power Spectral Density
#   99% power below: 12.3 Hz
# 
# Step 2: Residual Analysis
#   Optimal cut-off: 10.0 Hz
# 
# Step 3: Compare Candidate Cut-offs
# [Plots shown]
# 
# === Recommendation ===
# Based on analysis:
#   PSD suggests: <12.3 Hz
#   Residual analysis suggests: 10.0 Hz
# 
# Recommended cut-off: 10.0 Hz
#   (preserves signal, removes high-frequency noise)
```

### Determine Cut-off for Marker Data

```python
# Load marker
marker = laban.Point3D.from_tdf("gait.tdf", marker_name="ankle_L")
y = marker['y'].data / 1000  # mm → m
freq = marker.sampling_frequency

# Residual analysis
cutoffs, rms = laban.residual_analysis(
    y,
    freq=freq,
    cutoff_range=(3, 15),
    step=0.25,
    order=4
)

# Find elbow
optimal = find_elbow(cutoffs, rms)

# Validate with PSD
freqs, psd = laban.psd(y, freq=freq)
freq_95 = freqs[np.where(np.cumsum(psd) >= 0.95 * np.sum(psd))[0][0]]

print(f"Marker vertical position:")
print(f"  Residual analysis: {optimal:.1f} Hz")
print(f"  PSD (95% power): {freq_95:.1f} Hz")
print(f"\nRecommended: {optimal:.1f} Hz")

# Output:
# Marker vertical position:
#   Residual analysis: 6.0 Hz
#   PSD (95% power): 6.3 Hz
# 
# Recommended: 6.0 Hz
```

## Signal-Specific Guidelines

### Force Platform Signals

**Typical frequency content:**
- Quiet standing: 0-2 Hz
- Jumping: 0-15 Hz
- Running/gait: 0-20 Hz

**Recommended cut-offs:**
```python
# Quiet standing / balance
cut = 10  # Hz

# Jumping (CMJ, SJ, DJ)
cut = 12  # Hz

# Running / gait
cut = 15  # Hz

# Always verify with residual analysis!
```

### Motion Capture Markers

**Typical frequency content:**
- Walking: 0-6 Hz
- Running: 0-10 Hz
- Jumping: 0-12 Hz

**Recommended cut-offs:**
```python
# Walking
cut = 6  # Hz (Winter 2009 recommendation)

# Running
cut = 8  # Hz

# Jumping
cut = 10  # Hz

# High-speed movements
cut = 15  # Hz
```

### EMG Signals

**Different approach - band-pass filtering:**
```python
# EMG requires band-pass, not low-pass
# Typical: 20-450 Hz

# Use PSD to verify no line noise (50/60 Hz)
emg = laban.EMGSignal.from_tdf("trial.tdf", column="biceps_R")
freqs, psd_emg = laban.psd(emg.data, freq=emg.sampling_frequency)

# Check for 50 Hz line noise
noise_band = (freqs >= 48) & (freqs <= 52)
noise_power = np.mean(psd_emg[noise_band])
signal_power = np.mean(psd_emg[(freqs >= 20) & (freqs <= 450)])

if noise_power > 0.1 * signal_power:
    print("Warning: Significant 50 Hz line noise detected")
    print("Consider notch filter at 50 Hz")
```

## Advanced: Custom Frequency Analysis

### Multi-Signal Coherence

Check if two signals share frequency content (e.g., bilateral markers):

```python
from scipy import signal as sp_signal

# Load bilateral markers
ankle_L = laban.Point3D.from_tdf("gait.tdf", marker_name="ankle_L")
ankle_R = laban.Point3D.from_tdf("gait.tdf", marker_name="ankle_R")

y_L = ankle_L['y'].data
y_R = ankle_R['y'].data
freq = ankle_L.sampling_frequency

# Calculate coherence
freqs, coherence = sp_signal.coherence(
    y_L,
    y_R,
    fs=freq,
    nperseg=1024
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(freqs, coherence)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence')
plt.title('Left-Right Ankle Coherence')
plt.xlim(0, 10)
plt.grid(True, alpha=0.3)
plt.axhline(0.5, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()

# Find dominant coherent frequency
coherent_freqs = freqs[coherence > 0.7]
print(f"High coherence (>0.7) at: {coherent_freqs[0]:.2f}-{coherent_freqs[-1]:.2f} Hz")
# Output: High coherence (>0.7) at: 1.56-2.14 Hz (gait frequency)
```

### Spectrogram (Time-Frequency Analysis)

Visualize how frequency content changes over time:

```python
from scipy import signal as sp_signal

# Load long trial
fp = laban.ForcePlatform.from_tdf("balance_60s.tdf", platform_name="FP1")
fz = fp.force['Fz'].data
freq = fp.sampling_frequency

# Calculate spectrogram
f, t, Sxx = sp_signal.spectrogram(
    fz,
    fs=freq,
    nperseg=2048,
    noverlap=1536
)

# Plot
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Force Platform Spectrogram')
plt.ylim(0, 20)
plt.colorbar(label='Power (dB/Hz)')
plt.show()

# Useful for detecting:
# - Fatigue effects (frequency shift over time)
# - Transient events
# - Non-stationary signals
```

## Troubleshooting

### PSD shows unexpected peaks

```python
# Check for:
# 1. Line noise (50/60 Hz)
freqs, psd = laban.psd(signal, freq=freq)

if psd[np.abs(freqs - 50) < 0.5].max() > 10 * np.median(psd):
    print("50 Hz line noise detected - use notch filter")

# 2. Aliasing (power at Nyquist frequency)
if psd[-10:].mean() > 0.01 * psd.max():
    print("Possible aliasing - check if sampling frequency is sufficient")

# 3. DC offset (very high power at 0 Hz)
if psd[0] > 100 * psd[1:10].mean():
    print("Large DC offset - consider high-pass filtering")
```

### Residual analysis unclear

```python
# If elbow is not obvious:
# 1. Try different step size
cutoffs_fine, rms_fine = laban.residual_analysis(
    signal,
    freq=freq,
    cutoff_range=(5, 15),
    step=0.25  # Finer resolution
)

# 2. Compare with PSD
freqs, psd = laban.psd(signal, freq=freq)
cumsum = np.cumsum(psd)
freq_95 = freqs[np.where(cumsum >= 0.95 * cumsum[-1])[0][0]]
print(f"PSD suggests cut-off around: {freq_95:.1f} Hz")

# 3. Use conservative approach
# If uncertain, choose higher cut-off (preserve more signal)
```

## Best Practices

1. **Always visualize before filtering**
   - Calculate PSD
   - Check for unexpected content
   - Identify noise vs. signal regions

2. **Use residual analysis for systematic selection**
   - Don't guess cut-offs
   - Document your choice
   - Show the residual analysis plot in reports

3. **Validate your filter**
   - Compare PSDs before/after
   - Check time-domain signal visually
   - Verify metrics are not affected

4. **Be conservative**
   - When in doubt, use higher cut-off
   - Preserving signal > aggressive noise removal
   - Can always re-filter with lower cut-off

5. **Document signal-specific cut-offs**
   - Different signals need different cut-offs
   - Keep analysis logs
   - Use consistent cut-offs within study

## See Also

- **[Filtering](filtering.md)** - Apply filters after selecting cut-off
- **[Derivatives](derivatives.md)** - Filter before differentiation
- **[Peak Detection](peak-detection.md)** - Filtering improves peak detection
- **[API Reference: psd()](../../api/signalprocessing.md#psd)** - PSD function
- **[API Reference: residual_analysis()](../../api/signalprocessing.md#residual_analysis)** - Residual analysis function

---

**Module**: `src/labanalysis/signalprocessing.py`  
**Key Functions**: `psd()`, `residual_analysis()`  
**Reference**: Winter DA. Biomechanics and Motor Control of Human Movement. 4th ed. 2009. (Chapter 2: Signal Processing)
