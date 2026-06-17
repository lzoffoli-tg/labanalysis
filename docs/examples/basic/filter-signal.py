"""
Filter Signal Example
=====================

Demonstrates signal filtering and comparison workflow:
1. Load force platform data
2. Apply different filter types
3. Compare original vs filtered signals
4. Visualize filtering effects
5. Choose optimal filter parameters

Common use case: Remove noise from force/marker data while preserving signal features.
"""

import labanalysis as laban
from labanalysis.signalprocessing import (
    butterworth_filter,
    median_filter,
    running_mean_filter,
    power_spectral_density
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def main():
    # ===== 1. LOAD DATA =====
    print("Loading force platform data...")

    # Load force platform from BTS file
    fp = laban.ForcePlatform.from_tdf_file(
        "path/to/your/file.tdf",
        fp_label="FP1"  # Force platform label
    )

    # Extract vertical ground reaction force (GRF)
    grf_z = fp["FORCE", "Z"]  # Vertical component

    print(f"Loaded {len(grf_z)} samples")
    print(f"Duration: {grf_z.index[-1]:.2f} seconds")
    print(f"Sampling rate: {1/np.mean(np.diff(grf_z.index)):.1f} Hz")


    # ===== 2. APPLY DIFFERENT FILTERS =====
    print("\nApplying filters...")

    # Butterworth low-pass filter (most common for biomechanics)
    grf_butter_6hz = butterworth_filter(
        grf_z,
        frequency=6,  # Cutoff frequency in Hz
        order=4,      # Filter order (higher = sharper cutoff)
        type='low'
    )

    grf_butter_10hz = butterworth_filter(
        grf_z,
        frequency=10,
        order=4,
        type='low'
    )

    # Median filter (good for spike removal)
    grf_median = median_filter(
        grf_z,
        window=5  # Window size (samples)
    )

    # Running mean filter (simple averaging)
    grf_running = running_mean_filter(
        grf_z,
        window=10  # Window size (samples)
    )


    # ===== 3. FREQUENCY ANALYSIS (OPTIONAL) =====
    print("\nAnalyzing frequency content...")

    # Compute power spectral density (PSD) to see frequency components
    freq, psd = power_spectral_density(grf_z)

    # Find 99% power frequency (signal vs noise threshold)
    cumulative_power = np.cumsum(psd) / np.sum(psd)
    freq_99 = freq[np.where(cumulative_power >= 0.99)[0][0]]

    print(f"99% power frequency: {freq_99:.1f} Hz")
    print(f"Suggested cutoff: {freq_99:.1f} - {freq_99*1.5:.1f} Hz")


    # ===== 4. CREATE COMPARISON PLOTS =====
    print("\nCreating comparison plots...")

    # Plot 1: Time-domain comparison (original vs filters)
    fig1 = go.Figure()

    # Original signal
    fig1.add_trace(
        go.Scatter(
            x=grf_z.index,
            y=grf_z.to_numpy(),
            mode='lines',
            name='Original',
            line=dict(color='lightgray', width=1),
            opacity=0.7
        )
    )

    # Butterworth 6 Hz
    fig1.add_trace(
        go.Scatter(
            x=grf_butter_6hz.index,
            y=grf_butter_6hz.to_numpy(),
            mode='lines',
            name='Butterworth 6 Hz',
            line=dict(color='blue', width=2)
        )
    )

    # Butterworth 10 Hz
    fig1.add_trace(
        go.Scatter(
            x=grf_butter_10hz.index,
            y=grf_butter_10hz.to_numpy(),
            mode='lines',
            name='Butterworth 10 Hz',
            line=dict(color='red', width=2)
        )
    )

    # Median filter
    fig1.add_trace(
        go.Scatter(
            x=grf_median.index,
            y=grf_median.to_numpy(),
            mode='lines',
            name='Median (window=5)',
            line=dict(color='green', width=2, dash='dash')
        )
    )

    fig1.update_layout(
        title="Filter Comparison - Time Domain",
        xaxis_title="Time (s)",
        yaxis_title="Vertical GRF (N)",
        template='plotly_white',
        hovermode='x unified'
    )


    # Plot 2: Zoomed view to see filtering effects
    # Select a 1-second window
    t_start = 2.0  # Start time
    t_end = 3.0    # End time

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=grf_z.index,
            y=grf_z.to_numpy(),
            mode='lines',
            name='Original',
            line=dict(color='black', width=1)
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=grf_butter_6hz.index,
            y=grf_butter_6hz.to_numpy(),
            mode='lines',
            name='Butterworth 6 Hz',
            line=dict(color='blue', width=2)
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=grf_butter_10hz.index,
            y=grf_butter_10hz.to_numpy(),
            mode='lines',
            name='Butterworth 10 Hz',
            line=dict(color='red', width=2)
        )
    )

    fig2.update_layout(
        title="Filter Comparison - Zoomed View (1 second window)",
        xaxis_title="Time (s)",
        yaxis_title="Vertical GRF (N)",
        xaxis_range=[t_start, t_end],
        template='plotly_white',
        hovermode='x unified'
    )


    # Plot 3: Power spectral density (frequency domain)
    fig3 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Original Signal PSD', 'Filtered Signal PSD'),
        vertical_spacing=0.12
    )

    # Original PSD
    freq_orig, psd_orig = power_spectral_density(grf_z)
    fig3.add_trace(
        go.Scatter(
            x=freq_orig,
            y=psd_orig,
            mode='lines',
            name='Original',
            line=dict(color='black', width=2),
            fill='tozeroy'
        ),
        row=1, col=1
    )

    # Filtered PSD
    freq_filt, psd_filt = power_spectral_density(grf_butter_6hz)
    fig3.add_trace(
        go.Scatter(
            x=freq_filt,
            y=psd_filt,
            mode='lines',
            name='Butterworth 6 Hz',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )

    # Add vertical line at cutoff frequency
    fig3.add_vline(x=6, line_dash="dash", line_color="red", row=2, col=1)
    fig3.add_annotation(
        x=6, y=psd_filt.max()*0.8,
        text="Cutoff: 6 Hz",
        showarrow=True,
        arrowhead=2,
        row=2, col=1
    )

    fig3.update_xaxes(title_text="Frequency (Hz)", range=[0, 50])
    fig3.update_yaxes(title_text="Power")

    fig3.update_layout(
        title="Frequency Domain Analysis",
        height=800,
        template='plotly_white',
        showlegend=False
    )


    # ===== 5. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()
    fig2.show()
    fig3.show()


    # ===== 6. SAVE FILTERED DATA =====
    print("\nSaving filtered signal...")

    # Export to DataFrame
    import pandas as pd

    df = pd.DataFrame({
        'time': grf_z.index,
        'original': grf_z.to_numpy(),
        'butterworth_6hz': grf_butter_6hz.to_numpy(),
        'butterworth_10hz': grf_butter_10hz.to_numpy(),
        'median': grf_median.to_numpy()
    })

    df.to_csv("filtered_signal.csv", index=False)
    print("✓ Saved: filtered_signal.csv")


    # ===== 7. QUANTIFY FILTERING EFFECTS =====
    print("\n===== FILTERING EFFECTS =====")

    # Calculate noise reduction (RMS of difference)
    def rms_difference(signal1, signal2):
        diff = signal1.to_numpy() - signal2.to_numpy()
        return np.sqrt(np.mean(diff**2))

    print(f"\nRMS difference from original:")
    print(f"  Butterworth 6 Hz:  {rms_difference(grf_z, grf_butter_6hz):.2f} N")
    print(f"  Butterworth 10 Hz: {rms_difference(grf_z, grf_butter_10hz):.2f} N")
    print(f"  Median filter:     {rms_difference(grf_z, grf_median):.2f} N")

    # Calculate signal-to-noise ratio improvement
    original_std = grf_z.to_numpy().std()
    filtered_std = grf_butter_6hz.to_numpy().std()
    noise_reduction_pct = (1 - filtered_std/original_std) * 100

    print(f"\nNoise reduction (Butterworth 6 Hz): {noise_reduction_pct:.1f}%")


    # ===== 8. RECOMMENDATIONS =====
    print("\n===== RECOMMENDATIONS =====")
    print(f"Based on PSD analysis:")
    print(f"  - 99% power frequency: {freq_99:.1f} Hz")
    print(f"  - Recommended cutoff: {freq_99:.1f} - {freq_99*1.5:.1f} Hz")
    print(f"\nStandard cutoff frequencies for biomechanics:")
    print(f"  - Force platforms: 10-15 Hz")
    print(f"  - Marker data: 6-10 Hz")
    print(f"  - EMG (after rectification): 5-10 Hz")
    print(f"\nChosen filter: Butterworth 6 Hz (good balance of smoothing vs preservation)")


if __name__ == "__main__":
    main()
