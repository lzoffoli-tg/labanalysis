"""
Marker Tracking Example
========================

Demonstrates marker trajectory analysis and quality assessment:
1. Load 3D marker data
2. Assess data quality (gaps, noise, outliers)
3. Clean and interpolate missing data
4. Calculate marker velocities and accelerations
5. Visualize 3D trajectories
6. Compute derived metrics (displacement, path length)

Common use case: Motion capture quality control, marker trajectory analysis.
"""

import labanalysis as laban
from labanalysis.signalprocessing import (
    butterworth_filter,
    fillna,
    derivative,
    median_filter
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def main():
    # ===== 1. LOAD MARKER DATA =====
    print("Loading marker data...")

    # Load WholeBody markers
    body = laban.WholeBody.from_tdf_file(
        "path/to/your/file.tdf",
        labels="LABEL"
    )

    print(f"Loaded {len(body.labels)} markers")
    print(f"Sampling rate: {1/np.mean(np.diff(body.index)):.1f} Hz")
    print(f"Duration: {body.index[-1]:.2f} s")


    # ===== 2. SELECT MARKER FOR ANALYSIS =====
    print("\nSelecting marker for detailed analysis...")

    # Example: Track the left knee lateral marker
    marker_label = "knee_L_lat"
    marker = body.get(marker_label)

    if marker is None:
        print(f"ERROR: Marker '{marker_label}' not found")
        print(f"Available markers: {body.labels[:20]}...")
        return

    print(f"Analyzing marker: {marker_label}")


    # ===== 3. ASSESS DATA QUALITY =====
    print("\nAssessing data quality...")

    # Check for missing data (NaN values)
    marker_array = marker.to_numpy()  # Shape: (n_samples, 3) for X, Y, Z

    # Count NaNs per axis
    nan_counts = {
        'X': np.isnan(marker_array[:, 0]).sum(),
        'Y': np.isnan(marker_array[:, 1]).sum(),
        'Z': np.isnan(marker_array[:, 2]).sum()
    }

    total_samples = len(marker_array)
    print(f"\nMissing data:")
    print(f"  X: {nan_counts['X']} / {total_samples} ({nan_counts['X']/total_samples*100:.1f}%)")
    print(f"  Y: {nan_counts['Y']} / {total_samples} ({nan_counts['Y']/total_samples*100:.1f}%)")
    print(f"  Z: {nan_counts['Z']} / {total_samples} ({nan_counts['Z']/total_samples*100:.1f}%)")


    # Find gap lengths
    def find_gaps(data):
        """Find consecutive NaN sequences."""
        is_nan = np.isnan(data)
        gaps = []

        if is_nan.any():
            diff = np.diff(np.concatenate([[0], is_nan.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for start, end in zip(starts, ends):
                gaps.append((start, end - start))

        return gaps

    gaps_z = find_gaps(marker_array[:, 2])
    if gaps_z:
        max_gap = max(gap[1] for gap in gaps_z)
        print(f"\nLargest gap: {max_gap} samples ({max_gap/100:.2f} s at 100 Hz)")
    else:
        print("\nNo gaps detected")


    # ===== 4. CLEAN DATA =====
    print("\nCleaning data...")

    # Step 1: Remove outliers with median filter
    marker_cleaned = median_filter(marker, window=3)

    # Step 2: Interpolate missing data
    marker_filled = fillna(
        marker_cleaned,
        method='cubic',  # Cubic spline interpolation
        axis=0
    )

    # Step 3: Apply low-pass filter to reduce noise
    marker_smooth = butterworth_filter(
        marker_filled,
        frequency=6,  # Hz (standard for marker data)
        order=4
    )

    print("✓ Data cleaned: outliers removed, gaps filled, noise filtered")


    # ===== 5. CALCULATE VELOCITIES =====
    print("\nCalculating velocities...")

    # Compute velocity using Winter's derivative method
    velocity = derivative(
        marker_smooth,
        order=1,
        method='winter'
    )

    # Velocity magnitude
    vel_array = velocity.to_numpy()
    velocity_mag = np.sqrt(vel_array[:, 0]**2 + vel_array[:, 1]**2 + vel_array[:, 2]**2)

    max_velocity = velocity_mag.max()
    avg_velocity = velocity_mag.mean()

    print(f"Max velocity: {max_velocity:.2f} m/s")
    print(f"Average velocity: {avg_velocity:.3f} m/s")


    # ===== 6. CALCULATE ACCELERATIONS =====
    print("\nCalculating accelerations...")

    # Compute acceleration (2nd derivative)
    acceleration = derivative(
        marker_smooth,
        order=2,
        method='winter'
    )

    # Acceleration magnitude
    acc_array = acceleration.to_numpy()
    acceleration_mag = np.sqrt(acc_array[:, 0]**2 + acc_array[:, 1]**2 + acc_array[:, 2]**2)

    max_acceleration = acceleration_mag.max()
    print(f"Max acceleration: {max_acceleration:.2f} m/s²")


    # ===== 7. COMPUTE DERIVED METRICS =====
    print("\nComputing trajectory metrics...")

    # Path length (total distance traveled)
    diff = np.diff(marker_smooth.to_numpy(), axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    path_length = np.sum(segment_lengths)

    print(f"Path length: {path_length:.3f} m")

    # Displacement (straight-line distance from start to end)
    start_pos = marker_smooth.to_numpy()[0]
    end_pos = marker_smooth.to_numpy()[-1]
    displacement = np.linalg.norm(end_pos - start_pos)

    print(f"Displacement: {displacement:.3f} m")
    print(f"Path efficiency: {displacement/path_length*100:.1f}%")

    # Range of motion (ROM) in each axis
    rom_x = marker_smooth["X"].to_numpy().max() - marker_smooth["X"].to_numpy().min()
    rom_y = marker_smooth["Y"].to_numpy().max() - marker_smooth["Y"].to_numpy().min()
    rom_z = marker_smooth["Z"].to_numpy().max() - marker_smooth["Z"].to_numpy().min()

    print(f"\nRange of motion:")
    print(f"  X (A-P): {rom_x*1000:.1f} mm")
    print(f"  Y (M-L): {rom_y*1000:.1f} mm")
    print(f"  Z (Vertical): {rom_z*1000:.1f} mm")


    # ===== 8. CREATE VISUALIZATIONS =====
    print("\nCreating visualizations...")

    # Plot 1: 3D trajectory
    fig1 = go.Figure()

    # Original trajectory
    fig1.add_trace(
        go.Scatter3d(
            x=marker["X"].to_numpy(),
            y=marker["Y"].to_numpy(),
            z=marker["Z"].to_numpy(),
            mode='lines',
            name='Original',
            line=dict(color='lightgray', width=2),
            opacity=0.5
        )
    )

    # Smoothed trajectory
    fig1.add_trace(
        go.Scatter3d(
            x=marker_smooth["X"].to_numpy(),
            y=marker_smooth["Y"].to_numpy(),
            z=marker_smooth["Z"].to_numpy(),
            mode='lines',
            name='Filtered',
            line=dict(color='blue', width=4)
        )
    )

    # Mark start and end points
    fig1.add_trace(
        go.Scatter3d(
            x=[start_pos[0]], y=[start_pos[1]], z=[start_pos[2]],
            mode='markers',
            name='Start',
            marker=dict(size=8, color='green')
        )
    )

    fig1.add_trace(
        go.Scatter3d(
            x=[end_pos[0]], y=[end_pos[1]], z=[end_pos[2]],
            mode='markers',
            name='End',
            marker=dict(size=8, color='red')
        )
    )

    fig1.update_layout(
        title=f"3D Trajectory: {marker_label}",
        scene=dict(
            xaxis_title='X - Anterior (m)',
            yaxis_title='Y - Lateral (m)',
            zaxis_title='Z - Vertical (m)',
            aspectmode='data'
        ),
        template='plotly_white'
    )


    # Plot 2: Position components over time
    fig2 = make_subplots(
        rows=3, cols=1,
        subplot_titles=('X Position', 'Y Position', 'Z Position'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )

    # X component
    fig2.add_trace(
        go.Scatter(x=marker.index, y=marker["X"].to_numpy(),
                   name='Original', line=dict(color='lightgray', width=1)),
        row=1, col=1
    )
    fig2.add_trace(
        go.Scatter(x=marker_smooth.index, y=marker_smooth["X"].to_numpy(),
                   name='Filtered', line=dict(color='red', width=2)),
        row=1, col=1
    )

    # Y component
    fig2.add_trace(
        go.Scatter(x=marker.index, y=marker["Y"].to_numpy(),
                   name='Original', line=dict(color='lightgray', width=1), showlegend=False),
        row=2, col=1
    )
    fig2.add_trace(
        go.Scatter(x=marker_smooth.index, y=marker_smooth["Y"].to_numpy(),
                   name='Filtered', line=dict(color='green', width=2), showlegend=False),
        row=2, col=1
    )

    # Z component
    fig2.add_trace(
        go.Scatter(x=marker.index, y=marker["Z"].to_numpy(),
                   name='Original', line=dict(color='lightgray', width=1), showlegend=False),
        row=3, col=1
    )
    fig2.add_trace(
        go.Scatter(x=marker_smooth.index, y=marker_smooth["Z"].to_numpy(),
                   name='Filtered', line=dict(color='blue', width=2), showlegend=False),
        row=3, col=1
    )

    # Highlight gaps
    for start_idx, length in gaps_z:
        t_start = marker.index[start_idx]
        t_end = marker.index[min(start_idx + length, len(marker.index) - 1)]
        fig2.add_vrect(
            x0=t_start, x1=t_end,
            fillcolor="red", opacity=0.2,
            layer="below", line_width=0,
            row=3, col=1
        )

    fig2.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig2.update_yaxes(title_text="Position (m)")

    fig2.update_layout(
        title=f"Position Components: {marker_label}",
        height=800,
        template='plotly_white',
        hovermode='x unified'
    )


    # Plot 3: Velocity magnitude
    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            x=velocity.index,
            y=velocity_mag,
            mode='lines',
            name='Velocity',
            line=dict(color='purple', width=2),
            fill='tozeroy'
        )
    )

    fig3.add_hline(y=avg_velocity, line_dash="dash", line_color="red",
                   annotation_text=f"Average: {avg_velocity:.3f} m/s")

    fig3.update_layout(
        title=f"Velocity Magnitude: {marker_label}",
        xaxis_title="Time (s)",
        yaxis_title="Velocity (m/s)",
        template='plotly_white'
    )


    # ===== 9. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()
    fig2.show()
    fig3.show()


    # ===== 10. EXPORT RESULTS =====
    print("\nExporting results...")

    # Summary metrics
    summary = pd.DataFrame({
        'Metric': [
            'Path Length (m)',
            'Displacement (m)',
            'Path Efficiency (%)',
            'Max Velocity (m/s)',
            'Avg Velocity (m/s)',
            'Max Acceleration (m/s²)',
            'ROM X (mm)',
            'ROM Y (mm)',
            'ROM Z (mm)',
            'Missing Data (%)'
        ],
        'Value': [
            round(path_length, 3),
            round(displacement, 3),
            round(displacement/path_length*100, 1),
            round(max_velocity, 2),
            round(avg_velocity, 3),
            round(max_acceleration, 2),
            round(rom_x*1000, 1),
            round(rom_y*1000, 1),
            round(rom_z*1000, 1),
            round(nan_counts['Z']/total_samples*100, 1)
        ]
    })

    summary.to_csv(f"{marker_label}_analysis.csv", index=False)
    print(f"✓ Saved: {marker_label}_analysis.csv")

    # Export cleaned trajectory
    df_trajectory = pd.DataFrame({
        'Time': marker_smooth.index,
        'X': marker_smooth["X"].to_numpy(),
        'Y': marker_smooth["Y"].to_numpy(),
        'Z': marker_smooth["Z"].to_numpy(),
        'Velocity_X': vel_array[:, 0],
        'Velocity_Y': vel_array[:, 1],
        'Velocity_Z': vel_array[:, 2],
        'Velocity_Mag': velocity_mag
    })

    df_trajectory.to_csv(f"{marker_label}_trajectory.csv", index=False)
    print(f"✓ Saved: {marker_label}_trajectory.csv")


    print("\n===== QUALITY ASSESSMENT =====")
    if max(nan_counts.values()) / total_samples < 0.05:
        print("✓ Data quality: GOOD (< 5% missing)")
    elif max(nan_counts.values()) / total_samples < 0.15:
        print("⚠ Data quality: ACCEPTABLE (5-15% missing)")
    else:
        print("✗ Data quality: POOR (> 15% missing)")


if __name__ == "__main__":
    main()
