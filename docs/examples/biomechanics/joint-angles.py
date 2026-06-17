"""
Joint Angles Analysis Example
==============================

Demonstrates joint angle extraction and analysis:
1. Load full-body kinematics
2. Extract joint angles (hip, knee, ankle)
3. Identify gait events from angles
4. Calculate range of motion (ROM)
5. Compare bilateral symmetry
6. Visualize angle profiles

Common use case: Gait analysis, running mechanics, squat assessment.
"""

import labanalysis as laban
from labanalysis.signalprocessing import butterworth_filter, find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def main():
    # ===== 1. LOAD FULL-BODY KINEMATICS =====
    print("Loading full-body kinematics...")

    # Load WholeBody from gait trial
    body = laban.WholeBody.from_tdf_file(
        "path/to/gait_trial.tdf",
        labels="LABEL"
    )

    print(f"Loaded {len(body.labels)} markers")
    print(f"Duration: {body.index[-1]:.2f} seconds")


    # ===== 2. EXTRACT LOWER LIMB JOINT ANGLES =====
    print("\nExtracting joint angles...")

    # Left leg angles
    left_ankle_flexext = body.left_ankle_flexionextension  # Dorsi/plantarflexion
    left_knee_flexext = body.left_knee_flexionextension    # Knee flexion
    left_hip_flexext = body.left_hip_flexionextension      # Hip flexion/extension

    # Right leg angles
    right_ankle_flexext = body.right_ankle_flexionextension
    right_knee_flexext = body.right_knee_flexionextension
    right_hip_flexext = body.right_hip_flexionextension

    # Frontal plane angles
    left_knee_varusvalgus = body.left_knee_varusvalgus   # Knee varus/valgus
    left_hip_abductionadduction = body.left_hip_abductionadduction  # Hip abduction

    print("✓ Extracted 8 joint angles")


    # ===== 3. FILTER ANGLES (REMOVE NOISE) =====
    print("\nFiltering angles...")

    # Apply Butterworth filter (6 Hz cutoff recommended for marker data)
    left_knee_smooth = butterworth_filter(left_knee_flexext, frequency=6, order=4)
    right_knee_smooth = butterworth_filter(right_knee_flexext, frequency=6, order=4)


    # ===== 4. IDENTIFY GAIT EVENTS =====
    print("\nDetecting gait events...")

    # Method 1: Using knee angle peaks (max flexion = swing phase)
    knee_peaks, _ = find_peaks(
        left_knee_smooth.to_numpy(),
        threshold=20,  # Minimum knee flexion (degrees)
        distance=50    # Minimum samples between peaks
    )

    # Convert peak indices to time
    peak_times = body.index[knee_peaks]
    print(f"Found {len(peak_times)} gait cycles (left leg)")

    # Calculate stride time
    if len(peak_times) > 1:
        stride_times = np.diff(peak_times)
        avg_stride_time = np.mean(stride_times)
        cadence = 60 / avg_stride_time  # steps/min

        print(f"Average stride time: {avg_stride_time:.2f} s")
        print(f"Cadence: {cadence:.1f} steps/min")


    # ===== 5. CALCULATE RANGE OF MOTION (ROM) =====
    print("\nCalculating ROM...")

    def calculate_rom(angle_signal):
        """Calculate range of motion (max - min)."""
        return angle_signal.to_numpy().max() - angle_signal.to_numpy().min()

    # ROM for each joint
    rom_data = {
        'Joint': [
            'Left Ankle',
            'Right Ankle',
            'Left Knee',
            'Right Knee',
            'Left Hip',
            'Right Hip'
        ],
        'ROM (deg)': [
            calculate_rom(left_ankle_flexext),
            calculate_rom(right_ankle_flexext),
            calculate_rom(left_knee_flexext),
            calculate_rom(right_knee_flexext),
            calculate_rom(left_hip_flexext),
            calculate_rom(right_hip_flexext)
        ]
    }

    df_rom = pd.DataFrame(rom_data)
    print("\nRange of Motion:")
    print(df_rom.to_string(index=False))


    # ===== 6. BILATERAL SYMMETRY ANALYSIS =====
    print("\nAnalyzing bilateral symmetry...")

    def symmetry_index(left, right):
        """Calculate symmetry index: (L-R) / (0.5*(L+R)) * 100"""
        return (left - right) / (0.5 * (left + right)) * 100

    symmetry = {
        'Joint': ['Ankle', 'Knee', 'Hip'],
        'Left ROM': [
            rom_data['ROM (deg)'][0],
            rom_data['ROM (deg)'][2],
            rom_data['ROM (deg)'][4]
        ],
        'Right ROM': [
            rom_data['ROM (deg)'][1],
            rom_data['ROM (deg)'][3],
            rom_data['ROM (deg)'][5]
        ],
        'Symmetry Index (%)': [
            symmetry_index(rom_data['ROM (deg)'][0], rom_data['ROM (deg)'][1]),
            symmetry_index(rom_data['ROM (deg)'][2], rom_data['ROM (deg)'][3]),
            symmetry_index(rom_data['ROM (deg)'][4], rom_data['ROM (deg)'][5])
        ]
    }

    df_symmetry = pd.DataFrame(symmetry)
    print("\nBilateral Symmetry:")
    print(df_symmetry.to_string(index=False))
    print("\nNote: Symmetry index >10% may indicate asymmetry")


    # ===== 7. EXTRACT SINGLE GAIT CYCLE =====
    print("\nExtracting representative gait cycle...")

    if len(peak_times) >= 2:
        # Use middle gait cycle
        cycle_start = peak_times[len(peak_times)//2]
        cycle_end = peak_times[len(peak_times)//2 + 1]

        # Extract angles for this cycle
        cycle_knee = left_knee_smooth[cycle_start:cycle_end]
        cycle_hip = butterworth_filter(
            left_hip_flexext[cycle_start:cycle_end],
            frequency=6, order=4
        )
        cycle_ankle = butterworth_filter(
            left_ankle_flexext[cycle_start:cycle_end],
            frequency=6, order=4
        )

        # Normalize to 0-100% gait cycle
        gait_percent = np.linspace(0, 100, len(cycle_knee))


    # ===== 8. CREATE VISUALIZATIONS =====
    print("\nCreating visualizations...")

    # Plot 1: Time series of all angles
    fig1 = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Hip Flexion/Extension', 'Knee Flexion/Extension', 'Ankle Dorsi/Plantarflexion'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )

    # Hip angles
    fig1.add_trace(
        go.Scatter(x=body.index, y=left_hip_flexext.to_numpy(),
                   name='Left', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig1.add_trace(
        go.Scatter(x=body.index, y=right_hip_flexext.to_numpy(),
                   name='Right', line=dict(color='red', width=2)),
        row=1, col=1
    )

    # Knee angles
    fig1.add_trace(
        go.Scatter(x=body.index, y=left_knee_smooth.to_numpy(),
                   name='Left', line=dict(color='blue', width=2), showlegend=False),
        row=2, col=1
    )
    fig1.add_trace(
        go.Scatter(x=body.index, y=right_knee_smooth.to_numpy(),
                   name='Right', line=dict(color='red', width=2), showlegend=False),
        row=2, col=1
    )

    # Ankle angles
    fig1.add_trace(
        go.Scatter(x=body.index, y=left_ankle_flexext.to_numpy(),
                   name='Left', line=dict(color='blue', width=2), showlegend=False),
        row=3, col=1
    )
    fig1.add_trace(
        go.Scatter(x=body.index, y=right_ankle_flexext.to_numpy(),
                   name='Right', line=dict(color='red', width=2), showlegend=False),
        row=3, col=1
    )

    # Mark gait events
    for t in peak_times:
        fig1.add_vline(x=t, line_dash="dash", line_color="gray", opacity=0.5)

    fig1.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig1.update_yaxes(title_text="Angle (deg)")

    fig1.update_layout(
        title="Lower Limb Joint Angles - Time Series",
        height=900,
        template='plotly_white',
        hovermode='x unified'
    )


    # Plot 2: Normalized gait cycle (if available)
    if len(peak_times) >= 2:
        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=gait_percent, y=cycle_hip.to_numpy(),
                name='Hip', line=dict(color='red', width=3)
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=gait_percent, y=cycle_knee.to_numpy(),
                name='Knee', line=dict(color='blue', width=3)
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=gait_percent, y=cycle_ankle.to_numpy(),
                name='Ankle', line=dict(color='green', width=3)
            )
        )

        # Mark stance/swing phases (approximate)
        fig2.add_vrect(x0=0, x1=60, fillcolor="lightblue", opacity=0.2,
                       annotation_text="Stance", annotation_position="top left")
        fig2.add_vrect(x0=60, x1=100, fillcolor="lightyellow", opacity=0.2,
                       annotation_text="Swing", annotation_position="top left")

        fig2.update_layout(
            title="Representative Gait Cycle - Left Leg",
            xaxis_title="Gait Cycle (%)",
            yaxis_title="Joint Angle (deg)",
            template='plotly_white',
            hovermode='x unified'
        )

        fig2.show()


    # Plot 3: Frontal plane angles (varus/valgus)
    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            x=body.index,
            y=left_knee_varusvalgus.to_numpy(),
            name='Left Knee',
            line=dict(color='blue', width=2)
        )
    )

    fig3.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

    fig3.update_layout(
        title="Knee Varus/Valgus Angle",
        xaxis_title="Time (s)",
        yaxis_title="Varus (-) / Valgus (+) [deg]",
        template='plotly_white'
    )


    # ===== 9. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()
    fig3.show()


    # ===== 10. EXPORT RESULTS =====
    print("\nExporting results...")

    # Save ROM data
    df_rom.to_csv("joint_rom_analysis.csv", index=False)
    print("✓ Saved: joint_rom_analysis.csv")

    # Save symmetry data
    df_symmetry.to_csv("bilateral_symmetry.csv", index=False)
    print("✓ Saved: bilateral_symmetry.csv")

    # Save normalized gait cycle
    if len(peak_times) >= 2:
        df_cycle = pd.DataFrame({
            'Gait_Percent': gait_percent,
            'Hip_Angle': cycle_hip.to_numpy(),
            'Knee_Angle': cycle_knee.to_numpy(),
            'Ankle_Angle': cycle_ankle.to_numpy()
        })
        df_cycle.to_csv("gait_cycle_normalized.csv", index=False)
        print("✓ Saved: gait_cycle_normalized.csv")


    # ===== 11. CLINICAL INTERPRETATION =====
    print("\n===== CLINICAL INTERPRETATION =====")
    print("\nNormal ROM ranges (walking):")
    print("  Hip: 30-40° flexion/extension")
    print("  Knee: 60-70° flexion/extension")
    print("  Ankle: 25-30° dorsi/plantarflexion")

    print("\nBilateral asymmetry:")
    for idx, row in df_symmetry.iterrows():
        si = abs(row['Symmetry Index (%)'])
        status = "NORMAL" if si < 10 else "ASYMMETRIC"
        print(f"  {row['Joint']}: {si:.1f}% - {status}")


if __name__ == "__main__":
    main()
