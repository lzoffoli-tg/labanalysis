"""
Ground Reaction Force (GRF) Analysis Example
=============================================

Demonstrates force platform analysis:
1. Load multi-platform force data
2. Calculate GRF components and resultant
3. Compute center of pressure (COP) trajectory
4. Analyze impulse and loading rates
5. Identify gait events from force
6. Compare bilateral loading patterns

Common use case: Gait analysis, landing mechanics, force asymmetry.
"""

import labanalysis as laban
from labanalysis.signalprocessing import butterworth_filter, find_peaks, derivative
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def main():
    # ===== 1. LOAD FORCE PLATFORM DATA =====
    print("Loading force platform data...")

    # Load two force platforms (typical gait setup)
    fp1 = laban.ForcePlatform.from_tdf_file(
        "path/to/gait_trial.tdf",
        fp_label="FP1"
    )

    fp2 = laban.ForcePlatform.from_tdf_file(
        "path/to/gait_trial.tdf",
        fp_label="FP2"
    )

    print(f"FP1 duration: {fp1.index[-1]:.2f} s")
    print(f"FP2 duration: {fp2.index[-1]:.2f} s")


    # ===== 2. EXTRACT GRF COMPONENTS =====
    print("\nExtracting GRF components...")

    # FP1 forces (Newtons)
    fp1_force_x = fp1["FORCE", "X"]  # Anterior-posterior (A-P)
    fp1_force_y = fp1["FORCE", "Y"]  # Medial-lateral (M-L)
    fp1_force_z = fp1["FORCE", "Z"]  # Vertical

    # FP2 forces
    fp2_force_x = fp2["FORCE", "X"]
    fp2_force_y = fp2["FORCE", "Y"]
    fp2_force_z = fp2["FORCE", "Z"]

    # Filter forces (10 Hz cutoff standard for GRF)
    fp1_fz_smooth = butterworth_filter(fp1_force_z, frequency=10, order=4)
    fp2_fz_smooth = butterworth_filter(fp2_force_z, frequency=10, order=4)


    # ===== 3. CALCULATE RESULTANT FORCE =====
    print("\nCalculating resultant force...")

    # Resultant force magnitude
    fp1_resultant = np.sqrt(
        fp1_force_x.to_numpy()**2 +
        fp1_force_y.to_numpy()**2 +
        fp1_force_z.to_numpy()**2
    )

    fp2_resultant = np.sqrt(
        fp2_force_x.to_numpy()**2 +
        fp2_force_y.to_numpy()**2 +
        fp2_force_z.to_numpy()**2
    )


    # ===== 4. DETECT FOOT CONTACTS =====
    print("\nDetecting foot contacts...")

    # Threshold for contact detection (typically 20-50 N)
    contact_threshold = 30  # N

    # Find contact periods for FP1
    fp1_contact = fp1_fz_smooth.to_numpy() < -contact_threshold  # Negative = downward

    # Find contact onset/offset
    fp1_contacts_idx = np.where(np.diff(fp1_contact.astype(int)) == 1)[0]
    fp1_liftoffs_idx = np.where(np.diff(fp1_contact.astype(int)) == -1)[0]

    print(f"FP1: {len(fp1_contacts_idx)} contacts detected")

    # Same for FP2
    fp2_contact = fp2_fz_smooth.to_numpy() < -contact_threshold
    fp2_contacts_idx = np.where(np.diff(fp2_contact.astype(int)) == 1)[0]
    fp2_liftoffs_idx = np.where(np.diff(fp2_contact.astype(int)) == -1)[0]

    print(f"FP2: {len(fp2_contacts_idx)} contacts detected")


    # ===== 5. ANALYZE FIRST CONTACT (FP1) =====
    if len(fp1_contacts_idx) > 0 and len(fp1_liftoffs_idx) > 0:
        print("\nAnalyzing first contact on FP1...")

        # Extract first contact phase
        contact_start_idx = fp1_contacts_idx[0]
        contact_end_idx = fp1_liftoffs_idx[0]

        contact_start_time = fp1.index[contact_start_idx]
        contact_end_time = fp1.index[contact_end_idx]
        contact_duration = contact_end_time - contact_start_time

        print(f"Contact duration: {contact_duration*1000:.0f} ms")

        # Extract forces during contact
        contact_fz = fp1_fz_smooth[contact_start_time:contact_end_time]
        contact_fx = butterworth_filter(
            fp1_force_x[contact_start_time:contact_end_time],
            frequency=10, order=4
        )

        # Peak vertical force
        peak_fz = abs(contact_fz.to_numpy().min())  # Most negative value
        print(f"Peak vertical force: {peak_fz:.1f} N")

        # Loading rate (force/time to peak)
        peak_idx = np.argmin(contact_fz.to_numpy())
        time_to_peak = contact_fz.index[peak_idx] - contact_start_time
        loading_rate = peak_fz / time_to_peak if time_to_peak > 0 else 0
        print(f"Loading rate: {loading_rate:.0f} N/s")

        # Impulse (area under force-time curve)
        dt = np.mean(np.diff(contact_fz.index))
        impulse_z = abs(np.sum(contact_fz.to_numpy()) * dt)
        print(f"Vertical impulse: {impulse_z:.1f} N·s")


    # ===== 6. CENTER OF PRESSURE (COP) TRAJECTORY =====
    print("\nCalculating COP trajectory...")

    # Extract COP from force platform
    # COP is stored in ORIGIN field
    cop1_x = fp1["ORIGIN", "X"]
    cop1_y = fp1["ORIGIN", "Y"]

    cop2_x = fp2["ORIGIN", "X"]
    cop2_y = fp2["ORIGIN", "Y"]

    # Filter COP (smoother visualization)
    cop1_x_smooth = butterworth_filter(cop1_x, frequency=6, order=4)
    cop1_y_smooth = butterworth_filter(cop1_y, frequency=6, order=4)


    # ===== 7. BILATERAL COMPARISON =====
    print("\nComparing bilateral loading...")

    if len(fp1_contacts_idx) > 0 and len(fp2_contacts_idx) > 0:
        # Peak forces
        fp1_peaks = []
        for i in range(min(len(fp1_contacts_idx), len(fp1_liftoffs_idx))):
            start_t = fp1.index[fp1_contacts_idx[i]]
            end_t = fp1.index[fp1_liftoffs_idx[i]]
            segment = fp1_fz_smooth[start_t:end_t]
            fp1_peaks.append(abs(segment.to_numpy().min()))

        fp2_peaks = []
        for i in range(min(len(fp2_contacts_idx), len(fp2_liftoffs_idx))):
            start_t = fp2.index[fp2_contacts_idx[i]]
            end_t = fp2.index[fp2_liftoffs_idx[i]]
            segment = fp2_fz_smooth[start_t:end_t]
            fp2_peaks.append(abs(segment.to_numpy().min()))

        # Average peak forces
        avg_fp1 = np.mean(fp1_peaks) if fp1_peaks else 0
        avg_fp2 = np.mean(fp2_peaks) if fp2_peaks else 0

        # Symmetry index
        symmetry_index = (avg_fp1 - avg_fp2) / (0.5 * (avg_fp1 + avg_fp2)) * 100

        print(f"FP1 average peak: {avg_fp1:.1f} N")
        print(f"FP2 average peak: {avg_fp2:.1f} N")
        print(f"Symmetry index: {symmetry_index:.1f}%")
        print(f"  ({'Left dominant' if symmetry_index > 0 else 'Right dominant'})")


    # ===== 8. CREATE VISUALIZATIONS =====
    print("\nCreating visualizations...")

    # Plot 1: GRF components over time
    fig1 = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Vertical GRF', 'Anterior-Posterior GRF', 'Medial-Lateral GRF'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )

    # Vertical forces
    fig1.add_trace(
        go.Scatter(x=fp1.index, y=fp1_fz_smooth.to_numpy(),
                   name='FP1', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig1.add_trace(
        go.Scatter(x=fp2.index, y=fp2_fz_smooth.to_numpy(),
                   name='FP2', line=dict(color='red', width=2)),
        row=1, col=1
    )

    # A-P forces
    fig1.add_trace(
        go.Scatter(x=fp1.index, y=fp1_force_x.to_numpy(),
                   name='FP1', line=dict(color='blue', width=2), showlegend=False),
        row=2, col=1
    )
    fig1.add_trace(
        go.Scatter(x=fp2.index, y=fp2_force_x.to_numpy(),
                   name='FP2', line=dict(color='red', width=2), showlegend=False),
        row=2, col=1
    )

    # M-L forces
    fig1.add_trace(
        go.Scatter(x=fp1.index, y=fp1_force_y.to_numpy(),
                   name='FP1', line=dict(color='blue', width=2), showlegend=False),
        row=3, col=1
    )
    fig1.add_trace(
        go.Scatter(x=fp2.index, y=fp2_force_y.to_numpy(),
                   name='FP2', line=dict(color='red', width=2), showlegend=False),
        row=3, col=1
    )

    # Mark contacts
    for idx in fp1_contacts_idx:
        t = fp1.index[idx]
        fig1.add_vline(x=t, line_dash="dot", line_color="blue", opacity=0.3)

    fig1.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig1.update_yaxes(title_text="Force (N)")

    fig1.update_layout(
        title="Ground Reaction Forces - Both Platforms",
        height=900,
        template='plotly_white',
        hovermode='x unified'
    )


    # Plot 2: COP trajectory (bird's eye view)
    fig2 = go.Figure()

    # FP1 COP path
    fig2.add_trace(
        go.Scatter(
            x=cop1_x_smooth.to_numpy(),
            y=cop1_y_smooth.to_numpy(),
            mode='lines',
            name='FP1 COP',
            line=dict(color='blue', width=2)
        )
    )

    # FP2 COP path
    fig2.add_trace(
        go.Scatter(
            x=cop2_x_smooth.to_numpy(),
            y=cop2_y_smooth.to_numpy(),
            mode='lines',
            name='FP2 COP',
            line=dict(color='red', width=2)
        )
    )

    fig2.update_layout(
        title="Center of Pressure Trajectory",
        xaxis_title="A-P Position (m)",
        yaxis_title="M-L Position (m)",
        template='plotly_white',
        yaxis_scaleanchor="x"  # Equal aspect ratio
    )


    # Plot 3: Force-time integral (impulse visualization)
    if len(fp1_contacts_idx) > 0:
        fig3 = go.Figure()

        fig3.add_trace(
            go.Scatter(
                x=contact_fz.index,
                y=contact_fz.to_numpy(),
                mode='lines',
                name='Vertical GRF',
                line=dict(color='blue', width=3),
                fill='tozeroy'
            )
        )

        fig3.add_hline(y=-contact_threshold, line_dash="dash",
                       line_color="red", annotation_text="Contact threshold")

        fig3.update_layout(
            title=f"GRF During Contact (Impulse = {impulse_z:.1f} N·s)",
            xaxis_title="Time (s)",
            yaxis_title="Vertical Force (N)",
            template='plotly_white'
        )

        fig3.show()


    # ===== 9. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()
    fig2.show()


    # ===== 10. EXPORT RESULTS =====
    print("\nExporting results...")

    # Create summary DataFrame
    results = {
        'Platform': ['FP1', 'FP2'],
        'Contacts': [len(fp1_contacts_idx), len(fp2_contacts_idx)],
        'Avg Peak Force (N)': [avg_fp1 if 'avg_fp1' in locals() else 0,
                                avg_fp2 if 'avg_fp2' in locals() else 0],
        'Loading Rate (N/s)': [loading_rate if 'loading_rate' in locals() else 0,
                               np.nan]
    }

    df_results = pd.DataFrame(results)
    df_results.to_csv("grf_analysis_summary.csv", index=False)
    print("✓ Saved: grf_analysis_summary.csv")


if __name__ == "__main__":
    main()
