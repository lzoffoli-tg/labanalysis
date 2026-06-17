"""
Running Test Analysis Example
==============================

Demonstrates running gait analysis:
1. Load running trial with kinematics and force data
2. Identify gait cycles and foot contacts
3. Calculate spatiotemporal parameters
4. Analyze GRF characteristics
5. Compute bilateral asymmetry
6. Generate running report

Common use case: Running mechanics assessment, injury screening.
"""

import labanalysis as laban
from labanalysis.signalprocessing import butterworth_filter, find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def main():
    # ===== 1. LOAD RUNNING DATA =====
    print("Loading running trial...")

    # Load full-body kinematics
    body = laban.WholeBody.from_tdf_file(
        "path/to/running_trial.tdf",
        labels="LABEL"
    )

    # Load force platforms
    fp1 = laban.ForcePlatform.from_tdf_file(
        "path/to/running_trial.tdf",
        fp_label="FP1"
    )

    fp2 = laban.ForcePlatform.from_tdf_file(
        "path/to/running_trial.tdf",
        fp_label="FP2"
    )

    print(f"✓ Loaded running data")
    print(f"Duration: {body.index[-1]:.2f} s")
    print(f"Sampling rate: {1/np.mean(np.diff(body.index)):.1f} Hz")


    # ===== 2. EXTRACT PELVIS TRAJECTORY =====
    print("\nExtracting pelvis trajectory...")

    # Get pelvis center (represents runner's CoM approximation)
    pelvis_center = body.pelvis_center

    # Vertical displacement
    pelvis_z = pelvis_center["Z"]

    # Forward velocity (anterior-posterior)
    pelvis_x = pelvis_center["X"]


    # ===== 3. DETECT FOOT CONTACTS FROM FORCE =====
    print("\nDetecting foot contacts...")

    # Smooth vertical GRF
    fp1_fz = butterworth_filter(fp1["FORCE", "Z"], frequency=10, order=4)
    fp2_fz = butterworth_filter(fp2["FORCE", "Z"], frequency=10, order=4)

    # Contact threshold
    contact_threshold = 50  # N

    # Detect contacts (force exceeds threshold)
    fp1_contacts = fp1_fz.to_numpy() < -contact_threshold
    fp2_contacts = fp2_fz.to_numpy() < -contact_threshold

    # Find contact events
    fp1_strike_idx = np.where(np.diff(fp1_contacts.astype(int)) == 1)[0]
    fp1_toeoff_idx = np.where(np.diff(fp1_contacts.astype(int)) == -1)[0]

    fp2_strike_idx = np.where(np.diff(fp2_contacts.astype(int)) == 1)[0]
    fp2_toeoff_idx = np.where(np.diff(fp2_contacts.astype(int)) == -1)[0]

    print(f"FP1: {len(fp1_strike_idx)} contacts")
    print(f"FP2: {len(fp2_strike_idx)} contacts")


    # ===== 4. CALCULATE SPATIOTEMPORAL PARAMETERS =====
    print("\n===== SPATIOTEMPORAL PARAMETERS =====")

    # Function to calculate gait metrics from contact times
    def calculate_gait_metrics(strike_times, toeoff_times):
        """Calculate contact time, flight time, step frequency."""
        metrics = []

        for i in range(min(len(strike_times), len(toeoff_times))):
            contact_time = toeoff_times[i] - strike_times[i]

            # Flight time (to next contact)
            if i < len(strike_times) - 1:
                flight_time = strike_times[i+1] - toeoff_times[i]
                step_time = strike_times[i+1] - strike_times[i]
            else:
                flight_time = np.nan
                step_time = np.nan

            metrics.append({
                'contact_time': contact_time,
                'flight_time': flight_time,
                'step_time': step_time
            })

        return metrics

    # Convert indices to times
    fp1_strike_times = body.index[fp1_strike_idx]
    fp1_toeoff_times = body.index[fp1_toeoff_idx]
    fp2_strike_times = body.index[fp2_strike_idx]
    fp2_toeoff_times = body.index[fp2_toeoff_idx]

    # Calculate metrics
    fp1_metrics = calculate_gait_metrics(fp1_strike_times, fp1_toeoff_times)
    fp2_metrics = calculate_gait_metrics(fp2_strike_times, fp2_toeoff_times)

    # Average metrics
    if fp1_metrics:
        avg_contact_fp1 = np.nanmean([m['contact_time'] for m in fp1_metrics]) * 1000  # ms
        avg_flight_fp1 = np.nanmean([m['flight_time'] for m in fp1_metrics]) * 1000
        avg_step_fp1 = np.nanmean([m['step_time'] for m in fp1_metrics]) * 1000
        step_freq_fp1 = 60 / (avg_step_fp1 / 1000)  # steps/min

        print(f"\nFP1 (Left foot):")
        print(f"  Contact time: {avg_contact_fp1:.0f} ms")
        print(f"  Flight time: {avg_flight_fp1:.0f} ms")
        print(f"  Step frequency: {step_freq_fp1:.1f} steps/min")

    if fp2_metrics:
        avg_contact_fp2 = np.nanmean([m['contact_time'] for m in fp2_metrics]) * 1000
        avg_flight_fp2 = np.nanmean([m['flight_time'] for m in fp2_metrics]) * 1000
        avg_step_fp2 = np.nanmean([m['step_time'] for m in fp2_metrics]) * 1000
        step_freq_fp2 = 60 / (avg_step_fp2 / 1000)

        print(f"\nFP2 (Right foot):")
        print(f"  Contact time: {avg_contact_fp2:.0f} ms")
        print(f"  Flight time: {avg_flight_fp2:.0f} ms")
        print(f"  Step frequency: {step_freq_fp2:.1f} steps/min")


    # ===== 5. ANALYZE GRF CHARACTERISTICS =====
    print("\n===== GRF CHARACTERISTICS =====")

    def analyze_grf_contact(grf_z, strike_idx, toeoff_idx):
        """Analyze GRF during single contact."""
        strike_time = body.index[strike_idx]
        toeoff_time = body.index[toeoff_idx]

        contact_grf = grf_z[strike_time:toeoff_time]

        # Peak vertical force
        peak_force = abs(contact_grf.to_numpy().min())

        # Loading rate (first peak)
        # Find first 50% of contact
        mid_idx = len(contact_grf) // 2
        first_half = contact_grf.to_numpy()[:mid_idx]

        if len(first_half) > 0:
            peak_idx = np.argmin(first_half)
            time_to_peak = contact_grf.index[peak_idx] - strike_time
            loading_rate = peak_force / time_to_peak if time_to_peak > 0 else 0
        else:
            loading_rate = 0

        # Impulse
        dt = np.mean(np.diff(contact_grf.index))
        impulse = abs(np.sum(contact_grf.to_numpy()) * dt)

        return {
            'peak_force': peak_force,
            'loading_rate': loading_rate,
            'impulse': impulse
        }

    # Analyze first contact on each platform
    if len(fp1_strike_idx) > 0 and len(fp1_toeoff_idx) > 0:
        grf1_analysis = analyze_grf_contact(fp1_fz, fp1_strike_idx[0], fp1_toeoff_idx[0])

        print(f"\nFP1 GRF:")
        print(f"  Peak force: {grf1_analysis['peak_force']:.1f} N")
        print(f"  Loading rate: {grf1_analysis['loading_rate']:.0f} N/s")
        print(f"  Impulse: {grf1_analysis['impulse']:.1f} N·s")

    if len(fp2_strike_idx) > 0 and len(fp2_toeoff_idx) > 0:
        grf2_analysis = analyze_grf_contact(fp2_fz, fp2_strike_idx[0], fp2_toeoff_idx[0])

        print(f"\nFP2 GRF:")
        print(f"  Peak force: {grf2_analysis['peak_force']:.1f} N")
        print(f"  Loading rate: {grf2_analysis['loading_rate']:.0f} N/s")
        print(f"  Impulse: {grf2_analysis['impulse']:.1f} N·s")


    # ===== 6. CALCULATE BILATERAL ASYMMETRY =====
    print("\n===== BILATERAL ASYMMETRY =====")

    def symmetry_index(left, right):
        """Symmetry index: (L-R) / (0.5*(L+R)) * 100"""
        return (left - right) / (0.5 * (left + right)) * 100

    if fp1_metrics and fp2_metrics:
        si_contact = symmetry_index(avg_contact_fp1, avg_contact_fp2)
        si_flight = symmetry_index(avg_flight_fp1, avg_flight_fp2)

        print(f"Contact time SI: {si_contact:.1f}%")
        print(f"Flight time SI: {si_flight:.1f}%")

        if 'grf1_analysis' in locals() and 'grf2_analysis' in locals():
            si_peak_force = symmetry_index(
                grf1_analysis['peak_force'],
                grf2_analysis['peak_force']
            )
            print(f"Peak force SI: {si_peak_force:.1f}%")

            status = "SYMMETRIC" if abs(si_peak_force) < 10 else "ASYMMETRIC"
            print(f"\nAsymmetry status: {status}")


    # ===== 7. EXTRACT VERTICAL DISPLACEMENT =====
    print("\n===== VERTICAL DISPLACEMENT =====")

    # Vertical oscillation during running
    pelvis_z_smooth = butterworth_filter(pelvis_z, frequency=6, order=4)

    # Find peaks (highest points) and valleys (lowest points)
    peaks_idx, _ = find_peaks(pelvis_z_smooth.to_numpy(), threshold=None, distance=20)
    valleys_idx, _ = find_peaks(-pelvis_z_smooth.to_numpy(), threshold=None, distance=20)

    if len(peaks_idx) > 0 and len(valleys_idx) > 0:
        peak_heights = pelvis_z_smooth.to_numpy()[peaks_idx]
        valley_heights = pelvis_z_smooth.to_numpy()[valleys_idx]

        # Average vertical oscillation
        vertical_oscillation = np.mean(peak_heights) - np.mean(valley_heights)

        print(f"Vertical oscillation: {vertical_oscillation*100:.1f} cm")
        print(f"  (Lower values indicate more economical running)")


    # ===== 8. CREATE VISUALIZATIONS =====
    print("\nCreating visualizations...")

    # Plot 1: GRF and pelvis trajectory
    fig1 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Vertical Ground Reaction Force', 'Pelvis Vertical Position'),
        vertical_spacing=0.12,
        shared_xaxes=True
    )

    # GRF
    fig1.add_trace(
        go.Scatter(x=fp1_fz.index, y=fp1_fz.to_numpy(),
                   name='Left (FP1)', line=dict(color='blue', width=2)),
        row=1, col=1
    )

    fig1.add_trace(
        go.Scatter(x=fp2_fz.index, y=fp2_fz.to_numpy(),
                   name='Right (FP2)', line=dict(color='red', width=2)),
        row=1, col=1
    )

    # Mark contacts
    for t in fp1_strike_times:
        fig1.add_vline(x=t, line_dash="dot", line_color="blue", opacity=0.3, row=1, col=1)
    for t in fp2_strike_times:
        fig1.add_vline(x=t, line_dash="dot", line_color="red", opacity=0.3, row=1, col=1)

    # Pelvis height
    fig1.add_trace(
        go.Scatter(x=pelvis_z_smooth.index, y=pelvis_z_smooth.to_numpy()*100,
                   name='Pelvis', line=dict(color='purple', width=2)),
        row=2, col=1
    )

    # Mark peaks/valleys
    if len(peaks_idx) > 0:
        fig1.add_trace(
            go.Scatter(
                x=pelvis_z_smooth.index[peaks_idx],
                y=pelvis_z_smooth.to_numpy()[peaks_idx]*100,
                mode='markers',
                marker=dict(size=8, color='green', symbol='triangle-up'),
                name='Peaks',
                showlegend=False
            ),
            row=2, col=1
        )

    fig1.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig1.update_yaxes(title_text="Force (N)", row=1, col=1)
    fig1.update_yaxes(title_text="Height (cm)", row=2, col=1)

    fig1.update_layout(
        title="Running Gait Analysis",
        height=700,
        template='plotly_white',
        hovermode='x unified'
    )


    # Plot 2: Spatiotemporal bar chart
    if fp1_metrics and fp2_metrics:
        fig2 = go.Figure()

        metrics_names = ['Contact Time', 'Flight Time', 'Step Time']
        left_vals = [avg_contact_fp1, avg_flight_fp1, avg_step_fp1]
        right_vals = [avg_contact_fp2, avg_flight_fp2, avg_step_fp2]

        fig2.add_trace(
            go.Bar(x=metrics_names, y=left_vals, name='Left', marker_color='blue')
        )

        fig2.add_trace(
            go.Bar(x=metrics_names, y=right_vals, name='Right', marker_color='red')
        )

        fig2.update_layout(
            title="Spatiotemporal Parameters Comparison",
            yaxis_title="Time (ms)",
            template='plotly_white',
            barmode='group'
        )

        fig2.show()


    # ===== 9. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()


    # ===== 10. EXPORT RESULTS =====
    print("\nExporting results...")

    # Summary table
    if fp1_metrics and fp2_metrics:
        summary = {
            'Parameter': [
                'Contact Time', 'Flight Time', 'Step Frequency',
                'Peak Force', 'Loading Rate', 'Vertical Oscillation'
            ],
            'Left': [
                f"{avg_contact_fp1:.0f}",
                f"{avg_flight_fp1:.0f}",
                f"{step_freq_fp1:.1f}",
                f"{grf1_analysis['peak_force']:.0f}" if 'grf1_analysis' in locals() else 'N/A',
                f"{grf1_analysis['loading_rate']:.0f}" if 'grf1_analysis' in locals() else 'N/A',
                f"{vertical_oscillation*100:.1f}" if 'vertical_oscillation' in locals() else 'N/A'
            ],
            'Right': [
                f"{avg_contact_fp2:.0f}",
                f"{avg_flight_fp2:.0f}",
                f"{step_freq_fp2:.1f}",
                f"{grf2_analysis['peak_force']:.0f}" if 'grf2_analysis' in locals() else 'N/A',
                f"{grf2_analysis['loading_rate']:.0f}" if 'grf2_analysis' in locals() else 'N/A',
                'N/A'
            ],
            'Unit': ['ms', 'ms', 'steps/min', 'N', 'N/s', 'cm']
        }

        df_summary = pd.DataFrame(summary)
        df_summary.to_csv("running_test_summary.csv", index=False)
        print("✓ Saved: running_test_summary.csv")


if __name__ == "__main__":
    main()
