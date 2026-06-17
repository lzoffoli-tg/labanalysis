"""
Balance Test Example
=====================

Demonstrates static balance assessment:
1. Load force platform data from quiet standing
2. Calculate center of pressure (COP) metrics
3. Analyze sway area and path length
4. Compare eyes open vs eyes closed conditions
5. Calculate Romberg quotient
6. Visualize sway pattern

Common use case: Balance assessment, fall risk screening, injury rehabilitation.
"""

import labanalysis as laban
from labanalysis.signalprocessing import butterworth_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def main():
    # ===== 1. LOAD BALANCE TEST DATA =====
    print("Loading balance test data...")

    # Load eyes open condition
    fp_eo = laban.ForcePlatform.from_tdf_file(
        "path/to/balance_eyes_open.tdf",
        fp_label="FP1"
    )

    # Load eyes closed condition
    fp_ec = laban.ForcePlatform.from_tdf_file(
        "path/to/balance_eyes_closed.tdf",
        fp_label="FP1"
    )

    print(f"✓ Loaded balance tests")
    print(f"Eyes open duration: {fp_eo.index[-1]:.1f} s")
    print(f"Eyes closed duration: {fp_ec.index[-1]:.1f} s")


    # ===== 2. EXTRACT COP TRAJECTORY =====
    print("\nExtracting COP trajectories...")

    # Eyes open COP
    cop_eo_x = fp_eo["ORIGIN", "X"]  # Anterior-posterior
    cop_eo_y = fp_eo["ORIGIN", "Y"]  # Medial-lateral

    # Eyes closed COP
    cop_ec_x = fp_ec["ORIGIN", "X"]
    cop_ec_y = fp_ec["ORIGIN", "Y"]

    # Filter COP (reduce high-frequency noise)
    cop_eo_x_smooth = butterworth_filter(cop_eo_x, frequency=5, order=4)
    cop_eo_y_smooth = butterworth_filter(cop_eo_y, frequency=5, order=4)

    cop_ec_x_smooth = butterworth_filter(cop_ec_x, frequency=5, order=4)
    cop_ec_y_smooth = butterworth_filter(cop_ec_y, frequency=5, order=4)


    # ===== 3. CALCULATE COP METRICS =====
    print("\n===== COP METRICS =====")

    def calculate_cop_metrics(cop_x, cop_y):
        """Calculate standard COP metrics."""
        # Convert to numpy arrays
        x = cop_x.to_numpy()
        y = cop_y.to_numpy()

        # Remove NaN values
        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]

        if len(x) < 10:
            return None

        # Mean position
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        # RMS (root mean square) displacement
        rms_x = np.sqrt(np.mean((x - mean_x)**2))
        rms_y = np.sqrt(np.mean((y - mean_y)**2))
        rms_total = np.sqrt(rms_x**2 + rms_y**2)

        # Range (peak-to-peak)
        range_x = x.max() - x.min()
        range_y = y.max() - y.min()

        # Path length (total distance traveled)
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        path_length = np.sum(distances)

        # Mean velocity
        dt = np.mean(np.diff(cop_x.index))
        mean_velocity = path_length / (len(x) * dt)

        # 95% confidence ellipse area
        # Using 2 standard deviations
        area_ellipse = np.pi * (2 * rms_x) * (2 * rms_y)

        # Sway area (95% confidence ellipse)
        try:
            points = np.column_stack((x, y))
            hull = ConvexHull(points)
            area_hull = hull.volume  # In 2D, volume is area
        except:
            area_hull = area_ellipse

        return {
            'mean_x': mean_x,
            'mean_y': mean_y,
            'rms_x': rms_x,
            'rms_y': rms_y,
            'rms_total': rms_total,
            'range_x': range_x,
            'range_y': range_y,
            'path_length': path_length,
            'mean_velocity': mean_velocity,
            'area_ellipse': area_ellipse,
            'area_hull': area_hull
        }

    # Calculate metrics for both conditions
    metrics_eo = calculate_cop_metrics(cop_eo_x_smooth, cop_eo_y_smooth)
    metrics_ec = calculate_cop_metrics(cop_ec_x_smooth, cop_ec_y_smooth)

    # Print results
    print("\nEyes Open:")
    print(f"  RMS sway: {metrics_eo['rms_total']*100:.2f} cm")
    print(f"  Path length: {metrics_eo['path_length']:.2f} m")
    print(f"  Mean velocity: {metrics_eo['mean_velocity']*100:.2f} cm/s")
    print(f"  95% ellipse area: {metrics_eo['area_ellipse']*10000:.1f} cm²")

    print("\nEyes Closed:")
    print(f"  RMS sway: {metrics_ec['rms_total']*100:.2f} cm")
    print(f"  Path length: {metrics_ec['path_length']:.2f} m")
    print(f"  Mean velocity: {metrics_ec['mean_velocity']*100:.2f} cm/s")
    print(f"  95% ellipse area: {metrics_ec['area_ellipse']*10000:.1f} cm²")


    # ===== 4. CALCULATE ROMBERG QUOTIENT =====
    print("\n===== ROMBERG QUOTIENT =====")

    # Romberg quotient = EC metric / EO metric
    # Values > 1 indicate increased reliance on vision
    romberg_sway = metrics_ec['rms_total'] / metrics_eo['rms_total']
    romberg_path = metrics_ec['path_length'] / metrics_eo['path_length']
    romberg_velocity = metrics_ec['mean_velocity'] / metrics_eo['mean_velocity']

    print(f"RQ (sway): {romberg_sway:.2f}")
    print(f"RQ (path): {romberg_path:.2f}")
    print(f"RQ (velocity): {romberg_velocity:.2f}")

    if romberg_sway > 2.0:
        print("\n⚠ High visual dependence detected (RQ > 2.0)")
    elif romberg_sway > 1.5:
        print("\n⚠ Moderate visual dependence (RQ = 1.5-2.0)")
    else:
        print("\n✓ Normal visual-vestibular integration (RQ < 1.5)")


    # ===== 5. FREQUENCY ANALYSIS (OPTIONAL) =====
    print("\n===== FREQUENCY ANALYSIS =====")

    # Dominant frequency indicates control strategy
    # Low freq (< 0.5 Hz) = vestibular
    # Mid freq (0.5-1.5 Hz) = visual
    # High freq (> 1.5 Hz) = proprioceptive

    from labanalysis.signalprocessing import power_spectral_density

    freq_eo, psd_eo = power_spectral_density(cop_eo_x_smooth)
    freq_ec, psd_ec = power_spectral_density(cop_ec_x_smooth)

    # Find dominant frequency (peak in 0-3 Hz range)
    freq_range = (freq_eo >= 0) & (freq_eo <= 3)
    dominant_freq_eo = freq_eo[freq_range][np.argmax(psd_eo[freq_range])]
    dominant_freq_ec = freq_ec[freq_range][np.argmax(psd_ec[freq_range])]

    print(f"Dominant frequency (EO): {dominant_freq_eo:.2f} Hz")
    print(f"Dominant frequency (EC): {dominant_freq_ec:.2f} Hz")


    # ===== 6. CREATE VISUALIZATIONS =====
    print("\nCreating visualizations...")

    # Plot 1: COP trajectories (bird's eye view)
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Eyes Open', 'Eyes Closed'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    # Eyes open trajectory
    fig1.add_trace(
        go.Scatter(
            x=cop_eo_x_smooth.to_numpy()*100,  # Convert to cm
            y=cop_eo_y_smooth.to_numpy()*100,
            mode='lines',
            line=dict(color='blue', width=1),
            name='EO trajectory'
        ),
        row=1, col=1
    )

    # Mean position
    fig1.add_trace(
        go.Scatter(
            x=[metrics_eo['mean_x']*100],
            y=[metrics_eo['mean_y']*100],
            mode='markers',
            marker=dict(size=10, color='red', symbol='cross'),
            name='Mean EO'
        ),
        row=1, col=1
    )

    # Eyes closed trajectory
    fig1.add_trace(
        go.Scatter(
            x=cop_ec_x_smooth.to_numpy()*100,
            y=cop_ec_y_smooth.to_numpy()*100,
            mode='lines',
            line=dict(color='green', width=1),
            name='EC trajectory',
            showlegend=False
        ),
        row=1, col=2
    )

    # Mean position
    fig1.add_trace(
        go.Scatter(
            x=[metrics_ec['mean_x']*100],
            y=[metrics_ec['mean_y']*100],
            mode='markers',
            marker=dict(size=10, color='red', symbol='cross'),
            name='Mean EC',
            showlegend=False
        ),
        row=1, col=2
    )

    fig1.update_xaxes(title_text="A-P (cm)", row=1, col=1)
    fig1.update_xaxes(title_text="A-P (cm)", row=1, col=2)
    fig1.update_yaxes(title_text="M-L (cm)", row=1, col=1)
    fig1.update_yaxes(title_text="M-L (cm)", row=1, col=2)

    # Equal aspect ratio
    fig1.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig1.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)

    fig1.update_layout(
        title="COP Trajectories",
        height=500,
        template='plotly_white'
    )


    # Plot 2: Time series comparison
    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Anterior-Posterior Displacement', 'Medial-Lateral Displacement'),
        vertical_spacing=0.12,
        shared_xaxes=True
    )

    # A-P displacement
    fig2.add_trace(
        go.Scatter(x=cop_eo_x_smooth.index, y=cop_eo_x_smooth.to_numpy()*100,
                   name='Eyes Open', line=dict(color='blue', width=1)),
        row=1, col=1
    )

    fig2.add_trace(
        go.Scatter(x=cop_ec_x_smooth.index, y=cop_ec_x_smooth.to_numpy()*100,
                   name='Eyes Closed', line=dict(color='green', width=1)),
        row=1, col=1
    )

    # M-L displacement
    fig2.add_trace(
        go.Scatter(x=cop_eo_y_smooth.index, y=cop_eo_y_smooth.to_numpy()*100,
                   name='Eyes Open', line=dict(color='blue', width=1), showlegend=False),
        row=2, col=1
    )

    fig2.add_trace(
        go.Scatter(x=cop_ec_y_smooth.index, y=cop_ec_y_smooth.to_numpy()*100,
                   name='Eyes Closed', line=dict(color='green', width=1), showlegend=False),
        row=2, col=1
    )

    fig2.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig2.update_yaxes(title_text="Displacement (cm)")

    fig2.update_layout(
        title="COP Displacement Over Time",
        height=700,
        template='plotly_white',
        hovermode='x unified'
    )


    # Plot 3: Metrics comparison bar chart
    fig3 = go.Figure()

    metrics_names = [
        'RMS Sway\n(cm)',
        'Path Length\n(m)',
        'Mean Velocity\n(cm/s)',
        'Sway Area\n(cm²)'
    ]

    eo_values = [
        metrics_eo['rms_total'] * 100,
        metrics_eo['path_length'],
        metrics_eo['mean_velocity'] * 100,
        metrics_eo['area_ellipse'] * 10000
    ]

    ec_values = [
        metrics_ec['rms_total'] * 100,
        metrics_ec['path_length'],
        metrics_ec['mean_velocity'] * 100,
        metrics_ec['area_ellipse'] * 10000
    ]

    fig3.add_trace(
        go.Bar(x=metrics_names, y=eo_values, name='Eyes Open', marker_color='blue')
    )

    fig3.add_trace(
        go.Bar(x=metrics_names, y=ec_values, name='Eyes Closed', marker_color='green')
    )

    fig3.update_layout(
        title="Balance Metrics Comparison",
        yaxis_title="Value",
        template='plotly_white',
        barmode='group'
    )

    fig3.show()


    # ===== 7. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()
    fig2.show()


    # ===== 8. EXPORT RESULTS =====
    print("\nExporting results...")

    # Create comparison table
    comparison = {
        'Metric': [
            'RMS Sway (cm)',
            'Range A-P (cm)',
            'Range M-L (cm)',
            'Path Length (m)',
            'Mean Velocity (cm/s)',
            'Sway Area (cm²)',
            'Dominant Freq (Hz)'
        ],
        'Eyes Open': [
            f"{metrics_eo['rms_total']*100:.2f}",
            f"{metrics_eo['range_x']*100:.2f}",
            f"{metrics_eo['range_y']*100:.2f}",
            f"{metrics_eo['path_length']:.2f}",
            f"{metrics_eo['mean_velocity']*100:.2f}",
            f"{metrics_eo['area_ellipse']*10000:.1f}",
            f"{dominant_freq_eo:.2f}"
        ],
        'Eyes Closed': [
            f"{metrics_ec['rms_total']*100:.2f}",
            f"{metrics_ec['range_x']*100:.2f}",
            f"{metrics_ec['range_y']*100:.2f}",
            f"{metrics_ec['path_length']:.2f}",
            f"{metrics_ec['mean_velocity']*100:.2f}",
            f"{metrics_ec['area_ellipse']*10000:.1f}",
            f"{dominant_freq_ec:.2f}"
        ],
        'Romberg Quotient': [
            f"{romberg_sway:.2f}",
            '-',
            '-',
            f"{romberg_path:.2f}",
            f"{romberg_velocity:.2f}",
            '-',
            '-'
        ]
    }

    df_comparison = pd.DataFrame(comparison)
    df_comparison.to_csv("balance_test_results.csv", index=False)
    print("✓ Saved: balance_test_results.csv")


    # ===== 9. CLINICAL INTERPRETATION =====
    print("\n===== CLINICAL INTERPRETATION =====")

    print("\nNormative values (healthy adults):")
    print("  RMS sway (EO): 0.5-1.5 cm")
    print("  RMS sway (EC): 0.8-2.5 cm")
    print("  Romberg quotient: 1.0-1.5")

    print("\nTest results:")
    if metrics_eo['rms_total']*100 < 1.5:
        print("  ✓ Eyes open sway: NORMAL")
    else:
        print("  ⚠ Eyes open sway: ELEVATED")

    if metrics_ec['rms_total']*100 < 2.5:
        print("  ✓ Eyes closed sway: NORMAL")
    else:
        print("  ⚠ Eyes closed sway: ELEVATED")


if __name__ == "__main__":
    main()
