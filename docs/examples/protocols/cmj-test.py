"""
Countermovement Jump (CMJ) Test Example
========================================

Demonstrates complete CMJ protocol analysis:
1. Load jump test data
2. Automatically detect jump phases
3. Extract all jump metrics
4. Compare to normative data
5. Generate athlete report
6. Visualize jump profile

Common use case: Athlete assessment, jump performance monitoring.
"""

import labanalysis as laban
from labanalysis.protocols import JumpTest, JumpTestResults
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def main():
    # ===== 1. LOAD JUMP TEST =====
    print("Loading CMJ test...")

    # Load using SingleJump class
    jump = laban.SingleJump.from_tdf_file(
        "path/to/cmj_test.tdf",
        labels="LABEL",
        fp_label="FP1"
    )

    print(f"✓ Loaded jump test")
    print(f"Duration: {jump.index[-1]:.2f} s")
    print(f"Sampling rate: {1/np.mean(np.diff(jump.index)):.1f} Hz")


    # ===== 2. EXTRACT KEY METRICS =====
    print("\n===== JUMP METRICS =====")

    # Performance metrics
    print(f"\nPerformance:")
    print(f"  Jump height: {jump.height*100:.1f} cm")
    print(f"  Flight time: {jump.flight_time*1000:.0f} ms")
    print(f"  Takeoff velocity: {jump.takeoff_velocity:.2f} m/s")
    print(f"  RSI-modified: {jump.rsi_modified:.2f}")

    # Force/Power metrics
    print(f"\nForce & Power:")
    print(f"  Peak force: {jump.peak_force:.1f} N ({jump.peak_force/9.81:.1f} kg)")
    print(f"  Peak power: {jump.peak_power:.1f} W")
    print(f"  Peak velocity: {jump.peak_velocity:.2f} m/s")

    # Temporal metrics
    print(f"\nTemporal:")
    print(f"  Eccentric duration: {jump.eccentric_duration*1000:.0f} ms")
    print(f"  Concentric duration: {jump.concentric_duration*1000:.0f} ms")
    print(f"  Countermovement depth: {abs(jump.countermovement_depth)*100:.1f} cm")

    # Key time points
    print(f"\nPhase transitions:")
    print(f"  Propulsion start: {jump.propulsion_start:.3f} s")
    print(f"  Takeoff: {jump.takeoff:.3f} s")
    print(f"  Landing: {jump.landing:.3f} s")


    # ===== 3. NORMATIVE COMPARISON =====
    print("\n===== NORMATIVE COMPARISON =====")

    # Example normative data (replace with actual sport-specific norms)
    athlete_mass = 75  # kg

    normative_data = {
        'Metric': [
            'Jump Height',
            'Peak Force',
            'Peak Power',
            'RSI-modified'
        ],
        'Athlete': [
            jump.height * 100,
            jump.peak_force / athlete_mass,
            jump.peak_power / athlete_mass,
            jump.rsi_modified
        ],
        'Elite Avg': [45.0, 25.0, 60.0, 0.50],  # Example values
        'Recreational Avg': [30.0, 20.0, 40.0, 0.30],
        'Unit': ['cm', 'N/kg', 'W/kg', '-']
    }

    df_norm = pd.DataFrame(normative_data)

    # Calculate percentile rank (simplified)
    df_norm['vs Elite (%)'] = ((df_norm['Athlete'] - df_norm['Elite Avg']) /
                                df_norm['Elite Avg'] * 100).round(1)

    print(df_norm.to_string(index=False))


    # ===== 4. PHASE ANALYSIS =====
    print("\n===== PHASE ANALYSIS =====")

    # Get force platform data
    fp = jump.force_platform
    grf_z = fp["FORCE", "Z"]

    # Extract each phase
    phases = {
        'Propulsion': (jump.propulsion_start, jump.takeoff),
        'Flight': (jump.takeoff, jump.landing),
    }

    print("\nPhase metrics:")
    for phase_name, (t_start, t_end) in phases.items():
        phase_grf = grf_z[t_start:t_end]
        duration = (t_end - t_start) * 1000  # ms

        if phase_name == 'Propulsion':
            avg_force = abs(phase_grf.to_numpy().mean())
            peak_force = abs(phase_grf.to_numpy().min())
            print(f"\n{phase_name}:")
            print(f"  Duration: {duration:.0f} ms")
            print(f"  Average force: {avg_force:.1f} N")
            print(f"  Peak force: {peak_force:.1f} N")
        else:
            print(f"\n{phase_name}:")
            print(f"  Duration: {duration:.0f} ms")


    # ===== 5. CALCULATE FORCE-VELOCITY PROFILE =====
    print("\n===== FORCE-VELOCITY PROFILE =====")

    # Get force and velocity time series
    if hasattr(jump, 'velocity') and hasattr(jump, 'force_platform'):
        # During propulsion phase
        t_start = jump.propulsion_start
        t_end = jump.takeoff

        phase_vel = jump.velocity[t_start:t_end].to_numpy()
        phase_force = abs(grf_z[t_start:t_end].to_numpy())

        # Find peak force and corresponding velocity
        idx_peak_force = np.argmax(phase_force)
        vel_at_peak_force = phase_vel[idx_peak_force]

        # Find peak velocity and corresponding force
        idx_peak_vel = np.argmax(phase_vel)
        force_at_peak_vel = phase_force[idx_peak_vel]

        print(f"At peak force: velocity = {vel_at_peak_force:.2f} m/s")
        print(f"At peak velocity: force = {force_at_peak_vel:.1f} N")

        # F-V slope (simple linear approximation)
        fv_slope = (jump.peak_force - force_at_peak_vel) / (vel_at_peak_force - jump.peak_velocity)
        print(f"F-V slope: {fv_slope:.1f} N·s/m")


    # ===== 6. CREATE VISUALIZATIONS =====
    print("\nCreating visualizations...")

    # Plot 1: Complete jump profile (force, velocity, power, displacement)
    fig1 = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Force', 'Velocity', 'Power', 'Displacement'),
        vertical_spacing=0.06,
        shared_xaxes=True
    )

    # Force
    fig1.add_trace(
        go.Scatter(x=grf_z.index, y=grf_z.to_numpy(),
                   name='Force', line=dict(color='blue', width=2)),
        row=1, col=1
    )

    # Velocity (if available)
    if hasattr(jump, 'velocity'):
        fig1.add_trace(
            go.Scatter(x=jump.velocity.index, y=jump.velocity.to_numpy(),
                       name='Velocity', line=dict(color='green', width=2)),
            row=2, col=1
        )

    # Power (if available)
    if hasattr(jump, 'power'):
        fig1.add_trace(
            go.Scatter(x=jump.power.index, y=jump.power.to_numpy(),
                       name='Power', line=dict(color='red', width=2)),
            row=3, col=1
        )

    # Displacement (if available)
    if hasattr(jump, 'center_of_mass_position'):
        fig1.add_trace(
            go.Scatter(x=jump.center_of_mass_position.index,
                       y=jump.center_of_mass_position.to_numpy()*100,  # Convert to cm
                       name='CoM Height', line=dict(color='purple', width=2)),
            row=4, col=1
        )

    # Mark key events
    for row in range(1, 5):
        fig1.add_vline(x=jump.propulsion_start, line_dash="dot", line_color="green",
                       annotation_text="Propulsion" if row == 1 else "", row=row, col=1)
        fig1.add_vline(x=jump.takeoff, line_dash="solid", line_color="red",
                       annotation_text="Takeoff" if row == 1 else "", row=row, col=1)
        fig1.add_vline(x=jump.landing, line_dash="solid", line_color="orange",
                       annotation_text="Landing" if row == 1 else "", row=row, col=1)

    fig1.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig1.update_yaxes(title_text="Force (N)", row=1, col=1)
    fig1.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    fig1.update_yaxes(title_text="Power (W)", row=3, col=1)
    fig1.update_yaxes(title_text="Height (cm)", row=4, col=1)

    fig1.update_layout(
        title="Countermovement Jump Profile",
        height=1000,
        template='plotly_white',
        showlegend=False,
        hovermode='x unified'
    )


    # Plot 2: Force-Velocity relationship (if data available)
    if hasattr(jump, 'velocity'):
        fig2 = go.Figure()

        # Scatter plot of force vs velocity during propulsion
        fig2.add_trace(
            go.Scatter(
                x=phase_vel,
                y=phase_force,
                mode='markers',
                marker=dict(
                    size=4,
                    color=np.arange(len(phase_vel)),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time")
                ),
                name='F-V data'
            )
        )

        # Mark peak force and peak velocity
        fig2.add_trace(
            go.Scatter(
                x=[vel_at_peak_force],
                y=[jump.peak_force],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Peak Force'
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=[jump.peak_velocity],
                y=[force_at_peak_vel],
                mode='markers',
                marker=dict(size=12, color='blue', symbol='star'),
                name='Peak Velocity'
            )
        )

        fig2.update_layout(
            title="Force-Velocity Profile (Propulsion Phase)",
            xaxis_title="Velocity (m/s)",
            yaxis_title="Force (N)",
            template='plotly_white'
        )

        fig2.show()


    # Plot 3: Comparison to normative bands
    fig3 = go.Figure()

    metrics = ['Jump Height\n(cm)', 'Peak Force\n(N/kg)', 'Peak Power\n(W/kg)', 'RSI-mod']
    athlete_vals = df_norm['Athlete'].values
    elite_vals = df_norm['Elite Avg'].values
    rec_vals = df_norm['Recreational Avg'].values

    x_pos = np.arange(len(metrics))

    # Normative bands
    fig3.add_trace(
        go.Bar(x=metrics, y=rec_vals, name='Recreational',
               marker=dict(color='lightgray'), opacity=0.5)
    )

    fig3.add_trace(
        go.Bar(x=metrics, y=elite_vals, name='Elite',
               marker=dict(color='lightblue'), opacity=0.5)
    )

    # Athlete values
    fig3.add_trace(
        go.Scatter(x=metrics, y=athlete_vals, mode='markers+lines',
                   name='Athlete', marker=dict(size=12, color='red'),
                   line=dict(color='red', width=2))
    )

    fig3.update_layout(
        title="Performance vs Normative Data",
        yaxis_title="Value",
        template='plotly_white',
        barmode='group'
    )

    fig3.show()


    # ===== 7. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()


    # ===== 8. EXPORT RESULTS =====
    print("\nExporting results...")

    # Comprehensive results table
    results = {
        'Category': [
            'Performance', '', '', '',
            'Force/Power', '', '',
            'Temporal', '', '',
            'Phases', ''
        ],
        'Metric': [
            'Jump Height', 'Flight Time', 'Takeoff Velocity', 'RSI-modified',
            'Peak Force', 'Peak Power', 'Peak Velocity',
            'Eccentric Duration', 'Concentric Duration', 'CM Depth',
            'Takeoff Time', 'Landing Time'
        ],
        'Value': [
            jump.height * 100,
            jump.flight_time * 1000,
            jump.takeoff_velocity,
            jump.rsi_modified,
            jump.peak_force,
            jump.peak_power,
            jump.peak_velocity,
            jump.eccentric_duration * 1000,
            jump.concentric_duration * 1000,
            abs(jump.countermovement_depth) * 100,
            jump.takeoff,
            jump.landing
        ],
        'Unit': [
            'cm', 'ms', 'm/s', '-',
            'N', 'W', 'm/s',
            'ms', 'ms', 'cm',
            's', 's'
        ]
    }

    df_results = pd.DataFrame(results)
    df_results.to_csv("cmj_test_results.csv", index=False)
    print("✓ Saved: cmj_test_results.csv")

    # Export normative comparison
    df_norm.to_csv("cmj_normative_comparison.csv", index=False)
    print("✓ Saved: cmj_normative_comparison.csv")


    # ===== 9. GENERATE ATHLETE REPORT =====
    print("\n===== ATHLETE REPORT =====")

    report = f"""
COUNTERMOVEMENT JUMP TEST REPORT
=================================

Athlete ID: [Replace with athlete info]
Test Date: 2026-06-17
Test Type: CMJ (Countermovement Jump)

KEY PERFORMANCE INDICATORS
---------------------------
Jump Height:        {jump.height*100:.1f} cm
Flight Time:        {jump.flight_time*1000:.0f} ms
Peak Power:         {jump.peak_power:.0f} W ({jump.peak_power/athlete_mass:.1f} W/kg)
RSI-modified:       {jump.rsi_modified:.2f}

FORCE CHARACTERISTICS
---------------------
Peak Force:         {jump.peak_force:.0f} N ({jump.peak_force/athlete_mass:.1f} N/kg)
Peak Velocity:      {jump.peak_velocity:.2f} m/s
Takeoff Velocity:   {jump.takeoff_velocity:.2f} m/s

MOVEMENT STRATEGY
-----------------
CM Depth:           {abs(jump.countermovement_depth)*100:.1f} cm
Eccentric Phase:    {jump.eccentric_duration*1000:.0f} ms
Concentric Phase:   {jump.concentric_duration*1000:.0f} ms

INTERPRETATION
--------------
Jump height of {jump.height*100:.1f} cm is {'above' if jump.height*100 > 45 else 'below'} elite average (45 cm).
RSI-modified of {jump.rsi_modified:.2f} indicates {'good' if jump.rsi_modified > 0.5 else 'moderate'} reactive strength.

RECOMMENDATIONS
---------------
[Add sport-specific recommendations based on results]
"""

    with open("cmj_athlete_report.txt", "w") as f:
        f.write(report)

    print(report)
    print("✓ Saved: cmj_athlete_report.txt")


if __name__ == "__main__":
    main()
