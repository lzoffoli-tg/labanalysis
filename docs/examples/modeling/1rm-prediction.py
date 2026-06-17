"""
1RM Prediction from Isokinetic Test Example
============================================

Demonstrates 1RM estimation from isokinetic strength testing:
1. Load isokinetic test data from Biostrength equipment
2. Extract peak force at different velocities
3. Fit force-velocity relationship
4. Predict 1RM using Brzycki equation
5. Compare multiple prediction methods
6. Generate strength profile report

Common use case: Strength assessment without maximal testing, athlete monitoring.
"""

import labanalysis as laban
from labanalysis.equations import Brzycki1RM
from labanalysis.modelling.ols import PolynomialRegression, PowerRegression
from labanalysis.signalprocessing import butterworth_filter, find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def main():
    # ===== 1. LOAD ISOKINETIC TEST DATA =====
    print("Loading isokinetic test data...")

    # Load Biostrength data
    # Typically contains multiple velocity conditions (60°/s, 180°/s, 300°/s)
    test_file = "path/to/isokinetic_test.csv"

    # For this example, we'll simulate isokinetic data at multiple velocities
    # In practice, load from Biostrength CSV files

    # Simulated data: Force measurements at different angular velocities
    test_velocities = [60, 120, 180, 240, 300]  # deg/s
    peak_forces = [850, 720, 620, 540, 480]     # N

    # Convert to linear velocity (approximate for knee extension)
    # v_linear ≈ v_angular × lever_arm
    lever_arm = 0.35  # meters (approximate lower leg length)
    linear_velocities = [np.deg2rad(v) * lever_arm for v in test_velocities]  # m/s

    print(f"✓ Loaded {len(test_velocities)} velocity conditions")
    print(f"Velocity range: {min(test_velocities)} - {max(test_velocities)} deg/s")
    print(f"Force range: {min(peak_forces)} - {max(peak_forces)} N")


    # ===== 2. DISPLAY RAW DATA =====
    print("\n===== RAW TEST DATA =====")
    for v_ang, v_lin, f in zip(test_velocities, linear_velocities, peak_forces):
        print(f"  {v_ang:3d} deg/s ({v_lin:.3f} m/s): {f:4.0f} N")


    # ===== 3. FIT FORCE-VELOCITY MODEL =====
    print("\n===== FORCE-VELOCITY MODELING =====")

    # Convert to numpy arrays
    velocities = np.array(linear_velocities).reshape(-1, 1)
    forces = np.array(peak_forces)

    # Method 1: Linear regression
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()
    linear_model.fit(velocities, forces)

    F0_linear = linear_model.predict(np.array([[0]]))[0]  # Force at zero velocity
    slope = linear_model.coef_[0]
    V0_linear = -F0_linear / slope  # Velocity at zero force

    print(f"\nLinear F-V Model:")
    print(f"  F0 (maximal force): {F0_linear:.1f} N")
    print(f"  V0 (maximal velocity): {V0_linear:.3f} m/s")
    print(f"  Slope: {slope:.1f} N·s/m")

    # Method 2: Polynomial regression (2nd degree)
    poly_model = PolynomialRegression(degree=2)
    poly_model.fit(velocities, forces)

    F0_poly = poly_model.predict(np.array([[0]]))[0]

    print(f"\nPolynomial F-V Model (degree 2):")
    print(f"  F0 (maximal force): {F0_poly:.1f} N")
    print(f"  R²: {poly_model.r_squared:.4f}")


    # ===== 4. PREDICT 1RM USING BRZYCKI EQUATION =====
    print("\n===== 1RM PREDICTION =====")

    # Brzycki equation: 1RM = weight / (1.0278 - 0.0278 × reps)
    # For single rep max from force: 1RM ≈ F0 / g

    # Method 1: Direct from F0 (assumes perfect form, no assistance)
    g = 9.81  # m/s²
    one_rm_direct = F0_linear / g

    print(f"\nDirect method (F0 / g):")
    print(f"  1RM: {one_rm_direct:.1f} kg")

    # Method 2: Using Brzycki with estimated reps from velocity
    # Higher velocities correspond to lighter loads (more reps possible)
    # This is a simplified relationship for demonstration

    # Estimate load from force (% of 1RM)
    loads_pct = forces / F0_linear * 100  # % of max force
    estimated_reps = 1 + (100 - loads_pct) / 2.5  # Simplified rep estimation

    print(f"\nBrzycki method (from submaximal velocities):")
    for v, f, pct, reps in zip(test_velocities, peak_forces, loads_pct, estimated_reps):
        brzycki_1rm = f / g / (1.0278 - 0.0278 * reps)
        print(f"  {v:3d} deg/s ({pct:5.1f}% max): {reps:4.1f} reps → 1RM = {brzycki_1rm:.1f} kg")

    # Average Brzycki prediction
    brzycki_predictions = [
        f / g / (1.0278 - 0.0278 * reps)
        for f, reps in zip(peak_forces, estimated_reps)
    ]
    one_rm_brzycki = np.mean(brzycki_predictions)

    print(f"\n  Average 1RM (Brzycki): {one_rm_brzycki:.1f} kg")


    # Method 3: Regression-based prediction
    # Use polynomial model to predict force at very low velocity
    very_low_velocity = 0.05  # m/s (very slow movement)
    force_at_low_v = poly_model.predict(np.array([[very_low_velocity]]))[0]
    one_rm_regression = force_at_low_v / g

    print(f"\nRegression method (force at {very_low_velocity} m/s):")
    print(f"  1RM: {one_rm_regression:.1f} kg")


    # ===== 5. CALCULATE CONFIDENCE INTERVALS =====
    print("\n===== PREDICTION CONFIDENCE =====")

    # Standard error of predictions
    std_predictions = np.std(brzycki_predictions)
    se_predictions = std_predictions / np.sqrt(len(brzycki_predictions))

    # 95% confidence interval
    ci_95 = 1.96 * se_predictions

    print(f"Standard deviation: {std_predictions:.1f} kg")
    print(f"95% CI: {one_rm_brzycki:.1f} ± {ci_95:.1f} kg")
    print(f"Range: {one_rm_brzycki - ci_95:.1f} - {one_rm_brzycki + ci_95:.1f} kg")


    # ===== 6. STRENGTH PROFILE ANALYSIS =====
    print("\n===== STRENGTH PROFILE =====")

    # Calculate strength deficit (difference between actual and predicted)
    # at each velocity

    # Predict force from linear model
    forces_predicted = linear_model.predict(velocities)
    deficits = forces - forces_predicted

    print("\nForce deficit analysis:")
    for v, actual, pred, deficit in zip(test_velocities, peak_forces, forces_predicted, deficits):
        deficit_pct = (deficit / pred) * 100
        print(f"  {v:3d} deg/s: {deficit:+6.1f} N ({deficit_pct:+5.1f}%)")

    # Identify strength characteristics
    deficit_at_low = deficits[0]  # Low velocity (high force)
    deficit_at_high = deficits[-1]  # High velocity (low force)

    if deficit_at_low < -20:
        print("\n⚠ Maximal strength deficit detected")
        print("  → Recommendation: Focus on heavy resistance training")
    elif deficit_at_high < -20:
        print("\n⚠ High-velocity strength deficit detected")
        print("  → Recommendation: Focus on power/explosive training")
    else:
        print("\n✓ Balanced strength profile")


    # ===== 7. CREATE VISUALIZATIONS =====
    print("\nCreating visualizations...")

    # Plot 1: Force-velocity relationship with models
    fig1 = go.Figure()

    # Measured data points
    fig1.add_trace(
        go.Scatter(
            x=linear_velocities,
            y=peak_forces,
            mode='markers',
            marker=dict(size=12, color='blue'),
            name='Measured data'
        )
    )

    # Model predictions
    v_range = np.linspace(0, max(linear_velocities) * 1.1, 100)
    f_linear = linear_model.predict(v_range.reshape(-1, 1))
    f_poly = poly_model.predict(v_range.reshape(-1, 1))

    fig1.add_trace(
        go.Scatter(
            x=v_range,
            y=f_linear,
            mode='lines',
            line=dict(color='red', width=2),
            name='Linear model'
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=v_range,
            y=f_poly,
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Polynomial model'
        )
    )

    # Mark F0
    fig1.add_trace(
        go.Scatter(
            x=[0],
            y=[F0_linear],
            mode='markers+text',
            marker=dict(size=14, color='red', symbol='star'),
            text=[f'F0 = {F0_linear:.0f} N'],
            textposition='top right',
            name='F0'
        )
    )

    fig1.update_layout(
        title="Force-Velocity Profile",
        xaxis_title="Velocity (m/s)",
        yaxis_title="Force (N)",
        template='plotly_white'
    )


    # Plot 2: 1RM predictions comparison
    fig2 = go.Figure()

    methods = ['Direct\n(F0/g)', 'Brzycki\n(avg)', 'Regression\n(low v)']
    predictions = [one_rm_direct, one_rm_brzycki, one_rm_regression]
    colors = ['blue', 'green', 'red']

    fig2.add_trace(
        go.Bar(
            x=methods,
            y=predictions,
            marker_color=colors,
            text=[f'{p:.1f} kg' for p in predictions],
            textposition='outside'
        )
    )

    # Add error bars for Brzycki method
    fig2.add_trace(
        go.Scatter(
            x=[methods[1]],
            y=[one_rm_brzycki],
            error_y=dict(
                type='data',
                array=[ci_95],
                visible=True,
                color='black'
            ),
            mode='markers',
            marker=dict(size=0),
            showlegend=False
        )
    )

    fig2.update_layout(
        title="1RM Prediction Methods Comparison",
        yaxis_title="Predicted 1RM (kg)",
        template='plotly_white',
        showlegend=False
    )


    # Plot 3: Strength deficit profile
    fig3 = go.Figure()

    fig3.add_trace(
        go.Bar(
            x=[f'{v} deg/s' for v in test_velocities],
            y=(deficits / forces_predicted * 100),  # % deficit
            marker_color=['red' if d < -5 else 'green' if d > 5 else 'gray'
                          for d in (deficits / forces_predicted * 100)],
            text=[f'{d:+.1f}%' for d in (deficits / forces_predicted * 100)],
            textposition='outside'
        )
    )

    fig3.add_hline(y=0, line_dash="dash", line_color="black")

    fig3.update_layout(
        title="Strength Deficit Profile",
        xaxis_title="Test Velocity",
        yaxis_title="Force Deficit (%)",
        template='plotly_white',
        showlegend=False
    )


    # ===== 8. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()
    fig2.show()
    fig3.show()


    # ===== 9. EXPORT RESULTS =====
    print("\nExporting results...")

    # Summary report
    report = {
        'Method': [
            'Direct (F0/g)',
            'Brzycki Average',
            'Regression (low v)',
            'Final Estimate'
        ],
        'Predicted 1RM (kg)': [
            f"{one_rm_direct:.1f}",
            f"{one_rm_brzycki:.1f}",
            f"{one_rm_regression:.1f}",
            f"{np.mean([one_rm_direct, one_rm_brzycki, one_rm_regression]):.1f}"
        ],
        '95% CI': [
            '-',
            f'± {ci_95:.1f}',
            '-',
            '-'
        ]
    }

    df_report = pd.DataFrame(report)
    df_report.to_csv("1rm_prediction_report.csv", index=False)
    print("✓ Saved: 1rm_prediction_report.csv")

    # Detailed measurements
    measurements = {
        'Velocity (deg/s)': test_velocities,
        'Velocity (m/s)': [f'{v:.3f}' for v in linear_velocities],
        'Peak Force (N)': peak_forces,
        'Predicted Force (N)': [f'{f:.1f}' for f in forces_predicted],
        'Deficit (N)': [f'{d:+.1f}' for d in deficits],
        'Deficit (%)': [f'{d:+.1f}' for d in (deficits / forces_predicted * 100)]
    }

    df_measurements = pd.DataFrame(measurements)
    df_measurements.to_csv("isokinetic_measurements.csv", index=False)
    print("✓ Saved: isokinetic_measurements.csv")


    # ===== 10. FINAL SUMMARY =====
    print("\n===== 1RM PREDICTION SUMMARY =====")
    print(f"\nForce-Velocity Parameters:")
    print(f"  F0: {F0_linear:.1f} N")
    print(f"  V0: {V0_linear:.3f} m/s")

    print(f"\n1RM Predictions:")
    print(f"  Direct method: {one_rm_direct:.1f} kg")
    print(f"  Brzycki method: {one_rm_brzycki:.1f} ± {ci_95:.1f} kg")
    print(f"  Regression method: {one_rm_regression:.1f} kg")

    final_estimate = np.mean([one_rm_direct, one_rm_brzycki, one_rm_regression])
    print(f"\n  Recommended 1RM estimate: {final_estimate:.1f} kg")

    print(f"\nStrength Profile:")
    if deficit_at_low < -20:
        print("  ⚠ Maximal strength deficit → Train heavy loads")
    elif deficit_at_high < -20:
        print("  ⚠ High-velocity deficit → Train explosive power")
    else:
        print("  ✓ Balanced profile")


if __name__ == "__main__":
    main()
