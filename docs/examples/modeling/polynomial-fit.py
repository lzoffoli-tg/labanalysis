"""
Polynomial Regression Example
==============================

Demonstrates polynomial regression for biomechanical modeling:
1. Load force-velocity data from jump test
2. Fit polynomial regression model
3. Validate model with residual analysis
4. Extract biomechanical parameters (F0, V0, Pmax)
5. Visualize force-velocity profile
6. Compare to theoretical optimal profile

Common use case: Force-velocity profiling, power-force-velocity assessment.
"""

import labanalysis as laban
from labanalysis.modelling.ols import PolynomialRegression
from labanalysis.signalprocessing import butterworth_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def main():
    # ===== 1. LOAD JUMP TEST DATA =====
    print("Loading jump test data...")

    # Load CMJ test
    jump = laban.SingleJump.from_tdf_file(
        "path/to/cmj_test.tdf",
        labels="LABEL",
        fp_label="FP1"
    )

    print(f"✓ Loaded jump test")
    print(f"Jump height: {jump.height*100:.1f} cm")
    print(f"Peak power: {jump.peak_power:.1f} W")


    # ===== 2. EXTRACT FORCE-VELOCITY DATA =====
    print("\nExtracting force-velocity data...")

    # Get propulsion phase
    t_start = jump.propulsion_start
    t_end = jump.takeoff

    # Extract force and velocity during propulsion
    grf_z = jump.force_platform["FORCE", "Z"][t_start:t_end]
    grf_filtered = butterworth_filter(grf_z, frequency=10, order=4)

    # Get velocity (if available)
    if hasattr(jump, 'velocity'):
        velocity = jump.velocity[t_start:t_end]
    else:
        print("ERROR: Velocity data not available")
        return

    # Convert to numpy arrays
    force_data = abs(grf_filtered.to_numpy())  # Absolute values
    velocity_data = velocity.to_numpy()

    # Remove any NaN values
    valid = ~(np.isnan(force_data) | np.isnan(velocity_data))
    force_data = force_data[valid]
    velocity_data = velocity_data[valid]

    print(f"Data points: {len(force_data)}")
    print(f"Force range: {force_data.min():.1f} - {force_data.max():.1f} N")
    print(f"Velocity range: {velocity_data.min():.2f} - {velocity_data.max():.2f} m/s")


    # ===== 3. FIT POLYNOMIAL MODELS =====
    print("\n===== POLYNOMIAL REGRESSION =====")

    # Fit different polynomial orders
    models = {}
    for degree in [1, 2, 3]:
        print(f"\nFitting {degree}-degree polynomial...")

        model = PolynomialRegression(degree=degree)
        model.fit(velocity_data.reshape(-1, 1), force_data)

        # Get model statistics
        r2 = model.r_squared
        rmse = model.rmse
        aic = model.aic

        print(f"  R² = {r2:.4f}")
        print(f"  RMSE = {rmse:.2f} N")
        print(f"  AIC = {aic:.2f}")

        models[degree] = model


    # ===== 4. SELECT BEST MODEL =====
    print("\n===== MODEL SELECTION =====")

    # Choose model with lowest AIC (balances fit quality and complexity)
    best_degree = min(models.keys(), key=lambda d: models[d].aic)
    best_model = models[best_degree]

    print(f"\nBest model: {best_degree}-degree polynomial")
    print(f"  R² = {best_model.r_squared:.4f}")
    print(f"  RMSE = {best_model.rmse:.2f} N")
    print(f"  AIC = {best_model.aic:.2f}")

    # Get model coefficients
    print(f"\nCoefficients:")
    for i, coef in enumerate(best_model.coefficients):
        print(f"  β{i} = {coef:.4f}")


    # ===== 5. EXTRACT BIOMECHANICAL PARAMETERS =====
    print("\n===== BIOMECHANICAL PARAMETERS =====")

    # Create prediction grid
    velocity_range = np.linspace(0, velocity_data.max() * 1.2, 100)
    force_pred = best_model.predict(velocity_range.reshape(-1, 1))

    # F0: Theoretical maximal force (force at zero velocity)
    F0 = best_model.predict(np.array([[0]]))[0]

    # V0: Theoretical maximal velocity (velocity at zero force)
    # Find where force prediction crosses zero
    zero_crossings = np.where(np.diff(np.sign(force_pred)))[0]
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        V0 = velocity_range[idx]
    else:
        # Extrapolate to find V0
        V0 = velocity_data.max() * 1.5  # Estimate

    # Pmax: Maximal power (F × V)
    power_pred = force_pred * velocity_range
    Pmax = power_pred.max()
    Vopt = velocity_range[np.argmax(power_pred)]  # Velocity at Pmax
    Fopt = force_pred[np.argmax(power_pred)]       # Force at Pmax

    print(f"\nF-V Profile Parameters:")
    print(f"  F0 (maximal force): {F0:.1f} N")
    print(f"  V0 (maximal velocity): {V0:.2f} m/s")
    print(f"  Pmax (maximal power): {Pmax:.1f} W")
    print(f"  Vopt (optimal velocity): {Vopt:.2f} m/s")
    print(f"  Fopt (force at Pmax): {Fopt:.1f} N")

    # F-V slope (mechanical characteristic)
    fv_slope = -F0 / V0
    print(f"  F-V slope: {fv_slope:.1f} N·s/m")


    # ===== 6. CALCULATE FV IMBALANCE =====
    print("\n===== FORCE-VELOCITY IMBALANCE =====")

    # FV imbalance: deviation from optimal (50% F0, 50% V0)
    # Optimal occurs at V = 0.5 * V0
    actual_velocity_ratio = Vopt / V0
    fv_imbalance = (actual_velocity_ratio - 0.5) * 100

    print(f"Actual V/V0 ratio: {actual_velocity_ratio:.2%}")
    print(f"FV imbalance: {fv_imbalance:+.1f}%")

    if abs(fv_imbalance) < 10:
        print("  ✓ Well-balanced profile")
    elif fv_imbalance > 0:
        print("  ⚠ Velocity-dominant (train maximal strength)")
    else:
        print("  ⚠ Force-dominant (train power/speed)")


    # ===== 7. RESIDUAL ANALYSIS =====
    print("\n===== RESIDUAL ANALYSIS =====")

    # Calculate residuals
    force_fitted = best_model.predict(velocity_data.reshape(-1, 1))
    residuals = force_data - force_fitted

    # Residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    print(f"Mean residual: {mean_residual:.2f} N (should be ~0)")
    print(f"Std residual: {std_residual:.2f} N")

    # Check for systematic bias
    if abs(mean_residual) > 10:
        print("  ⚠ Warning: Systematic bias detected")
    else:
        print("  ✓ No systematic bias")


    # ===== 8. CREATE VISUALIZATIONS =====
    print("\nCreating visualizations...")

    # Plot 1: Force-velocity relationship with model fit
    fig1 = go.Figure()

    # Scatter plot of actual data
    fig1.add_trace(
        go.Scatter(
            x=velocity_data,
            y=force_data,
            mode='markers',
            marker=dict(size=4, color='lightblue', opacity=0.6),
            name='Measured data'
        )
    )

    # Model prediction
    fig1.add_trace(
        go.Scatter(
            x=velocity_range,
            y=force_pred,
            mode='lines',
            line=dict(color='red', width=3),
            name=f'{best_degree}° polynomial fit'
        )
    )

    # Mark key points
    fig1.add_trace(
        go.Scatter(
            x=[0, Vopt, V0],
            y=[F0, Fopt, 0],
            mode='markers+text',
            marker=dict(size=10, color=['blue', 'green', 'purple']),
            text=['F0', 'Pmax', 'V0'],
            textposition='top center',
            name='Key points'
        )
    )

    fig1.update_layout(
        title=f"Force-Velocity Profile (R² = {best_model.r_squared:.3f})",
        xaxis_title="Velocity (m/s)",
        yaxis_title="Force (N)",
        template='plotly_white',
        hovermode='closest'
    )


    # Plot 2: Power-velocity relationship
    fig2 = go.Figure()

    # Actual power
    power_actual = force_data * velocity_data
    fig2.add_trace(
        go.Scatter(
            x=velocity_data,
            y=power_actual,
            mode='markers',
            marker=dict(size=4, color='lightgreen', opacity=0.6),
            name='Measured power'
        )
    )

    # Predicted power
    fig2.add_trace(
        go.Scatter(
            x=velocity_range,
            y=power_pred,
            mode='lines',
            line=dict(color='green', width=3),
            name='Predicted power'
        )
    )

    # Mark Pmax
    fig2.add_trace(
        go.Scatter(
            x=[Vopt],
            y=[Pmax],
            mode='markers+text',
            marker=dict(size=12, color='red', symbol='star'),
            text=['Pmax'],
            textposition='top center',
            name='Pmax'
        )
    )

    fig2.update_layout(
        title=f"Power-Velocity Profile (Pmax = {Pmax:.0f} W)",
        xaxis_title="Velocity (m/s)",
        yaxis_title="Power (W)",
        template='plotly_white'
    )


    # Plot 3: Residual plot
    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals vs Velocity', 'Residual Distribution')
    )

    # Residuals vs velocity
    fig3.add_trace(
        go.Scatter(
            x=velocity_data,
            y=residuals,
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.6),
            name='Residuals'
        ),
        row=1, col=1
    )

    fig3.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Residual histogram
    fig3.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=20,
            marker_color='blue',
            opacity=0.7,
            name='Distribution'
        ),
        row=1, col=2
    )

    fig3.update_xaxes(title_text="Velocity (m/s)", row=1, col=1)
    fig3.update_xaxes(title_text="Residual (N)", row=1, col=2)
    fig3.update_yaxes(title_text="Residual (N)", row=1, col=1)
    fig3.update_yaxes(title_text="Count", row=1, col=2)

    fig3.update_layout(
        title="Residual Analysis",
        height=400,
        template='plotly_white',
        showlegend=False
    )


    # Plot 4: Model comparison (all degrees)
    fig4 = go.Figure()

    # Data points
    fig4.add_trace(
        go.Scatter(
            x=velocity_data,
            y=force_data,
            mode='markers',
            marker=dict(size=4, color='lightgray'),
            name='Data'
        )
    )

    # All models
    colors = ['blue', 'green', 'red']
    for degree, color in zip([1, 2, 3], colors):
        model = models[degree]
        force_model = model.predict(velocity_range.reshape(-1, 1))

        fig4.add_trace(
            go.Scatter(
                x=velocity_range,
                y=force_model,
                mode='lines',
                line=dict(color=color, width=2, dash='solid' if degree == best_degree else 'dash'),
                name=f'Degree {degree} (R²={model.r_squared:.3f})'
            )
        )

    fig4.update_layout(
        title="Polynomial Model Comparison",
        xaxis_title="Velocity (m/s)",
        yaxis_title="Force (N)",
        template='plotly_white'
    )


    # ===== 9. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()


    # ===== 10. EXPORT RESULTS =====
    print("\nExporting results...")

    # Summary table
    summary = {
        'Parameter': [
            'Model Degree',
            'R²',
            'RMSE (N)',
            'AIC',
            'F0 (N)',
            'V0 (m/s)',
            'Pmax (W)',
            'Vopt (m/s)',
            'Fopt (N)',
            'FV Slope (N·s/m)',
            'FV Imbalance (%)'
        ],
        'Value': [
            best_degree,
            f"{best_model.r_squared:.4f}",
            f"{best_model.rmse:.2f}",
            f"{best_model.aic:.2f}",
            f"{F0:.1f}",
            f"{V0:.2f}",
            f"{Pmax:.1f}",
            f"{Vopt:.2f}",
            f"{Fopt:.1f}",
            f"{fv_slope:.1f}",
            f"{fv_imbalance:+.1f}"
        ]
    }

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("fv_profile_results.csv", index=False)
    print("✓ Saved: fv_profile_results.csv")

    # Export model predictions
    df_model = pd.DataFrame({
        'Velocity': velocity_range,
        'Force_Predicted': force_pred,
        'Power_Predicted': power_pred
    })

    df_model.to_csv("fv_model_predictions.csv", index=False)
    print("✓ Saved: fv_model_predictions.csv")


    print("\n===== ANALYSIS COMPLETE =====")
    print(f"Best fit: {best_degree}-degree polynomial (R² = {best_model.r_squared:.3f})")
    print(f"FV Profile: F0 = {F0:.0f} N, V0 = {V0:.2f} m/s, Pmax = {Pmax:.0f} W")


if __name__ == "__main__":
    main()
