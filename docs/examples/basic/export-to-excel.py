"""
Export to Excel Example
========================

Demonstrates data export workflow:
1. Load and process test data
2. Extract key metrics and time series
3. Export to Excel with multiple sheets
4. Format Excel file for readability
5. Create summary reports

Common use case: Generate athlete reports or analysis summaries in Excel format.
"""

import labanalysis as laban
from labanalysis.signalprocessing import butterworth_filter
import pandas as pd
import numpy as np


def main():
    # ===== 1. LOAD AND PROCESS DATA =====
    print("Loading jump test data...")

    # Load CMJ (countermovement jump) test
    jump = laban.SingleJump.from_tdf_file(
        "path/to/jump_test.tdf",
        labels="LABEL",
        fp_label="FP1"
    )

    print(f"Jump height: {jump.height*100:.1f} cm")
    print(f"Peak force: {jump.peak_force:.1f} N")
    print(f"Peak power: {jump.peak_power:.1f} W")


    # Filter force data for cleaner signals
    grf_vertical = jump.force_platform["FORCE", "Z"]
    grf_filtered = butterworth_filter(grf_vertical, frequency=10, order=4)


    # ===== 2. EXTRACT KEY METRICS =====
    print("\nExtracting metrics...")

    # Summary metrics
    metrics = {
        'Metric': [
            'Jump Height',
            'Flight Time',
            'Takeoff Velocity',
            'Peak Force',
            'Peak Power',
            'Peak Velocity',
            'Countermovement Depth',
            'Eccentric Duration',
            'Concentric Duration',
            'RSI-modified',
            'Takeoff Time',
            'Landing Time'
        ],
        'Value': [
            jump.height * 100,  # Convert to cm
            jump.flight_time * 1000,  # Convert to ms
            jump.takeoff_velocity,
            jump.peak_force,
            jump.peak_power,
            jump.peak_velocity,
            abs(jump.countermovement_depth) * 100,  # Convert to cm
            jump.eccentric_duration * 1000,  # Convert to ms
            jump.concentric_duration * 1000,  # Convert to ms
            jump.rsi_modified,
            jump.takeoff,
            jump.landing
        ],
        'Unit': [
            'cm',
            'ms',
            'm/s',
            'N',
            'W',
            'm/s',
            'cm',
            'ms',
            'ms',
            '-',
            's',
            's'
        ]
    }

    df_metrics = pd.DataFrame(metrics)


    # ===== 3. EXTRACT TIME SERIES DATA =====
    print("Extracting time series...")

    # Get key phases
    fp = jump.force_platform

    # Time series: Force, velocity, power
    time_series = pd.DataFrame({
        'Time (s)': grf_filtered.index,
        'Force (N)': grf_filtered.to_numpy(),
        'Velocity (m/s)': jump.velocity.to_numpy() if hasattr(jump, 'velocity') else np.nan,
        'Power (W)': jump.power.to_numpy() if hasattr(jump, 'power') else np.nan,
        'Position (m)': jump.center_of_mass_position.to_numpy() if hasattr(jump, 'center_of_mass_position') else np.nan
    })


    # ===== 4. CREATE PHASE MARKERS =====
    print("Creating phase information...")

    phases = pd.DataFrame({
        'Phase': [
            'Weighing',
            'Unweighing',
            'Braking',
            'Propulsion',
            'Flight',
            'Landing'
        ],
        'Start Time (s)': [
            0.0,
            jump.unweighing_start if hasattr(jump, 'unweighing_start') else np.nan,
            jump.braking_start if hasattr(jump, 'braking_start') else np.nan,
            jump.propulsion_start,
            jump.takeoff,
            jump.landing
        ],
        'End Time (s)': [
            jump.unweighing_start if hasattr(jump, 'unweighing_start') else np.nan,
            jump.braking_start if hasattr(jump, 'braking_start') else np.nan,
            jump.propulsion_start,
            jump.takeoff,
            jump.landing,
            jump.index[-1]
        ],
        'Duration (ms)': [
            np.nan,
            np.nan,
            np.nan,
            jump.concentric_duration * 1000,
            jump.flight_time * 1000,
            np.nan
        ]
    })


    # ===== 5. CREATE NORMATIVE COMPARISON (OPTIONAL) =====
    print("Creating normative comparison...")

    # Example normative data (replace with actual norms)
    athlete_mass = 75  # kg (replace with actual)

    normative = pd.DataFrame({
        'Metric': [
            'Jump Height',
            'Relative Peak Force',
            'Relative Peak Power'
        ],
        'Athlete Value': [
            jump.height * 100,
            jump.peak_force / athlete_mass,
            jump.peak_power / athlete_mass
        ],
        'Elite Average': [
            50.0,  # cm (example)
            25.0,  # N/kg (example)
            60.0   # W/kg (example)
        ],
        'Difference': [
            (jump.height * 100) - 50.0,
            (jump.peak_force / athlete_mass) - 25.0,
            (jump.peak_power / athlete_mass) - 60.0
        ],
        'Unit': [
            'cm',
            'N/kg',
            'W/kg'
        ]
    })

    # Calculate percentage difference
    normative['% Difference'] = (normative['Difference'] / normative['Elite Average'] * 100).round(1)


    # ===== 6. EXPORT TO EXCEL =====
    print("\nExporting to Excel...")

    # Create Excel writer
    output_file = "jump_analysis_report.xlsx"

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # Sheet 1: Summary metrics
        df_metrics.to_excel(
            writer,
            sheet_name='Summary',
            index=False,
            startrow=1
        )

        # Add header to Summary sheet
        worksheet = writer.sheets['Summary']
        worksheet['A1'] = 'COUNTERMOVEMENT JUMP ANALYSIS'
        worksheet['A1'].font = worksheet['A1'].font.copy(bold=True, size=14)

        # Format numeric columns
        for row in range(2, len(df_metrics) + 2):
            cell = worksheet[f'B{row}']
            if isinstance(cell.value, (int, float)):
                cell.number_format = '0.00'


        # Sheet 2: Time series data
        time_series.to_excel(
            writer,
            sheet_name='Time Series',
            index=False
        )


        # Sheet 3: Phase information
        phases.to_excel(
            writer,
            sheet_name='Phases',
            index=False
        )


        # Sheet 4: Normative comparison
        normative.to_excel(
            writer,
            sheet_name='Normative Comparison',
            index=False,
            startrow=1
        )

        worksheet_norm = writer.sheets['Normative Comparison']
        worksheet_norm['A1'] = 'COMPARISON TO NORMATIVE DATA'
        worksheet_norm['A1'].font = worksheet_norm['A1'].font.copy(bold=True, size=14)


        # Sheet 5: Metadata
        metadata = pd.DataFrame({
            'Field': [
                'Test Type',
                'Date',
                'Athlete ID',
                'Sampling Rate',
                'Filter Cutoff',
                'Analysis Software'
            ],
            'Value': [
                'Countermovement Jump',
                '2026-06-17',  # Replace with actual
                'Athlete_001',  # Replace with actual
                f"{1/np.mean(np.diff(jump.index)):.1f} Hz",
                '10 Hz Butterworth',
                'labanalysis v1.0'
            ]
        })

        metadata.to_excel(
            writer,
            sheet_name='Metadata',
            index=False
        )

    print(f"✓ Saved: {output_file}")


    # ===== 7. CREATE SIMPLIFIED CSV EXPORT =====
    print("\nExporting simplified CSV...")

    # Combine key metrics into single row for database/tracking
    summary_row = {
        'date': '2026-06-17',
        'athlete_id': 'Athlete_001',
        'test_type': 'CMJ',
        'jump_height_cm': round(jump.height * 100, 2),
        'flight_time_ms': round(jump.flight_time * 1000, 1),
        'peak_force_N': round(jump.peak_force, 1),
        'peak_power_W': round(jump.peak_power, 1),
        'peak_velocity_ms': round(jump.peak_velocity, 2),
        'rsi_modified': round(jump.rsi_modified, 2),
        'concentric_duration_ms': round(jump.concentric_duration * 1000, 1),
        'eccentric_duration_ms': round(jump.eccentric_duration * 1000, 1)
    }

    df_summary_row = pd.DataFrame([summary_row])
    df_summary_row.to_csv("jump_summary.csv", index=False)
    print("✓ Saved: jump_summary.csv")


    # ===== 8. PRINT SUMMARY =====
    print("\n===== EXPORT SUMMARY =====")
    print(f"\nExcel file created: {output_file}")
    print(f"  - Summary sheet: {len(df_metrics)} metrics")
    print(f"  - Time series: {len(time_series)} samples")
    print(f"  - Phases: {len(phases)} phases")
    print(f"  - Normative comparison: {len(normative)} metrics")
    print(f"\nCSV file created: jump_summary.csv")
    print(f"  - Single row summary for tracking/database")


    # ===== 9. ADVANCED: BATCH EXPORT MULTIPLE TESTS =====
    print("\n===== BATCH EXPORT EXAMPLE =====")
    print("For multiple tests, use this pattern:")
    print("""
    # Collect multiple tests
    test_results = []

    for file in test_files:
        jump = laban.SingleJump.from_tdf_file(file, labels="LABEL", fp_label="FP1")

        test_results.append({
            'file': file,
            'height_cm': jump.height * 100,
            'peak_force_N': jump.peak_force,
            # ... other metrics
        })

    # Create DataFrame and export
    df_batch = pd.DataFrame(test_results)
    df_batch.to_excel("batch_jump_analysis.xlsx", index=False)
    """)


if __name__ == "__main__":
    main()
