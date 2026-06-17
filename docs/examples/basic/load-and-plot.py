"""
Load and Plot Example
======================

Demonstrates basic data loading and visualization workflow:
1. Load marker data from BTS TDF file
2. Extract specific markers
3. Create interactive plots with Plotly
4. Display and save figures

This is a minimal working example showing the most common operations.
"""

import labanalysis as laban
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    # ===== 1. LOAD DATA =====
    print("Loading data from TDF file...")

    # Load WholeBody from BTS file
    # Replace with your actual file path
    body = laban.WholeBody.from_tdf_file(
        "path/to/your/file.tdf",
        labels="LABEL"  # BTS label field
    )

    print(f"Loaded {len(body.labels)} markers")
    print(f"Duration: {body.index[-1]:.2f} seconds")
    print(f"Sampling rate: {1/body.index.to_series().diff().mean():.1f} Hz")


    # ===== 2. EXTRACT SPECIFIC MARKERS =====
    print("\nExtracting pelvis markers...")

    # Get individual markers
    asis_l = body.get("ASIS_L")  # Left anterior superior iliac spine
    asis_r = body.get("ASIS_R")  # Right ASIS
    psis_l = body.get("PSIS_L")  # Left posterior superior iliac spine
    psis_r = body.get("PSIS_R")  # Right PSIS

    # Check if markers exist
    if asis_l is None:
        print("ERROR: ASIS_L marker not found in data")
        print(f"Available markers: {body.labels[:10]}...")  # Show first 10
        return


    # ===== 3. CREATE PLOTS =====
    print("\nCreating plots...")

    # Plot 1: Single marker trajectory (3D position over time)
    fig1 = make_subplots(
        rows=3, cols=1,
        subplot_titles=('X (Anterior-Posterior)', 'Y (Medial-Lateral)', 'Z (Vertical)'),
        vertical_spacing=0.08
    )

    # X component (anterior-posterior)
    fig1.add_trace(
        go.Scatter(
            x=asis_l.index,
            y=asis_l["X"].to_numpy(),
            mode='lines',
            name='X',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # Y component (medial-lateral)
    fig1.add_trace(
        go.Scatter(
            x=asis_l.index,
            y=asis_l["Y"].to_numpy(),
            mode='lines',
            name='Y',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )

    # Z component (vertical)
    fig1.add_trace(
        go.Scatter(
            x=asis_l.index,
            y=asis_l["Z"].to_numpy(),
            mode='lines',
            name='Z',
            line=dict(color='blue', width=2)
        ),
        row=3, col=1
    )

    fig1.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig1.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig1.update_yaxes(title_text="Position (m)", row=2, col=1)
    fig1.update_yaxes(title_text="Position (m)", row=3, col=1)

    fig1.update_layout(
        title="Left ASIS Marker Trajectory",
        height=800,
        showlegend=False,
        template='plotly_white'
    )


    # Plot 2: Multiple markers comparison (vertical displacement)
    fig2 = go.Figure()

    markers = {
        'ASIS Left': asis_l,
        'ASIS Right': asis_r,
        'PSIS Left': psis_l,
        'PSIS Right': psis_r
    }

    colors = ['blue', 'red', 'green', 'orange']

    for (name, marker), color in zip(markers.items(), colors):
        if marker is not None:
            fig2.add_trace(
                go.Scatter(
                    x=marker.index,
                    y=marker["Z"].to_numpy(),  # Vertical component
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2)
                )
            )

    fig2.update_layout(
        title="Pelvis Markers - Vertical Position",
        xaxis_title="Time (s)",
        yaxis_title="Vertical Position (m)",
        template='plotly_white',
        hovermode='x unified'
    )


    # Plot 3: 3D trajectory visualization
    fig3 = go.Figure()

    # Add ASIS left trajectory in 3D space
    fig3.add_trace(
        go.Scatter3d(
            x=asis_l["X"].to_numpy(),
            y=asis_l["Y"].to_numpy(),
            z=asis_l["Z"].to_numpy(),
            mode='lines',
            name='Left ASIS',
            line=dict(color='blue', width=4)
        )
    )

    # Add ASIS right trajectory
    if asis_r is not None:
        fig3.add_trace(
            go.Scatter3d(
                x=asis_r["X"].to_numpy(),
                y=asis_r["Y"].to_numpy(),
                z=asis_r["Z"].to_numpy(),
                mode='lines',
                name='Right ASIS',
                line=dict(color='red', width=4)
            )
        )

    fig3.update_layout(
        title="3D Marker Trajectories",
        scene=dict(
            xaxis_title='X - Anterior (m)',
            yaxis_title='Y - Lateral (m)',
            zaxis_title='Z - Vertical (m)',
            aspectmode='data'
        ),
        template='plotly_white'
    )


    # ===== 4. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()
    fig2.show()
    fig3.show()


    # ===== 5. SAVE PLOTS (OPTIONAL) =====
    print("\nSaving plots to HTML files...")
    fig1.write_html("output_trajectory_components.html")
    fig2.write_html("output_pelvis_vertical.html")
    fig3.write_html("output_3d_trajectory.html")
    print("✓ Saved: output_trajectory_components.html")
    print("✓ Saved: output_pelvis_vertical.html")
    print("✓ Saved: output_3d_trajectory.html")

    # Can also save as static images (requires kaleido)
    # fig1.write_image("output_trajectory.png", width=1200, height=800)


    # ===== 6. PRINT SUMMARY STATISTICS =====
    print("\n===== SUMMARY STATISTICS =====")
    print(f"\nLeft ASIS position:")
    print(f"  X: mean={asis_l['X'].to_numpy().mean():.3f} m, "
          f"std={asis_l['X'].to_numpy().std():.3f} m")
    print(f"  Y: mean={asis_l['Y'].to_numpy().mean():.3f} m, "
          f"std={asis_l['Y'].to_numpy().std():.3f} m")
    print(f"  Z: mean={asis_l['Z'].to_numpy().mean():.3f} m, "
          f"std={asis_l['Z'].to_numpy().std():.3f} m")

    # Calculate range of motion
    z_displacement = asis_l["Z"].to_numpy()
    rom = z_displacement.max() - z_displacement.min()
    print(f"\nVertical range of motion: {rom*1000:.1f} mm")


if __name__ == "__main__":
    main()
