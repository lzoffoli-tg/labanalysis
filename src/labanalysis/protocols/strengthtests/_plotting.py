"""Shared plotting utilities for strength tests."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...constants import RANK_4COLORS, SIDE_COLORS


def _get_force_figure(
    tracks: pd.DataFrame,
    summary: pd.DataFrame,
    include_emg: bool = True,
    time_mode: str = 'percentage',  # 'percentage' or 'absolute'
    time_points: list[int] = [100, 200, 500, 1000],  # Time points for markers (ms)
    max_time_ms: int = 2000,  # Maximum time for X-axis (ms)
):

    # generate the figure
    def get_muscles():
        lbls = [i for i in summary.parameter if i.endswith(" (%)")]
        return np.unique([i.rsplit(" ", 1)[0] for i in lbls]).tolist()

    def balance_string(left: str, right: str, sep: str = " | "):
        width = max(len(left), len(right))
        nbsp = "\u00a0"  # non-breaking
        ljust = left.rjust(width).replace(" ", nbsp)
        rjust = right.ljust(width).replace(" ", nbsp)
        return sep.join([ljust, rjust])

    plot_emg_muscle_balance = len(get_muscles()) > 0 and include_emg
    sides = np.unique(tracks.side)
    titles = [i.capitalize() for i in sides]
    ncols = len(sides)
    if plot_emg_muscle_balance:
        ncols += 1
        titles += ["Muscle Inbalance"]
    fig = make_subplots(
        rows=1,
        cols=ncols,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=titles,
        horizontal_spacing=0.1,
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
        width=500 * len(titles),
    )
    if plot_emg_muscle_balance:
        labels = list(RANK_4COLORS.keys())
        colors = list(RANK_4COLORS.values())
        values = [10, 20, 30, 40]
        cscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
        fig.update_layout(
            coloraxis=dict(
                colorscale=cscale,
                cmin=0,
                cmax=50,
                colorbar=dict(
                    title=dict(text="Imbalance<br>Levels"),
                    len=0.95,
                    y=0.5,
                    tickmode="array",
                    tickvals=values,
                    ticktext=labels,
                ),
            ),
        )

    # plot force profiles
    f_data = tracks.loc[tracks.parameter == "force_amplitude"]

    # Determine time column based on mode
    time_col = "time_ms" if time_mode == 'absolute' else "time_%"

    # Group by appropriate time column
    group_cols = [time_col, "side", "limb"]
    f_data = f_data.groupby(group_cols, as_index=False).max()

    for i, side in enumerate(sides):
        y = f_data.loc[f_data.side == side, "value"].to_numpy()  # type: ignore
        y = y.astype(float).flatten()
        x = f_data.loc[f_data.side == side, time_col].to_numpy()  # type: ignore
        x = x.astype(float).flatten()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name="force profile",
                showlegend=False,
                line_color=SIDE_COLORS[side],
            ),
            row=1,
            col=i + 1,
        )

        # Add markers for specific time points (only in absolute mode)
        if time_mode == 'absolute' and len(time_points) > 0:
            # Define colors for markers (distinct colors, same across subplots)
            marker_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

            # Pre-calculate all marker positions
            marker_data = []
            legend_lines = []

            for idx, tp in enumerate(time_points):
                if tp <= x.max():
                    force_val_n = float(np.interp(tp, x, y))
                    force_val_kn = force_val_n / 1000.0  # Convert to kN

                    # Get RFD from summary (now in kN/s)
                    rfd_param = f"RFD 0-{tp} ms (kN/s)"
                    rfd_row = summary.loc[summary.parameter == rfd_param]
                    rfd_val = None
                    if not rfd_row.empty and side in rfd_row.columns:
                        try:
                            rfd_val = float(rfd_row[side].iloc[0])
                        except (IndexError, ValueError, TypeError):
                            pass

                    # Get color for this marker (consistent across subplots)
                    color = marker_colors[idx % len(marker_colors)]

                    # Add marker on the curve
                    fig.add_trace(
                        go.Scatter(
                            x=[tp],
                            y=[force_val_n],  # Plot in N (original scale)
                            mode="markers",
                            marker=dict(
                                size=12,  # Large markers
                                color=color,
                                symbol='circle',
                                line=dict(width=2, color='white')
                            ),
                            showlegend=False,
                            hovertemplate=f"{tp}ms: {force_val_kn:.2f}kN<br>RFD: {rfd_val:.2f} kN/s<extra></extra>" if rfd_val else f"{tp}ms: {force_val_kn:.2f}kN<extra></extra>",
                        ),
                        row=1,
                        col=i + 1,
                    )

                    # Build legend text for this subplot (show in kN and kN/s)
                    legend_text = f"● {tp}ms: {force_val_kn:.2f}kN"
                    if rfd_val is not None:
                        legend_text += f" | RFD: {rfd_val:.2f} kN/s"
                    legend_lines.append(f'<span style="color:{color}">{legend_text}</span>')

            # Add peak force to legend with RFD
            peak_force_kn = np.max(y) / 1000.0
            peak_time_ms = x[np.argmax(y)]

            # Get overall RFD from summary
            rfd_overall_param = "rate of force development (kN/s)"
            rfd_overall_row = summary.loc[summary.parameter == rfd_overall_param]
            rfd_overall_text = ""
            if not rfd_overall_row.empty and side in rfd_overall_row.columns:
                try:
                    rfd_overall = float(rfd_overall_row[side].iloc[0])
                    rfd_overall_text = f" | RFD: {rfd_overall:.2f} kN/s"
                except (IndexError, ValueError, TypeError):
                    pass

            peak_legend = f"<b>Peak: {peak_time_ms:.0f}ms | {peak_force_kn:.2f}kN{rfd_overall_text}</b>"
            legend_lines.append(peak_legend)

            # Add legend as annotation in bottom-right corner of this subplot
            legend_text_html = "<br>".join(legend_lines)

            # Determine subplot domain for positioning
            # Each subplot has domain [col_start, col_end] in x-axis
            col_width = 1.0 / ncols
            x_domain_start = i * col_width
            x_domain_end = (i + 1) * col_width

            fig.add_annotation(
                text=legend_text_html,
                xref="paper",
                yref="paper",
                x=x_domain_end - 0.01,  # Bottom-right corner with small margin
                y=0.05,  # Near bottom
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=9, color="black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4,
            )

        # Add peak marker (same size as time point markers)
        x_peak = x[np.argmax(y)]
        fig.add_trace(
            go.Scatter(
                x=[x_peak],
                y=[np.max(y)],
                mode="markers",
                marker=dict(
                    size=12,  # Same size as time point markers
                    color='black',
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                showlegend=False,
                hovertemplate=f"Peak: {x_peak:.0f}ms | {np.max(y)/1000:.2f}kN<extra></extra>",
            ),
            row=1,
            col=i + 1,
        )


    # update force profiles figure layout
    yrange = f_data["value"].to_numpy().flatten()  # type: ignore
    yrange = np.array([np.min(yrange), np.max(yrange)])
    yrange *= np.array([0.95, 1.15])  # Reduced expansion (was 0.9, 1.3)
    yrange = yrange.tolist()

    # Configure x-axis based on time mode
    if time_mode == 'absolute':
        # Use max_time_ms as upper limit for X-axis
        xrange = [0, max_time_ms]
        # Create ticks at key intervals for absolute time
        xticks = [0, 500, 1000, 1500, 2000, 2500, 3000]
        # Filter ticks to only those within the range
        xticks = [t for t in xticks if xrange[0] <= t <= xrange[1]]
        xlabel = "Time (ms)"
    else:
        xrange = f_data["time_%"].to_numpy().flatten()  # type: ignore
        xrange = [np.min(xrange), np.max(xrange)]
        xticks = np.linspace(xrange[0], xrange[1], 5)
        xticks = [int(round(i / 5) * 5) for i in xticks]
        xlabel = "Concentric Phase (%)"

    for i in range(len(sides)):
        fig.update_xaxes(
            title=xlabel,
            row=1,
            col=i + 1,
            showline=True,
            linewidth=2,
            linecolor="black",
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            tickvals=xticks,
            tickmode="array",
            range=[min(xticks), max(xticks)],
        )
        # Generate y-axis ticks in kN
        yticks_n = np.linspace(yrange[0], yrange[1], 6)
        yticks_kn = [f"{yn/1000:.1f}" for yn in yticks_n]

        fig.update_yaxes(
            title="Force (kN)",
            range=yrange,  # Keep range in N for plotting
            tickvals=yticks_n,
            ticktext=yticks_kn,
            row=1,
            col=i + 1,
            showline=True,
            linewidth=2,
            linecolor="black",
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            matches='y' if i > 0 else None,  # Share Y-axis across subplots
        )

    # plot muscle data
    if plot_emg_muscle_balance:
        if "symmetry (%)" not in summary.columns:
            raise ValueError("'symmetry (%)' missing from summary dataframe")
        if "parameter" not in summary.columns:
            raise ValueError("'parameter' missing from summary dataframe")
        parameters = summary.parameter.to_numpy().flatten()
        symmetries = summary["symmetry (%)"].to_numpy().flatten()
        vals = []
        for i, muscle in enumerate(get_muscles()):
            idx = [j for j, v in enumerate(parameters) if muscle in v]
            if len(idx) == 0:
                continue
            idx = idx[0]
            symm = float(symmetries[idx])
            val = max(-50, min(50, symm))
            vals.append(val)
            lbl = f"{abs(symm):+0.1f}%" if -50 <= symm <= 50 else ">50.0%"
            fig.add_trace(
                trace=go.Bar(
                    x=[val],
                    y=[muscle.replace(" ", "<br>")],
                    orientation="h",
                    text=[lbl],
                    textangle=0,
                    textposition="outside",
                    name="Muscle imbalance",
                    legendgroup="Muscle imbalance",
                    showlegend=False,
                    marker=dict(
                        color=[abs(symm)],
                        coloraxis="coloraxis",
                    ),
                ),
                row=1,
                col=ncols,
            )

        # adjust the muscle balance axes
        vrange = np.max(abs(np.array(vals))) * 1.5
        vrange = max(50, np.max(abs(np.array(vals))) * 1.5)
        vrange = [-vrange, vrange]
        fig.update_xaxes(
            row=1,
            col=ncols,
            title=f"{balance_string("Left", "Right", "|")}<br>(%)",
            zeroline=False,
            showline=False,
            showgrid=False,
            showticklabels=False,
            range=vrange,
        )
        fig.update_yaxes(
            row=1,
            col=ncols,
            title="",
            showgrid=False,
            zeroline=True,
            showline=False,
            zerolinecolor="black",
            zerolinewidth=2,
        )
        fig.add_vline(
            x=0,
            row=1,  # type: ignore
            col=ncols,  # type: ignore
            showlegend=False,
            line_color="black",
            line_width=2,
            line_dash="solid",
        )

    return fig


