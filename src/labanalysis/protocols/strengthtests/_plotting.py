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
        if time_mode == 'absolute':
            time_points = [100, 200, 500, 1000]
            for tp in time_points:
                # Get force at this time point from summary
                param_name = f"force at {tp} ms (N)"
                force_row = summary.loc[summary.parameter == param_name]
                if not force_row.empty and side in force_row.columns:
                    try:
                        force_val = float(force_row[side].iloc[0])
                        # Add marker on the curve
                        fig.add_trace(
                            go.Scatter(
                                x=[tp],
                                y=[force_val],
                                mode="markers+text",
                                text=f" {force_val:0.0f}N",  # Space before text for padding
                                textposition="middle right",
                                marker=dict(
                                    size=6,
                                    color=SIDE_COLORS[side],
                                    symbol='circle',
                                    opacity=0.6,
                                    line=dict(width=1, color='white')
                                ),
                                textfont=dict(size=9, color=SIDE_COLORS[side]),
                                showlegend=False,
                                name=f"force @ {tp}ms",
                            ),
                            row=1,
                            col=i + 1,
                        )
                    except (IndexError, ValueError, TypeError):
                        continue

        x_peak = x[np.argmax(y)]
        fig.add_trace(
            go.Scatter(
                x=[x_peak, x_peak],
                y=[0, np.max(y)],
                name="peak",
                line_dash="dash",
                line_color="black",
                opacity=0.5,
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )
        note = [f"{"Peak:"}{np.max(y):0.1f}N"]
        extras = [
            "estimated 1RM (kg)",
            "rate of force development (kN/s)",
            "time to peak force (ms)",
        ]
        for ext in extras:
            est_row = summary.loc[summary.parameter == ext]
            if est_row.empty:
                continue
            if side not in est_row.columns:
                continue
            # Extract the value directly using iloc
            try:
                est = float(est_row[side].iloc[0])
            except (IndexError, ValueError, TypeError):
                continue
            key = (
                ext.replace("estimated", "Est.")
                .replace("rate of force development", "RFD")
                .replace("time to peak force", "Time@F<sub>MAX</sub>")
            )
            note += [f"{key}: {est:0.1f}"]
        note = "<br>".join(note)
        if x_peak / np.max(x) < 0.40:
            dx = 20
            textposition = "top right"
        elif x_peak / np.max(x) > 0.60:
            dx = -20
            textposition = "top left"
        else:
            dx = 0
            textposition = "top center"
        fig.add_trace(
            row=1,
            col=i + 1,
            trace=go.Scatter(
                x=[x_peak],
                y=[np.max(y)],
                dx=dx,
                text=note,
                mode="markers+text",
                textposition=textposition,
                marker=dict(size=12, color="black"),
                textfont=dict(size=12, color="black"),
                showlegend=False,
                name="force profile",
            ),
        )

    # update force profiles figure layout
    yrange = f_data["value"].to_numpy().flatten()  # type: ignore
    yrange = np.array([np.min(yrange), np.max(yrange)])
    yrange *= np.array([0.9, 1.3])
    yrange = yrange.tolist()

    # Configure x-axis based on time mode
    if time_mode == 'absolute':
        xrange = f_data["time_ms"].to_numpy().flatten()  # type: ignore
        xrange = [np.min(xrange), np.max(xrange)]
        # Create ticks at key intervals for absolute time
        xticks = [0, 500, 1000, 1500, 2000]
        # Filter ticks to only those within the data range
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
        fig.update_yaxes(
            title="Force (N)",
            range=yrange,
            row=1,
            col=i + 1,
            showline=True,
            linewidth=2,
            linecolor="black",
            showgrid=False,
            zeroline=False,
            showticklabels=True,
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


