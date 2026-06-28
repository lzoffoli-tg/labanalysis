"""Shared plotting utilities for balance tests."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...constants import RANK_4COLORS
from ...modelling import Ellipse
from ...records import TimeseriesRecord
from ...utils import FloatArray1D


def _get_sway_figure(
    cop_x: FloatArray1D,
    cop_y: FloatArray1D,
    normative_data: pd.DataFrame = pd.DataFrame(),
    emgsignals: TimeseriesRecord = TimeseriesRecord(),
):

    def balance_string(left: str, right: str, sep: str = " | "):
        width = max(len(left), len(right))
        nbsp = " "  # non-breaking
        ljust = left.rjust(width).replace(" ", nbsp)
        rjust = right.ljust(width).replace(" ", nbsp)
        return sep.join([ljust, rjust])

    # collect emg signals
    emg_signals = {}
    for chn in emgsignals.values():
        if chn.muscle_name not in emg_signals:
            emg_signals[chn.muscle_name] = {}
        emg_signals[chn.muscle_name][chn.side] = chn.to_dataframe()

    # generate the figure and setup the layout
    if len(emg_signals) == 0 and normative_data.empty:
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=["Sway"],
        )
        ncols = 1
    elif len(emg_signals) == 0 and not normative_data.empty:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Sway", "Performance Overview"],
        )
        ncols = 2
    elif len(emg_signals) > 0 and normative_data.empty:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Sway", "Muscle imbalance"],
        )
        ncols = 2
    else:
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["Sway", "Performance Overview", "Muscle imbalance"],
        )
        ncols = 3

    # update overall layout
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Peformance<br>Levels",
        height=500,
        width=500 * ncols,
    )

    # plot normative data
    if not normative_data.empty:

        # get normative data
        norms_idx = normative_data.parameter == "area_of_stability_mm2"
        norms = normative_data.loc[norms_idx, ["mean", "std"]]
        avg, std = norms.values.astype(float).flatten()
        areas = np.array([avg, avg + 1 * std, avg + 2 * std])
        ranks = list(RANK_4COLORS.keys())[:-1][::-1]
        rank_colors = list(RANK_4COLORS.values())[:-1][::-1]

        # plot the background on the sway plot
        fig.add_shape(
            type="rect",
            xref="x domain",
            yref="y domain",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            fillcolor=RANK_4COLORS["Poor"],
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )

        # plot normative ellipses
        def build_ellipse(a, b, t, x0, y0):

            # Generate ellipse points
            theta = np.linspace(0, 2 * np.pi, 100)
            xy = np.column_stack([a * np.cos(theta), b * np.sin(theta)])

            # Apply rotation
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            xy_rot = xy @ R.T

            # translate to the center
            return (xy_rot + np.array([x0, y0])).T

        def is_within_ellipse(
            x: np.ndarray,
            y: np.ndarray,
            x0: float,
            y0: float,
            a: float,
            b: float,
            t: float,
        ):
            p1 = (((x - x0) * np.cos(t) + (y - y0) * np.sin(t)) ** 2) / a**2
            p2 = ((-(x - x0) * np.sin(t) + (y - y0) * np.cos(t)) ** 2) / b**2
            return np.asarray((p1 + p2) <= 1, dtype=bool)

        # get the sway ellipse
        ellipse = Ellipse().fit(cop_x, cop_y)

        # get the ellipse properties
        cop_x0, cop_y0 = ellipse.center
        semiaxis_a, semiaxis_b = ellipse.semi_axes
        cop_angle_rad = ellipse.rotation_angle / 180 * np.pi
        cop_area = ellipse.area

        samples_within = {}
        for area, color, label in zip(areas[::-1], rank_colors, ranks):

            # scale the axes according to the ratio between the ellipses area
            ratio = (area / cop_area) ** 0.5

            # add the ellipse
            x_ell, y_ell = build_ellipse(
                semiaxis_a * ratio,
                semiaxis_b * ratio,
                cop_angle_rad,
                cop_x0,
                cop_y0,
            )
            fig.add_trace(
                trace=go.Scatter(
                    x=x_ell,
                    y=y_ell,
                    fill="toself",
                    fillcolor=color,
                    line_width=0,
                    mode="lines",
                    name=label,
                    showlegend=False,
                    legendgroup=label,
                ),
                row=1,
                col=1,
            )

            # check the count of cop samples within the current norm
            within = is_within_ellipse(
                cop_x,
                cop_y,
                cop_x0,
                cop_y0,
                semiaxis_a * ratio,
                semiaxis_b * ratio,
                cop_angle_rad,
            )
            samples_within[label] = np.sum(within) / len(within) * 100

        # get the time spent within each norm interval
        ranks = ranks[::-1]
        for i in range(len(ranks) - 1):
            ranki = ranks[i]
            for j in range(i + 1, len(ranks)):
                rankj = ranks[j]
                samples_within[rankj] -= samples_within[ranki]
        samples_within["Poor"] = 100 - sum(samples_within.values())

        # plot the cumulative time spent at each level of norm
        ranks = list(RANK_4COLORS.keys())[::-1]
        colors = list(RANK_4COLORS.values())[::-1]
        for rank, color in zip(ranks, colors):
            value = samples_within[rank]
            fig.add_trace(
                trace=go.Bar(
                    x=[float(value)],
                    y=[rank],
                    text=[f"{value:0.2f}%"],
                    marker_color=color,
                    name=rank,
                    textposition="outside",
                    textangle=0,
                    orientation="h",
                    legendgroup=rank,
                ),
                row=1,
                col=2,
            )

        # update axes
        vrange = [0, max(20, np.max(list(samples_within.values())) * 1.25)]
        fig.update_yaxes(
            row=1,
            col=2,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor="black",
            showgrid=False,
        )
        fig.update_xaxes(
            row=1,
            col=2,
            title="Time lapsed (%)",
            zeroline=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            range=vrange,
        )

    # plot the emg signals
    if len(emg_signals) > 0:
        col = 2 if normative_data.empty else 3

        # plot the muscle balance
        cscales = [
            [i / (len(RANK_4COLORS) - 1), col]
            for i, col in enumerate(RANK_4COLORS.values())
        ]
        vals = []
        for i, (muscle, dct) in enumerate(emg_signals.items()):
            lt, rt = [float(m.mean().iloc[0]) for m in dct.values()]
            symm = (lt - rt) / (rt + lt) * 100
            val = max(-50, min(50, symm))
            vals.append(val)
            lbl = f"{abs(symm):0.1f}%" if -50 <= symm <= 50 else ">50.0%"
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
                        colorscale=cscales,
                        cmin=0,  # range colore
                        cmax=50,
                        colorbar=dict(
                            title=dict(
                                text="Muscle<br>Imbalance<br>Levels",
                            ),
                            len=0.50,
                            y=0.5,
                            tickvals=[10, 20, 30, 40],
                            ticktext=["Minimal", "Low", "Moderate", "High"],
                        ),
                        showscale=False,
                    ),
                ),
                row=1,
                col=col,
            )

        # adjust the muscle balance axes
        vrange = max(50, np.max(abs(np.array(vals))) * 1.5)
        vrange = [-vrange, vrange]
        fig.update_xaxes(
            row=1,
            col=col,
            title=f"{balance_string("Left", "Right", "|")}<br>(%)",
            zeroline=False,
            showline=False,
            showgrid=False,
            showticklabels=False,
            range=vrange,
        )
        fig.update_yaxes(
            row=1,
            col=col,
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
            col=col,  # type: ignore
            showlegend=False,
            line_color="black",
            line_width=2,
            line_dash="solid",
        )

    # adjust the sway plot axes
    fig.update_xaxes(
        row=1,
        col=1,
        title=f"{balance_string("Left", "Right", "|")}<br>(mm)",
        scaleanchor="y",
        scaleratio=1,
        showgrid=False,
        zeroline=False,
        showline=False,
    )
    fig.update_yaxes(
        row=1,
        col=1,
        title=f"{balance_string("Backward", "Forward", "|")}<br>(mm)",
        showgrid=False,
        zeroline=False,
        showline=False,
    )

    # add the sway
    fig.add_trace(
        trace=go.Scatter(
            x=cop_x,
            y=cop_y,
            mode="lines",
            opacity=0.5,
            line_width=1,
            line_color="black",
            name="Sway",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # add vertical and horizontal axes to the sway plot
    fig.add_hline(
        y=0,
        line_color="black",
        line_dash="dash",
        line_width=2,
        showlegend=False,
        row=1,  # type: ignore
        col=1,  # type: ignore
    )
    fig.add_vline(
        x=0,
        line_color="black",
        line_dash="dash",
        line_width=2,
        showlegend=False,
        row=1,  # type: ignore
        col=1,  # type: ignore
    )

    return fig
