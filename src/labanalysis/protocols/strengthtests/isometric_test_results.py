"""Isometric test results implementation."""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as cmap
from scipy.interpolate import PchipInterpolator

from ...exercises.strength import IsometricExercise
from ..test_results import TestResults


class IsometricTestResults(TestResults):
    """
    Results container for isometric strength test analysis.

    IsometricTestResults processes IsometricTest data to generate comprehensive
    performance summaries including peak force, rate of force development (RFD),
    time to peak force, and muscle activation patterns. The class provides
    automated reporting with EMG analysis and bilateral symmetry calculations.

    Parameters
    ----------
    test : IsometricTest
        Processed isometric test data to analyze.
    include_emg : bool
        Whether to include EMG analysis in results (mean amplitude per muscle).

    Attributes
    ----------
    summary : pd.DataFrame
        Comprehensive table of isometric metrics including:
        - Peak force (N)
        - Force at 100 ms (N)
        - Force at 200 ms (N)
        - Force at 500 ms (N)
        - Force at 1000 ms (N)
        - Rate of force development (kN/s)
        - Time to peak force (ms)
        - EMG mean amplitude (% or µV) per muscle
        - Left/right symmetry (%)
    analytics : pd.DataFrame
        Time-series data for all trials in long format.
    figures : dict of str -> go.Figure
        Dictionary of interactive Plotly figures:
        - 'force_traces': Force-time curves for all trials

    Notes
    -----
    Metric Calculations:
    - Peak Force: Maximum force value during MVIC
    - RFD: (Peak Force - Baseline) / (Time to Peak - Baseline Time), in kN/s
    - Time to Peak: Time from contraction onset to peak force, in ms
    - Symmetry: 100 * (Right - Left) / mean(Right, Left)

    EMG Processing:
    - Mean amplitude computed over entire contraction phase
    - Values expressed as % of reference if normalization applied
    - Values in µV if no normalization reference provided

    The class automatically identifies valid repetitions and computes metrics
    for left, right, and bilateral trials when available.

    Examples
    --------
    >>> from labanalysis.protocols import IsometricTest, Participant
    >>> from labanalysis.exercises.strength import IsometricExercise
    >>>
    >>> # Create and process test
    >>> participant = Participant(surname='Athlete', weight=75)
    >>> left_ex = IsometricExercise.from_biostrength("left.txt")
    >>> test = IsometricTest(left=left_ex, right=None, bilateral=None,
    ...                       participant=participant)
    >>> results = test.get_results(include_emg=True)
    >>>
    >>> # View summary metrics
    >>> print(results.summary)
    >>>
    >>> # Display force-time curve
    >>> results.figures['force_traces'].show()

    See Also
    --------
    IsometricTest : Test protocol for isometric strength assessment.
    TestResults : Parent class for test results.
    """

    def __init__(self, test: "IsometricTest", include_emg: bool):
        from .isometric_test import IsometricTest

        if not isinstance(test, IsometricTest):
            raise ValueError("'test' must be an IsometricTest instance.")
        super().__init__(test, include_emg)

    def _calculate_rfd(self, force: np.ndarray, time: np.ndarray):
        rfd = (force[1:] - force[0]) / (time[1:] - time[0])
        return np.append([0], rfd)

    def _get_summary(self, test: "IsometricTest"):
        trials = [test.left, test.right, test.bilateral]
        sides = ["left", "right", "bilateral"]

        # Initialize metrics with all required columns
        metrics = []
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            emg_norms = test.emg_normalization_values

            # Iterate over all repetitions to get metrics for each
            for i, rep in enumerate(trial.repetitions):
                new = {}

                # EMG data for this repetition
                if self.include_emg:
                    for m in rep.emgsignals.values():
                        ename = str(m.muscle_name)
                        eside = str(m.side)
                        if eside != side:
                            continue
                        keys = emg_norms.keys()
                        check = [i[0] == ename and i[1] == eside for i in keys]
                        ename += " (%)" if any(check) else " (uV)"
                        new[ename] = m.to_numpy().mean()

                # Force metrics for this repetition
                force = rep.force.to_numpy().flatten()
                time = (rep.index - rep.index[0]) * 1000
                new["peak force (N)"] = round(np.max(force), 1)
                new["time to peak force (ms)"] = round(time[np.argmax(force)], 0)

                # get rfd metrics
                rfd = self._calculate_rfd(force, time / 1000)
                new["peak RFD (N/s)"] = round(np.max(rfd), 1)
                new["time to peak RFD (ms)"] = round(time[np.argmax(rfd)], 0)

                # get force and rfd interpolators
                fint = PchipInterpolator(time, force)
                rint = PchipInterpolator(time, rfd)

                # Get time points from exercise
                for tp in trial.time_points:
                    new[f"force at {tp}ms (N)"] = round(float(fint(tp)), 1)
                    new[f"RFD 0-{tp}ms (N/s)"] = round(float(rint(tp)), 1)

                # add rep and side options
                new = pd.DataFrame(pd.Series(new))
                new.columns = pd.Index(["value"])
                new.reset_index(names="parameter", inplace=True)
                new.insert(0, "repetition", i + 1)
                new.insert(0, "side", side)
                metrics.append(new)

        # aggregate
        metrics = (
            pd.concat(metrics, ignore_index=True)
            .pivot_table(
                index=["parameter", "repetition"],
                columns="side",
                values="value",
            )
            .reset_index()
        )

        return metrics

    def _get_analytics(self, test: "IsometricTest"):
        processed = test.processed_data
        analytics = []
        trials = [processed.left, processed.right, processed.bilateral]
        sides = ["left", "right", "bilateral"]
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            for i, rep in enumerate(trial.repetitions):
                cycle = rep.copy()
                if not self.include_emg:
                    cycle.drop(cycle.emgsignals.keys(), inplace=True)
                cycle = cycle.to_dataframe()
                time = cycle.index - cycle.index.min()
                cycle.insert(0, "time_s", time)
                cycle.insert(0, "repetition", i + 1)
                cycle.insert(0, "side", side)
                analytics.append(cycle)
        return pd.concat(analytics, ignore_index=True)

    def _get_profiles_with_time_intervals(self, test: "IsometricTest"):

        # force data
        analytics = self.analytics
        summary: pd.DataFrame = self.summary  # type: ignore
        if analytics is None or summary is None:
            return None

        # Find actual force and position column names
        available_cols = analytics.columns.tolist()

        # Find force column (e.g., "force N")
        force_col = None
        for col in available_cols:
            if "force" in col.lower():
                force_col = col
                break

        # Find position column (e.g., "position m")
        position_col = None
        for col in available_cols:
            if "position" in col.lower():
                position_col = col
                break

        if force_col is None or position_col is None:
            # If we can't find these columns, skip figure generation
            return {}

        # Determine time limit and time points from exercises
        max_time_s = None
        time_points = []
        for exe in [test.left, test.right, test.bilateral]:
            if exe is not None:
                max_time_s = exe.max_time_s
                time_points = exe.time_points
                break

        # Default to 2000 ms if max_time_s is not set
        max_time_ms = max_time_s * 1000 if max_time_s is not None else 2000

        # Process force data for each side and repetition
        # Build tracks with absolute time (ms) - include ALL repetitions
        tracks_data = []
        for (side, rep_num), group in analytics.groupby(["side", "repetition"]):
            if group.empty:
                continue

            # Get time in ms and force
            time_ms = (group["time_s"].to_numpy() * 1000).flatten()
            force = group[force_col].to_numpy().flatten()

            # Limit to max_time_ms
            mask = time_ms <= max_time_ms
            time_ms = time_ms[mask]
            force = force[mask]

            # get RFD
            rfd = self._calculate_rfd(force, time_ms / 1000) / 1000

            # Store in tracks_data with repetition number
            dff = pd.DataFrame({"time": time_ms, "value": force})
            dff.insert(0, "parameter", "Force")
            dff.insert(0, "unit", "N")
            dfp = pd.DataFrame({"time": time_ms, "value": rfd})
            dfp.insert(0, "parameter", "Rate of Force Development")
            dfp.insert(0, "unit", "kN/s")
            df = pd.concat([dff, dfp], ignore_index=True)
            df.insert(0, "side", side)
            df.insert(0, "repetition", rep_num)
            tracks_data.append(df)

        # get the dataframe
        tracks = pd.concat(tracks_data, ignore_index=True)

        # generate the figures
        out: dict[str, go.Figure] = {}
        colormap = cmap.Plotly

        def add_point(
            fig: go.Figure,
            row: int,
            col: int,
            x: float,
            y: float,
            text: str,
            color: str,
            name: str,
            legendgroup: str,
            showlegend: bool,
            textposition: Literal["middle right", "top right"],
        ):
            # vertical line
            fig.add_trace(
                row=row,
                col=col,
                trace=go.Scatter(
                    x=[x, x],
                    y=[0, y],
                    line_dash="dash",
                    mode="lines",
                    line_width=2,
                    opacity=0.3,
                    line_color=color,
                    showlegend=False,
                    name=name,
                    legendgroup=legendgroup,
                    legendgrouptitle_text=legendgroup,
                ),
            )

            # horizontal line
            fig.add_trace(
                row=row,
                col=col,
                trace=go.Scatter(
                    x=[0, x],
                    y=[y, y],
                    line_dash="dash",
                    mode="lines",
                    line_width=2,
                    opacity=0.3,
                    line_color=color,
                    showlegend=False,
                    name=name,
                    legendgroup=legendgroup,
                    legendgrouptitle_text=legendgroup,
                ),
            )

            # marker
            fig.add_trace(
                col=i,
                row=1,
                trace=go.Scatter(
                    x=[x],
                    y=[y],
                    text=[text],
                    textposition="middle right",
                    textfont_color=color,
                    textfont_size=12,
                    mode="markers+text",
                    marker_size=12,
                    marker_color=color,
                    opacity=1,
                    name=name,
                    legendgroup="Time",
                    legendgrouptitle_text="Time",
                    showlegend=showlegend,
                ),
            )

        for (parameter, unit), dfp in tracks.groupby(["parameter", "unit"]):

            # generate the figure
            sides = dfp.side.unique()
            y_range = [0, dfp.value.max() * 1.1]
            fig = make_subplots(
                rows=1,
                cols=len(sides),
                subplot_titles=[i.upper() for i in sides],
            )
            fig.update_yaxes(showticklabels=False, range=y_range)
            fig.update_yaxes(title=unit, col=1, showticklabels=True)
            fig.update_xaxes(title="Time (ms)")
            fig.update_layout(
                title=f"{parameter.capitalize()} Profile",
                template="simple_white",
                height=500,
                width=1250,
            )

            # add the data to each subplot
            for i, side in enumerate(sides, 1):
                dfs = dfp.loc[dfp.side == side].copy()

                # plot the trace of each repetition
                for r, dfr in dfs.groupby("repetition"):
                    x = dfr["time"].to_numpy().flatten()
                    y = dfr["value"].to_numpy().flatten()
                    rep = f"Repetition {r}"
                    fig.add_trace(
                        col=i,
                        row=1,
                        trace=go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            line_color=colormap[r],
                            line_width=3,
                            opacity=0.5,
                            name=rep,
                            legendgroup="Repetitions",
                            legendgrouptitle_text="Repetitions",
                            showlegend=i == 1,
                        ),
                    )

                # get interpolated signals to each time point
                y_vals = {}
                for rep, dfr in dfs.groupby("repetition"):
                    x = dfr["time"].to_numpy().flatten()
                    y = dfr["value"].to_numpy().flatten()
                    cs = PchipInterpolator(x, y)

                    # get the values at each timepoint
                    for t in time_points:
                        if t not in y_vals:
                            y_vals[t] = []
                        y_vals[t].append(cs(t))

                # pick the highest value for each timepoint
                y_vals = {t: np.max(v) for t, v in y_vals.items()}

                # plot the vertical and horizontal lines corresponding to
                # the time intervals
                n_reps = int(dfs.repetition.max())
                for n, (tp, y) in enumerate(y_vals.items()):
                    name = f"{tp:0.0f}ms"
                    text = f"{y:0.1f}{unit}"
                    add_point(
                        fig=fig,
                        row=1,
                        col=i,
                        x=tp,
                        y=y,
                        text=text,
                        color=colormap[(n + n_reps + 1) % len(colormap)],
                        name=name,
                        legendgroup="Time",
                        showlegend=i == 1,
                        textposition="middle right",
                    )

                # get the highest value
                maxv = dfs.value.max()
                maxt = dfs["time"].to_numpy()[np.argmax(dfs["value"].to_numpy())]
                name = f"Peak: {maxv:0.1f}{unit}<br>Time: {maxt:0.0f}ms"
                add_point(
                    fig=fig,
                    row=1,
                    col=i,
                    x=maxt,
                    y=maxv,
                    text=name,
                    color="black",
                    name="Peak",
                    legendgroup="Peak",
                    showlegend=i == 1,
                    textposition="top right",
                )

                # ensure the text of all samples is visible
                max_time = float(dfs["time"].to_numpy().max())
                max_time = max(
                    [max_time, *[float(t) + 1000 for t in time_points + [maxt]]]
                )
                fig.update_xaxes(range=[0, max_time], col=i)

            # append the figure
            out[parameter] = fig

        return out

    def _get_figures(self, test: "IsometricTest"):
        out: dict[str, go.Figure] = {}

        force_fig = self._get_profiles_with_time_intervals(test)
        if force_fig is not None:
            for k, v in force_fig.items():
                out[k] = v

        return out


__all__ = ["IsometricTestResults"]
