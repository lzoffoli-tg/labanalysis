"""Jump test results implementation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as colormaps
from plotly.subplots import make_subplots

from ..test_protocol import TestProtocol
from ..test_results import TestResults
from ...constants import RANK_5COLORS, SIDE_COLORS
from ...utils import hex_to_rgba


class JumpTestResults(TestResults):
    """
    Results container for jump test analysis with automated reporting.

    JumpTestResults processes JumpTest data to generate comprehensive
    performance summaries, interactive visualizations, and normative
    comparisons. The class automatically computes all relevant jump metrics,
    generates publication-ready figures, and provides structured data export.

    Parameters
    ----------
    test : JumpTest
        Processed jump test data to analyze.
    include_emg : bool
        Whether to include EMG analysis in results (activation timing,
        amplitude, and pre-activation ratios).

    Attributes
    ----------
    summary : pd.DataFrame
        Comprehensive table of all jump metrics including elevation, contact
        time, flight time, RSI, force symmetry, and EMG metrics.
    analytics : pd.DataFrame
        Time-series data for all jumps in long format for detailed analysis.
    figures : dict of str -> go.Figure or dict
        Dictionary of interactive Plotly figures:
        - 'ground_reaction_forces': Force-time curves for all jumps
        - 'elevation': Jump height with normative bands and symmetry
        - 'contact_time': Contact time with normative ranking (DJ/repeated only)
        - 'rsi': Reactive strength index (DJ/repeated only)
        - 'muscle_activation_ratio': EMG pre-activation (DJ only, if include_emg)
        - 'muscle_activation_time': EMG onset timing (DJ only, if include_emg)

    Notes
    -----
    Metric Calculations:
    - Elevation: min(flight_time_method, impulse_method) for conservative estimate
    - Flight time method: h = (t_flight^2 * g) / 8
    - Impulse method: h = v_takeoff^2 / (2*g), where v from force integral
    - RSI: elevation / (contact_time / 1000)
    - Force symmetry: 100 * (R - L) / mean(R, L)

    EMG Metrics (Drop Jumps only):
    - Activation time: Time from landing to sustained EMG > threshold
    - Pre-activation ratio: mean(EMG_pre) / max(EMG_loading) * 100
    - Pre-window: 25ms before landing
    - Loading window: Landing to bodyweight crossing

    Figure Organization:
    Each figure type may contain multiple subplots organized by:
    - Jump type (SJ, CMJ, DJ, Repeated)
    - Side (bilateral, left, right)
    - Box height (for drop jumps)

    Normative data (if available) is displayed as colored bands overlaying
    the performance bars, with ranks typically defined as:
    - 5-level: Elite, Above Average, Average, Below Average, Poor
    - 3-level: Good, Average, Poor

    See Also
    --------
    JumpTest : Test protocol for jump assessment.
    TestResults : Parent class for test results.
    """

    def __init__(self, test: TestProtocol, include_emg: bool):
        """
        Initialize JumpTestResults with test data.

        Parameters
        ----------
        test : TestProtocol
            JumpTest instance containing processed jump data.
        include_emg : bool
            Whether to include EMG analysis in results.

        Raises
        ------
        ValueError
            If test is not a JumpTest instance.
        """
        from .jump_test import JumpTest

        if not isinstance(test, JumpTest):
            raise ValueError("'test' must be an JumpTest instance.")
        super().__init__(test, include_emg)

    def _get_test_metrics(self, test: "JumpTest"):
        """
        Generate comprehensive summary table of all jump metrics.

        Processes all jumps (squat, counter-movement, drop, repeated) to create
        a detailed summary DataFrame with metrics including elevation, contact time,
        flight time, RSI, force symmetry, and EMG parameters.

        Parameters
        ----------
        test : JumpTest
            The processed jump test data.

        Returns
        -------
        pd.DataFrame
        """
        # get metrics
        out = [jump.output_metrics for jump in test.jumps]
        summary = pd.concat(out, ignore_index=True)

        # remove rows with nans in value
        summary = summary.loc[summary.value.notna()].reset_index(drop=True)

        # set non defined options
        opts = ["free hands", "box height", "straight leg"]
        for opt in opts:
            if opt in summary.columns:
                if opt == "box height":
                    summary.loc[summary[opt].isna(), opt] = 0
                else:
                    summary.loc[summary[opt].isna(), opt] = False

        # add jump number
        summary.insert(summary.shape[1] - 1, "jump", 0)
        for grp, dfr in summary.groupby([i for i in summary.columns if i != "value"]):
            summary.loc[dfr.index, "jump"] = np.arange(dfr.shape[0]) + 1

        return summary

    def _get_summary(self, test: "JumpTest"):
        """
        Generate comprehensive summary table of all jump metrics.

        Processes all jumps (squat, counter-movement, drop, repeated) to create
        a detailed summary DataFrame with metrics including elevation, contact time,
        flight time, RSI, force symmetry, and EMG parameters.

        Parameters
        ----------
        test : JumpTest
            The processed jump test data.

        Returns
        -------
        pd.DataFrame
            Summary table with columns:
            - type: Jump type (squat jump, counter movement jump, drop jump, repeated jump)
            - side: bilateral/left/right
            - jump: Jump number
            - box height (cm): Box height for drop jumps (if applicable)
            - free hands: Whether hands were free during jump
            - parameter: Metric name
            - left: Left side value
            - right: Right side value
            - bilateral: Bilateral value
            - symmetry (%): Left-right symmetry percentage
        """
        out = self._get_test_metrics(test)
        indices = [i for i in out.columns if i not in ["value", "side", "jump"]]
        out = (
            out.pivot_table(index=indices + ["jump"], columns="side", values="value")
            .reset_index()
            .drop("jump", axis=1)
            .groupby(indices)
            .agg(["mean", "std", "min", "max"])
        )
        return out

    def _get_analytics(self, test: "JumpTest"):
        """
        Generate detailed analytics with time-series data for all jumps.

        Parameters
        ----------
        test : JumpTest
            The processed jump test data.

        Returns
        -------
        pd.DataFrame
            Time-series analytics table with columns:
            - type: Jump type (squat jump, CMJ, drop jump, repeated jump)
            - jump: Jump number
            - side: bilateral/left/right
            - box height (cm): Box height for drop jumps (if applicable)
            - free hands: Whether hands were free during jump
            - phase: Contact or flight phase
            - time_s: Time relative to contact phase start
            - force columns: Force platform data
            - emg columns: EMG signals (if include_emg is True)
        """
        jumps: dict[str, list[pd.DataFrame]] = {}
        for jump in test.jumps:

            # get the label
            lbl = jump.name.lower()

            # add box height
            if hasattr(jump, "box_height_cm"):
                lbl += f" ({jump.box_height_cm}cm)"

            # add side
            if hasattr(jump, "side"):
                lbl += f" - {jump.side}"

            # add free hands
            if hasattr(jump, "free_hands") and jump.free_hands:
                lbl += " - free hands"

            # add straight legs
            if hasattr(jump, "straight_legs") and jump.straight_legs:
                lbl += " - straight legs"

            # get the df
            df = jump.to_dataframe().reset_index(drop=True)

            # add the jump number
            if lbl not in jumps:
                jumps[lbl] = []

            jump_count = len(jumps[lbl]) + 1
            df.insert(0, "jump", jump_count)

            # append
            jumps[lbl].append(df)

        # concatenate and return
        analytics = {i: pd.concat(v, ignore_index=True) for i, v in jumps.items()}

        return analytics

    def _get_performance_data(
        self,
        test: "JumpTest",
        metric: str,
        include_force_balance: bool,
        ranks: dict[str, str],
        symmetric_ranks: bool,
        reversed_ranks: bool,
    ):
        """
        Generate performance bar chart with optional normative bands and balance plot.

        Creates a 1-column or 2-column figure showing performance bars colored by
        normative ranking, with optional left-right balance subplot.

        Parameters
        ----------
        performance_data : dict of str to list of float
            Performance values organized by side: {side: [jump1_val, jump2_val, ...]}.
        performance_norms : tuple of (list, list, list, list)
            Normative data as (lower_bounds, upper_bounds, labels, colors).
        performance_unit : str
            Unit of measurement for display (e.g., "cm", "ms").
        performance_metric : str
            Metric name for subplot title.
        balance_data : list of float or None, optional
            Left-right imbalance percentages (negative = left bias, positive = right bias).
            Default: None.
        balance_norms : tuple or None, optional
            Normative data for balance subplot, same structure as performance_norms.
            Default: None.

        Returns
        -------
        go.Figure
            Plotly figure with 1-2 columns showing performance bars overlaid on
            colored normative bands, plus optional balance subplot.
        """

        # extract the metric
        metrics = self._get_test_metrics(test)
        df = metrics.loc[metrics.metric == metric]
        if df.empty:
            raise RuntimeError(f"{metric} not found in summary data")

        # add limb
        df.loc[df.index, "limb"] = df.side.map(
            lambda x: x if x == "bilateral" else "unilateral"
        )

        # get the trial labels
        trial_cols = [
            i
            for i in df.columns
            if i not in ["unit", "jump", "value", "metric", "side"]
        ]

        # get the normative data
        normative_data = test.normative_data.copy()
        normative_data = normative_data.loc[normative_data.metric == metric]
        normative_data = normative_data.loc[
            normative_data.gender.str.startswith(test.participant.gender.lower()[0])
        ]

        # get balance data if required
        if include_force_balance:
            balance_data = metrics.loc[metrics.metric == "force asymmetry"]
            balance_data.loc[balance_data.index, "limb"] = balance_data.side.map(
                lambda x: x if x == "bilateral" else "unilateral"
            )
        else:
            balance_data = pd.DataFrame(columns=trial_cols)

        # prepare the balance norms (although they might be not used)
        vals = np.array([0, 10, 20, 30, 40, 100])
        lows = vals[:-1].copy().tolist()
        tops = vals[1:].copy().tolist()
        clrs = list(RANK_5COLORS.values())
        lbls = list(RANK_5COLORS.keys())
        balance_norms = {
            l: {"color": c, "from": b, "to": t}
            for l, c, b, t in zip(lbls, clrs, lows, tops)
        }

        # split the data into trials and add norms if available
        trials = {}
        for grp, dfr in df.groupby(trial_cols):

            # setup the figure title
            title = ""
            params = dict(zip(trial_cols, grp))
            for k, v in params.items():
                if k == "type":
                    title += v
                elif k == "box height" and v > 0:
                    title += f" ({v}cm)"
                elif k == "free hands" and v:
                    title += f" - free hands"
                elif k == "straight legs" and v:
                    title += f" - straight legs"
                elif k == "limb":
                    title += f" - {v}"

            # add the unit of measurement
            trials[title] = {}
            trials[title]["performance"] = {}
            trials[title]["performance"]["unit"] = str(df.unit.unique()[0])
            trials[title]["performance"]["metric"] = metric

            # add the data
            trials[title]["performance"]["data"] = {}
            for side, dfs in dfr.groupby("side"):
                vals = dfs.value.to_numpy().flatten()
                trials[title]["performance"]["data"][side] = vals

            # get the specific norms
            param_type = params["type"]
            if (
                "free hands" in params
                and params["free hands"]
                and param_type == "CounterMovementJump"
            ):
                param_type += " - free hands"
            norms = normative_data.copy().loc[normative_data["type"] == param_type]
            norms = norms.loc[norms.side == params["limb"]]

            # check if norms are properly found
            if norms.shape[0] > 1:
                raise ValueError("Multiple normative values found.")
            elif not norms.empty:

                # extract the normative ranges
                avg, std = norms[["mean", "std"]].to_numpy().flatten().astype(float)
                rank_clrs = list(ranks.values())
                rank_lbls = list(ranks.keys())
                if reversed_ranks:
                    rank_clrs = rank_clrs[::-1]
                    rank_lbls = rank_lbls[::-1]
                n_vals = len(ranks)
                if symmetric_ranks:
                    rank_clrs = rank_clrs[::-1] + rank_clrs
                    rank_lbls = rank_lbls[::-1] + rank_lbls
                    rank_vals = np.arange(n_vals + 1)
                else:
                    if n_vals % 2 == 1:
                        rank_vals = np.arange((n_vals + 1) // 2) + 1
                    else:
                        rank_vals = np.arange((n_vals + 2) // 2)
                rank_vals = np.concatenate([rank_vals, -rank_vals]) * std + avg
                rank_vals = np.unique(rank_vals)[::-1]
                rank_lows = rank_vals[1:].copy().tolist()
                rank_tops = rank_vals[:-1].copy().tolist()

                # add norms to trial
                trials[title]["performance"]["norms"] = {}
                for rank_color, rank_name, rank_min, rank_max in zip(
                    rank_clrs,
                    rank_lbls,
                    rank_lows,
                    rank_tops,
                ):
                    trials[title]["performance"]["norms"][rank_name] = {
                        "color": rank_color,
                        "from": rank_min,
                        "to": rank_max,
                    }

            else:
                trials[title]["performance"]["norms"] = None

            # get balance data
            valid_idx = pd.DataFrame([params]).to_dict("list")
            valid_idx = balance_data[trial_cols].isin(valid_idx).all(axis=1)
            bdf = balance_data.loc[valid_idx]
            if bdf.empty:
                trials[title]["balance"] = None
                continue

            # add the unit of measurement
            trials[title]["balance"] = {}
            trials[title]["balance"]["unit"] = "%"
            trials[title]["balance"]["metric"] = "Left/Right Imbalance"

            # add the balance data
            trials[title]["balance"]["data"] = {}
            for side, dfs in bdf.groupby("side"):
                vals = dfs.value.to_numpy().flatten()
                trials[title]["balance"]["data"][side] = vals

            # add the balance norms
            trials[title]["balance"]["norms"] = balance_norms

        return trials

    def _get_single_performance_figure(
        self,
        performance: dict[str, str | dict],
        balance: dict[str, str | dict] | None,
    ):

        def get_color_from_value(value: float, norms: dict | None):
            froms = [i["from"] for i in norms.values()]
            idx = np.argsort(froms)
            froms = np.array(froms)[idx]
            colors = [i["color"] for i in norms.values()]
            colors = np.array(colors)[idx]
            if abs(value) < np.min(froms):
                return colors[0]
            return colors[np.where(froms < value)[0][-1]]

        # generate the figure
        subplot_titles = [performance["metric"].upper()]
        ncols = 1
        if balance is not None:
            subplot_titles += [balance["metric"].upper()]
            ncols = 2

        fig = make_subplots(
            rows=1,
            cols=ncols,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.01,
        )
        fig.update_layout(
            template="plotly_white",
            legend=dict(title_text="Legend"),
            width=1000,
            height=400,
            bargroupgap=0.25,
            # margin = dict(t = 100, r = 100, b=50, l=100),
        )
        fig.update_xaxes(
            showgrid=False,
            showline=False,
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False,
        )

        # plot the bars representing the performance values
        colors_plotted = []
        values = []
        for k, (side, performances) in enumerate(performance["data"].items()):
            for j, y in enumerate(performances):

                # get the value
                value = round(y, 1)
                values.append(value)

                # get the color
                if performance["norms"] is not None:
                    color = get_color_from_value(value, performance["norms"])
                else:
                    color = SIDE_COLORS[side]
                    colors_plotted.append(color)

                # plot the bar
                fig.add_trace(
                    row=1,
                    col=1,
                    trace=go.Bar(
                        x=[k + 1],
                        y=[value],
                        text=[f"Jump {j+1}<br>{value} {performance['unit']}"],
                        textposition="outside",
                        textangle=0,
                        marker_color=[color],
                        marker_line_color=["black"],
                        name=f"Jump {j + 1}",
                        # legendgroup="Limb",
                        # legendgrouptitle_text="Limb",
                        offsetgroup=str(j + 1),
                        showlegend=False,
                    ),
                )

        # update the xaxes
        fig.update_xaxes(
            col=1,
            row=1,
            range=[0, len(performance["data"]) + 1],
            showticklabels=False,
        )
        if len(performance["data"]) > 1:
            fig.update_xaxes(
                col=1,
                row=1,
                showticklabels=True,
                tickvals=np.arange(len(performance["data"])) + 1,
                tickmode="array",
                ticktext=[
                    str(i).capitalize() for i in list(performance["data"].keys())
                ],
            )

        # plot average line
        avg = round(np.mean(values), 1)
        fig.add_hline(
            y=avg,
            col=1,  # type: ignore
            line_dash="dash",
            line_color="red",
            line_width=1.5,
            opacity=0.7,
        )
        fig.add_annotation(
            col=1,
            row=1,
            x=0,
            y=avg,
            text=f"mean<br>{avg} {performance['unit']}",
            font=dict(color="red"),
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            xref="x",
            yref="y",
        )

        # plot the norms as colored boxes behind the bars
        yrange = [np.min(values) * 0.9, np.max(values) * 1.2]
        rank_min = np.min([i["from"] for i in performance["norms"].values()])
        rank_top = np.max([i["to"] for i in performance["norms"].values()])
        for rank_name, obj in performance["norms"].items():

            # adjust the edges of the plot to consider values outside the given ranges
            if obj["from"] == rank_min and obj["from"] > yrange[0]:
                rlow = yrange[0]
            else:
                rlow = obj["from"]
            if obj["to"] == rank_top and obj["to"] < yrange[1]:
                rtop = yrange[1]
            else:
                rtop = obj["to"]
            yrange.append(rlow)
            yrange.append(rtop)

            # add the shape
            fig.add_shape(
                type="rect",
                x0=0,
                x1=len(performance["data"]) + 1,
                y0=rlow,
                y1=rtop,
                line_width=0,
                fillcolor=hex_to_rgba(obj["color"], 0.25),
                layer="below",
                name=str(rank_name).capitalize(),
                legendgroup=str(rank_name).capitalize(),
                # legendgrouptitle_text="Rank",
                showlegend=obj["color"] not in colors_plotted,
                col=1,
                row=1,
            )
            if rtop < rank_top:
                fig.add_annotation(
                    x=len(performance["data"]) + 1,
                    y=rtop,
                    text=f"{rtop:0.1f} {performance["unit"]}",
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    font=dict(color=obj["color"]),
                    valign="top",
                    yshift=0,
                    name=str(rank_name).capitalize(),
                    col=1,  # type: ignore
                    row=1,  # type: ignore
                )

            # ensure that the legend is plotted once
            colors_plotted.append(obj["color"])

        # update the yaxes
        fig.update_yaxes(row=1, col=1, range=[np.min(yrange), np.max(yrange)])

        # plot balance
        values = []
        if balance is not None:

            # plot the balance of each single jump
            values = list(balance["data"].values())
            values = values[
                0
            ].tolist()  # here we must have just one occasion (bilateral)
            vals = []
            for j, x in enumerate(values):

                # get the value
                value = max(-50, min(50, x))
                vals.append(value)

                # get the color
                color = get_color_from_value(value, balance["norms"])

                # get the label
                title = f"{abs(value):0.1f}%" if -50 <= value <= 50 else ">50.0%"
                title = f"Jump {j+1}<br>{title}"

                # plot the bar
                fig.add_trace(
                    col=2,
                    row=1,
                    trace=go.Bar(
                        y=[len(values) - 1 - j],
                        x=[value],
                        text=[title],
                        textposition="outside",
                        textangle=0,
                        showlegend=False,
                        marker_color=[color],
                        marker_line_color=["black"],
                        name=f"Jump {j+1}",
                        legendgroup="Jump",
                        legendgrouptitle_text="Jump",
                        orientation="h",
                    ),
                )

            # plot the norms as colored boxes behind the bars
            for rank_name, obj in balance["norms"].items():
                fig.add_shape(
                    type="rect",
                    y0=-1,
                    y1=len(values),
                    x0=obj["from"],
                    x1=obj["to"],
                    line_width=0,
                    fillcolor=hex_to_rgba(obj["color"], 0.25),
                    layer="below",
                    name=str(rank_name).capitalize(),
                    legendgroup="Rank",
                    legendgrouptitle_text="Rank",
                    showlegend=color not in colors_plotted,
                    col=2,
                    row=1,
                )
                fig.add_shape(
                    type="rect",
                    y0=-1,
                    y1=len(values),
                    x0=-obj["from"],
                    x1=-obj["to"],
                    line_width=0,
                    fillcolor=hex_to_rgba(obj["color"], 0.25),
                    layer="below",
                    name=str(rank_name).capitalize(),
                    legendgroup="Rank",
                    legendgrouptitle_text="Rank",
                    showlegend=False,
                    col=2,
                    row=1,
                )

                # ensure that the legend is plotted once
                colors_plotted.append(obj["color"])

            # plot the zero line
            fig.add_vline(
                col=2,  # type: ignore
                row=1,  # type: ignore
                x=0,
                line_width=2,
                line_dash="solid",
                showlegend=False,
            )

            # update the xaxes
            xrange = [-np.max(rank_top), np.max(rank_top)]
            fig.update_xaxes(
                col=2,
                row=1,
                range=xrange,
                tickmode="array",
                tickvals=[xrange[0] * 0.9, 0, xrange[1] * 0.9],
                ticktext=["Left", "Perfect<br>Balance", "Right"],
                ticklen=0,
            )

            # update the yaxes
            fig.update_yaxes(
                col=2,
                row=1,
                range=[-1, len(vals)],
            )

        # check
        return fig

    def _get_performance_figures(
        self,
        test: "JumpTest",
        metric: str,
        include_force_balance: bool,
        ranks: dict[str, str],
        symmetric_ranks: bool,
        reversed_ranks: bool,
    ):
        trials = self._get_performance_data(
            test,
            metric,
            include_force_balance,
            ranks,
            symmetric_ranks,
            reversed_ranks,
        )
        return {
            str(i): self._get_single_performance_figure(v["performance"], v["balance"])
            for i, v in trials.items()
        }

    def _get_raw_signals_figures(self, test: "JumpTest"):
        """
        Generate raw signal figures for all jumps.

        Creates a dictionary of interactive Plotly figures showing force-time
        curves and EMG signals (if available) for each jump type and condition.

        Parameters
        ----------
        test : JumpTest
            The processed jump test data.

        Returns
        -------
        go.Figure
        """
        # generate the raw signals
        out = {}
        for jump in test.jumps:
            key = jump.name.lower()
            if hasattr(jump, "box_height_cm"):
                key += f" ({jump.box_height_cm}cm)"
            if hasattr(jump, "side"):
                key += f" - {jump.side}"
            if hasattr(jump, "free_hands") and jump.free_hands:
                key += " - free hands"
            if hasattr(jump, "straight_legs") and jump.straight_legs:
                key += " - straight legs"
            if key not in out:
                out[key] = []
            df = jump.to_dataframe()
            df.index = pd.Index(df["time"])
            cols = [
                i
                for i in df.columns
                if i == "phase"
                or (i.endswith(jump.vertical_axis) and (" force " in i or "s2 " in i))
            ]
            out[key].append(df[cols])

        # generate the figures
        figures: dict[str, go.Figure] = {}
        CMAP = colormaps.Plotly
        for key, dfl in out.items():

            # create a new figure
            has_s2 = [any(["s2" in i for i in dfr.columns]) for dfr in dfl]
            fig = make_subplots(
                rows=len(has_s2),
                cols=1,
                specs=[[{"secondary_y": i}] for i in has_s2],
                subplot_titles=[f"Jump {i+1}" for i in range(len(dfl))],
                vertical_spacing=0.05,
            )

            # populate each subplot with the corresponding jump data
            for i, (dfr, s2_in) in enumerate(zip(dfl, has_s2), 1):

                # plot the signals
                for n, col in enumerate(dfr.columns):
                    dfs = dfr[[col]].copy().dropna()
                    if col == "phase":
                        continue
                    fig.add_trace(
                        row=i,
                        col=1,
                        secondary_y="s2" in col if s2_in else False,
                        trace=go.Scatter(
                            x=dfs.index.tolist(),
                            y=dfs[col].tolist(),
                            legendgroup=col,
                            legendgrouptitle_text="Signal",
                            name=col,
                            showlegend=i == 1,
                            line_color=CMAP[n % len(CMAP)],
                        ),
                    )
                # plot each phase
                for n, (phase, dff) in enumerate(dfr.groupby("phase")):
                    if dff.empty:
                        continue
                    color = CMAP[(n + len(dfr.columns) - 1) % len(CMAP)]
                    fig.add_vrect(
                        row=i,
                        col=1,
                        x0=dff.index[0],
                        x1=dff.index[-1],
                        fillcolor=color,
                        opacity=0.25,
                        line_width=0,
                        showlegend=i == 1,
                        legendgroup=phase,
                        legendgrouptitle_text="Phase",
                        name=phase,
                        annotation_text=phase,
                        annotation_position="top",
                        annotation=dict(
                            font_size=10,
                            font_color=color,
                            textangle=0,
                            xanchor="center",
                            yanchor="top",
                        ),
                    )

                # update the layout
                fig.update_yaxes(
                    row=i,
                    col=1,
                    secondary_y=False,
                    title_text="Vertical Force (N)",
                )
                if s2_in:
                    fig.update_yaxes(
                        row=i,
                        col=1,
                        secondary_y=True,
                        title_text="S2",
                    )

            # update the overall layout
            fig.update_xaxes(title_text="Time (s)", row=len(dfl), col=1)
            fig.update_layout(
                title_text=f"Raw Signals - {key}",
                template="plotly_white",
                height=400 * len(dfl),
                width=1000,
            )
            figures[key] = fig

        return figures

    def _get_figures(self, test: "JumpTest"):
        """
        Generate all visualization figures for the jump test results.

        Creates a comprehensive set of interactive Plotly figures including
        force curves, elevation, contact time, RSI, and EMG metrics based on
        available jump types and configuration.

        Parameters
        ----------
        test : JumpTest
            The processed jump test data.

        Returns
        -------
        dict of str to go.Figure or dict
            Dictionary of figures with keys:
            - 'ground_reaction_forces': Force-time curves for all jumps
            - 'elevation': Jump height figures (dict of figures by condition)
            - 'contact_time': Contact time figures (if drop/repeated jumps present)
            - 'rsi': RSI figures (if drop/repeated jumps present)
            - 'muscle_activation_ratio': Pre-activation figures (if drop jumps + EMG)
            - 'muscle_activation_time': Activation timing figures (if drop jumps + EMG)
        """
        metrics = ["elevation"]
        reverse = [False]
        if len(test.drop_jumps) > 0 or len(test.repeated_jumps) > 0:
            metrics += ["contact time", "reactive strength index"]
            reverse += [True, False]
            if self.include_emg:
                metrics.append("muscular reactivity index")
                reverse += [False]
        out = {
            metric: self._get_performance_figures(
                test=test,
                metric=metric,
                include_force_balance=metric == "elevation",
                ranks=RANK_5COLORS,
                symmetric_ranks=False,
                reversed_ranks=rev,
            )
            for metric, rev in zip(metrics, reverse)
        }
        out["raw_signals"] = self._get_raw_signals_figures(test)

        return out


__all__ = ["JumpTestResults"]
