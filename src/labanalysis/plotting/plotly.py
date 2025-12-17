"""
plotting module

a set of functions for standard plots creation.

Functions
---------
plot_comparisons
    A combination of regression and bland-altmann plots which returns a
    Plotly FigureWidget object.

bars_with_normative_bands
    Return a plotly FigureWidget and a dataframe with bars defining values
    and normative data in the background.


"""

#! IMPORTS

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.express.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, ttest_ind, ttest_rel

__all__ = ["plot_comparisons", "bars_with_normative_bands"]


#! FUNCTION


def plot_comparisons(
    data_frame: pd.DataFrame | None,
    true_data: np.ndarray | str,
    pred_data: np.ndarray | str,
    color_data: np.ndarray | str | None = None,
    confidence: float = 0.95,
    parametric: bool = False,
    color_scale: str = "temps",
):

    # * PREPARATION

    # check the inputs
    if data_frame is None:
        msg = "'{}' must be a list or a numpy ndarray if data_frame is None."
        if not isinstance(true_data, (list, np.ndarray)):
            raise ValueError(msg.format("true_data"))
        if not isinstance(pred_data, (list, np.ndarray)):
            raise ValueError(msg.format("pred_data"))
        if color_data is not None:
            if not isinstance(color_data, (list, np.ndarray)):
                raise ValueError(msg.format("color_data"))

    elif isinstance(data_frame, pd.DataFrame):
        msg = "if data_frame is provided, '{}' must be the name of one column "
        msg += "in data_frame."
        if not isinstance(true_data, str) or true_data not in data_frame.columns:  # type: ignore
            raise ValueError(msg.format("true_data"))
        if not isinstance(pred_data, str) or pred_data not in data_frame.columns:  # type: ignore
            raise ValueError(msg.format("pred_data"))
        true_data, pred_data = (
            data_frame[[true_data, pred_data]].to_numpy().astype(float).T
        )
        if color_data is None:
            color_data = np.tile("ALL", len(true_data))
        else:
            if not isinstance(color_data, str) or color_data not in data_frame.columns:
                raise ValueError(msg.format("color_data"))
            color_data = data_frame[color_data].to_numpy().flatten()

    else:
        raise ValueError("'data_frame' must be None or a pandas DataFrame.")

    if not isinstance(confidence, float) or not (0 < confidence < 1):
        raise ValueError("'confidence' must be a value within the (0, 1) range.")

    if not isinstance(parametric, bool):
        raise ValueError("'parametric' must be True or False.")

    # prepare the axes and labels for the bland-altmann plot
    loa_lbl = f"{confidence * 100:0.0f}% LIMITS OF AGREEMENT"
    eps = 1e-15
    x_rng2 = (true_data + pred_data) / 2
    x_rng2 = [np.min(x_rng2), np.max(x_rng2)]
    x_diff = abs(np.diff(x_rng2))[0]
    x_rng2 = [x_rng2[0] - x_diff * 0.05, x_rng2[1] + x_diff * 0.05]

    # get the colormap
    pmap = pcolors.qualitative.Plotly
    colmap = np.unique(np.array(color_data).flatten().astype(str))
    colmap = [(i, color_data == i, k) for i, k in zip(colmap, pmap)]
    n_colors = len(colmap)

    # get the colorscale based on the deviation between true and pred values
    # this colors will be used to color the lines in the link plot
    diffs_data = pred_data - true_data
    colorscale = pcolors.get_colorscale(color_scale)
    max_diff = np.max(abs(diffs_data))

    def map_to_color(val):
        t = val / max_diff
        t = max(0, min(1, t))  # clamp

        def as_rgb(cs):
            try:
                rgb = pcolors.hex_to_rgb(cs)
            except Exception as e:
                rgb = pcolors.unlabel_rgb(colorscale[i][1])
            return pcolors.label_rgb(rgb)

        for i in range(len(colorscale) - 1):
            if t <= colorscale[i + 1][0]:
                frac = (t - colorscale[i][0]) / (
                    colorscale[i + 1][0] - colorscale[i][0]
                )
                c1 = pcolors.unlabel_rgb(as_rgb(colorscale[i][1]))
                c2 = pcolors.unlabel_rgb(as_rgb(colorscale[i + 1][1]))
                rgb = [int(c1[j] + frac * (c2[j] - c1[j])) for j in range(3)]
                return pcolors.label_rgb(rgb)
        return as_rgb(colorscale[-1][1])

    diff_colors = np.array(list(map(map_to_color, abs(diffs_data))))

    # generate the figure
    fig = make_subplots(
        rows=2,
        cols=3,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=[
            "",
            "TRUE vs PREDICTED",
            ("NON-" if not parametric else "") + "PARAMETRIC BLAND-ALTMAN",
            "ERRORS DISTRIBUTION",
            "LINK PLOT",
        ],
        specs=[[{"rowspan": 2, "type": "table"}, {}, {}], [None, {}, {}]],
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    # prepare the data storage for the fitting metrics
    headers = [""]
    rows = [
        ["#"],
        ["RMSE"],
        ["MAPE"],
        ["R<sup>2</sup>"],
        ["T<sub>Paired</sub>"],
        ["T<sub>Independent</sub>"],
        ["Bias"],
        ["LOA<sub>Upper</sub>"],
        ["LOA<sub>Lower</sub>"],
    ]

    # * ADD DATA TO FIGURE

    for name, idx, col in colmap:
        xarri = true_data[idx]
        yarri = pred_data[idx]
        diffi = diffs_data[idx]
        diffc = diff_colors[idx]

        # add the fitting metrics
        rmse = np.mean((yarri - xarri) ** 2) ** 0.5
        mape = np.mean((abs(yarri - xarri) + eps) / (xarri + eps)) * 100
        r2 = np.corrcoef(xarri, yarri)[0][1] ** 2
        tt_rel = ttest_rel(xarri, yarri)
        tt_ind = ttest_ind(xarri, yarri)
        means = (xarri + yarri) / 2
        diffs = yarri - xarri
        if not parametric:
            ref = (1 - confidence) / 2
            loalow, loasup, bias = np.quantile(diffi, [ref, 1 - ref, 0.5])
        else:
            bias = np.mean(diffs)
            scale = np.std(diffs)
            loalow, loasup = norm.interval(confidence, loc=bias, scale=scale)
        headers += [name]
        rows[0] += [str(len(xarri))]
        rows[1] += [f"{rmse:0.4f}"]
        rows[2] += [f"{mape:0.1f}%"]
        rows[3] += [f"{r2:0.3f}"]
        rows[4] += [
            f"df={tt_rel.df:0.0f}<br>t={tt_rel.statistic:0.2f}<br>p={tt_rel.pvalue:0.3f}"
        ]
        rows[5] += [f"df={tt_ind.df:0.0f}<br>t={tt_ind.statistic:0.2f}<br>p={tt_ind.pvalue:0.3f}"]  # type: ignore
        rows[6] += [f"{bias:+0.3f}"]
        rows[7] += [f"{loalow:+0.3f}"]
        rows[8] += [f"{loasup:+0.3f}"]

        # plot the true vs predicted values in the regression plot
        fig.add_trace(
            row=1,
            col=2,
            trace=go.Scatter(
                x=xarri,
                y=yarri,
                mode="markers",
                marker_color=col,
                showlegend=n_colors > 1,
                opacity=0.5,
                name=name,
                legendgroup=name,
            ),
        )

        # plot the data on the bland-altman subplot
        fig.add_trace(
            row=1,
            col=3,
            trace=go.Scatter(
                x=means,
                y=diffs,
                mode="markers",
                marker_color=col,
                showlegend=False,
                opacity=0.5,
                name=name,
                legendgroup=name,
            ),
        )

        # plot the trend of the errors
        f_bias = np.polyfit(means, diffs, 1)
        y_bias = np.polyval(f_bias, x_rng2)
        fig.add_trace(
            row=1,
            col=3,
            trace=go.Scatter(
                x=x_rng2,
                y=y_bias,
                mode="lines",
                line_color=col,
                line_dash="dot",
                name=name,
                legendgroup=name,
                opacity=0.7,
                showlegend=False,
            ),
        )

        # plot the limits of agreement
        fig.add_trace(
            row=1,
            col=3,
            trace=go.Scatter(
                x=x_rng2,
                y=[loalow, loalow],
                mode="lines",
                line_color=col,
                line_dash="dashdot",
                name=loa_lbl,
                legendgroup=name,
                opacity=0.3,
                showlegend=False,
            ),
        )
        fig.add_trace(
            row=1,
            col=3,
            trace=go.Scatter(
                x=x_rng2,
                y=[loasup, loasup],
                mode="lines",
                line_color=col,
                line_dash="dashdot",
                name=name,
                legendgroup=name,
                opacity=0.3,
                showlegend=False,
            ),
        )

        # plot the errors distribution plot
        fig.add_trace(
            row=2,
            col=2,
            trace=go.Histogram(
                x=diffi,
                marker_color=col,
                name=name,
                legendgroup=name,
                opacity=0.3,
                showlegend=False,
                textposition="outside",
            ),
        )

        # fill the link-plot
        for x, y, c in zip(xarri, yarri, diffc):
            fig.add_trace(
                row=2,
                col=3,
                trace=go.Scatter(
                    x=["TRUE", "PREDICTED"],
                    y=[x, y],
                    mode="markers+lines",
                    marker=dict(color=col, coloraxis="coloraxis", showscale=True),
                    line=dict(color=c),
                    name=name,
                    legendgroup=name,
                    opacity=0.3,
                    showlegend=False,
                ),
            )

    # add the table with the fitting metrics
    fig.add_trace(
        row=1,
        col=1,
        trace=go.Table(
            header=dict(
                values=headers,
                fill_color="lightgrey",
                align="center",
                font=dict(size=12),
            ),
            cells=dict(
                values=np.array(rows).T.tolist(),
                fill_color="white",
                align="center",
                font=dict(size=12),
                height=50,
            ),
            name="fitting metrics",
        ),
    )

    # add the identity line to the regression plot
    ymin = min(np.min(pred_data), np.min(true_data))
    ymax = max(np.max(pred_data), np.max(true_data))
    fig.add_trace(
        row=1,
        col=2,
        trace=go.Scatter(
            x=[ymin, ymax],
            y=[ymin, ymax],
            mode="lines",
            line_dash="dash",
            line_color="black",
            name="IDENTITY LINE",
            legendgroup="IDENTITY LINE",
            showlegend=False,
            opacity=0.3,
        ),
    )

    # add the zero lines to the errors and bland-altman plots
    fig.add_hline(
        row=1,  # type: ignore
        col=3,  # type: ignore
        y=0,
        line_dash="dash",
        line_color="black",
        name="ZERO LINE",
        legendgroup="ZERO LINE",
        showlegend=False,
        opacity=0.3,
        exclude_empty_subplots=False,
    )
    fig.add_vline(
        row=2,  # type: ignore
        col=2,  # type: ignore
        x=0,
        line_dash="dash",
        line_color="black",
        name="ZERO LINE",
        legendgroup="ZERO LINE",
        showlegend=False,
        opacity=0.3,
        exclude_empty_subplots=False,
    )

    # update the layout
    fig.update_layout(
        barmode="group",
        bargap=0.05,
        template="plotly",
        legend=dict(
            title="groups",
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=50),
        coloraxis=dict(
            colorscale=color_scale,
            cmin=0,
            cmax=max(abs(diffs_data)),
            colorbar=dict(
                title="absolute<br>difference",
                x=1.05,
                y=0.5,
                len=0.8,
            ),
        ),
    )

    # regression figure axes labels
    fig.update_xaxes(title="TRUE", col=2, row=1)
    fig.update_yaxes(title="PREDICTED", col=2, row=1)

    # bland-altmann axes labels
    fig.update_xaxes(title="MEAN", col=3, row=1)
    fig.update_yaxes(title="DELTA", col=3, row=1)

    # errors distribution axes labels
    e_rng = max(abs(np.min(diffs_data)), abs(np.max(diffs_data)))
    e_rng = [-e_rng, e_rng]
    fig.update_xaxes(title="ERROR", range=e_rng, col=2, row=2)
    fig.update_yaxes(title="FREQUENCY (#)", col=2, row=2)

    # link plot axes labels
    fig.update_xaxes(title="", col=3, row=2)
    fig.update_yaxes(title="", col=3, row=2)

    return fig


def bars_with_normative_bands(
    data_frame: pd.DataFrame | None = None,
    xarr: str | np.ndarray | list = np.array([]),
    yarr: str | np.ndarray | list = np.array([]),
    patterns: str | np.ndarray | list | None = None,
    orientation: Literal["h", "v"] = "v",
    unit: str | None = None,
    intervals: pd.DataFrame = pd.DataFrame(),
):
    """
    Return a plotly FigureWidget and a dataframe with bars defining values
    and normative data in the background.

    Parameters
    ----------
    data_frame : pd.DataFrame | None, optional
        the dataframe containing the data, by default None

    xarr : str | np.ndarray | list, optional
        the xaxis label in data_frame or the array defining the xaxes labels.
        Please note that if data_frame is provided, xarr must be a str defining
        a column of the provided dataframe. by default np.array([])

    yarr : str | np.ndarray | list, optional
        the yaxis label in data_frame or the array defining the yaxes labels.
        Please note that if data_frame is provided, yarr must be a str defining
        a column of the provided dataframe. by default np.array([])

    patterns : str | np.ndarray | list | None, optional
        the column in data_frame defining the bar patterns or the array
        defining the  labels. Please note that if data_frame is provided,
        patterns must be a str defining a column of the provided dataframe.
        by default np.array([])

    orientation : Literal["h", "v"], optional
        the bars orientation, by default "v"

    unit: str | None, optional
        The unit of measurement of the bars.

    intervals: pd.DataFrame, optional
            all the normative intervals. The dataframe must have the following
            columns:

                Rank: str
                    the label defining the interpretation of the value

                Lower: int | float
                    the lower bound of the interval.

                Upper: int | float
                    the upper bound of the interval.

                Color: str
                    code that can be interpreted as a color.

    Returns
    -------
    fig: FigureWidget
        a plotly FigureWidget instance

    dfr: DataFrame
        the dataframe used to generate the figure
    """

    #  check the input data
    def _as_array(obj: object, lbl: str):
        """check if obj is a numeric 1D numpy array"""
        if not isinstance(lbl, str):
            msg = "if data_frame is provided, xarr, yarr, facet_col and "
            msg += "facet_row must be strings denoting a single row in"
            msg += " data_frame."
            raise ValueError(msg)
        msg = f"{lbl} must be a 1D array/list of int or float."
        if not isinstance(obj, (np.ndarray, list)):
            raise ValueError(msg)
        arr = np.array(obj) if isinstance(obj, list) else obj
        if arr.ndim > 1:
            raise ValueError(msg)
        return arr

    def _is_part(dfr: pd.DataFrame, lbl: object):
        """check if the lbl is a column of dfr"""
        msg = f"{lbl} must be a string defining one column of data_frame."
        if not isinstance(lbl, str):
            raise ValueError(msg)
        if not any([i == lbl for i in dfr.columns]):
            raise ValueError(msg)

    def _get_rank(x: float | int, intervals: pd.DataFrame):
        """return the rank corresponding to the x value"""
        if intervals.shape[0] > 0:
            for row in np.arange(intervals.shape[0]):
                rnk, low, upp, clr = intervals.iloc[row].values[-4:]
                if x >= low and x <= upp:
                    return str(rnk), str(clr)
        return None, None

    def _format_value(x: float | int, unit: str | None):
        """return formatted value with unit of measurement"""
        return str(x)[:5] + ("" if unit is None else f" {unit}")

    if data_frame is None:
        xlbl = "X"
        ylbl = "Y"
        dfr = {xlbl: _as_array(xarr, "xarr"), ylbl: _as_array(yarr, "yarr")}
        if not np.diff([len(v) for v in dfr.values()]).sum() == 0:
            raise ValueError("all the provided arrays must have the same length.")
        dfr = pd.DataFrame(dfr)
    else:
        if not isinstance(data_frame, pd.DataFrame):
            msg = "data_frame must be None or a valid pandas DataFrame."
            raise ValueError(msg)
        _is_part(data_frame, xarr)
        xlbl = str(xarr)
        _is_part(data_frame, yarr)
        ylbl = str(yarr)
        dfr = data_frame

    if patterns is not None:
        if data_frame is None:
            dfr.insert(0, "Pattern", _as_array(patterns, "patterns"))
            plbl = "Pattern"
        else:
            _is_part(dfr, patterns)
            plbl = str(patterns)
    else:
        plbl = None

    if not isinstance(orientation, str) or orientation not in ["h", "v"]:
        raise ValueError('orientation must be "h" or "v".')

    if unit is not None:
        if not isinstance(unit, str):
            raise ValueError("unit must be a str object or None.")

    # check the intervals
    columns = ["Rank", "Lower", "Upper", "Color"]
    msg = "intervals must be a pandas.DataFrame containing the "
    msg += "following columns: " + str(columns)
    msg2 = "Lower and Upper columns must contain only int or float-like values."
    if not isinstance(intervals, pd.DataFrame):
        raise ValueError(msg)
    if intervals.shape[0] > 0:
        for col in columns:
            if col not in intervals.columns.tolist():
                raise ValueError(msg)
            if col in ["Lower", "Upper"]:
                try:
                    _ = intervals[col].astype(float)
                except Exception:
                    raise ValueError(msg2)

    if orientation == "h":
        dfr.loc[dfr.index, ["_Text"]] = dfr[xlbl].map(lambda x: _format_value(x, unit))
        amp = dfr[xlbl].abs().max() * 2
        rng = [-amp, amp]
        rank_arr = dfr[xlbl]
    else:
        dfr.loc[dfr.index, ["_Text"]] = dfr[ylbl].map(lambda x: _format_value(x, unit))
        rng = [float(dfr[ylbl].min()) * 0.9, float(dfr[ylbl].max()) * 1.1]
        rank_arr = dfr[ylbl]
    for i, v in enumerate(rank_arr):
        index = dfr.index[i]
        rnk, clr = _get_rank(v, intervals)
        dfr.loc[index, ["_Rank"]] = rnk
        dfr.loc[index, ["_Color"]] = clr

    # get the output figure
    fig = px.bar(
        data_frame=dfr.reset_index(drop=True),
        x=xlbl,
        y=ylbl,
        pattern_shape=plbl,
        orientation=orientation,
        text="_Text",
        barmode="stack" if plbl is None else "group",
        template="simple_white",
    )

    # add the intervals
    if intervals.shape[0] > 0:
        rnks = []
        for row in np.arange(intervals.shape[0]):
            rnk, low, upp, clr = intervals.iloc[row].values[-4:]
            if not np.isfinite(low):
                low = rng[0]
            if not np.isfinite(upp):
                upp = rng[1]
            if rnk not in rnks:
                showlegend = True
                rnks += [rnk]
            else:
                showlegend = False
            if orientation == "h":
                fig.add_vrect(
                    x0=low,
                    x1=upp,
                    name=rnk,
                    showlegend=showlegend,
                    fillcolor=clr,
                    line_width=0,
                    opacity=0.1,
                    legendgroup="norms",
                    legendgrouptitle_text="Normative data",
                )
            else:
                fig.add_hrect(
                    y0=low,
                    y1=upp,
                    name=rnk,
                    showlegend=showlegend,
                    fillcolor=clr,
                    line_width=0,
                    opacity=0.1,
                    legendgroup="norms",
                    legendgrouptitle_text="Normative data",
                )

    # update the layout
    fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[1]))
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    if orientation == "v":
        fig.update_yaxes(title="" if unit is None else unit, range=rng)
        fig.update_xaxes(title="")
    if orientation == "h":
        fig.update_yaxes(title="")
        fig.update_xaxes(title="" if unit is None else unit, range=rng)

    if intervals.shape[0] > 0:
        if plbl is not None:
            for pattern in dfr[plbl].unique():
                for trace in fig.data:
                    if trace["name"] == pattern:  # type: ignore
                        clrs = dfr.loc[dfr[plbl] == pattern, "_Color"].values  # type: ignore
                        clrs = clrs.astype(str)
                        trace.update(  # type: ignore
                            marker_color=clrs,
                            marker_line_color=clrs,
                        )
        else:
            colors = dfr["_Color"].values.tolist()
            fig.update_traces(marker_color=colors, marker_line_color=colors)
    else:
        colors = pcolors.qualitative.Plotly[0]
        fig.update_traces(marker_color=colors, marker_line_color=colors)
    hover_tpl = f"<i>{xlbl}</i>: " + "%{x}<br>" + f"<i>{ylbl}</i>: " + "%{y}"
    fig.update_traces(
        marker_cornerradius="30%",
        marker_line_width=3,
        marker_pattern_fillmode="replace",
        textposition="outside",
        hovertemplate=hover_tpl,
        opacity=1,
    )

    return fig, dfr
