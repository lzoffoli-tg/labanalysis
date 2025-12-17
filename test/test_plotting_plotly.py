"plotting.plotly testing module"

import sys
from os.path import abspath, dirname

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.append(dirname(dirname(abspath(__file__))))

from src.labanalysis import bars_with_normative_bands, plot_comparisons


def test_plot_comparisons_from_values():
    true_data = np.array([1, 2, 3, 4, 5])
    pred_data = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    color_data = np.array(["A", "A", "B", "B", "A"])
    fig = plot_comparisons(
        data_frame=None,
        true_data=true_data,
        pred_data=pred_data,
        color_data=color_data,
        confidence=0.95,
        parametric=True,
    )
    assert isinstance(fig, go.Figure)


def test_plot_comparisons_from_dataframe():
    df = pd.DataFrame(
        dict(
            true_data=np.array([1, 2, 3, 4, 5]),
            pred_data=np.array([1.1, 1.9, 3.2, 3.8, 5.1]),
        )
    )
    fig = plot_comparisons(
        data_frame=df,
        true_data="true_data",
        pred_data="pred_data",
        confidence=0.99,
        parametric=False,
        color_scale="Viridis",
    )
    assert isinstance(fig, go.Figure)


def test_bars_with_normative_bands():
    xarr = ["A", "B", "C"]
    yarr = [10, 20, 30]
    patterns = ["p1", "p2", "p3"]
    intervals = pd.DataFrame(
        {
            "Rank": ["Low", "Medium", "High"],
            "Lower": [0, 15, 25],
            "Upper": [14, 24, 35],
            "Color": ["#FF0000", "#00FF6A", "#1100FF"],
        }
    )
    fig, dfr = bars_with_normative_bands(
        data_frame=None,
        xarr=xarr,
        yarr=yarr,
        patterns=patterns,
        orientation="v",
        unit="units",
        intervals=intervals,
    )
    assert isinstance(fig, go.Figure)
    assert isinstance(dfr, pd.DataFrame)
