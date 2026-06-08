"plotting.plotly testing module"

import sys
from os.path import abspath, dirname, join

import pandas as pd
import plotly.graph_objects as go

sys.path.append(dirname(dirname(abspath(__file__))))

from src import labanalysis as laban

PATH = dirname(__file__)
DATA_PATH = join(PATH, "assets", "plotting")
DATA_WITH_NO_VARIANCE_PATH = join(DATA_PATH, "no_variance_data.csv")


def test_plot_comparisons_from_values():
    data = pd.read_csv(DATA_WITH_NO_VARIANCE_PATH)
    true_data = data.hr.to_numpy()
    pred_data = data.hr_global.to_numpy()
    fig = laban.plot_comparisons(
        data_frame=None,
        true_data=true_data,
        pred_data=pred_data,
        confidence=0.95,
        parametric=True,
    )
    assert isinstance(fig, go.Figure)


def test_plot_comparisons_from_dataframe():
    data = pd.read_csv(DATA_WITH_NO_VARIANCE_PATH)
    fig = laban.plot_comparisons(
        data_frame=data,
        true_data="hr",
        pred_data="hr_global",
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
