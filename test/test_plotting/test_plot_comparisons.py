import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from scipy.stats import shapiro, ttest_ind, ttest_rel

from labanalysis import plot_comparisons


def test_plot_comparisons_returns_numeric_grouped_metrics():
    data = pd.DataFrame(
        {
            "true": [1.0, 2.0, 3.0, 4.0, 5.0],
            "pred": [1.1, 2.2, 3.3, 4.4, 5.5],
            "group": ["A", "A", "A", "A", "B"],
        }
    )

    fig, metrics = plot_comparisons(data, "true", "pred", color_data="group")

    assert isinstance(fig, Figure)
    assert list(metrics.columns) == ["A", "B"]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in metrics.dtypes)
    assert metrics.loc["#", "A"] == 4
    assert metrics.loc["#", "B"] == 1
    assert metrics.loc["RMSE", "A"] == np.sqrt(
        np.mean((data.loc[:3, "pred"] - data.loc[:3, "true"]) ** 2)
    )
    assert metrics.loc["T<sub>Paired</sub> df", "A"] == ttest_rel(
        data.loc[:3, "true"], data.loc[:3, "pred"]
    ).df
    assert metrics.loc["T<sub>Independent</sub> p", "A"] == ttest_ind(
        data.loc[:3, "true"], data.loc[:3, "pred"]
    ).pvalue
    assert metrics.loc["Shapiro-Wilk W", "A"] == shapiro(
        data.loc[:3, "pred"] - data.loc[:3, "true"]
    ).statistic
    assert metrics.loc["LOA<sub>Upper</sub>", "A"] > metrics.loc[
        "LOA<sub>Lower</sub>", "A"
    ]
    assert np.isnan(metrics.loc["T<sub>Paired</sub> t", "B"])

    table = next(trace for trace in fig.data if trace.type == "table")
    assert list(table.header.values) == ["", "A", "B"]
    table_rows = list(table.cells.values[0])
    a_values = list(table.cells.values[1])
    upper_row = table_rows.index("LOA<sub>Upper</sub>")
    lower_row = table_rows.index("LOA<sub>Lower</sub>")
    assert a_values[upper_row] == f"{metrics.loc['LOA<sub>Upper</sub>', 'A']:+0.3f}"
    assert a_values[lower_row] == f"{metrics.loc['LOA<sub>Lower</sub>', 'A']:+0.3f}"


def test_plot_comparisons_splits_multivalue_statistics_into_rows():
    data = pd.DataFrame(
        {
            "true": [1.0, 2.0, 3.0],
            "pred": [1.2, 2.1, 3.4],
        }
    )

    _, metrics = plot_comparisons(data, "true", "pred")

    assert "T<sub>Paired</sub> df" in metrics.index
    assert "T<sub>Paired</sub> t" in metrics.index
    assert "T<sub>Paired</sub> p" in metrics.index
    assert "T<sub>Independent</sub> df" in metrics.index
    assert "T<sub>Independent</sub> t" in metrics.index
    assert "T<sub>Independent</sub> p" in metrics.index
    assert "Shapiro-Wilk W" in metrics.index
    assert "Shapiro-Wilk p" in metrics.index


def test_plot_comparisons_defaults_to_no_dataframe_and_all_group():
    fig, metrics = plot_comparisons(
        true_data=np.array([1.0, 2.0, 3.0]),
        pred_data=np.array([1.2, 2.1, 3.4]),
    )

    assert isinstance(fig, Figure)
    assert list(metrics.columns) == ["ALL"]
