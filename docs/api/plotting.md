# labanalysis.plotting

Plotly-based visualization functions for biomechanical data.

**Source**: `src/labanalysis/plotting/plotly.py`

## Overview

Visualization utilities for creating interactive Plotly charts with normative bands and comparison plots.

## Functions

### plot_comparisons()

Create a figure for comparing reference and predicted values, together with
the numeric statistics shown in its summary table.

```python
def plot_comparisons(
    data_frame: pd.DataFrame | None = None,
    true_data: np.ndarray | str | None = None,
    pred_data: np.ndarray | str | None = None,
    color_data: np.ndarray | str | None = None,
    confidence: float = 0.95,
    parametric: bool = False,
    color_scale: str = "temps",
) -> tuple[go.Figure, pd.DataFrame]
```

**Parameters:**
- `data_frame`: Optional DataFrame containing the values and grouping column
  (defaults to `None`).
- `true_data`, `pred_data`: Column names when `data_frame` is provided;
  otherwise arrays of reference and predicted values. Both are required.
- `color_data`: Optional grouping column name.
- `confidence`: Confidence level for limits of agreement.
- `parametric`: Whether to calculate parametric limits of agreement.
- `color_scale`: Plotly color scale used by the link plot.

**Returns:**
- `fig` (`plotly.graph_objects.Figure`): Interactive comparison plot.
- `metrics` (`pandas.DataFrame`): Numeric metrics with groups as columns and
  statistic names as the index. T-test and Shapiro-Wilk results are split into
  separate rows; unavailable values are `NaN`.

**Example:**
```python
import labanalysis as laban
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([1.1, 2.0, 3.2, 3.9, 5.1])

fig, metrics = laban.plot_comparisons(
    true_data=x,
    pred_data=y,
)
fig.show()
print(metrics)
```

---

### bars_with_normative_bands()

Bar chart with normative reference bands.

```python
def bars_with_normative_bands(
    values: list,
    labels: list,
    normative_intervals: dict,
    title: str = '',
    y_label: str = ''
) -> go.Figure
```

**Parameters:**
- `values` (list): Values to plot
- `labels` (list): Bar labels
- `normative_intervals` (dict): Ranges for bands (e.g., `{'excellent': (0.5, inf), 'good': (0.4, 0.5)}`)
- `title`, `y_label` (str): Chart labels

**Example:**
```python
from labanalysis.plotting import bars_with_normative_bands

values = [0.45, 0.38, 0.52]
labels = ['Athlete 1', 'Athlete 2', 'Athlete 3']

norms = {
    'excellent': (0.50, float('inf')),
    'good': (0.40, 0.50),
    'average': (0.30, 0.40),
}

fig = bars_with_normative_bands(
    values, labels, norms,
    title="Jump Height", y_label="Height (m)"
)
fig.show()
```

---

## See Also

- [Visualization Guide](../guides/visualization/plotly.md)

---

**Create interactive Plotly visualizations with normative references.**
