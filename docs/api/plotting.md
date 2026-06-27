# labanalysis.plotting

Plotly-based visualization functions for biomechanical data.

**Source**: `src/labanalysis/plotting/plotly.py`

## Overview

Visualization utilities for creating interactive Plotly charts with normative bands and comparison plots.

## Functions

### plot_comparisons()

Create regression, Bland-Altman, or error plots.

```python
def plot_comparisons(
    x: np.ndarray,
    y: np.ndarray,
    plot_type: str = 'regression',
    **kwargs
) -> go.Figure
```

**Parameters:**
- `x`, `y` (ndarray): Data arrays
- `plot_type` (str): 'regression', 'bland_altman', or 'error'

**Returns:**
- `plotly.graph_objects.Figure`: Interactive plot

**Example:**
```python
import labanalysis as laban
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([1.1, 2.0, 3.2, 3.9, 5.1])

fig = laban.plot_comparisons(x, y, plot_type='regression')
fig.show()
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
