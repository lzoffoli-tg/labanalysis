# Plotly Basics

Guide to interactive visualization in labanalysis using Plotly for creating publication-quality, web-based figures.

## Overview

labanalysis uses Plotly to create interactive, browser-based visualizations that can be explored, customized, and exported in multiple formats. Plotly provides features like zoom, pan, hover tooltips, and export capabilities that make it ideal for data exploration and presentation.

**Key Features:**
- **Interactive plots** - Zoom, pan, hover for details
- **Multiple export formats** - PNG, SVG, PDF, HTML
- **Jupyter integration** - Display inline in notebooks
- **Two main functions** - `plot_comparisons()` for method validation, `bars_with_normative_bands()` for athlete reports
- **Customizable** - Modify colors, layouts, annotations after creation

## Installation & Setup

### Installing Dependencies

Plotly and kaleido (for static image export) are required:

```bash
pip install plotly kaleido
```

For conda environments:

```bash
conda install -c plotly plotly
pip install kaleido
```

### Jupyter Notebook Integration

Plotly works seamlessly in Jupyter notebooks with automatic inline display:

```python
import labanalysis as laban
import plotly.graph_objects as go

# Plots display automatically in notebook cells
fig = laban.plot_comparisons(...)
fig.show()  # Displays inline
```

### Browser Rendering

Outside Jupyter, `fig.show()` opens plots in your default web browser:

```python
# Opens in browser automatically
fig.show()
```

To specify a different renderer:

```python
import plotly.io as pio

# Use browser
pio.renderers.default = "browser"

# Use VS Code plotly extension
pio.renderers.default = "vscode"
```

## Quick Start

### Example 1: Comparison Plot

Create a comprehensive validation plot comparing two measurement methods:

```python
import labanalysis as laban
import numpy as np

# Generate example data
np.random.seed(42)
true = np.array([10, 20, 30, 40, 50])
pred = true + np.random.normal(0, 2, 5)

# Create comparison figure with 5 subplots
fig = laban.plot_comparisons(
    data_frame=None,
    true_data=true,
    pred_data=pred
)

# Display (browser or notebook)
fig.show()

# Output: Interactive figure with statistics table, regression plot,
# Bland-Altman plot, error distribution, and link plot
```

### Example 2: Normative Bands Chart

Create a bar chart with normative reference bands for athlete assessment:

```python
import pandas as pd

# Athlete test results
xarr = ["Jump Height", "Sprint 10m", "VO2max"]
yarr = [35, 1.85, 58]

# Define normative intervals
intervals = pd.DataFrame({
    "Rank": ["Low", "Medium", "High"],
    "Lower": [0, 30, 45],
    "Upper": [29, 44, 100],
    "Color": ["#FF6B6B", "#FFD93D", "#6BCF7F"]
})

# Create figure
fig, dfr = laban.bars_with_normative_bands(
    data_frame=None,
    xarr=xarr,
    yarr=yarr,
    orientation="v",
    unit="cm / s / ml/kg/min",
    intervals=intervals
)

fig.show()

# Output: Bar chart with color-coded bars and semi-transparent normative bands
# dfr contains enriched data with assigned ranks and colors
```

### Saving Figures

```python
# Save as HTML (interactive)
fig.write_html("comparison_plot.html")

# Save as static image (requires kaleido)
fig.write_image("comparison_plot.png", width=1200, height=800)
fig.write_image("comparison_plot.svg")
fig.write_image("comparison_plot.pdf")
```

## Understanding Plotly Objects

### Figure vs FigureWidget

labanalysis functions return different Plotly object types:

```python
# plot_comparisons returns Figure
fig1 = laban.plot_comparisons(...)
print(type(fig1))  # plotly.graph_objects.Figure

# bars_with_normative_bands returns FigureWidget
fig2, dfr = laban.bars_with_normative_bands(...)
print(type(fig2))  # plotly.graph_objects.FigureWidget
```

**Figure** - Standard Plotly figure object, immutable after creation
**FigureWidget** - Interactive widget that can be updated dynamically in Jupyter

Both support the same operations (.show(), .write_html(), etc.).

### Graph Objects Structure

Plotly figures consist of two main components:

```python
# Data traces (one or more plot elements)
print(len(fig.data))  # Number of traces
print(fig.data[0].type)  # 'scatter', 'bar', 'table', etc.

# Layout (appearance settings)
print(fig.layout.title.text)  # Figure title
print(fig.layout.xaxis.title.text)  # X-axis label
```

Access trace properties:

```python
# First scatter trace
trace = fig.data[0]
print(trace.x)  # X-axis data
print(trace.y)  # Y-axis data
print(trace.marker.color)  # Point colors
```

Modify layout properties:

```python
# Update title
fig.update_layout(title="Custom Title")

# Update axis labels
fig.update_xaxes(title_text="Time (s)")
fig.update_yaxes(title_text="Force (N)")
```

### Subplot Architecture

`plot_comparisons()` creates a figure with 5 subplots arranged in a grid:

```python
fig = laban.plot_comparisons(...)

# Access subplot information
print(fig.layout.grid)  # Grid structure

# Subplots are organized as:
# Row 1: [Statistics Table] [True vs Pred] [Bland-Altman]
# Row 2: [Error Distribution] [Link Plot]
```

Each subplot is referenced by row and column indices when customizing.

## Interactive Features

### Zoom and Pan

All plots support interactive exploration:

- **Box zoom** - Click and drag to zoom into a region
- **Autoscale** - Double-click to reset zoom
- **Pan** - Click and drag while in pan mode
- **Scroll zoom** - Mouse wheel to zoom in/out

```python
# Zoom is enabled by default
fig.show()  # Users can zoom interactively
```

Disable zoom for specific axes:

```python
fig.update_xaxes(fixedrange=True)  # Disable x-axis zoom
fig.update_yaxes(fixedrange=True)  # Disable y-axis zoom
```

### Hover Tooltips

Hovering over data points shows contextual information:

```python
import plotly.graph_objects as go

# Hover shows x, y values by default
fig = go.Figure(data=go.Scatter(
    x=[1, 2, 3],
    y=[10, 20, 15],
    mode='markers',
    hovertemplate='X: %{x}<br>Y: %{y}<br><extra></extra>'
))
fig.show()
```

Customize hover information:

```python
# Add custom hover text
fig.update_traces(
    hovertemplate='<b>Value:</b> %{y:.2f} N<br><extra></extra>'
)
```

### Modebar Buttons

The modebar appears in the top-right corner with tools:

- **Download plot as PNG** - Save static image
- **Zoom** - Box zoom tool
- **Pan** - Pan mode
- **Box/Lasso select** - Select data points
- **Reset axes** - Return to original view
- **Autoscale** - Fit data to view

Configure modebar:

```python
# Hide modebar
fig.show(config={'displayModeBar': False})

# Show only specific buttons
fig.show(config={'modeBarButtonsToRemove': ['pan2d', 'lasso2d']})
```

### Selection Tools

Select data points interactively:

```python
# Enable box and lasso selection
fig.update_traces(
    selected=dict(marker=dict(color='red')),
    unselected=dict(marker=dict(opacity=0.3))
)
```

## Rendering Options

### Jupyter Inline Display

In Jupyter notebooks, figures display automatically:

```python
# Cell 1: Create figure
fig = laban.plot_comparisons(...)

# Cell 2: Display (automatic if fig is last line, explicit with .show())
fig.show()
```

Control figure size in notebooks:

```python
fig.update_layout(width=900, height=600)
fig.show()
```

### Browser Rendering

Outside Jupyter, figures open in the default browser:

```python
# Opens browser window
fig.show()
```

### HTML Export

Export interactive plots as standalone HTML files:

```python
# Standalone HTML with embedded plotly.js
fig.write_html("interactive_plot.html")

# Include plotly.js from CDN (smaller file)
fig.write_html("plot_cdn.html", include_plotlyjs='cdn')

# Div only (for embedding in existing HTML)
fig.write_html("plot_div.html", include_plotlyjs=False, full_html=False)
```

HTML files can be:
- Opened directly in any browser
- Embedded in web pages
- Shared via email or file sharing
- Hosted on web servers

### Static Image Export

Export publication-quality static images (requires kaleido):

```python
# PNG (raster)
fig.write_image("figure.png", width=1200, height=800, scale=2)

# SVG (vector, ideal for publications)
fig.write_image("figure.svg", width=1200, height=800)

# PDF (vector)
fig.write_image("figure.pdf", width=1200, height=800)
```

Adjust resolution for print quality:

```python
# High DPI for print (300 DPI equivalent)
fig.write_image("high_res.png", width=2400, height=1600, scale=1)
```

## Performance Considerations

### Large Datasets

Plotly handles large datasets well, but performance degrades beyond ~10k points:

```python
# Example with 50k points
large_data = np.random.randn(50000)

# This may be slow to render
fig = go.Figure(data=go.Scatter(y=large_data, mode='markers'))
fig.show()  # Slow in browser
```

### Strategies for Large Data

**1. Downsample data for visualization:**

```python
# Keep every 10th point
downsampled = large_data[::10]
fig = go.Figure(data=go.Scatter(y=downsampled, mode='markers'))
fig.show()  # Much faster
```

**2. Use WebGL rendering (Scattergl):**

```python
# WebGL-accelerated scatter (faster for >1k points)
fig = go.Figure(data=go.Scattergl(
    x=np.arange(50000),
    y=large_data,
    mode='markers'
))
fig.show()  # Faster rendering
```

**3. Use aggregation:**

```python
# Bin data into histogram
fig = go.Figure(data=go.Histogram(x=large_data, nbinsx=100))
fig.show()  # Fast even with millions of points
```

### Memory Management

Large figures can consume significant memory:

```python
import sys

# Check figure size in memory
fig = laban.plot_comparisons(...)
size_mb = sys.getsizeof(fig) / 1024 / 1024
print(f"Figure size: {size_mb:.2f} MB")
```

For batch processing, explicitly delete figures:

```python
for i in range(100):
    fig = laban.plot_comparisons(...)
    fig.write_html(f"report_{i}.html")
    del fig  # Free memory
```

## Integration with labanalysis

### Using Signal1D Data

Plot Signal1D objects directly with Plotly:

```python
import labanalysis as laban
import plotly.graph_objects as go

# Load force platform data
record = laban.TimeseriesRecord.from_tdf("jump_test.tdf")
fp = record['FP1']
fz = fp.force['Fz']

# Create time-series plot
fig = go.Figure(data=go.Scatter(
    x=fz.index,      # Time axis
    y=fz.data,       # Force values
    mode='lines',
    name=fz.label
))

fig.update_layout(
    title="Vertical Ground Reaction Force",
    xaxis_title="Time (s)",
    yaxis_title=f"Force ({fz.unit})"
)
fig.show()
```

### Converting Record Objects

Extract data from Record objects for plotting:

```python
# Multiple force components
fx = fp.force['Fx'].data
fy = fp.force['Fy'].data
fz = fp.force['Fz'].data
time = fp.force['Fz'].index

# Create multi-trace figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=fx, name='Fx', line=dict(color='red')))
fig.add_trace(go.Scatter(x=time, y=fy, name='Fy', line=dict(color='green')))
fig.add_trace(go.Scatter(x=time, y=fz, name='Fz', line=dict(color='blue')))

fig.update_layout(
    title="3D Ground Reaction Forces",
    xaxis_title="Time (s)",
    yaxis_title="Force (N)"
)
fig.show()
```

### Working with DataFrames

labanalysis plotting functions accept both DataFrames and arrays:

```python
import pandas as pd

# Create DataFrame from Signal objects
df = pd.DataFrame({
    'time': time,
    'Fx': fx,
    'Fy': fy,
    'Fz': fz
})

# Use with labanalysis functions
fig = laban.plot_comparisons(
    data_frame=df,
    true_data='Fx',
    pred_data='Fy'
)
fig.show()
```

### Example: Force Platform Data Workflow

Complete workflow from loading to visualization:

```python
import labanalysis as laban
import plotly.graph_objects as go

# Load and filter force data
record = laban.TimeseriesRecord.from_tdf("jump_test.tdf")
fp = record['FP1']
fz_raw = fp.force['Fz']

# Apply low-pass filter
fz_filtered = laban.butterworth_filt(
    signal=fz_raw.data,
    freq=fz_raw.sampling_frequency,
    cut=10,
    order=4,
    filt_type='low'
)

# Compare raw vs filtered
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fz_raw.index,
    y=fz_raw.data,
    name='Raw',
    line=dict(color='lightgray', width=1)
))
fig.add_trace(go.Scatter(
    x=fz_raw.index,
    y=fz_filtered,
    name='Filtered (10 Hz)',
    line=dict(color='blue', width=2)
))

fig.update_layout(
    title="Effect of Low-Pass Filtering on Force Data",
    xaxis_title="Time (s)",
    yaxis_title="Vertical Force (N)",
    hovermode='x unified'
)
fig.show()
```

## Troubleshooting

### Installation Issues

**Problem:** `ModuleNotFoundError: No module named 'plotly'`

```bash
# Solution: Install plotly
pip install plotly
```

**Problem:** Static image export fails

```bash
# Solution: Install kaleido
pip install kaleido

# If still failing, try reinstalling
pip uninstall kaleido
pip install kaleido
```

### Rendering Problems

**Problem:** Figure doesn't display in Jupyter

```python
# Solution 1: Explicitly call .show()
fig.show()

# Solution 2: Restart Jupyter kernel and reimport
```

**Problem:** Browser window doesn't open

```python
import plotly.io as pio

# Check renderer
print(pio.renderers.default)  # Should be 'browser'

# Set explicitly
pio.renderers.default = 'browser'
```

### Export Failures

**Problem:** `write_image()` fails with kaleido error

```python
# Solution: Specify image format explicitly
fig.write_image("figure.png", format="png")

# Ensure dimensions are reasonable
fig.write_image("figure.png", width=1200, height=800)
```

**Problem:** HTML file is too large

```python
# Solution: Use CDN for plotly.js instead of embedding
fig.write_html("plot.html", include_plotlyjs='cdn')
```

### Performance Issues

**Problem:** Plot is slow with many points

```python
# Solution 1: Use WebGL rendering
import plotly.graph_objects as go
fig = go.Figure(data=go.Scattergl(x=x, y=y, mode='markers'))

# Solution 2: Downsample data
fig = go.Figure(data=go.Scatter(x=x[::10], y=y[::10], mode='markers'))
```

## See Also

- [Comparison Plots](comparison-plots.md) - Detailed guide to `plot_comparisons()`
- [Protocol Reports](protocol-reports.md) - Using `bars_with_normative_bands()` for athlete reports
- [Custom Figures](custom-figures.md) - Advanced customization and publication formatting
- [Plotly Official Documentation](https://plotly.com/python/) - Complete Plotly reference

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
