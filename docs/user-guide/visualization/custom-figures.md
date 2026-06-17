# Custom Figures

Advanced guide to customizing Plotly figures created by labanalysis and building custom visualizations for publication and presentation.

## Overview

This guide shows you how to customize figures returned by `plot_comparisons()` and `bars_with_normative_bands()`, and how to create fully custom visualizations from scratch using labanalysis data structures. Master these techniques to create publication-quality figures, interactive dashboards, and presentation-ready graphics.

**Topics Covered:**
- Modifying colors, fonts, and styles in existing figures
- Customizing subplot layouts and annotations
- Publication-quality formatting (journal requirements, high-resolution export)
- Creating custom figures from Signal1D/Signal3D data
- Multi-signal plots and phase-specific highlighting
- Interactive dashboards with linked plots
- 3D visualizations of marker trajectories

## Understanding Figure Structure

### Plotly Figure Anatomy

Every Plotly figure consists of two main components:

```python
import labanalysis as laban
import plotly.graph_objects as go

fig = laban.plot_comparisons(...)

# Component 1: Data traces (plot elements)
print(len(fig.data))  # Number of traces
print(fig.data[0])    # First trace (scatter, bar, table, etc.)

# Component 2: Layout (appearance settings)
print(fig.layout)     # Title, axes, fonts, margins, etc.
```

### Accessing Components

**Traces:**

```python
# Iterate through all traces
for i, trace in enumerate(fig.data):
    print(f"Trace {i}: {trace.type}, name={trace.name}")
    
# Access specific trace
first_scatter = fig.data[0]
print(first_scatter.x)  # X-axis data
print(first_scatter.y)  # Y-axis data
print(first_scatter.marker.color)  # Point colors
```

**Layout:**

```python
# Global layout properties
print(fig.layout.title.text)
print(fig.layout.font.size)
print(fig.layout.width)
print(fig.layout.height)

# Axis properties
print(fig.layout.xaxis.title.text)
print(fig.layout.yaxis.range)
```

### Subplot Indexing

`plot_comparisons()` creates a 2×3 grid with 5 subplots:

```
Row 1:  [Table (spans 2 rows)]  [Scatter]     [Bland-Altman]
Row 2:                           [Histogram]   [Link Plot]
```

Access subplots by (row, col) indices:

```python
fig = laban.plot_comparisons(...)

# Update specific subplot axis
fig.update_xaxes(title_text="Custom X Label", row=1, col=2)  # Scatter plot
fig.update_yaxes(range=[0, 100], row=1, col=3)  # Bland-Altman
```

## Customizing plot_comparisons()

### Modifying Colors and Styles

**Change scatter point colors:**

```python
import labanalysis as laban
import numpy as np

fig = laban.plot_comparisons(
    data_frame=None,
    true_data=np.array([10, 20, 30, 40, 50]),
    pred_data=np.array([9, 21, 29, 41, 48])
)

# Change scatter plot colors (row=1, col=2)
fig.update_traces(
    marker=dict(
        color='purple',
        size=10,
        opacity=0.7,
        line=dict(color='black', width=1)
    ),
    selector=dict(type='scatter', row=1, col=2)
)

fig.show()
```

**Modify line styles:**

```python
# Change regression line style
fig.update_traces(
    line=dict(color='red', width=3, dash='dot'),
    selector=dict(mode='lines', row=1, col=2)
)
```

**Marker customization:**

```python
# Custom marker symbols and sizes
fig.update_traces(
    marker=dict(
        symbol='diamond',  # 'circle', 'square', 'diamond', 'cross', 'x'
        size=12,
        line_width=2
    ),
    selector=dict(type='scatter')
)
```

### Adjusting Subplot Titles

```python
# Update subplot annotations
fig.for_each_annotation(lambda a: a.update(font=dict(size=16, color='darkblue')))

# Change specific subplot title
annotations = list(fig.layout.annotations)
annotations[1].text = "Regression Analysis"  # Second subplot title
fig.layout.annotations = annotations
```

### Modifying Statistical Table

```python
# Access table trace (first trace, usually)
table_trace = fig.data[0]

# Modify table appearance
table_trace.update(
    header=dict(
        fill_color='darkblue',
        font=dict(color='white', size=14)
    ),
    cells=dict(
        fill_color='lightgray',
        font=dict(size=12),
        align='center'
    )
)
```

### Changing Axis Ranges and Labels

```python
# Scatter plot (row=1, col=2)
fig.update_xaxes(
    title_text="Reference Method (cm)",
    range=[0, 60],
    row=1, col=2
)
fig.update_yaxes(
    title_text="Test Method (cm)",
    range=[0, 60],
    row=1, col=2
)

# Bland-Altman plot (row=1, col=3)
fig.update_xaxes(
    title_text="Mean of Two Methods (cm)",
    row=1, col=3
)
fig.update_yaxes(
    title_text="Difference (cm)",
    row=1, col=3
)
```

### Adding Custom Annotations

```python
# Add text annotation to scatter plot
fig.add_annotation(
    text="Outlier",
    x=45, y=50,
    xref="x2", yref="y2",  # Subplot 2 (row=1, col=2)
    showarrow=True,
    arrowhead=2,
    arrowcolor="red",
    font=dict(size=12, color="red")
)

# Add horizontal line to Bland-Altman
fig.add_hline(
    y=0, 
    line_dash="dash", 
    line_color="black",
    row=1, col=3
)

# Add vertical span (shaded region)
fig.add_vrect(
    x0=20, x1=30,
    fillcolor="lightgray",
    opacity=0.3,
    layer="below",
    row=1, col=2
)
```

### Modifying Color Scales

```python
# Link plot uses colorscale parameter
fig = laban.plot_comparisons(
    ...,
    color_scale='Viridis'  # Change at creation
)

# Or modify after creation (advanced)
for trace in fig.data:
    if hasattr(trace, 'line') and hasattr(trace.line, 'color'):
        # Update line colors in link plot
        trace.line.colorscale = 'RdBu'
```

## Customizing bars_with_normative_bands()

### Bar Appearance

```python
import labanalysis as laban
import pandas as pd

fig, dfr = laban.bars_with_normative_bands(
    xarr=['Test A', 'Test B', 'Test C'],
    yarr=[35, 42, 38],
    intervals=intervals
)

# Modify bar width and gap
fig.update_traces(width=0.5)  # Thinner bars
fig.update_layout(bargap=0.3)  # More spacing

# Corner radius
fig.update_traces(marker_cornerradius='20%')  # Less rounded

# Opacity
fig.update_traces(opacity=0.8)

# Border
fig.update_traces(
    marker_line_color='black',
    marker_line_width=2
)

fig.show()
```

### Normative Band Styling

```python
# Bands are created as shapes (rectangles)
# Modify existing shapes
for shape in fig.layout.shapes:
    shape.opacity = 0.2  # More transparent
    shape.line.width = 1  # Add border
    shape.line.color = 'gray'

# Or remove bands entirely
fig.layout.shapes = []  # Remove all shapes
```

### Legend Customization

```python
# Position and orientation
fig.update_layout(
    legend=dict(
        orientation="h",  # Horizontal
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    )
)

# Hide legend
fig.update_layout(showlegend=False)

# Show only specific items
for trace in fig.data:
    if 'Normative' in trace.name:
        trace.showlegend = False
```

### Adding Target Lines

```python
# Add horizontal goal line
target_value = 45

fig.add_hline(
    y=target_value,
    line_dash="dash",
    line_color="red",
    line_width=3,
    annotation_text="Team Goal",
    annotation_position="right"
)

# Add range highlighting
fig.add_hrect(
    y0=40, y1=50,
    fillcolor="green",
    opacity=0.1,
    layer="below",
    annotation_text="Target Range"
)
```

## Publication-Quality Formatting

### Journal-Specific Requirements

**Nature/Science Style:**

```python
fig = laban.plot_comparisons(...)

# Nature requirements: Arial font, specific sizing
fig.update_layout(
    font_family="Arial",
    font_size=8,  # 6-8 pt for figure text
    title_font_size=10,
    width=89,  # mm (single column) or 183 (double column)
    height=89,
    margin=dict(l=10, r=10, t=30, b=10)
)

# Remove background
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Thinner lines
fig.update_traces(line_width=1)

# Export
fig.write_image("figure1.pdf", width=89*3.78, height=89*3.78)  # Convert mm to px
```

**JAMA Style:**

```python
# JAMA: Times New Roman, grayscale preferred
fig.update_layout(
    font_family="Times New Roman",
    font_size=9,
    width=600,
    height=400
)

# Convert to grayscale
grayscale_colors = ['black', 'gray', 'lightgray', 'darkgray']
for i, trace in enumerate(fig.data):
    if hasattr(trace, 'marker'):
        trace.marker.color = grayscale_colors[i % len(grayscale_colors)]
```

### High-Resolution Export

```python
# Vector formats (preferred for publications)
fig.write_image("figure.svg", width=1200, height=800)
fig.write_image("figure.pdf", width=1200, height=800)

# High-DPI raster (300 DPI for print)
# Formula: pixels = inches × DPI
# Example: 4" × 3" at 300 DPI = 1200×900 px
fig.write_image(
    "figure.png",
    width=1200,   # 4 inches at 300 DPI
    height=900,   # 3 inches at 300 DPI
    scale=1       # Already at target size
)

# Alternative: use scale parameter
fig.write_image(
    "figure_high_res.png",
    width=400,    # Base size
    height=300,
    scale=3       # 3× scaling = 1200×900 final
)
```

### Grayscale Conversion

```python
import plotly.graph_objects as go

def convert_to_grayscale(fig):
    """Convert color figure to grayscale."""
    
    gray_map = {
        'red': 'rgb(100, 100, 100)',
        'blue': 'rgb(150, 150, 150)',
        'green': 'rgb(200, 200, 200)',
        'orange': 'rgb(120, 120, 120)',
    }
    
    for trace in fig.data:
        if hasattr(trace, 'marker'):
            # Simple approach: use patterns instead of colors
            trace.marker.color = 'white'
            trace.marker.line.color = 'black'
            trace.marker.line.width = 2
            
    return fig

# Apply conversion
fig_gray = convert_to_grayscale(fig)
fig_gray.write_image("figure_grayscale.pdf")
```

**Alternative: Use patterns for distinction**

```python
# Use different line styles instead of colors
line_styles = ['solid', 'dash', 'dot', 'dashdot']

for i, trace in enumerate(fig.data):
    if hasattr(trace, 'line'):
        trace.line.color = 'black'
        trace.line.dash = line_styles[i % len(line_styles)]
        trace.line.width = 2
```

## Creating Custom Figures from Scratch

### Using Signal Data Directly

```python
import labanalysis as laban
import plotly.graph_objects as go

# Load force platform data
record = laban.TimeseriesRecord.from_tdf("jump_test.tdf")
fp = record['FP1']
fz = fp.force['Fz']

# Create custom time-series plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=fz.index,
    y=fz.data,
    mode='lines',
    name='Vertical Force',
    line=dict(color='blue', width=2)
))

fig.update_layout(
    title="Ground Reaction Force - Countermovement Jump",
    xaxis_title="Time (s)",
    yaxis_title=f"Force ({fz.unit})",
    hovermode='x unified',
    template='plotly_white'
)

fig.show()
```

### Multi-Signal Plots

**Shared time axis:**

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load 3D force data
fx = fp.force['Fx']
fy = fp.force['Fy']
fz = fp.force['Fz']

# Create figure with shared x-axis
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=['Fx (Anteroposterior)', 'Fy (Mediolateral)', 'Fz (Vertical)']
)

# Add traces
fig.add_trace(go.Scatter(x=fx.index, y=fx.data, name='Fx', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=fy.index, y=fy.data, name='Fy', line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=fz.index, y=fz.data, name='Fz', line=dict(color='blue')), row=3, col=1)

# Update axes
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Force (N)", row=1, col=1)
fig.update_yaxes(title_text="Force (N)", row=2, col=1)
fig.update_yaxes(title_text="Force (N)", row=3, col=1)

fig.update_layout(height=800, title="3D Ground Reaction Forces")
fig.show()
```

**Overlaid signals:**

```python
# Plot all on same axes
fig = go.Figure()

fig.add_trace(go.Scatter(x=fx.index, y=fx.data, name='Fx', line=dict(color='red')))
fig.add_trace(go.Scatter(x=fy.index, y=fy.data, name='Fy', line=dict(color='green')))
fig.add_trace(go.Scatter(x=fz.index, y=fz.data, name='Fz', line=dict(color='blue')))

fig.update_layout(
    title="3D Forces (Overlaid)",
    xaxis_title="Time (s)",
    yaxis_title="Force (N)"
)
fig.show()
```

### Phase-Specific Highlighting

**Highlight jump phases:**

```python
import labanalysis as laban
import plotly.graph_objects as go

# Detect jump phases (example indices)
unweighting_start = 0.5
unweighting_end = 0.8
braking_end = 1.1
propulsion_end = 1.4

fig = go.Figure()

# Plot force trace
fig.add_trace(go.Scatter(
    x=fz.index,
    y=fz.data,
    mode='lines',
    name='Vertical Force',
    line=dict(color='black', width=2)
))

# Add phase regions
fig.add_vrect(
    x0=unweighting_start, x1=unweighting_end,
    fillcolor="blue", opacity=0.2,
    layer="below", line_width=0,
    annotation_text="Unweighting",
    annotation_position="top left"
)

fig.add_vrect(
    x0=unweighting_end, x1=braking_end,
    fillcolor="orange", opacity=0.2,
    layer="below", line_width=0,
    annotation_text="Braking",
    annotation_position="top left"
)

fig.add_vrect(
    x0=braking_end, x1=propulsion_end,
    fillcolor="green", opacity=0.2,
    layer="below", line_width=0,
    annotation_text="Propulsion",
    annotation_position="top left"
)

fig.update_layout(
    title="CMJ Phases",
    xaxis_title="Time (s)",
    yaxis_title="Force (N)"
)

fig.show()
```

**Event markers:**

```python
# Mark takeoff and landing
takeoff_time = 1.4
landing_time = 1.9

fig.add_vline(
    x=takeoff_time,
    line_dash="dash",
    line_color="red",
    annotation_text="Takeoff",
    annotation_position="top"
)

fig.add_vline(
    x=landing_time,
    line_dash="dash",
    line_color="red",
    annotation_text="Landing",
    annotation_position="top"
)
```

### Custom Subplots

**Complex layouts:**

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create 2×2 grid with mixed plot types
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"type": "scatter"}, {"type": "bar"}],
        [{"type": "table", "colspan": 2}, None]
    ],
    subplot_titles=['Force Trace', 'Peak Forces', 'Jump Metrics']
)

# Row 1, Col 1: Time series
fig.add_trace(
    go.Scatter(x=fz.index, y=fz.data, name='Fz'),
    row=1, col=1
)

# Row 1, Col 2: Bar chart
fig.add_trace(
    go.Bar(x=['Fx', 'Fy', 'Fz'], y=[120, 85, 1850], name='Peaks'),
    row=1, col=2
)

# Row 2, Cols 1-2: Table
fig.add_trace(
    go.Table(
        header=dict(values=['Metric', 'Value']),
        cells=dict(values=[['Height', 'Power', 'Flight Time'], ['35 cm', '4500 W', '0.55 s']])
    ),
    row=2, col=1
)

fig.update_layout(height=700, title="Comprehensive Jump Report")
fig.show()
```

### 3D Visualizations

**Marker trajectory:**

```python
import plotly.graph_objects as go

# Extract marker 3D position
marker = record.body['R_ASIS']  # Right anterior superior iliac spine
x = marker.data[:, 0]  # X coordinate
y = marker.data[:, 1]  # Y coordinate  
z = marker.data[:, 2]  # Z coordinate

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='lines',
    line=dict(color='blue', width=4),
    name='ASIS Trajectory'
)])

fig.update_layout(
    scene=dict(
        xaxis_title='X (mm)',
        yaxis_title='Y (mm)',
        zaxis_title='Z (mm)',
        aspectmode='data'  # Equal aspect ratio
    ),
    title="Marker Trajectory During Jump"
)

fig.show()
```

**3D scatter with color-coded time:**

```python
import numpy as np

# Color by time
time_colors = np.linspace(0, 1, len(x))

fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=3,
        color=time_colors,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Time (normalized)")
    )
)])

fig.show()
```

**Camera positioning:**

```python
# Set custom view angle
fig.update_layout(
    scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5),  # Camera position
        center=dict(x=0, y=0, z=0),      # Look-at point
        up=dict(x=0, y=0, z=1)           # Up direction
    )
)
```

## Interactive Dashboards

### Linked Plots

**Cross-filtering with shared selection:**

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create linked scatter plots
fig = make_subplots(rows=1, cols=2, subplot_titles=['Jump Height vs Power', 'Height vs Flight Time'])

# Left plot
fig.add_trace(
    go.Scatter(x=[4200, 4500, 3800], y=[32, 35, 28], mode='markers', name='Athletes'),
    row=1, col=1
)

# Right plot
fig.add_trace(
    go.Scatter(x=[0.52, 0.55, 0.48], y=[32, 35, 28], mode='markers', name='Athletes', showlegend=False),
    row=1, col=2
)

# Enable selection
fig.update_traces(
    selected=dict(marker=dict(color='red', size=12)),
    unselected=dict(marker=dict(opacity=0.3))
)

fig.update_layout(dragmode='select')  # Enable box/lasso select
fig.show()
```

### Button Controls

**Toggle between views:**

```python
import plotly.graph_objects as go

fig = go.Figure()

# Add both raw and filtered data
fig.add_trace(go.Scatter(x=fz.index, y=fz.data, name='Raw', visible=True))
fig.add_trace(go.Scatter(x=fz.index, y=filtered_fz, name='Filtered', visible=False))

# Add buttons to toggle
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": [True, False]}],
                    label="Raw",
                    method="update"
                ),
                dict(
                    args=[{"visible": [False, True]}],
                    label="Filtered",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True, True]}],
                    label="Both",
                    method="update"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.11,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ]
)

fig.show()
```

### Slider Controls

**Time navigation:**

```python
import numpy as np
import plotly.graph_objects as go

# Create frames for animation
frames = []
for i in range(0, len(fz.data), 10):
    frames.append(go.Frame(
        data=[go.Scatter(x=fz.index[:i], y=fz.data[:i])],
        name=f"frame_{i}"
    ))

fig = go.Figure(
    data=[go.Scatter(x=[], y=[])],
    frames=frames
)

# Add slider
fig.update_layout(
    sliders=[{
        "active": 0,
        "steps": [
            {
                "args": [[f.name], {"frame": {"duration": 0}, "mode": "immediate"}],
                "label": f"{i/1000:.2f}s",
                "method": "animate"
            }
            for i, f in enumerate(frames)
        ]
    }],
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 50}}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
        ]
    }]
)

fig.show()
```

## Combining Multiple Reports

**Merge comparison and normative plots:**

```python
from plotly.subplots import make_subplots

# Create combined figure
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"colspan": 2}, None],
        [{}, {}]
    ],
    subplot_titles=['Method Validation', 'Current Performance', 'Team Ranking']
)

# Top: Comparison plot (extract key subplot)
comparison_fig = laban.plot_comparisons(...)
# Extract Bland-Altman subplot (requires advanced trace manipulation)
# ... (simplified here)

# Bottom-left: Individual athlete bars
bars_fig1, _ = laban.bars_with_normative_bands(...)
for trace in bars_fig1.data:
    fig.add_trace(trace, row=2, col=1)

# Bottom-right: Team bars
bars_fig2, _ = laban.bars_with_normative_bands(...)
for trace in bars_fig2.data:
    fig.add_trace(trace, row=2, col=2)

fig.update_layout(height=900, title="Comprehensive Assessment Report")
fig.show()
```

## Best Practices

**Color Scheme Consistency:**
- Use same color palette across all figures in a publication
- Define colors once, reuse:

```python
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'neutral': '#C73E1D'
}

fig.update_traces(marker_color=COLORS['primary'])
```

**Font Size Guidelines:**

| Element | Size (pt) | Use Case |
|---------|-----------|----------|
| Title | 14-18 | Main figure title |
| Axis labels | 12-14 | X/Y axis titles |
| Tick labels | 10-12 | Axis tick marks |
| Legend | 10-12 | Legend text |
| Annotations | 9-11 | In-plot notes |

**Legend Placement:**
- Inside plot area: Use when space is limited
- Outside plot area: Better for readability

```python
# Inside, top-right
fig.update_layout(legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top'))

# Outside, right
fig.update_layout(legend=dict(x=1.02, y=0.5, xanchor='left', yanchor='middle'))
```

**Aspect Ratios:**
- Square (1:1): Scatter plots, Bland-Altman
- Landscape (16:9 or 4:3): Time series, multi-panel
- Portrait (3:4): Vertical comparisons

```python
# Square
fig.update_layout(width=600, height=600)

# Landscape 16:9
fig.update_layout(width=1600, height=900)
```

**Accessibility:**
- Use colorblind-friendly palettes
- Add patterns/line styles in addition to colors
- Ensure sufficient contrast (text vs background)

## Advanced Examples

### Example 1: Custom Bland-Altman with CI

```python
import labanalysis as laban
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Generate data
true_data = np.random.uniform(20, 50, 50)
pred_data = true_data + np.random.normal(0, 3, 50)

# Compute Bland-Altman metrics
mean_vals = (true_data + pred_data) / 2
diffs = pred_data - true_data
bias = np.mean(diffs)
std_diffs = np.std(diffs)
loa_upper = bias + 1.96 * std_diffs
loa_lower = bias - 1.96 * std_diffs

# Confidence intervals for LOA
n = len(diffs)
se_loa = std_diffs * np.sqrt(3 / n)
ci_upper = loa_upper + 1.96 * se_loa
ci_lower = loa_lower - 1.96 * se_loa

# Create custom Bland-Altman
fig = go.Figure()

# Scatter points
fig.add_trace(go.Scatter(
    x=mean_vals, y=diffs,
    mode='markers',
    marker=dict(color='blue', size=8, opacity=0.6),
    name='Data'
))

# Bias line
fig.add_hline(y=bias, line_dash='solid', line_color='red', annotation_text=f'Bias: {bias:.2f}')

# LOA lines
fig.add_hline(y=loa_upper, line_dash='dash', line_color='red', annotation_text=f'Upper LOA: {loa_upper:.2f}')
fig.add_hline(y=loa_lower, line_dash='dash', line_color='red', annotation_text=f'Lower LOA: {loa_lower:.2f}')

# Confidence intervals (shaded)
x_range = [np.min(mean_vals), np.max(mean_vals)]
fig.add_trace(go.Scatter(
    x=x_range + x_range[::-1],
    y=[ci_upper, ci_upper, loa_upper, loa_upper],
    fill='toself',
    fillcolor='rgba(255,0,0,0.1)',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig.update_layout(
    title="Bland-Altman Plot with 95% CI for LOA",
    xaxis_title="Mean of Two Methods",
    yaxis_title="Difference (Pred - True)",
    template='plotly_white'
)

fig.show()
```

### Example 2: Multi-Panel Athlete Report

```python
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=3, cols=2,
    specs=[
        [{"type": "indicator"}, {"type": "indicator"}],
        [{"colspan": 2}, None],
        [{"type": "bar"}, {"type": "scatter"}]
    ],
    subplot_titles=['', '', 'Performance Profile', 'Progress Over Time', 'Peer Comparison']
)

# Top: Key metrics as indicators
fig.add_trace(go.Indicator(
    mode="number+delta",
    value=35,
    delta={'reference': 32, 'relative': False},
    title={'text': "CMJ Height (cm)"},
    domain={'x': [0, 0.5], 'y': [0.7, 1]}
), row=1, col=1)

fig.add_trace(go.Indicator(
    mode="number+delta",
    value=4500,
    delta={'reference': 4200},
    title={'text': "Peak Power (W)"},
), row=1, col=2)

# Middle: Bar chart (performance profile)
# ... add bars

# Bottom-left: Line chart (progress)
# ... add time series

# Bottom-right: Scatter (peer comparison)
# ... add scatter

fig.update_layout(height=1000, title="Athlete Report: John Doe")
fig.show()
```

### Example 3: Time-Series Comparison with Highlighting

```python
# Longitudinal tracking with goal highlighting
weeks = [0, 2, 4, 6, 8, 10, 12]
cmj = [28, 30, 32, 33, 35, 36, 38]
goal = 35

fig = go.Figure()

# Line plot
fig.add_trace(go.Scatter(
    x=weeks, y=cmj,
    mode='lines+markers',
    line=dict(color='blue', width=3),
    marker=dict(size=10),
    name='CMJ Height'
))

# Goal line
fig.add_hline(
    y=goal,
    line_dash='dash',
    line_color='green',
    annotation_text='Goal',
    annotation_position='right'
)

# Highlight when goal reached
goal_week = next(w for w, c in zip(weeks, cmj) if c >= goal)
fig.add_vrect(
    x0=goal_week, x1=weeks[-1],
    fillcolor='green', opacity=0.1,
    annotation_text='Goal Achieved',
    annotation_position='top left'
)

fig.update_layout(
    title="12-Week Training Progress",
    xaxis_title="Week",
    yaxis_title="CMJ Height (cm)"
)

fig.show()
```

## Troubleshooting

**Problem:** Subplot indexing errors

```python
# Error: Invalid subplot reference
fig.update_xaxes(title="X", row=5, col=1)  # But only 2 rows exist!

# Solution: Check subplot specs
print(fig.layout)  # Verify actual row/col count
```

**Problem:** Trace visibility

```python
# Traces not showing
fig.add_trace(go.Scatter(...), row=1, col=2)

# Solution: Check if trace was added to correct subplot
for trace in fig.data:
    print(trace.xaxis, trace.yaxis)  # Should show x2, y2 for subplot (1,2)
```

**Problem:** Export resolution issues

```python
# Image too small or pixelated
fig.write_image("low_res.png")

# Solution: Specify dimensions explicitly
fig.write_image("high_res.png", width=1200, height=800, scale=2)
```

**Problem:** Memory issues with large datasets

```python
# Figure with 100k points crashes browser
fig = go.Figure(data=go.Scatter(y=large_array))

# Solution: Downsample or use Scattergl
fig = go.Figure(data=go.Scattergl(y=large_array[::10]))  # Every 10th point, WebGL
```

## See Also

- [Plotly Basics](plotly-basics.md) - Plotly integration fundamentals
- [Comparison Plots](comparison-plots.md) - Using `plot_comparisons()` 
- [Protocol Reports](protocol-reports.md) - Using `bars_with_normative_bands()`
- [Signal Processing](../signal-processing/filtering.md) - Preparing data for visualization
- [Plotly Documentation](https://plotly.com/python/) - Complete Plotly reference
- [Plotly Figure Reference](https://plotly.com/python/reference/) - All trace types and properties

**Questions?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
