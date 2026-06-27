# labanalysis.modelling.ols

Ordinary Least Squares regression and geometric modeling.

**Source**: `src/labanalysis/modelling/ols/`

## Overview

OLS-based regression models and geometric object fitting:

**Regression Models:**
- **BaseRegression**: Base class with input transformation support
- **PolynomialRegression**: Polynomial regression (degree 1-n)
- **PowerRegression**: Power law regression (y = a × x^b)
- **ExponentialRegression**: Exponential regression (y = a × e^(b×x))
- **MultiSegmentRegression**: Piecewise regression with breakpoints

**Geometric Objects:**
- **Line2D**: 2D line fitting (A×x + B×y + C = 0)
- **Line3D**: 3D line fitting
- **Circle**: Circle fitting with center and radius
- **Ellipse**: Ellipse fitting with semi-axes, rotation, and area

All models use ordinary least squares for parameter estimation.

## Regression Classes

### BaseRegression

Base class for all regression models.

```python
class BaseRegression(LinearRegression):
    """
    Base regression class with input transformation support.
    
    Extends sklearn's LinearRegression with callable transformation
    functions and intercept control.
    
    Parameters
    ----------
    fit_intercept : bool, optional
        Whether to calculate intercept
        Default: True
    transform : Callable, optional
        Function applied element-wise to input X
        Default: lambda x: x (identity)
    positive : bool, optional
        Force non-negative coefficients
        Default: False
    
    Attributes
    ----------
    betas : pd.DataFrame
        Estimated coefficients (including intercept if applicable)
    transform : Callable
        Input transformation function
    
    Methods
    -------
    fit(X, y)
        Fit regression model
    predict(X)
        Predict using fitted model
    copy()
        Create deep copy of model
    
    Examples
    --------
    >>> from labanalysis.modelling import BaseRegression
    >>> import numpy as np
    >>> 
    >>> # Simple linear regression
    >>> X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> 
    >>> model = BaseRegression()
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> 
    >>> print(model.betas)
    """
```

---

### PolynomialRegression

Polynomial regression of arbitrary degree.

```python
class PolynomialRegression(BaseRegression):
    """
    Polynomial regression: Y = b0 + b1×fn(X)^1 + ... + bn×fn(X)^n + e
    
    Parameters
    ----------
    degree : int, optional
        Polynomial degree
        Default: 1 (linear)
    fit_intercept : bool, optional
        Include intercept term
        Default: True
    transform : Callable, optional
        Input transformation before polynomial expansion
        Default: lambda x: x
    positive : bool, optional
        Force non-negative coefficients
        Default: False
    
    Examples
    --------
    >>> from labanalysis.modelling import PolynomialRegression
    >>> import numpy as np
    >>> 
    >>> # Quadratic fit
    >>> X = np.linspace(0, 10, 50).reshape(-1, 1)
    >>> y = 2 + 3*X.flatten() - 0.5*X.flatten()**2 + np.random.normal(0, 1, 50)
    >>> 
    >>> model = PolynomialRegression(degree=2)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> 
    >>> print(f"R²: {model.score(X, y):.3f}")
    >>> print(model.betas)
    """
```

**Use Cases:**
- Curved relationships (force-velocity, length-tension)
- Non-linear trends in biomechanical data
- Smoothing with polynomial basis

---

### PowerRegression

Power law regression.

```python
class PowerRegression(BaseRegression):
    """
    Power law regression: Y = a × X^b
    
    Linearized via log transformation: log(Y) = log(a) + b×log(X)
    
    Parameters
    ----------
    fit_intercept : bool, optional
        Include intercept (a parameter)
        Default: True
    positive : bool, optional
        Force non-negative coefficients
        Default: False
    
    Examples
    --------
    >>> from labanalysis.modelling import PowerRegression
    >>> import numpy as np
    >>> 
    >>> # Allometric relationship
    >>> X = np.array([50, 60, 70, 80, 90]).reshape(-1, 1)  # Body mass (kg)
    >>> y = np.array([45, 52, 58, 63, 68])  # VO2max (ml/kg/min)
    >>> 
    >>> model = PowerRegression()
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> 
    >>> print(f"Power exponent (b): {model.coef_[0]:.3f}")
    """
```

**Use Cases:**
- Allometric scaling (body size relationships)
- Force-length curves
- Metabolic scaling laws

---

### ExponentialRegression

Exponential regression.

```python
class ExponentialRegression(BaseRegression):
    """
    Exponential regression: Y = a × e^(b×X)
    
    Linearized via log transformation: log(Y) = log(a) + b×X
    
    Parameters
    ----------
    fit_intercept : bool, optional
        Include intercept (a parameter)
        Default: True
    positive : bool, optional
        Force non-negative coefficients
        Default: False
    
    Examples
    --------
    >>> from labanalysis.modelling import ExponentialRegression
    >>> import numpy as np
    >>> 
    >>> # Lactate accumulation
    >>> X = np.array([8, 10, 12, 14, 16]).reshape(-1, 1)  # Speed (km/h)
    >>> y = np.array([1.2, 1.8, 3.5, 7.2, 15.8])  # Blood lactate (mmol/L)
    >>> 
    >>> model = ExponentialRegression()
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> 
    >>> print(f"Exponential rate (b): {model.coef_[0]:.3f}")
    """
```

**Use Cases:**
- Lactate accumulation curves
- Fatigue accumulation
- Recovery kinetics (exponential decay with b < 0)

---

### MultiSegmentRegression

Piecewise linear regression with breakpoints.

```python
class MultiSegmentRegression(BaseRegression):
    """
    Piecewise regression with multiple segments.
    
    Fits separate linear models for different ranges of X,
    optimizing breakpoint locations.
    
    Parameters
    ----------
    n_segments : int, optional
        Number of linear segments
        Default: 2
    fit_intercept : bool, optional
        Include intercept for each segment
        Default: True
    
    Examples
    --------
    >>> from labanalysis.modelling import MultiSegmentRegression
    >>> import numpy as np
    >>> 
    >>> # Force-velocity relationship with two phases
    >>> X = np.array([0, 20, 40, 60, 80, 100]).reshape(-1, 1)  # % Vmax
    >>> y = np.array([1800, 1600, 1200, 800, 400, 0])  # Force (N)
    >>> 
    >>> model = MultiSegmentRegression(n_segments=2)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> 
    >>> print(f"Breakpoint: {model.breakpoints_[0]:.1f}% Vmax")
    """
```

**Use Cases:**
- Force-velocity curves (distinct concentric/eccentric phases)
- Fatigue profiles (different rates in early/late phases)
- Gait transition detection (walk-to-run)

---

## Geometric Classes

### Line2D

2D line fitting.

```python
class Line2D(GeometricObject):
    """
    2D line: A×x + B×y + C = 0
    
    Fits line to 2D point cloud using least squares.
    
    Attributes
    ----------
    betas : dict
        Fitted coefficients {'A': float, 'B': float, 'C': float}
    general_equation : sympy.Eq
        Symbolic equation
    fitted_equation : sympy.Eq
        Equation with fitted coefficients
    
    Methods
    -------
    fit(x, y)
        Fit line to points
    predict(x=None, y=None)
        Predict y from x (or vice versa)
    
    Examples
    --------
    >>> from labanalysis.modelling import Line2D
    >>> import numpy as np
    >>> 
    >>> # Fit line to noisy data
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
    >>> 
    >>> line = Line2D()
    >>> line.fit(x, y)
    >>> 
    >>> # Predict y for new x
    >>> y_pred = line.predict(x=np.array([6, 7]))
    >>> 
    >>> print(line.fitted_equation)
    """
```

---

### Line3D

3D line fitting.

```python
class Line3D(GeometricObject):
    """
    3D line fitting.
    
    Fits line to 3D point cloud using principal component analysis.
    
    Methods
    -------
    fit(x, y, z)
        Fit line to 3D points
    predict(x=None, y=None, z=None)
        Predict missing coordinate
    
    Examples
    --------
    >>> from labanalysis.modelling import Line3D
    >>> import numpy as np
    >>> 
    >>> # Fit line to 3D trajectory
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> y = np.array([0, 2, 4, 6, 8])
    >>> z = np.array([0, 1, 2, 3, 4])
    >>> 
    >>> line = Line3D()
    >>> line.fit(x, y, z)
    >>> 
    >>> print(line.fitted_equation)
    """
```

---

### Circle

Circle fitting.

```python
class Circle(GeometricObject):
    """
    Circle fitting: (x - x0)^2 + (y - y0)^2 = r^2
    
    Fits circle to 2D points using least squares.
    
    Attributes
    ----------
    center : tuple
        Circle center (x0, y0)
    radius : float
        Circle radius
    area : float
        Circle area (π × r^2)
    
    Methods
    -------
    fit(x, y)
        Fit circle to points
    
    Examples
    --------
    >>> from labanalysis.modelling import Circle
    >>> import numpy as np
    >>> 
    >>> # Fit circle to COP trajectory
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> x = 5 + 3*np.cos(theta) + np.random.normal(0, 0.1, 100)
    >>> y = 2 + 3*np.sin(theta) + np.random.normal(0, 0.1, 100)
    >>> 
    >>> circle = Circle()
    >>> circle.fit(x, y)
    >>> 
    >>> print(f"Center: ({circle.center[0]:.2f}, {circle.center[1]:.2f})")
    >>> print(f"Radius: {circle.radius:.2f}")
    >>> print(f"Area: {circle.area:.2f}")
    """
```

**Use Cases:**
- COP sway area estimation
- Joint center location
- Circular motion analysis

---

### Ellipse

Ellipse fitting.

```python
class Ellipse(GeometricObject):
    """
    Ellipse fitting with full parameterization.
    
    Fits ellipse to 2D points, extracting center, semi-axes,
    rotation angle, and area.
    
    Attributes
    ----------
    center : tuple
        Ellipse center (x0, y0)
    semi_axes : tuple
        Semi-major and semi-minor axes (a, b)
    rotation_angle : float
        Rotation angle in degrees
    area : float
        Ellipse area (π × a × b)
    eccentricity : float
        Eccentricity (0 = circle, approaching 1 = line)
    
    Methods
    -------
    fit(x, y)
        Fit ellipse to points
    
    Examples
    --------
    >>> from labanalysis.modelling import Ellipse
    >>> import numpy as np
    >>> 
    >>> # Fit ellipse to COP sway
    >>> cop_x = np.random.normal(0, 3, 1000)  # mm
    >>> cop_y = np.random.normal(0, 5, 1000)  # mm
    >>> 
    >>> ellipse = Ellipse()
    >>> ellipse.fit(cop_x, cop_y)
    >>> 
    >>> print(f"Center: ({ellipse.center[0]:.2f}, {ellipse.center[1]:.2f})")
    >>> print(f"Semi-axes: {ellipse.semi_axes[0]:.2f} × {ellipse.semi_axes[1]:.2f}")
    >>> print(f"Rotation: {ellipse.rotation_angle:.1f}°")
    >>> print(f"Area: {ellipse.area:.1f} mm²")
    >>> print(f"Eccentricity: {ellipse.eccentricity:.3f}")
    """
```

**Use Cases:**
- **COP sway area**: 95% confidence ellipse for balance assessment
- Joint range of motion
- Gait trajectory analysis

**Note**: Used extensively in balance tests for quantifying postural stability.

---

## Complete Example Workflows

### Force-Velocity Profiling

```python
import numpy as np
import pandas as pd
from labanalysis.modelling import PolynomialRegression, PowerRegression
import plotly.graph_objects as go

# Sprint force-velocity data
velocity_pct = np.array([0, 20, 40, 60, 80, 100])
force_N = np.array([1800, 1600, 1200, 800, 400, 0])

X = velocity_pct.reshape(-1, 1)
y = force_N

# Fit linear model
linear = PolynomialRegression(degree=1)
linear.fit(X, y)

# Fit quadratic model (better for sprint mechanics)
quadratic = PolynomialRegression(degree=2)
quadratic.fit(X, y)

# Compare fits
print(f"Linear R²: {linear.score(X, y):.3f}")
print(f"Quadratic R²: {quadratic.score(X, y):.3f}")

# Predict for plotting
X_plot = np.linspace(0, 100, 100).reshape(-1, 1)
y_linear = linear.predict(X_plot)
y_quadratic = quadratic.predict(X_plot)

# Plot
fig = go.Figure()
fig.add_scatter(x=velocity_pct, y=force_N, mode='markers', name='Data', marker=dict(size=10))
fig.add_scatter(x=X_plot.flatten(), y=y_linear, mode='lines', name='Linear')
fig.add_scatter(x=X_plot.flatten(), y=y_quadratic, mode='lines', name='Quadratic')
fig.update_layout(
    title='Force-Velocity Profile',
    xaxis_title='Velocity (% Vmax)',
    yaxis_title='Force (N)'
)
fig.show()
```

### COP Sway Area Calculation

```python
import numpy as np
from labanalysis.modelling import Ellipse

# Simulate COP trajectory from balance test
np.random.seed(42)
n_samples = 1000
cop_x = np.random.normal(0, 3, n_samples)  # Mediolateral (mm)
cop_y = np.random.normal(0, 5, n_samples)  # Anteroposterior (mm)

# Fit ellipse (95% confidence)
ellipse = Ellipse()
ellipse.fit(cop_x, cop_y)

print("=== COP Sway Analysis ===")
print(f"Ellipse center: ({ellipse.center[0]:.2f}, {ellipse.center[1]:.2f}) mm")
print(f"Semi-axes: {ellipse.semi_axes[0]:.2f} × {ellipse.semi_axes[1]:.2f} mm")
print(f"Rotation: {ellipse.rotation_angle:.1f}°")
print(f"Sway area: {ellipse.area:.1f} mm²")
print(f"Eccentricity: {ellipse.eccentricity:.3f}")

# Interpretation
if ellipse.area < 200:
    print("  → Excellent postural control")
elif ellipse.area < 400:
    print("  → Good postural control")
elif ellipse.area < 600:
    print("  → Fair postural control")
else:
    print("  → Poor postural control (fall risk)")

# Check ellipse orientation
if abs(ellipse.rotation_angle) < 30:
    print("  → Predominantly AP sway")
elif abs(ellipse.rotation_angle) > 60:
    print("  → Predominantly ML sway")
else:
    print("  → Oblique sway pattern")
```

### Lactate Threshold Detection

```python
import numpy as np
from labanalysis.modelling import ExponentialRegression, MultiSegmentRegression

# Graded exercise test data
speed_kmh = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16])
lactate_mmol = np.array([1.0, 1.2, 1.4, 1.7, 2.3, 3.8, 6.5, 11.2, 18.5])

X = speed_kmh.reshape(-1, 1)
y = lactate_mmol

# Exponential fit (traditional method)
exp_model = ExponentialRegression()
exp_model.fit(X, y)

# Multi-segment fit (breakpoint = threshold)
seg_model = MultiSegmentRegression(n_segments=2)
seg_model.fit(X, y)

print("=== Lactate Threshold Analysis ===")
print(f"Exponential model R²: {exp_model.score(X, y):.3f}")
print(f"Breakpoint (threshold): {seg_model.breakpoints_[0]:.1f} km/h")

# Predict lactate at specific speeds
speeds_test = np.array([10, 12, 14]).reshape(-1, 1)
lactate_pred = exp_model.predict(speeds_test)

for speed, lac in zip(speeds_test.flatten(), lactate_pred):
    print(f"  Speed {speed:.0f} km/h → Lactate {lac:.1f} mmol/L")
```

---

## Advanced Features

### Custom Input Transformations

```python
from labanalysis.modelling import BaseRegression
import numpy as np

# Log-transform for exponential relationships
model = BaseRegression(
    fit_intercept=True,
    transform=np.log  # Apply log to X before fitting
)

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2.7, 7.4, 20.1, 54.6, 148.4])  # y = e^x

model.fit(X, y)
y_pred = model.predict(X)
```

### Force Non-Negative Coefficients

```python
from labanalysis.modelling import PolynomialRegression

# Ensure all coefficients are positive (physical constraint)
model = PolynomialRegression(
    degree=2,
    positive=True  # Force b0, b1, b2 >= 0
)

X = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
y = np.array([1, 3, 7, 13, 21])

model.fit(X, y)
print(f"Coefficients (all >= 0): {model.coef_}")
```

---

## See Also

- [Equations](../equations/) - Pre-defined biomechanical equations
- [PyTorch Models](pytorch.md) - Neural network models
- [Plotting](../plotting.md) - Visualization tools

---

**OLS regression and geometric modeling for biomechanical data analysis.**
