# Regression Models

Guide to fitting regression models using OLS (Ordinary Least Squares) approaches in labanalysis.

## Overview

labanalysis provides 5 regression model classes built on top of scikit-learn's `LinearRegression`:

1. **BaseRegression** - Simple OLS with optional input transformation
2. **PolynomialRegression** - Polynomial expansion (degree n)
3. **PowerRegression** - Power law models (Y = b0 * X^b1)
4. **ExponentialRegression** - Exponential models (Y = b0 + X^b1)
5. **MultiSegmentRegression** - Piecewise polynomial regression

All models return predictions as **pandas DataFrames** with proper column names and support multivariate inputs/outputs.

## Quick Reference

```python
import labanalysis as laban
import numpy as np

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 4.3, 6.8, 9.2, 11.5])

# Polynomial regression (degree 2)
model = laban.PolynomialRegression(degree=2)
model.fit(x, y)
y_pred = model.predict(x)

# Get coefficients
print(model.betas)
#        Y0
# beta0  1.1   (intercept)
# beta1  1.9   (linear term)
# beta2  0.05  (quadratic term)
```

## BaseRegression

Simple OLS linear regression with optional input transformation.

### Basic Linear Regression

```python
import labanalysis as laban
import numpy as np

# Generate linear data
x = np.array([1, 2, 3, 4, 5])
y = 2.5 * x + 1.2

# Fit linear model
model = laban.BaseRegression(fit_intercept=True)
model.fit(x, y)

# Predict
x_new = np.array([6, 7, 8])
y_pred = model.predict(x_new)

print(y_pred)
#        Y0
# 0   16.2
# 1   18.7
# 2   21.2

# Get coefficients
print(model.betas)
#        Y0
# beta0  1.2   (intercept)
# beta1  2.5   (slope)
```

### With Input Transformation

Apply a transformation function to inputs before fitting:

```python
# Log transformation
model_log = laban.BaseRegression(
    fit_intercept=True,
    transform=np.log  # Transform X before regression
)

x = np.array([1, 10, 100, 1000])
y = np.array([2, 3, 4, 5])

model_log.fit(x, y)
y_pred = model_log.predict(x)

# Model internally fits: Y = b0 + b1 * log(X)
```

### Multivariate Regression

Multiple input features:

```python
# 2 input features, 1 output
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])
y = np.array([5, 8, 11, 14])

model = laban.BaseRegression()
model.fit(X, y)

# Coefficients
print(model.betas)
#        Y0
# beta0  1.0    (intercept)
# beta1  1.5    (coefficient for X0)
# beta2  2.0    (coefficient for X1)

# Predict
X_new = np.array([[5, 6], [6, 7]])
y_pred = model.predict(X_new)
```

### Multi-Output Regression

Multiple outputs:

```python
# 1 input, 2 outputs
x = np.array([1, 2, 3, 4, 5])
Y = np.array([
    [2, 10],
    [4, 20],
    [6, 30],
    [8, 40],
    [10, 50]
])

model = laban.BaseRegression()
model.fit(x, Y)

print(model.betas)
#         Y0    Y1
# beta0  0.0   0.0
# beta1  2.0  10.0

# Predictions have both columns
y_pred = model.predict([6, 7])
print(y_pred)
#       Y0    Y1
# 0   12.0  60.0
# 1   14.0  70.0
```

## PolynomialRegression

Polynomial expansion of input features up to degree n.

**Model form**: Y = β₀ + β₁·X + β₂·X² + ... + βₙ·Xⁿ

### Quadratic Fit

```python
import labanalysis as laban
import numpy as np

# Generate quadratic data
x = np.linspace(0, 10, 50)
y = 0.5 * x**2 - 3 * x + 5 + np.random.randn(50) * 2

# Fit quadratic model (degree 2)
model = laban.PolynomialRegression(degree=2)
model.fit(x, y)

# Coefficients
print(model.betas)
#        Y0
# beta0  5.2    (intercept)
# beta1  -2.9   (linear term)
# beta2  0.48   (quadratic term)

# Predict
x_new = np.linspace(0, 10, 100)
y_pred = model.predict(x_new)
```

### Higher Degree Polynomials

```python
# Cubic polynomial (degree 3)
model_cubic = laban.PolynomialRegression(degree=3)
model_cubic.fit(x, y)

# Quartic polynomial (degree 4)
model_quartic = laban.PolynomialRegression(degree=4)
model_quartic.fit(x, y)
```

### Multivariate Polynomial

```python
# 2D polynomial expansion
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])
y = np.array([5, 12, 23, 38])

# Degree 2 with interactions: Y = b0 + b1*X0 + b2*X1 + b3*X0² + b4*X0*X1 + b5*X1²
model = laban.PolynomialRegression(degree=2)
model.fit(X, y)

print(model.get_feature_names_in())
# ['x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']

print(model.betas)
#          Y0
# beta0   1.0     (intercept)
# beta1   0.5     (X0)
# beta2   1.0     (X1)
# beta3   0.2     (X0²)
# beta4   0.1     (X0·X1)
# beta5   0.3     (X1²)
```

### Without Intercept

```python
# Force regression through origin
model_no_intercept = laban.PolynomialRegression(
    degree=2,
    fit_intercept=False
)
model_no_intercept.fit(x, y)
```

## PowerRegression

Power law model: **Y = β₀ · X₁^β₁ · X₂^β₂ · ... · Xₙ^βₙ**

Common in biomechanics for allometric scaling (e.g., force-velocity relationships).

### Simple Power Law

```python
import labanalysis as laban
import numpy as np

# Generate power law data: Y = 2 * X^0.7
x = np.linspace(1, 100, 50)
y = 2 * x**0.7 + np.random.randn(50) * 0.5

# Fit power model
model = laban.PowerRegression()
model.fit(x, y)

print(model.betas)
#        Y0
# beta0  2.1    (multiplicative constant)
# beta1  0.68   (exponent)

# Model equation: Y = 2.1 * X^0.68

# Predict
y_pred = model.predict([10, 50, 100])
print(y_pred)
#        Y0
# 0   10.5
# 1   28.3
# 2   42.1
```

### Force-Velocity Relationship

```python
# Fit force-velocity curve (Hill's equation approximation)
velocity = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # m/s
force = np.array([850, 650, 500, 400, 330, 280])     # N

model = laban.PowerRegression()
model.fit(velocity, force)

print(model.betas)
#          Force
# beta0   1020.5   (F₀, isometric force)
# beta1   -0.42    (power exponent)

# Predict force at new velocities
v_new = np.array([0.8, 1.2, 1.8])
f_pred = model.predict(v_new)
```

### Multivariate Power Law

```python
# Y = b0 * X0^b1 * X1^b2
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])
y = 5 * (X[:, 0] ** 0.5) * (X[:, 1] ** 0.3)

model = laban.PowerRegression()
model.fit(X, y)

print(model.betas)
#        Y0
# beta0  5.1    (multiplicative constant)
# beta1  0.48   (exponent for X0)
# beta2  0.31   (exponent for X1)
```

### Important: Positive Values Only

PowerRegression uses log transformation, so all X and Y values must be positive:

```python
# WRONG: Contains zero or negative values
x = np.array([0, 1, 2, 3])  # Error! Contains 0
y = np.array([1, 2, 3, 4])

model = laban.PowerRegression()
# model.fit(x, y)  # Raises ValueError

# RIGHT: All positive
x = np.array([0.1, 1, 2, 3])  # All > 0
model.fit(x, y)  # Works
```

## ExponentialRegression

Exponential model: **Y = β₀ + X₁^β₁ + X₂^β₂ + ... + Xₙ^βₙ**

Uses numerical optimization (BFGS) to find exponents.

### Simple Exponential

```python
import labanalysis as laban
import numpy as np

# Generate exponential data: Y = 2 + X^1.5
x = np.linspace(0.1, 5, 50)
y = 2 + x**1.5 + np.random.randn(50) * 0.3

# Fit exponential model
model = laban.ExponentialRegression(fit_intercept=True)
model.fit(x, y)

print(model.betas)
#        Y0
# beta0  2.1    (intercept)
# X0     1.48   (exponent)

# Model equation: Y = 2.1 + X^1.48

# Predict
y_pred = model.predict([1, 2, 3, 4])
```

### Without Intercept

```python
# Y = X^b (no constant term)
model_no_intercept = laban.ExponentialRegression(fit_intercept=False)
model_no_intercept.fit(x, y)

print(model_no_intercept.betas)
#        Y0
# X0     1.52   (exponent only)
```

### Multivariate Exponential

```python
# Y = b0 + X0^b1 + X1^b2
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])
y = 5 + X[:, 0]**1.2 + X[:, 1]**0.8

model = laban.ExponentialRegression()
model.fit(X, y)

print(model.betas)
#        Y0
# beta0  5.1    (intercept)
# X0     1.18   (exponent for X0)
# X1     0.82   (exponent for X1)
```

## MultiSegmentRegression

Piecewise polynomial regression with automatic segmentation.

**Use case**: When data exhibits different linear/polynomial behavior in different regions.

### Two-Segment Linear

```python
import labanalysis as laban
import numpy as np

# Generate two-segment data
x = np.linspace(0, 10, 100)
y = np.where(
    x < 5,
    2 * x + 1,           # Segment 1: Y = 2X + 1
    -1 * x + 16          # Segment 2: Y = -X + 16
) + np.random.randn(100) * 0.3

# Fit 2-segment linear model
model = laban.MultiSegmentRegression(
    degree=1,        # Linear in each segment
    n_segments=2,    # 2 segments
    min_samples=10   # At least 10 points per segment
)
model.fit(x, y)

# Coefficients (different for each segment)
print(model.betas)
# Columns are MultiIndex: (Feature, X0, X1)
#          (Y0, -inf, 5.0)  (Y0, 5.0, inf)
# alpha0        0.0            5.0
# beta0         1.1           15.8
# beta1         1.98          -0.97

# Segment 1: Y = 1.1 + 1.98·X  (for X in [-inf, 5.0])
# Segment 2: Y = 15.8 - 0.97·X (for X in [5.0, inf])

# Predict (automatically selects correct segment)
y_pred = model.predict(x)
```

### Three-Segment Quadratic

```python
# 3 segments, quadratic in each
model_3seg = laban.MultiSegmentRegression(
    degree=2,
    n_segments=3,
    min_samples=15
)

x = np.linspace(0, 15, 200)
y = np.piecewise(
    x,
    [x < 5, (x >= 5) & (x < 10), x >= 10],
    [lambda x: x**2, lambda x: -x**2 + 50, lambda x: 0.5*x - 5]
)

model_3seg.fit(x, y)
y_pred = model_3seg.predict(x)
```

### Automatic Breakpoint Detection

The model automatically finds optimal breakpoints by minimizing SSE:

```python
# Let the model find the best segmentation
model = laban.MultiSegmentRegression(degree=1, n_segments=2)
model.fit(x, y)

# Extract breakpoints from betas columns
breakpoints = []
for feat, x0, x1 in model.betas.columns:
    if x0 != -np.inf:
        breakpoints.append(x0)

breakpoints = sorted(set(breakpoints))
print(f"Breakpoints detected at: {breakpoints}")
# Output: Breakpoints detected at: [4.98]
```

## Model Comparison and Selection

### Comparing Models

```python
import labanalysis as laban
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Generate test data
x = np.linspace(1, 10, 50)
y_true = 2 * x**1.5 + 5 + np.random.randn(50) * 2

# Fit different models
models = {
    'Linear': laban.PolynomialRegression(degree=1),
    'Quadratic': laban.PolynomialRegression(degree=2),
    'Cubic': laban.PolynomialRegression(degree=3),
    'Power': laban.PowerRegression(),
    'Exponential': laban.ExponentialRegression(),
}

results = {}
for name, model in models.items():
    model.fit(x, y_true)
    y_pred = model.predict(x)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    results[name] = {'R²': r2, 'RMSE': rmse}

# Print comparison
import pandas as pd
df_results = pd.DataFrame(results).T
print(df_results)
#                  R²    RMSE
# Linear        0.92    3.5
# Quadratic     0.96    2.1
# Cubic         0.97    1.9
# Power         0.98    1.4  ← Best fit
# Exponential   0.97    1.8
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Use sklearn's cross-validation with labanalysis models
model = laban.PolynomialRegression(degree=2)

# labanalysis models are sklearn-compatible
scores = cross_val_score(
    model, 
    x.reshape(-1, 1), 
    y_true, 
    cv=5,  # 5-fold CV
    scoring='r2'
)

print(f"CV R² scores: {scores}")
print(f"Mean R²: {scores.mean():.3f} ± {scores.std():.3f}")
```

## Practical Applications

### 1RM Prediction (Strength Assessment)

```python
# Predict 1RM from submaximal loads
loads = np.array([60, 70, 80, 90, 100])  # kg
reps = np.array([12, 8, 5, 3, 1])

# Fit power model: Reps = b0 * Load^b1
model = laban.PowerRegression()
model.fit(loads, reps)

# Predict 1RM (1 rep max)
one_rm = model.predict([loads.max() + 10, loads.max() + 20])
print(f"Predicted 1RM: {one_rm.values[0, 0]:.1f} kg")
```

### Jump Height vs Body Mass

```python
# Allometric scaling
body_mass = np.array([60, 65, 70, 75, 80, 85])  # kg
jump_height = np.array([45, 43, 41, 39, 37, 35])  # cm

model = laban.PowerRegression()
model.fit(body_mass, jump_height)

print(model.betas)
#             Jump_Height
# beta0       120.5        (scaling constant)
# beta1       -0.35        (negative exponent)

# Heavier athletes jump lower (negative scaling)
```

### Force-Time Curve Segments

```python
# Analyze different phases of force production
time = np.linspace(0, 2, 200)  # seconds
force = np.concatenate([
    50 + 400 * time[:50],              # Rising phase
    450 - 50 * (time[50:150] - 0.5),  # Plateau
    400 - 200 * (time[150:] - 1.5)    # Descending
])

# 3-segment model
model = laban.MultiSegmentRegression(degree=1, n_segments=3)
model.fit(time, force)

# Extract phase characteristics from coefficients
print(model.betas)
# Each segment has different slope → different phase characteristics
```

## Export and Persistence

### Save Coefficients

```python
# Fit model
model = laban.PolynomialRegression(degree=2)
model.fit(x, y)

# Export coefficients to CSV
model.betas.to_csv("model_coefficients.csv")

# Export to Excel
model.betas.to_excel("model_coefficients.xlsx")
```

### Model Copying

```python
# Create independent copy
model_original = laban.PolynomialRegression(degree=2)
model_original.fit(x, y)

model_copy = model_original.copy()

# Modify copy without affecting original
model_copy.fit(x_new, y_new)
```

## Troubleshooting

### Issue: "All values must be positive for log transformation"

PowerRegression requires positive X and Y:

```python
# WRONG
x = np.array([0, 1, 2, 3])  # Contains 0
model = laban.PowerRegression()
# model.fit(x, y)  # ValueError

# RIGHT: Shift data or use different model
x = np.array([0.1, 1, 2, 3])  # All positive
model.fit(x, y)  # Works

# Or use ExponentialRegression instead
model_exp = laban.ExponentialRegression()
model_exp.fit([0, 1, 2, 3], y)  # Accepts 0
```

### Issue: MultiSegmentRegression "xarr must be a 1D array"

```python
# WRONG: 2D input
X = np.array([[1, 2], [3, 4]])
# model.fit(X, y)  # ValueError

# RIGHT: Only 1D input
x = np.array([1, 2, 3, 4])
model.fit(x, y)  # Works
```

### Issue: Poor fit with PolynomialRegression

Try different degree or use non-linear model:

```python
# If polynomial doesn't fit well
model_poly = laban.PolynomialRegression(degree=3)
model_poly.fit(x, y)
r2_poly = r2_score(y, model_poly.predict(x))

if r2_poly < 0.9:
    # Try power or exponential
    model_power = laban.PowerRegression()
    model_power.fit(x, y)
    r2_power = r2_score(y, model_power.predict(x))
    
    if r2_power > r2_poly:
        print("Power model fits better")
```

### Issue: Overfitting with high polynomial degree

Use cross-validation to select optimal degree:

```python
from sklearn.model_selection import cross_val_score

degrees = [1, 2, 3, 4, 5]
cv_scores = []

for deg in degrees:
    model = laban.PolynomialRegression(degree=deg)
    scores = cross_val_score(model, x.reshape(-1, 1), y, cv=5, scoring='r2')
    cv_scores.append(scores.mean())

best_degree = degrees[np.argmax(cv_scores)]
print(f"Best degree: {best_degree}")
```

## See Also

- [PyTorch Basics](pytorch-basics.md) - Deep learning models
- [TorchTrainer](torch-trainer.md) - Neural network training
- [User Guide: Test Protocols](../test-protocols/README.md) - Applying regression to test data
- [API Reference: OLS](../../api-reference/modelling/ols.md) - Complete regression API

---

**Regression Models**: Five OLS-based regression classes for fitting linear, polynomial, power, exponential, and piecewise models to biomechanical data.
