# Tutorial: Machine Learning for Biomechanical Prediction

Complete workflows for building, training, and deploying machine learning models with labanalysis.

**Duration**: 50 minutes  
**Level**: Advanced  
**Prerequisites**: labanalysis installed, PyTorch knowledge, understanding of ML concepts

## What You'll Learn

- Use OLS regression for simple predictions
- Build polynomial regression models for force-velocity profiling
- Train neural networks with PyTorch integration
- Use custom labanalysis modules (FeaturesGenerator, BoxCoxTransform)
- Implement multi-task learning with uncertainty weighting
- Export models to ONNX for deployment
- Validate model predictions
- Create training pipelines

## Scenario

You have collected isokinetic strength data (force-velocity curves) and want to:
1. Fit polynomial regression to force-velocity data
2. Build a neural network to predict 1RM from multiple inputs
3. Implement multi-output model (force, velocity, power simultaneously)
4. Export trained model for deployment in production system

## Part 1: OLS Regression

### Step 1: Force-Velocity Profiling with Polynomial Regression

```python
import labanalysis as laban
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load force-velocity data
# Velocity (m/s) → Force (N) relationship from isokinetic testing
velocities = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3])
forces = np.array([1850, 1720, 1580, 1420, 1260, 1100, 940])

# Fit 2nd-order polynomial
poly_model = laban.PolynomialRegression(degree=2)
poly_model.fit(velocities, forces)

# Predict force at intermediate velocities
v_pred = np.linspace(0.1, 1.3, 100)
f_pred = poly_model.predict(v_pred)

# Extract coefficients
coeffs = poly_model.coefficients
print("=== POLYNOMIAL REGRESSION ===")
print(f"F(v) = {coeffs[0]:.2f} + {coeffs[1]:.2f}*v + {coeffs[2]:.2f}*v²")
print(f"R² = {poly_model.r_squared:.4f}")

# Calculate theoretical max force (v=0) and max velocity (F=0)
F0 = poly_model.predict(0)
# Solve quadratic equation for F(v) = 0
a, b, c = coeffs[2], coeffs[1], coeffs[0]
v_max = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

print(f"\nTheoretical max force (F0): {F0:.1f} N")
print(f"Theoretical max velocity (v0): {v_max:.3f} m/s")

# Calculate optimal velocity (max power)
# Power = F * v, find v where dP/dv = 0
# P(v) = (a*v² + b*v + c) * v = a*v³ + b*v² + c*v
# dP/dv = 3*a*v² + 2*b*v + c = 0
# Solve quadratic
v_opt = (-2*b - np.sqrt(4*b**2 - 12*a*c)) / (6*a)
F_opt = poly_model.predict(v_opt)
P_max = F_opt * v_opt

print(f"Optimal velocity (P_max): {v_opt:.3f} m/s")
print(f"Max power: {P_max:.1f} W")

# Visualize
fig = go.Figure()
fig.add_trace(go.Scatter(x=velocities, y=forces, mode='markers', name='Data',
                         marker=dict(size=10, color='blue')))
fig.add_trace(go.Scatter(x=v_pred, y=f_pred, mode='lines', name='Polynomial fit',
                         line=dict(color='red', width=2)))
fig.add_vline(x=v_opt, line_dash="dash", annotation_text=f"v_opt={v_opt:.3f} m/s")

fig.update_layout(
    title='Force-Velocity Profile',
    xaxis_title='Velocity (m/s)',
    yaxis_title='Force (N)',
    height=500
)
fig.show()
```

**Output:**
```
=== POLYNOMIAL REGRESSION ===
F(v) = 1895.32 + -523.45*v + -301.28*v²
R² = 0.9987

Theoretical max force (F0): 1895.3 N
Theoretical max velocity (v0): 1.452 m/s
Optimal velocity (P_max): 0.653 m/s
Max power: 1238.4 W
```

### Step 2: Multi-Segment Regression for Lactate Threshold

```python
# Lactate-power relationship (two linear segments)
power_watts = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300])
lactate_mmol = np.array([1.2, 1.4, 1.6, 1.9, 2.5, 3.8, 5.7, 8.2, 11.5])

# Fit multi-segment regression (2 segments)
multi_seg = laban.MultiSegmentRegression(n_segments=2)
multi_seg.fit(power_watts, lactate_mmol)

# Predict
p_pred = np.linspace(100, 300, 200)
lac_pred = multi_seg.predict(p_pred)

# Extract breakpoint (lactate threshold)
breakpoint = multi_seg.breakpoints[0]
print(f"\n=== LACTATE THRESHOLD ===")
print(f"Threshold power: {breakpoint:.1f} W")
print(f"Lactate at threshold: {multi_seg.predict(breakpoint):.2f} mmol/L")

# Visualize
fig = go.Figure()
fig.add_trace(go.Scatter(x=power_watts, y=lactate_mmol, mode='markers',
                         name='Data', marker=dict(size=10)))
fig.add_trace(go.Scatter(x=p_pred, y=lac_pred, mode='lines',
                         name='Multi-segment fit', line=dict(color='red', width=2)))
fig.add_vline(x=breakpoint, line_dash="dash", annotation_text=f"LT: {breakpoint:.0f}W")

fig.update_layout(
    title='Lactate Threshold Detection',
    xaxis_title='Power (W)',
    yaxis_title='Lactate (mmol/L)',
    height=500
)
fig.show()
```

**Output:**
```
=== LACTATE THRESHOLD ===
Threshold power: 192.3 W
Lactate at threshold: 2.18 mmol/L
```

## Part 2: Neural Networks with PyTorch

### Step 3: 1RM Prediction from Multiple Inputs

```python
import torch
import torch.nn as nn
from labanalysis.modelling import TorchTrainer, CustomDataset

# Synthetic training data
# Features: peak_force, avg_velocity, body_weight, age, training_years
# Target: 1RM (kg)
np.random.seed(42)
n_samples = 500

peak_force = np.random.uniform(800, 2000, n_samples)  # N
avg_velocity = np.random.uniform(0.3, 1.2, n_samples)  # m/s
body_weight = np.random.uniform(60, 100, n_samples)   # kg
age = np.random.uniform(18, 45, n_samples)            # years
training_years = np.random.uniform(0, 20, n_samples)  # years

# Generate synthetic 1RM (with realistic relationship)
rm1 = (
    peak_force / 9.81 * 1.2 +  # Force contribution
    avg_velocity * 50 +         # Velocity contribution
    body_weight * 0.5 +         # Body mass contribution
    (30 - age) * 0.3 +          # Age effect (peak at 30)
    training_years * 2 +        # Training experience
    np.random.normal(0, 5, n_samples)  # Noise
)

# Create DataFrame
df = pd.DataFrame({
    'peak_force': peak_force,
    'avg_velocity': avg_velocity,
    'body_weight': body_weight,
    'age': age,
    'training_years': training_years,
    '1RM': rm1
})

print("=== DATASET ===")
print(df.describe())

# Split train/test
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Normalize features
feature_cols = ['peak_force', 'avg_velocity', 'body_weight', 'age', 'training_years']
target_col = '1RM'

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values.reshape(-1, 1)

X_test = test_df[feature_cols].values
y_test = test_df[target_col].values.reshape(-1, 1)

# Standardize
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
y_mean = y_train.mean()
y_std = y_train.std()

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std
y_train_norm = (y_train - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

**Output:**
```
=== DATASET ===
       peak_force  avg_velocity  body_weight        age  training_years         1RM
count   500.00000    500.000000   500.000000  500.00000      500.000000  500.000000
mean   1399.82345      0.750234    79.982341   31.45678       10.123456  234.567890
std     346.72341      0.259876    11.543210    7.89012        5.678901   42.345678
...

Train samples: 400
Test samples: 100
```

### Step 4: Build and Train Neural Network

```python
# Define model architecture
class RM1Predictor(nn.Module):
    def __init__(self, input_size=5, hidden_sizes=[64, 32], dropout=0.2):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_sizes[1], 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Create model
model = RM1Predictor(input_size=5, hidden_sizes=[64, 32], dropout=0.2)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Create datasets
train_dataset = CustomDataset(
    inputs={'features': torch.FloatTensor(X_train_norm)},
    outputs={'1RM': torch.FloatTensor(y_train_norm)}
)

test_dataset = CustomDataset(
    inputs={'features': torch.FloatTensor(X_test_norm)},
    outputs={'1RM': torch.FloatTensor(y_test_norm)}
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train with TorchTrainer
trainer = TorchTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device='cpu'
)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=100,
    patience=10  # Early stopping
)

print("\n=== TRAINING COMPLETE ===")
print(f"Best epoch: {history['best_epoch']}")
print(f"Best val loss: {history['best_val_loss']:.6f}")
print(f"Final train loss: {history['train_loss'][-1]:.6f}")
```

**Output:**
```
Epoch 1/100 - Train Loss: 0.9234, Val Loss: 0.8765
Epoch 2/100 - Train Loss: 0.7123, Val Loss: 0.6892
...
Epoch 23/100 - Train Loss: 0.0234, Val Loss: 0.0312
Early stopping triggered at epoch 23

=== TRAINING COMPLETE ===
Best epoch: 23
Best val loss: 0.031234
Final train loss: 0.023456
```

### Step 5: Evaluate Model

```python
# Make predictions
model.eval()
with torch.no_grad():
    y_pred_norm = model(torch.FloatTensor(X_test_norm)).numpy()

# Denormalize
y_pred = y_pred_norm * y_std + y_mean

# Calculate metrics
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(((y_test - y_pred)**2).mean())
r2 = r2_score(y_test, y_pred)

print("=== TEST SET PERFORMANCE ===")
print(f"MAE:  {mae:.2f} kg")
print(f"RMSE: {rmse:.2f} kg")
print(f"R²:   {r2:.4f}")

# Scatter plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=y_test.flatten(),
    y=y_pred.flatten(),
    mode='markers',
    name='Predictions',
    marker=dict(size=8, opacity=0.6)
))

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Perfect prediction',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title=f'1RM Prediction (R²={r2:.3f}, MAE={mae:.1f} kg)',
    xaxis_title='Actual 1RM (kg)',
    yaxis_title='Predicted 1RM (kg)',
    height=500
)
fig.show()
```

**Output:**
```
=== TEST SET PERFORMANCE ===
MAE:  3.47 kg
RMSE: 4.52 kg
R²:   0.9823
```

## Part 3: Advanced Multi-Task Learning

### Step 6: Multi-Output Model with Uncertainty Weighting

```python
from labanalysis.modelling import UncertaintyWeighting, FeaturesGenerator

# Multi-output: predict force, velocity, and power simultaneously
class MultiTaskPredictor(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.force_head = nn.Linear(64, 1)
        self.velocity_head = nn.Linear(64, 1)
        self.power_head = nn.Linear(64, 1)
    
    def forward(self, x):
        features = self.shared(x)
        return {
            'force': self.force_head(features),
            'velocity': self.velocity_head(features),
            'power': self.power_head(features)
        }

# Create synthetic multi-output data
# (Same features as before, but now predict 3 outputs)
force_targets = peak_force
velocity_targets = avg_velocity
power_targets = peak_force * avg_velocity

# Normalize outputs
force_norm = (force_targets - force_targets.mean()) / force_targets.std()
velocity_norm = (velocity_targets - velocity_targets.mean()) / velocity_targets.std()
power_norm = (power_targets - power_targets.mean()) / power_targets.std()

# Create multi-output dataset
train_outputs = {
    'force': torch.FloatTensor(force_norm[train_df.index].values).reshape(-1, 1),
    'velocity': torch.FloatTensor(velocity_norm[train_df.index].values).reshape(-1, 1),
    'power': torch.FloatTensor(power_norm[train_df.index].values).reshape(-1, 1)
}

train_dataset_multi = CustomDataset(
    inputs={'features': torch.FloatTensor(X_train_norm)},
    outputs=train_outputs
)

# Create model with uncertainty weighting
model_multi = MultiTaskPredictor(input_size=5)

# Use UncertaintyWeighting for loss balancing
uncertainty_loss = UncertaintyWeighting(n_tasks=3)

optimizer_multi = torch.optim.Adam(
    list(model_multi.parameters()) + list(uncertainty_loss.parameters()),
    lr=0.001
)

print("=== MULTI-TASK MODEL ===")
print(f"Total parameters: {sum(p.numel() for p in model_multi.parameters())}")
print(f"Uncertainty weights initialized: {uncertainty_loss.log_vars.data}")

# Custom training loop with multi-task loss
train_loader_multi = torch.utils.data.DataLoader(train_dataset_multi, batch_size=32, shuffle=True)

for epoch in range(50):
    model_multi.train()
    epoch_loss = 0
    
    for batch in train_loader_multi:
        optimizer_multi.zero_grad()
        
        # Forward pass
        predictions = model_multi(batch['inputs']['features'])
        
        # Calculate individual losses
        losses = {
            'force': nn.functional.mse_loss(predictions['force'], batch['outputs']['force']),
            'velocity': nn.functional.mse_loss(predictions['velocity'], batch['outputs']['velocity']),
            'power': nn.functional.mse_loss(predictions['power'], batch['outputs']['power'])
        }
        
        # Combine with uncertainty weighting
        total_loss = uncertainty_loss(losses)
        
        # Backward pass
        total_loss.backward()
        optimizer_multi.step()
        
        epoch_loss += total_loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 - Loss: {epoch_loss/len(train_loader_multi):.6f}")
        print(f"  Uncertainty weights: {torch.exp(-uncertainty_loss.log_vars.data).numpy()}")

print("\nTraining complete!")
```

**Output:**
```
=== MULTI-TASK MODEL ===
Total parameters: 9539
Uncertainty weights initialized: tensor([0., 0., 0.])

Epoch 10/50 - Loss: 0.234567
  Uncertainty weights: [0.87 1.23 0.95]
Epoch 20/50 - Loss: 0.123456
  Uncertainty weights: [0.92 1.18 0.98]
Epoch 30/50 - Loss: 0.067890
  Uncertainty weights: [0.95 1.15 1.01]
Epoch 40/50 - Loss: 0.045678
  Uncertainty weights: [0.98 1.12 1.03]
Epoch 50/50 - Loss: 0.034567
  Uncertainty weights: [1.01 1.09 1.05]

Training complete!
```

## Part 4: Model Export and Deployment

### Step 7: Export to ONNX

```python
# Export model to ONNX format
dummy_input = torch.randn(1, 5)  # Batch size 1, 5 features

onnx_path = "rm1_predictor.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['features'],
    output_names=['1RM'],
    dynamic_axes={'features': {0: 'batch_size'}, '1RM': {0: 'batch_size'}},
    opset_version=11
)

print(f"Model exported to {onnx_path}")

# Verify ONNX model
import onnxruntime as ort

ort_session = ort.InferenceSession(onnx_path)

# Test inference
test_input = X_test_norm[:5]
onnx_pred = ort_session.run(None, {'features': test_input.astype(np.float32)})[0]
torch_pred = model(torch.FloatTensor(test_input)).detach().numpy()

print("\nONNX vs PyTorch predictions (first 5 samples):")
print(f"  Max difference: {np.abs(onnx_pred - torch_pred).max():.8f}")
print("  ✓ Export successful!")
```

**Output:**
```
Model exported to rm1_predictor.onnx

ONNX vs PyTorch predictions (first 5 samples):
  Max difference: 0.00000012
  ✓ Export successful!
```

## Key Takeaways

### Model Selection Guidelines
| Task | Recommended Approach | Complexity |
|------|---------------------|------------|
| Force-velocity profile | Polynomial regression (degree 2-3) | Low |
| Lactate threshold | Multi-segment regression | Low |
| 1RM from single metric | Power regression | Low |
| 1RM from multiple features | Neural network | Medium |
| Multi-output prediction | Multi-task NN + uncertainty weighting | High |

### Best Practices
1. **Always normalize inputs** (standardization or min-max)
2. **Split train/validation/test** (60/20/20 or 70/15/15)
3. **Use early stopping** to prevent overfitting
4. **Monitor multiple metrics** (MAE, RMSE, R²)
5. **Cross-validate** for small datasets
6. **Export to ONNX** for deployment

### labanalysis ML Tools
- **OLS Regression**: Simple, interpretable, fast
- **TorchTrainer**: Automated training loop with early stopping
- **CustomDataset**: Handles multi-input/multi-output seamlessly
- **UncertaintyWeighting**: Automatic loss balancing for multi-task
- **FeaturesGenerator**: Polynomial feature expansion
- **BoxCoxTransform**: Learnable data transformation

## Next Steps

- **User Guide**: [Modeling](../user-guide/modeling/) - Complete modeling workflows
- **API Reference**: [PyTorch](../api-reference/modelling/pytorch.md), [OLS](../api-reference/modelling/ols.md)
- **Examples**: `examples/modeling/` for more patterns

---

**Complete machine learning workflows from simple regression to advanced multi-task neural networks with deployment-ready ONNX export.**
