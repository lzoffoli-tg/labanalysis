# PyTorch Basics

Introduction to using PyTorch modules and utilities in labanalysis for deep learning-based biomechanical analysis.

## Overview

labanalysis provides custom PyTorch modules specifically designed for biomechanical data:

1. **FeaturesGenerator** - Polynomial feature expansion with transformations
2. **BoxCoxTransform** - Learnable Box-Cox transformation layer
3. **SigmoidTransformer** - Parameterized sigmoid transformation
4. **CustomDataset** - Dataset for structured dict-based inputs/outputs
5. **TorchTrainer** - Training loop with early stopping and tracking
6. **Loss Functions** - Custom losses (Pinball, Quantile, Combo)

These modules integrate seamlessly with standard PyTorch workflows while handling biomechanical-specific data structures.

## Quick Reference

```python
import labanalysis as laban
import torch

# Create feature generator
feature_gen = laban.FeaturesGenerator(
    order=2,
    apply_log_transform=True,
    apply_inverse_transform=True,
    include_interactions=True
)

# Create neural network with custom modules
model = torch.nn.Sequential(
    feature_gen,
    torch.nn.Linear(n_features, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1)
)

# Train with TorchTrainer (see torch-trainer.md)
```

## FeaturesGenerator

Automatically generates polynomial features with various transformations.

### Basic Feature Expansion

```python
import labanalysis as laban
import torch

# Create feature generator
feature_gen = laban.FeaturesGenerator(
    order=2,                      # Up to X²
    apply_log_transform=True,     # Add log(X)
    apply_inverse_transform=True, # Add 1/X
    include_interactions=False    # No cross-terms
)

# Input as dictionary
inputs = {
    'velocity': torch.tensor([1.0, 2.0, 3.0, 4.0]),
    'force': torch.tensor([100.0, 200.0, 300.0, 400.0])
}

# Generate features
features = feature_gen(inputs)

print(features.keys())
# Output: dict_keys([
#   'velocity',           # Original
#   'velocity_log',       # log(velocity)
#   'velocity_inv',       # 1/velocity
#   'velocity_invlog',    # 1/log(velocity)
#   'velocity_pow2',      # velocity²
#   'velocity_invpow2',   # velocity⁻²
#   'force',             # Original
#   'force_log',         # log(force)
#   ...
# ])

# Each feature is a tensor
print(features['velocity_pow2'])
# tensor([1., 4., 9., 16.])
```

### With Interactions

Include cross-product terms between features:

```python
feature_gen = laban.FeaturesGenerator(
    order=2,
    include_interactions=True  # Add velocity×force, etc.
)

inputs = {
    'velocity': torch.tensor([1.0, 2.0, 3.0]),
    'force': torch.tensor([100.0, 200.0, 300.0])
}

features = feature_gen(inputs)

# Interaction terms are named with "_x_"
print('velocity_x_force' in features.keys())
# True

print(features['velocity_x_force'])
# tensor([100., 400., 900.])  # velocity * force element-wise
```

### Selective Features

Process only specific input keys:

```python
# Only expand 'velocity', leave 'force' unchanged
feature_gen = laban.FeaturesGenerator(
    order=2,
    input_keys=['velocity']  # Only process velocity
)

inputs = {
    'velocity': torch.tensor([1.0, 2.0, 3.0]),
    'force': torch.tensor([100.0, 200.0, 300.0]),
    'time': torch.tensor([0.0, 1.0, 2.0])
}

features = feature_gen(inputs)

# Only velocity is expanded
print('velocity_pow2' in features.keys())  # True
print('force_pow2' in features.keys())     # False (force not expanded)
print('time' in features.keys())           # True (passed through)
```

### In a Neural Network

```python
import torch.nn as nn

# Network with automatic feature expansion
class BiomechModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        
        # Feature generator (order=2 → ~6-10x more features)
        self.features = laban.FeaturesGenerator(order=2)
        
        # Estimate number of output features (rough estimate)
        n_expanded = n_inputs * 6  # ~6 features per input
        
        # Dense layers
        self.fc1 = nn.Linear(n_expanded, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_outputs)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x is a dict
        x = self.features(x)
        
        # Concatenate all features into single tensor
        x_cat = torch.cat([v.unsqueeze(1) if v.ndim == 1 else v 
                           for v in x.values()], dim=1)
        
        x = self.relu(self.fc1(x_cat))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create and use model
model = BiomechModel(n_inputs=3, n_outputs=1)

inputs = {
    'velocity': torch.randn(32, 1),  # Batch size 32
    'force': torch.randn(32, 1),
    'angle': torch.randn(32, 1)
}

output = model(inputs)
print(output.shape)  # torch.Size([32, 1])
```

## BoxCoxTransform

Learnable Box-Cox transformation for normalizing skewed distributions.

**Transformation**:
- If λ = 0: y = log(x)
- If λ ≠ 0: y = (x^λ - 1) / λ

### Basic Usage

```python
import labanalysis as laban
import torch

# Create Box-Cox layer for 3 features
boxcox = laban.BoxCoxTransform(n_features=3)

# Input data (batch_size=5, n_features=3)
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [1.5, 2.5, 3.5],
    [2.0, 3.0, 4.0],
    [2.5, 3.5, 4.5],
    [3.0, 4.0, 5.0]
])

# Forward transform
y = boxcox(x)

print(y.shape)  # torch.Size([5, 3])

# Initial lambda parameters (all initialized to 1.0)
print(boxcox.lambda_param)
# tensor([1., 1., 1.], requires_grad=True)
```

### Learning Lambda Parameters

Lambda parameters are learned during training:

```python
import torch.optim as optim

# Create model with Box-Cox transformation
model = torch.nn.Sequential(
    laban.BoxCoxTransform(n_features=3),
    torch.nn.Linear(3, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

# Optimizer will update both linear weights AND lambda parameters
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop (simplified)
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, targets)
    loss.backward()
    optimizer.step()

# After training, lambda parameters are optimized
print(model[0].lambda_param)
# tensor([0.52, 1.23, 0.87], requires_grad=True)
# Different lambda for each feature
```

### Inverse Transformation

Reconstruct original values:

```python
# Forward transform
y = boxcox(x)

# Inverse transform
x_reconstructed = boxcox.inverse(y)

print(torch.allclose(x, x_reconstructed, atol=1e-6))
# True (reconstruction is nearly perfect)
```

### Use Case: Normalizing Skewed Data

```python
# Highly skewed force data
force_data = torch.tensor([
    [10.0], [15.0], [25.0], [50.0], [100.0], [200.0], [500.0]
])

# Box-Cox transformation learns optimal lambda to normalize
boxcox = laban.BoxCoxTransform(n_features=1)

# Before training, lambda = 1.0 (identity)
y_before = boxcox(force_data)

# After training with suitable loss, lambda adapts
# (example: lambda → 0.3 for power-law distribution)
```

## SigmoidTransformer

Applies learnable sigmoid transformation: Y = 1 / (1 + exp(-((X - J) @ Q)))

### Basic Usage

```python
import labanalysis as laban
import torch

# Create sigmoid transformer
# Input: 5 features → Output: 3 features
sigmoid_transform = laban.SigmoidTransformer(
    input_dim=5,
    output_dim=3,
    transform_dim=-1  # Apply along last dimension
)

# Input tensor (batch_size=10, features=5)
x = torch.randn(10, 5)

# Transform
y = sigmoid_transform(x)

print(y.shape)  # torch.Size([10, 3])
print(y.min(), y.max())  # Values in [0, 1] due to sigmoid
# tensor(0.1234) tensor(0.9876)

# Learnable parameters
print(sigmoid_transform.J.shape)  # torch.Size([1, 5]) - shift
print(sigmoid_transform.Q.shape)  # torch.Size([5, 3]) - projection
```

## CustomDataset

PyTorch Dataset for dict-based inputs/outputs.

### Creating a Dataset

```python
import labanalysis as laban
import torch

# Input features as dictionary
x = {
    'velocity': torch.randn(100, 1),
    'force': torch.randn(100, 1),
    'angle': torch.randn(100, 1)
}

# Target outputs as dictionary
y = {
    'power': torch.randn(100, 1),
    'efficiency': torch.randn(100, 1)
}

# Create dataset
dataset = laban.CustomDataset(x=x, y=y)

print(len(dataset))  # 100

# Access single sample
x_sample, y_sample = dataset[0]
print(x_sample.keys())  # dict_keys(['velocity', 'force', 'angle'])
print(y_sample.keys())  # dict_keys(['power', 'efficiency'])
```

### With DataLoader

```python
from torch.utils.data import DataLoader

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

# Iterate over batches
for batch_x, batch_y in dataloader:
    print(batch_x['velocity'].shape)  # torch.Size([16, 1])
    print(batch_y['power'].shape)     # torch.Size([16, 1])
    break
```

### Single Tensor Inputs/Outputs

Also supports plain tensors (not just dicts):

```python
# Simple tensors instead of dicts
x_tensor = torch.randn(100, 5)
y_tensor = torch.randn(100, 2)

dataset = laban.CustomDataset(x=x_tensor, y=y_tensor)

x_sample, y_sample = dataset[0]
print(x_sample.shape)  # torch.Size([5])
print(y_sample.shape)  # torch.Size([2])
```

## Loss Functions

### PinballLoss (Quantile Loss)

For quantile regression:

```python
import labanalysis as laban
import torch

# Create pinball loss for 50th percentile (median)
loss_fn = laban.PinballLoss(quantile=0.5)

y_pred = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([1.5, 2.5, 2.0])

loss = loss_fn(y_pred, y_true)
print(loss)
# tensor(0.3333)  # Average pinball loss

# For 90th percentile
loss_90 = laban.PinballLoss(quantile=0.9)
```

### StandardizedMSELoss

MSE loss with automatic standardization:

```python
loss_fn = laban.StandardizedMSELoss()

y_pred = torch.tensor([100.0, 200.0, 300.0])
y_true = torch.tensor([120.0, 210.0, 280.0])

# Automatically standardizes before computing MSE
loss = loss_fn(y_pred, y_true)
```

### QuantilicRangeLoss

Loss for prediction intervals:

```python
# Predict lower (10th) and upper (90th) percentiles
loss_fn = laban.QuantilicRangeLoss(
    lower_quantile=0.1,
    upper_quantile=0.9
)

# y_pred: [batch, 2] where [:, 0] = lower, [:, 1] = upper
y_pred = torch.tensor([[80.0, 120.0], [190.0, 210.0]])
y_true = torch.tensor([100.0, 200.0])

loss = loss_fn(y_pred, y_true)
```

### ComboLoss

Combine multiple losses:

```python
# Combine MSE + Pinball loss
loss_fn = laban.ComboLoss(
    losses=[
        torch.nn.MSELoss(),
        laban.PinballLoss(quantile=0.9)
    ],
    weights=[0.7, 0.3]  # 70% MSE, 30% Pinball
)

loss = loss_fn(y_pred, y_true)
```

## Practical Example: Force-Velocity Model

```python
import labanalysis as laban
import torch
import torch.nn as nn

# Create dataset
velocity = torch.linspace(0.5, 3.0, 100).unsqueeze(1)
force = 1000 / (velocity + 0.5) + torch.randn(100, 1) * 10  # Hill's curve

x = {'velocity': velocity}
y = {'force': force}

dataset = laban.CustomDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Model with feature generation
class ForceVelocityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = laban.FeaturesGenerator(
            order=2,
            input_keys=['velocity']
        )
        self.fc1 = nn.Linear(6, 32)  # ~6 features from order=2
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.features(x)
        x = torch.cat([v.unsqueeze(1) if v.ndim == 1 else v 
                       for v in x.values()], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = ForceVelocityModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
for epoch in range(50):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y['force'])
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.2f}")

# Predict
with torch.no_grad():
    test_vel = {'velocity': torch.tensor([[1.5]])}
    pred_force = model(test_vel)
    print(f"Predicted force at 1.5 m/s: {pred_force.item():.1f} N")
```

## See Also

- [TorchTrainer](torch-trainer.md) - Complete training workflow with TorchTrainer
- [Custom Models](custom-models.md) - Building custom PyTorch models
- [ONNX Deployment](onnx-deployment.md) - Exporting models for production
- [API Reference: PyTorch](../../api-reference/modelling/pytorch.md) - Complete PyTorch API

---

**PyTorch Basics**: Custom PyTorch modules and utilities for biomechanical deep learning in labanalysis.
