# TorchTrainer

Complete guide to training PyTorch models using the TorchTrainer class with early stopping, automatic optimization, and metric tracking.

## Overview

`TorchTrainer` provides a high-level training interface for PyTorch models with:

- **Automatic optimizer management** - No manual optimizer creation needed
- **Early stopping** - Prevents overfitting with patience-based stopping
- **Learning rate decay** - Automatic LR reduction on plateau
- **Validation splitting** - Automatic train/val split
- **Best weight restoration** - Restores best model after training
- **Multi-output support** - Dict-based inputs/outputs
- **Uncertainty weighting** - Automatic task balancing (Kendall & Gal 2018)
- **Progress tracking** - Real-time training metrics

## Quick Reference

```python
import labanalysis as laban
import torch

# Create dataset
dataset = laban.CustomDataset(x=X_dict, y=Y_dict)

# Define model
model = torch.nn.Sequential(
    torch.nn.Linear(n_inputs, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, n_outputs)
)

# Create trainer
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    metrics=laban.MAEMetric(),
    optimizer_kwargs={"lr": 0.001},
    epochs=10000,
    early_stopping_patience=500,
    verbose='minimal'
)

# Train
history = trainer.fit(model, dataset)

# Best model weights are automatically loaded
```

## Basic Usage

### Simple Training Workflow

```python
import labanalysis as laban
import torch
import numpy as np

# Generate synthetic data
x = torch.randn(1000, 5)  # 1000 samples, 5 features
y = torch.randn(1000, 2)  # 1000 samples, 2 outputs

# Create dataset
dataset = laban.CustomDataset(x=x, y=y)

# Define model
model = torch.nn.Sequential(
    torch.nn.Linear(5, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 2)
)

# Create trainer
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    optimizer_kwargs={"lr": 0.001},
    epochs=5000,
    batch_size=64,
    early_stopping_patience=300,
    validation_split=0.2,
    verbose='minimal'
)

# Train
history = trainer.fit(model, dataset)

# Training output (updates in place):
# Epoch 1000 | train=0.1234 | val=0.1456 | lr=1.00e-03 | gap=250 | time=12s

print(f"Best validation loss: {history['validation_loss'].min():.6f}")
```

### Dict-Based Inputs/Outputs

```python
# Multi-feature inputs as dictionary
X = {
    'velocity': torch.randn(500, 1),
    'force': torch.randn(500, 1),
    'angle': torch.randn(500, 1)
}

# Multi-output targets as dictionary
Y = {
    'power': torch.randn(500, 1),
    'efficiency': torch.randn(500, 1)
}

dataset = laban.CustomDataset(x=X, y=Y)

# Model must handle dict inputs
class MultiOutputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 32)
        self.fc2 = torch.nn.Linear(32, 2)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x_dict):
        # Concatenate dict values
        x = torch.cat([x_dict['velocity'], 
                       x_dict['force'], 
                       x_dict['angle']], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Return as dict
        return {
            'power': x[:, 0:1],
            'efficiency': x[:, 1:2]
        }

model = MultiOutputModel()

trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    optimizer_kwargs={"lr": 0.001}
)

history = trainer.fit(model, dataset)
```

## Early Stopping

### Basic Early Stopping

```python
# Stop after 500 epochs without improvement
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    early_stopping_patience=500,
    early_stopping_threshold=1e-5,  # Minimum improvement
    restore_best_weights=True       # Restore best model after stopping
)

history = trainer.fit(model, dataset)

# Training stops automatically when validation loss plateaus
# Best weights are restored
```

### Learning Rate Schedule

Pass multiple learning rates to automatically reduce LR on plateau:

```python
# Start at 0.01, reduce to 0.001, then 0.0001
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    optimizer_kwargs={"lr": [0.01, 0.001, 0.0001]},
    early_stopping_patience=300
)

history = trainer.fit(model, dataset)

# Training process:
# 1. Start with lr=0.01
# 2. After 300 epochs without improvement, reduce to lr=0.001
# 3. Restore best weights and continue
# 4. After another 300 epochs without improvement, reduce to lr=0.0001
# 5. After final 300 epochs without improvement, stop
```

## Validation and Metrics

### Validation Split

```python
# Use 30% of data for validation
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    validation_split=0.3  # 30% validation, 70% training
)

history = trainer.fit(model, dataset)

# Access validation loss
val_losses = history['validation_loss']
train_losses = history['training_loss']

import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training')
plt.plot(val_losses, label='Validation')
plt.legend()
plt.show()
```

### Adding Metrics

```python
# Single metric
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    metrics=laban.MAEMetric()  # Track MAE
)

history = trainer.fit(model, dataset)

print(history.keys())
# ['epoch', 'training_loss', 'validation_loss', 
#  'training_mae', 'validation_mae', 'learning_rate']

# Multiple metrics
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    metrics=[
        laban.MAEMetric(),
        lambda y_pred, y_true: torch.sqrt(torch.nn.functional.mse_loss(y_pred, y_true))  # RMSE
    ]
)
```

## Advanced Features

### Uncertainty Weighting

Automatically balance multi-output losses using learnable task uncertainties (Kendall & Gal 2018):

```python
# Multi-output model
Y = {
    'power': torch.randn(500, 1),
    'efficiency': torch.randn(500, 1),
    'speed': torch.randn(500, 1)
}

dataset = laban.CustomDataset(x=X, y=Y)

# Enable uncertainty weighting
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    use_uncertainty_weighting=True  # Learns task weights automatically
)

history = trainer.fit(model, dataset)

# Uncertainty parameters are learned during training
# Tasks with higher noise get lower weight
# Tasks with lower noise get higher weight
```

### Custom Loss Functions

```python
# Use custom loss
trainer = laban.TorchTrainer(
    loss=laban.PinballLoss(quantile=0.9),  # Quantile regression
    optimizer_kwargs={"lr": 0.001}
)

# Combine losses
combo_loss = laban.ComboLoss(
    torch.nn.MSELoss(),
    laban.PinballLoss(quantile=0.5)
)

trainer = laban.TorchTrainer(loss=combo_loss)
```

### Batch Size Tuning

```python
# Small batch (more updates, noisier gradients)
trainer_small = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    batch_size=32  # Good for small datasets
)

# Large batch (fewer updates, smoother gradients)
trainer_large = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    batch_size=512  # Good for large datasets, better CPU vectorization
)

# Full batch (single update per epoch)
trainer_full = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    batch_size=None  # Use all data in one batch
)
```

### Verbosity Levels

```python
# Minimal output (single line, updates in place)
trainer = laban.TorchTrainer(loss=torch.nn.MSELoss(), verbose='minimal')
# Output: Epoch 1000 | train=0.1234 | val=0.1456 | lr=1.00e-03 | gap=250 | time=12s

# Full output (detailed multi-line)
trainer = laban.TorchTrainer(loss=torch.nn.MSELoss(), verbose='full')
# Output:
# ================================================================================
# Epoch 1000 | LR: 1.00e-03 | Best: 0.145623 | No Improve: 50 | Gap: 250 | Time: 12s
# --------------------------------------------------------------------------------
# Training Loss:   0.123456
# Validation Loss: 0.145623
# ================================================================================

# Silent
trainer = laban.TorchTrainer(loss=torch.nn.MSELoss(), verbose='off')
# No output during training
```

### Optimizer Configuration

```python
# Adam optimizer (default is AdamW)
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    optimizer_class=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001, "weight_decay": 1e-5}
)

# SGD with momentum
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    optimizer_class=torch.optim.SGD,
    optimizer_kwargs={"lr": 0.01, "momentum": 0.9}
)

# AdamW (recommended, better generalization than Adam)
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    optimizer_class=torch.optim.AdamW,  # Default
    optimizer_kwargs={"lr": 0.001, "weight_decay": 0.01}
)
```

## Performance Optimization

### CPU Optimization

```python
# Optimal batch size for CPU (256 recommended)
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    batch_size=256,  # Best CPU vectorization
    num_workers=0    # Disable multiprocessing (Windows)
)

# Linux/Mac with multiprocessing
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    batch_size=256,
    num_workers=4  # Auto-selected if None
)
```

### Torch Compile (PyTorch 2.0+)

```python
# Enable torch.compile() for ~50-100% speedup
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    use_torch_compile=True  # Default: True
)

# Disable if compilation fails
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    use_torch_compile=False
)
```

See [CPU Optimization Guide](../../advanced/CPU_OPTIMIZATION_GUIDE.md) for detailed performance tuning.

## Practical Examples

### Force-Velocity Modeling

```python
import labanalysis as laban
import torch

# Load data
velocity = torch.tensor([[1.0], [1.5], [2.0], [2.5], [3.0]])
force = torch.tensor([[850.0], [650.0], [500.0], [400.0], [330.0]])

dataset = laban.CustomDataset(x=velocity, y=force)

# Model with feature generation
class HillModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = laban.FeaturesGenerator(order=2)
        self.fc1 = torch.nn.Linear(6, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = {'velocity': x}
        x = self.features(x)
        x = torch.cat([v.unsqueeze(1) if v.ndim == 1 else v 
                       for v in x.values()], dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = HillModel()

# Train
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    metrics=laban.MAEMetric(),
    optimizer_kwargs={"lr": [0.01, 0.001, 0.0001]},
    early_stopping_patience=500,
    batch_size=None,  # Full batch for small dataset
    verbose='minimal'
)

history = trainer.fit(model, dataset)

# Predict
with torch.no_grad():
    v_test = torch.tensor([[1.2], [1.8], [2.3]])
    f_pred = model(v_test)
    print(f"Predicted forces: {f_pred.squeeze().numpy()}")
```

### Multi-Output Jump Analysis

```python
# Inputs: body mass, height, age
X = {
    'mass': torch.randn(200, 1) * 10 + 75,   # kg
    'height': torch.randn(200, 1) * 0.1 + 1.75,  # m
    'age': torch.randn(200, 1) * 10 + 25    # years
}

# Outputs: jump height, peak power
Y = {
    'jump_height': torch.randn(200, 1) * 0.1 + 0.4,  # m
    'peak_power': torch.randn(200, 1) * 500 + 2500   # W
}

dataset = laban.CustomDataset(x=X, y=Y)

# Model
class JumpPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc_height = torch.nn.Linear(32, 1)
        self.fc_power = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x_cat = torch.cat([x['mass'], x['height'], x['age']], dim=1)
        x = self.relu(self.fc1(x_cat))
        x = self.relu(self.fc2(x))
        return {
            'jump_height': self.fc_height(x),
            'peak_power': self.fc_power(x)
        }

model = JumpPredictor()

# Train with uncertainty weighting
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    use_uncertainty_weighting=True,  # Balance jump_height vs peak_power
    optimizer_kwargs={"lr": 0.001},
    early_stopping_patience=1000,
    validation_split=0.25,
    verbose='full'
)

history = trainer.fit(model, dataset)

# Predict new athlete
new_athlete = {
    'mass': torch.tensor([[80.0]]),
    'height': torch.tensor([[1.85]]),
    'age': torch.tensor([[28.0]])
}

with torch.no_grad():
    pred = model(new_athlete)
    print(f"Predicted jump height: {pred['jump_height'].item():.2f} m")
    print(f"Predicted peak power: {pred['peak_power'].item():.0f} W")
```

## Accessing Training History

```python
# Fit model
history = trainer.fit(model, dataset)

# history is a dict with training metrics
print(history.keys())
# ['epoch', 'training_loss', 'validation_loss', 'learning_rate', ...]

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(history)

print(df.head())
#    epoch  training_loss  validation_loss  learning_rate
# 0      0       0.523451         0.534212      0.001000
# 1      1       0.412356         0.445123      0.001000
# 2      2       0.356789         0.398765      0.001000

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['training_loss'], label='Train')
plt.plot(df['epoch'], df['validation_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Export to CSV
df.to_csv('training_history.csv', index=False)
```

## Troubleshooting

### Issue: Training too slow

```python
# Increase batch size (better CPU vectorization)
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    batch_size=512  # Increase from default 256
)

# Enable torch.compile (if PyTorch 2.0+)
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    use_torch_compile=True
)

# See CPU_OPTIMIZATION_GUIDE.md for 2-3x speedup
```

### Issue: Validation loss not improving

```python
# Increase patience
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    early_stopping_patience=2000  # Wait longer
)

# Use learning rate schedule
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    optimizer_kwargs={"lr": [0.01, 0.001, 0.0001]}
)

# Increase validation split (more stable validation)
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    validation_split=0.3  # 30% validation instead of 20%
)
```

### Issue: Loss becomes NaN

```python
# Reduce learning rate
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    optimizer_kwargs={"lr": 0.0001}  # Lower LR
)

# Enable gradient clipping
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    gradient_clip_val=1.0  # Clip gradients to max norm 1.0
)

# Check for numerical issues in data
# Normalize inputs to [0, 1] or standardize
```

### Issue: Model not training (loss constant)

```python
# Enable debug mode to check gradients
trainer = laban.TorchTrainer(
    loss=torch.nn.MSELoss(),
    debug=True  # Prints gradient information
)

# Check model architecture
# Ensure activation functions are present
# Verify input/output shapes match
```

## See Also

- [PyTorch Basics](pytorch-basics.md) - PyTorch modules and features
- [Custom Models](custom-models.md) - Building custom architectures
- [ONNX Deployment](onnx-deployment.md) - Exporting trained models
- [CPU Optimization Guide](../../advanced/CPU_OPTIMIZATION_GUIDE.md) - 2-3x speedup tips
- [API Reference: TorchTrainer](../../api/modelling/pytorch.md#torchtrainer) - Complete API

---

**TorchTrainer**: High-level PyTorch training interface with automatic optimization, early stopping, and metric tracking.
