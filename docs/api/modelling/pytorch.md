# labanalysis.modelling.pytorch

PyTorch-based deep learning utilities for biomechanical modeling.

**Source**: `src/labanalysis/modelling/pytorch/`

## Overview

PyTorch utilities for training neural networks on biomechanical data:

**Training Utilities:**
- **TorchTrainer**: Complete training loop with early stopping, learning rate scheduling, and metric tracking
- **CustomDataset**: Dataset class for structured input/output dictionaries
- **TrainingLogger**: Training progress logging

**Custom Modules:**
- **FeaturesGenerator**: Polynomial feature expansion with interactions
- **BoxCoxTransform**: Learnable Box-Cox transformation layer
- **SigmoidTransformer**: Sigmoid-based feature transformation
- **PCA**: Principal component analysis layer
- **Lasso**: L1-regularized linear layer

**Loss Functions:**
- **PinballLoss**: Quantile regression loss
- **StandardizedMSELoss**: Normalized MSE loss
- **QuantilicRangeLoss**: Multi-quantile range loss
- **ComboLoss**: Combined loss functions
- **UncertaintyWeighting**: Multi-task loss balancing

**Metrics:**
- **MAEMetric**: Mean absolute error metric

## Training Utilities

### TorchTrainer

Complete training loop with validation and early stopping.

```python
class TorchTrainer:
    """
    PyTorch model trainer with early stopping and learning rate scheduling.
    
    Provides a complete training loop with:
    - Automatic train/validation split
    - Early stopping based on validation loss
    - Learning rate decay on plateau
    - Metric tracking (train/val loss per epoch)
    - Progress logging
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to train
    loss_fn : Callable
        Loss function (e.g., torch.nn.MSELoss())
    optimizer : torch.optim.Optimizer
        Optimizer (e.g., torch.optim.Adam)
    device : str or torch.device, optional
        Device for training ('cpu', 'cuda', 'mps')
        Default: 'cpu'
    
    Methods
    -------
    fit(train_loader, val_loader, epochs, patience)
        Train model with early stopping
    evaluate(data_loader)
        Evaluate model on data
    
    Examples
    --------
    >>> import torch
    >>> from labanalysis.modelling import TorchTrainer, CustomDataset
    >>> 
    >>> # Define model
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(10, 64),
    ...     torch.nn.ReLU(),
    ...     torch.nn.Linear(64, 1)
    ... )
    >>> 
    >>> # Setup trainer
    >>> trainer = TorchTrainer(
    ...     model=model,
    ...     loss_fn=torch.nn.MSELoss(),
    ...     optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    ...     device='cpu'
    ... )
    >>> 
    >>> # Train
    >>> history = trainer.fit(
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     epochs=100,
    ...     patience=10
    ... )
    >>> 
    >>> # Plot training curves
    >>> import plotly.express as px
    >>> fig = px.line(history, x='epoch', y=['train_loss', 'val_loss'])
    >>> fig.show()
    """
```

**Key Features:**
- **Early Stopping**: Stops training when validation loss stops improving
- **Learning Rate Scheduling**: Reduces learning rate on plateau
- **Automatic Checkpointing**: Saves best model state
- **Progress Logging**: Real-time training progress

---

### CustomDataset

Dataset class for structured input/output dictionaries.

```python
class CustomDataset(torch.utils.data.Dataset):
    """
    Dataset for structured input/output tensors.
    
    Supports dictionaries of tensors as inputs/outputs, enabling
    multi-input/multi-output models.
    
    Parameters
    ----------
    x : dict of str to torch.Tensor or torch.Tensor
        Input tensors (all must have same first dimension)
    y : dict of str to torch.Tensor or torch.Tensor
        Target tensors (all must have same first dimension)
    
    Examples
    --------
    >>> import torch
    >>> from labanalysis.modelling import CustomDataset
    >>> 
    >>> # Multi-input dataset
    >>> x = {
    ...     'force': torch.randn(100, 1),
    ...     'velocity': torch.randn(100, 1),
    ...     'power': torch.randn(100, 1)
    ... }
    >>> y = {
    ...     'performance': torch.randn(100, 1)
    ... }
    >>> 
    >>> dataset = CustomDataset(x, y)
    >>> loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    >>> 
    >>> # Access sample
    >>> x_sample, y_sample = dataset[0]
    >>> print(x_sample.keys())  # dict_keys(['force', 'velocity', 'power'])
    """
```

---

## Custom Modules

### FeaturesGenerator

Polynomial feature expansion with transformations.

```python
class FeaturesGenerator(torch.nn.Module):
    """
    Generate polynomial features with transformations and interactions.
    
    Applies multiple transformations to input features:
    - Power transformations (x², x³, ...)
    - Logarithmic transformations (log(x + 1))
    - Inverse transformations (1/x)
    - Interaction terms (x₁ × x₂, x₁ × x₂ × x₃, ...)
    
    Parameters
    ----------
    order : int, optional
        Maximum polynomial order
        Default: 2
    apply_log_transform : bool, optional
        Apply logarithmic transformations
        Default: True
    apply_inverse_transform : bool, optional
        Apply inverse transformations
        Default: True
    include_interactions : bool, optional
        Generate interaction terms
        Default: True
    input_keys : list of str, optional
        Specific keys to process (None = all)
        Default: None
    
    Examples
    --------
    >>> import torch
    >>> from labanalysis.modelling import FeaturesGenerator
    >>> 
    >>> # Create feature generator
    >>> feat_gen = FeaturesGenerator(
    ...     order=2,
    ...     apply_log_transform=True,
    ...     apply_inverse_transform=True,
    ...     include_interactions=True
    ... )
    >>> 
    >>> # Input features
    >>> inputs = {
    ...     'x1': torch.tensor([1.0, 2.0, 3.0]),
    ...     'x2': torch.tensor([4.0, 5.0, 6.0])
    ... }
    >>> 
    >>> # Generate features
    >>> outputs = feat_gen(inputs)
    >>> print(outputs.keys())
    >>> # Keys: x1, x2, x1_pow2, x2_pow2, x1_log, x2_log, 
    >>> #       x1_inv, x2_inv, x1_x_x2, ...
    """
```

**Generated Features:**
- Original: `x`
- Powers: `x_pow2`, `x_pow3`, ..., `x_pow{order}`
- Log: `x_log` (= log(x + 1))
- Inverse: `x_inv` (= 1/x), `x_invpow2` (= 1/x²), ...
- Log-Inverse: `x_invlog` (= 1/log(x + 1))
- Interactions: `x1_x_x2`, `x1_x_x2_x_x3`, ...

**Use Cases:**
- Feature engineering for biomechanical models
- Capturing non-linear relationships
- Augmenting input feature space

---

### BoxCoxTransform

Learnable Box-Cox transformation layer.

```python
class BoxCoxTransform(torch.nn.Module):
    """
    Learnable Box-Cox transformation.
    
    Applies parametric Box-Cox transformation with learnable λ:
    - If λ = 0: y = log(x)
    - If λ ≠ 0: y = (x^λ - 1) / λ
    
    Lambda parameters are learned during training to optimize
    for the specific task.
    
    Parameters
    ----------
    n_features : int
        Number of input features (each gets own λ)
    
    Attributes
    ----------
    lambda_param : torch.nn.Parameter
        Learnable lambda parameters (shape: n_features)
    
    Methods
    -------
    forward(x)
        Apply Box-Cox transformation
    inverse(y)
        Apply inverse transformation
    
    Examples
    --------
    >>> import torch
    >>> from labanalysis.modelling import BoxCoxTransform
    >>> 
    >>> # Create layer
    >>> boxcox = BoxCoxTransform(n_features=3)
    >>> 
    >>> # Apply transformation
    >>> x = torch.randn(100, 3).abs() + 1  # Ensure positive
    >>> y = boxcox(x)
    >>> 
    >>> # Inverse transformation
    >>> x_reconstructed = boxcox.inverse(y)
    >>> 
    >>> # Check learned lambdas
    >>> print(f"Learned lambdas: {boxcox.lambda_param}")
    """
```

**Use Cases:**
- Normalizing skewed distributions
- Variance stabilization
- Improving model convergence

---

### UncertaintyWeighting

Multi-task loss balancing via learned uncertainty.

```python
class UncertaintyWeighting(torch.nn.Module):
    """
    Task uncertainty weighting for multi-output models.
    
    Implements method from Kendall & Gal (CVPR 2018).
    Learns optimal weights for each task based on uncertainty:
    
    L = Σᵢ (exp(-sᵢ) × Lᵢ + sᵢ)
    
    where sᵢ = log(σᵢ²) is the learned log-variance.
    
    Parameters
    ----------
    output_keys : list of str
        Names of outputs for loss weighting
    
    Attributes
    ----------
    log_vars : torch.nn.Parameter
        Learnable log-variances (one per output)
    
    Examples
    --------
    >>> import torch
    >>> from labanalysis.modelling import UncertaintyWeighting
    >>> 
    >>> # Multi-task model
    >>> output_keys = ['force', 'velocity', 'power']
    >>> uncertainty_weighting = UncertaintyWeighting(output_keys)
    >>> 
    >>> # Compute weighted loss
    >>> losses = {
    ...     'force': torch.tensor(0.5),
    ...     'velocity': torch.tensor(0.3),
    ...     'power': torch.tensor(0.8)
    ... }
    >>> 
    >>> weighted_loss = uncertainty_weighting(losses)
    >>> 
    >>> # Check learned weights
    >>> weights = torch.exp(-uncertainty_weighting.log_vars)
    >>> print(f"Task weights: {weights}")
    """
```

**Use Cases:**
- Multi-output regression (e.g., predicting force, velocity, power simultaneously)
- Balancing tasks with different scales
- Automatic loss weighting without manual tuning

---

## Loss Functions

### PinballLoss

Quantile regression loss.

```python
class PinballLoss(torch.nn.Module):
    """
    Pinball loss for quantile regression.
    
    Loss that penalizes under-prediction and over-prediction
    asymmetrically based on quantile τ.
    
    Parameters
    ----------
    quantile : float
        Target quantile (0 < τ < 1)
        τ = 0.5: median (symmetric)
        τ = 0.9: 90th percentile (penalize under-prediction more)
    
    Examples
    --------
    >>> import torch
    >>> from labanalysis.modelling import PinballLoss
    >>> 
    >>> # Predict 90th percentile
    >>> loss_fn = PinballLoss(quantile=0.9)
    >>> 
    >>> y_true = torch.tensor([10.0, 20.0, 30.0])
    >>> y_pred = torch.tensor([9.0, 21.0, 28.0])
    >>> 
    >>> loss = loss_fn(y_pred, y_true)
    >>> print(f"Pinball loss: {loss.item():.3f}")
    """
```

**Use Cases:**
- Predicting performance ranges
- Uncertainty quantification
- Risk-aware predictions

---

### QuantilicRangeLoss

Multi-quantile range prediction loss.

```python
class QuantilicRangeLoss(torch.nn.Module):
    """
    Loss for predicting quantile ranges.
    
    Combines pinball losses for multiple quantiles
    to predict prediction intervals.
    
    Parameters
    ----------
    quantiles : list of float
        Target quantiles (e.g., [0.1, 0.5, 0.9])
    
    Examples
    --------
    >>> from labanalysis.modelling import QuantilicRangeLoss
    >>> 
    >>> # Predict 80% prediction interval (10th-90th percentile)
    >>> loss_fn = QuantilicRangeLoss(quantiles=[0.1, 0.5, 0.9])
    """
```

**Use Cases:**
- Prediction intervals for performance metrics
- Uncertainty estimation
- Confidence bands

---

## Complete Example Workflow

### Multi-Output Biomechanical Model

```python
import torch
import torch.nn as nn
from labanalysis.modelling import (
    TorchTrainer,
    CustomDataset,
    FeaturesGenerator,
    UncertaintyWeighting
)
import pandas as pd
import numpy as np

# 1. Prepare data (force-velocity-power relationship)
np.random.seed(42)
n_samples = 1000

force = np.random.uniform(0, 2000, n_samples)
velocity = 3.0 - 0.0015 * force + np.random.normal(0, 0.1, n_samples)
power = force * velocity / 1000

# Convert to tensors
X = {'force': torch.tensor(force, dtype=torch.float32).reshape(-1, 1)}
y = {
    'velocity': torch.tensor(velocity, dtype=torch.float32).reshape(-1, 1),
    'power': torch.tensor(power, dtype=torch.float32).reshape(-1, 1)
}

# 2. Create dataset and loader
dataset = CustomDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# 3. Define model with feature generation
class ForceVelocityPowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_gen = FeaturesGenerator(
            order=2,
            apply_log_transform=False,
            apply_inverse_transform=True,
            include_interactions=False
        )
        
        # After feature generation, we have: force, force_pow2, force_inv, force_invpow2
        self.shared = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Separate heads for each output
        self.velocity_head = nn.Linear(32, 1)
        self.power_head = nn.Linear(32, 1)
    
    def forward(self, x):
        # Generate features
        features = self.feat_gen(x)
        
        # Stack features for linear layer
        feat_tensor = torch.stack([
            features['force'],
            features['force_pow2'],
            features['force_inv'],
            features['force_invpow2']
        ], dim=1)
        
        # Shared layers
        shared_out = self.shared(feat_tensor)
        
        # Output heads
        return {
            'velocity': self.velocity_head(shared_out),
            'power': self.power_head(shared_out)
        }

model = ForceVelocityPowerModel()

# 4. Setup multi-task loss with uncertainty weighting
uncertainty_weighting = UncertaintyWeighting(output_keys=['velocity', 'power'])

def multi_task_loss(y_pred, y_true):
    losses = {
        'velocity': nn.functional.mse_loss(y_pred['velocity'], y_true['velocity']),
        'power': nn.functional.mse_loss(y_pred['power'], y_true['power'])
    }
    return uncertainty_weighting(losses)

# 5. Setup trainer
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(uncertainty_weighting.parameters()),
    lr=0.001
)

trainer = TorchTrainer(
    model=model,
    loss_fn=multi_task_loss,
    optimizer=optimizer,
    device='cpu'
)

# 6. Train
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    patience=15
)

# 7. Evaluate
model.eval()
with torch.no_grad():
    test_force = torch.linspace(0, 2000, 100).reshape(-1, 1)
    predictions = model({'force': test_force})
    
    pred_velocity = predictions['velocity'].numpy()
    pred_power = predictions['power'].numpy()

# 8. Plot results
import plotly.graph_objects as go

fig = go.Figure()
fig.add_scatter(x=test_force.flatten(), y=pred_velocity.flatten(), name='Velocity', yaxis='y1')
fig.add_scatter(x=test_force.flatten(), y=pred_power.flatten(), name='Power', yaxis='y2')
fig.update_layout(
    title='Force-Velocity-Power Model',
    xaxis_title='Force (N)',
    yaxis=dict(title='Velocity (m/s)'),
    yaxis2=dict(title='Power (kW)', overlaying='y', side='right')
)
fig.show()

# Check learned task weights
weights = torch.exp(-uncertainty_weighting.log_vars)
print(f"Learned task weights: velocity={weights[0]:.3f}, power={weights[1]:.3f}")
```

---

## Advanced Features

### Custom Loss Functions

```python
import torch
import torch.nn as nn

class BiomechanicalConstraintLoss(nn.Module):
    """
    Custom loss with physical constraints.
    
    Combines MSE loss with penalty for violating
    biomechanical constraints (e.g., force-velocity relationship).
    """
    
    def __init__(self, constraint_weight=0.1):
        super().__init__()
        self.constraint_weight = constraint_weight
    
    def forward(self, y_pred, y_true, force):
        # Data fitting loss
        mse = nn.functional.mse_loss(y_pred, y_true)
        
        # Constraint: velocity decreases with force
        dv_df = torch.gradient(y_pred.flatten())[0] / torch.gradient(force.flatten())[0]
        constraint_violation = torch.relu(dv_df)  # Penalize positive slopes
        
        return mse + self.constraint_weight * constraint_violation.mean()
```

---

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size or use gradient accumulation
```python
# Smaller batch size
train_loader = DataLoader(dataset, batch_size=16)  # Instead of 32

# Or gradient accumulation
for i, (x, y) in enumerate(train_loader):
    loss = loss_fn(model(x), y)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue: Model not converging

**Solution**: Check learning rate, add normalization, use scheduler
```python
# Lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Add batch normalization
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Use learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

---

## See Also

- [OLS Models](ols.md) - Traditional regression models
- [Equations](../equations/) - Biomechanical equations

---

**PyTorch utilities for deep learning on biomechanical data with custom modules and loss functions.**
