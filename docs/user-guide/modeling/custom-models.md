# Custom Models

Guide to building custom PyTorch models for biomechanical analysis using labanalysis modules.

## Overview

labanalysis provides building blocks for creating custom neural network architectures:

- **FeaturesGenerator** - Automatic polynomial feature expansion
- **BoxCoxTransform** - Learnable normalization layer
- **SigmoidTransformer** - Parameterized sigmoid transformation
- **Standard PyTorch layers** - Linear, Conv, RNN, Transformer, etc.

Combine these to build domain-specific models for force-velocity curves, jump prediction, gait analysis, and more.

## Quick Reference

```python
import labanalysis as laban
import torch
import torch.nn as nn

class CustomBiomechModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        
        # Add custom labanalysis modules
        self.features = laban.FeaturesGenerator(order=2)
        self.boxcox = laban.BoxCoxTransform(n_features=n_inputs)
        
        # Standard PyTorch layers
        self.fc1 = nn.Linear(n_expanded_features, 128)
        self.fc2 = nn.Linear(128, n_outputs)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.boxcox(x)
        x = self.features(x)
        # ... rest of forward pass
        return x
```

## Basic Custom Model

### Simple Feedforward Network

```python
import labanalysis as laban
import torch
import torch.nn as nn

class BiomechRegressor(nn.Module):
    """Simple regression model for biomechanical data."""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Regularization
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, input_dim)
        return self.network(x)

# Create model
model = BiomechRegressor(input_dim=5, output_dim=2, hidden_dims=[64, 32, 16])

# Test forward pass
x = torch.randn(10, 5)  # Batch of 10 samples
y = model(x)
print(y.shape)  # torch.Size([10, 2])
```

### With Feature Generation

```python
class EnhancedBiomechModel(nn.Module):
    """Model with automatic polynomial feature expansion."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Feature generator (order 2 → ~6x features)
        self.features = laban.FeaturesGenerator(
            order=2,
            apply_log_transform=True,
            apply_inverse_transform=True,
            include_interactions=True
        )
        
        # Estimate expanded feature count
        # For N inputs: N original + N log + N inv + N invlog + N pow2 + N invpow2
        # + interactions ≈ 6N + N*(N-1)/2 for order=2
        expanded_dim = input_dim * 6 + input_dim * (input_dim - 1) // 2
        
        # Dense layers
        self.fc1 = nn.Linear(expanded_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x can be dict or tensor
        if isinstance(x, torch.Tensor):
            # Convert to dict for FeaturesGenerator
            x = {'input': x}
        
        # Generate features
        x = self.features(x)
        
        # Concatenate all features
        x = torch.cat([v.unsqueeze(1) if v.ndim == 1 else v 
                       for v in x.values()], dim=1)
        
        # Forward through network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Usage
model = EnhancedBiomechModel(input_dim=3, output_dim=1)

# Single tensor input
x_tensor = torch.randn(32, 3)
y = model(x_tensor)

# Dict input
x_dict = {'velocity': torch.randn(32, 1), 
          'force': torch.randn(32, 1),
          'angle': torch.randn(32, 1)}
y = model(x_dict)
```

## Advanced Architectures

### Multi-Output Model

```python
class MultiOutputModel(nn.Module):
    """Model with separate heads for different outputs."""
    
    def __init__(self, input_dim):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Output-specific heads
        self.head_power = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.head_efficiency = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.head_speed = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Extract shared features
        features = self.shared(x)
        
        # Compute each output
        return {
            'power': self.head_power(features),
            'efficiency': self.head_efficiency(features),
            'speed': self.head_speed(features)
        }

# Usage
model = MultiOutputModel(input_dim=5)

x = torch.randn(16, 5)
outputs = model(x)

print(outputs['power'].shape)       # torch.Size([16, 1])
print(outputs['efficiency'].shape)  # torch.Size([16, 1])
print(outputs['speed'].shape)       # torch.Size([16, 1])
```

### With Learnable Normalization

```python
class NormalizedModel(nn.Module):
    """Model with learnable Box-Cox normalization."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Learnable normalization (one lambda per feature)
        self.normalize = laban.BoxCoxTransform(n_features=input_dim)
        
        # Network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        # Apply learnable normalization
        x_norm = self.normalize(x)
        
        # Forward through network
        return self.network(x_norm)
    
    def inverse_normalize(self, x):
        """Transform data back to original scale."""
        return self.normalize.inverse(x)

# Usage
model = NormalizedModel(input_dim=3, output_dim=2)

# Lambda parameters are learned during training
print("Initial lambdas:", model.normalize.lambda_param)
# tensor([1., 1., 1.], requires_grad=True)

# After training, lambdas adapt to data distribution
# model.normalize.lambda_param might be: [0.5, 1.2, 0.3]
```

### Residual Connections

```python
class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class ResNetRegressor(nn.Module):
    """Regressor with residual connections."""
    
    def __init__(self, input_dim, output_dim, n_blocks=3):
        super().__init__()
        
        hidden_dim = 64
        
        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        
        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)
        
        return self.output_layer(x)

# Usage
model = ResNetRegressor(input_dim=10, output_dim=1, n_blocks=5)
```

## Domain-Specific Models

### Force-Velocity Model

```python
class HillModel(nn.Module):
    """Neural network approximation of Hill's force-velocity curve."""
    
    def __init__(self):
        super().__init__()
        
        # Feature generation for velocity
        self.features = laban.FeaturesGenerator(
            order=3,  # Cubic terms
            input_keys=['velocity']  # Only expand velocity
        )
        
        # Estimate ~10 features from velocity expansion
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # Force must be positive
        )
    
    def forward(self, x):
        # x: {'velocity': tensor}
        features = self.features(x)
        
        # Concatenate features
        x_cat = torch.cat([v.unsqueeze(1) if v.ndim == 1 else v 
                           for v in features.values()], dim=1)
        
        # Predict force
        return self.network(x_cat)

# Usage
model = HillModel()

velocity = {'velocity': torch.tensor([[1.0], [1.5], [2.0]])}
force = model(velocity)
print(force)
# tensor([[850.2],
#         [650.4],
#         [500.1]], grad_fn=...)
```

### Jump Height Predictor

```python
class JumpPredictor(nn.Module):
    """Predict jump height from anthropometric data."""
    
    def __init__(self):
        super().__init__()
        
        # Normalize inputs (height, weight, age vary on different scales)
        self.normalize = laban.BoxCoxTransform(n_features=3)
        
        # Network
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: {'height': ..., 'weight': ..., 'age': ...}
        
        # Concatenate inputs
        x_cat = torch.cat([x['height'], x['weight'], x['age']], dim=1)
        
        # Normalize
        x_norm = self.normalize(x_cat)
        
        # Forward
        x = self.relu(self.fc1(x_norm))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Scale to reasonable jump height range [0.2, 0.8] m
        return 0.2 + 0.6 * self.sigmoid(x)

# Usage
model = JumpPredictor()

athlete = {
    'height': torch.tensor([[1.85]]),  # meters
    'weight': torch.tensor([[80.0]]),  # kg
    'age': torch.tensor([[25.0]])      # years
}

jump_height = model(athlete)
print(f"Predicted jump height: {jump_height.item():.2f} m")
```

### Gait Phase Classifier

```python
class GaitPhaseClassifier(nn.Module):
    """Classify gait phase from kinematic features."""
    
    def __init__(self, n_features):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4)  # 4 phases: heel strike, mid-stance, toe-off, swing
        )
    
    def forward(self, x):
        logits = self.network(x)
        return logits  # Use with CrossEntropyLoss
    
    def predict_phase(self, x):
        """Get phase prediction as class index."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

# Usage
model = GaitPhaseClassifier(n_features=10)

# Training with CrossEntropyLoss
trainer = laban.TorchTrainer(
    loss=nn.CrossEntropyLoss(),
    optimizer_kwargs={"lr": 0.001}
)

# Prediction
kinematics = torch.randn(1, 10)  # One sample
phase = model.predict_phase(kinematics)
print(f"Gait phase: {phase.item()}")  # 0, 1, 2, or 3
```

## Training Custom Models

### With TorchTrainer

```python
# Create custom model
model = EnhancedBiomechModel(input_dim=5, output_dim=2)

# Prepare dataset
X = torch.randn(1000, 5)
Y = torch.randn(1000, 2)
dataset = laban.CustomDataset(x=X, y=Y)

# Train
trainer = laban.TorchTrainer(
    loss=nn.MSELoss(),
    metrics=laban.MAEMetric(),
    optimizer_kwargs={"lr": 0.001},
    early_stopping_patience=500
)

history = trainer.fit(model, dataset)
```

### Manual Training Loop

```python
model = CustomBiomechModel(input_dim=5, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(1000):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## Model Utilities

### Save and Load

```python
# Save model
torch.save(model.state_dict(), 'model_weights.pth')

# Save entire model
torch.save(model, 'full_model.pth')

# Load weights
model = CustomBiomechModel(input_dim=5, output_dim=1)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Load entire model
model = torch.load('full_model.pth')
model.eval()
```

### Model Summary

```python
# Print model architecture
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Example output:
# Total parameters: 12,345
# Trainable parameters: 12,345
```

### Freeze Layers

```python
# Freeze early layers (transfer learning)
for param in model.shared.parameters():
    param.requires_grad = False

# Only train output heads
for param in model.head_power.parameters():
    param.requires_grad = True

# Check frozen status
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
```

## Best Practices

### 1. Input Normalization

Always normalize inputs to similar scales:

```python
class NormalizedInputModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Option 1: Learnable normalization
        self.normalize = laban.BoxCoxTransform(n_features=input_dim)
        
        # Option 2: Fixed standardization (register as buffer)
        # self.register_buffer('mean', torch.zeros(input_dim))
        # self.register_buffer('std', torch.ones(input_dim))
        
        self.network = nn.Sequential(...)
    
    def forward(self, x):
        x = self.normalize(x)
        return self.network(x)
```

### 2. Regularization

Add dropout and weight decay:

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.2),  # 20% dropout
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 1)
)

# Weight decay in optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

### 3. Activation Functions

Choose appropriate activations:

```python
# ReLU: Default choice, fast
nn.ReLU()

# LeakyReLU: Prevents "dead" neurons
nn.LeakyReLU(0.01)

# Softplus: Smooth, always positive (for forces, powers)
nn.Softplus()

# Sigmoid: Output in [0, 1] (for probabilities, normalized values)
nn.Sigmoid()

# Tanh: Output in [-1, 1]
nn.Tanh()
```

### 4. Output Constraints

Enforce physical constraints:

```python
class PhysicalModel(nn.Module):
    def forward(self, x):
        x = self.network(x)
        
        # Force must be positive
        force = nn.functional.softplus(x[:, 0:1])
        
        # Efficiency in [0, 1]
        efficiency = torch.sigmoid(x[:, 1:2])
        
        # Angle in [-180, 180]
        angle = 360 * torch.tanh(x[:, 2:3])
        
        return torch.cat([force, efficiency, angle], dim=1)
```

## See Also

- [PyTorch Basics](pytorch-basics.md) - PyTorch modules and utilities
- [TorchTrainer](torch-trainer.md) - Training custom models
- [Regression Models](regression.md) - OLS-based models
- [ONNX Deployment](onnx-deployment.md) - Exporting models for production
- [API Reference: PyTorch](../../api-reference/modelling/pytorch.md) - Complete API

---

**Custom Models**: Building domain-specific neural network architectures for biomechanical analysis using labanalysis and PyTorch.
