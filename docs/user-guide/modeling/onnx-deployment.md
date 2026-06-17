# ONNX Deployment

Guide to exporting PyTorch models to ONNX format for production deployment, cross-platform inference, and integration with other frameworks.

## Overview

ONNX (Open Neural Network Exchange) is an open format for representing machine learning models. Converting PyTorch models to ONNX enables:

- **Cross-platform deployment** - Run models in C++, Java, JavaScript, mobile apps
- **Framework interoperability** - Use PyTorch-trained models in TensorFlow, scikit-learn
- **Production optimization** - ONNX Runtime provides optimized inference (~2-10x faster)
- **Hardware acceleration** - Easy deployment to GPUs, TPUs, edge devices
- **Model serving** - Integration with deployment platforms (Azure ML, AWS SageMaker)

## Quick Reference

```python
import labanalysis as laban
import torch

# Train model
model = MyModel()
trainer = laban.TorchTrainer(loss=torch.nn.MSELoss())
history = trainer.fit(model, dataset)

# Export to ONNX
dummy_input = torch.randn(1, n_features)  # Example input
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

# Load and run with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
result = session.run(None, {'input': input_data.numpy()})
```

## Basic Export

### Simple Model Export

```python
import labanalysis as laban
import torch
import torch.nn as nn

# Define and train model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleModel()

# Train (skipped for brevity)
# trainer.fit(model, dataset)

# Create dummy input (same shape as real input)
dummy_input = torch.randn(1, 5)  # Batch size 1, 5 features

# Export to ONNX
torch.onnx.export(
    model,                      # Model to export
    dummy_input,                # Example input tensor
    "simple_model.onnx",        # Output file path
    export_params=True,         # Store trained weights
    opset_version=14,           # ONNX version
    do_constant_folding=True,   # Optimize constant operations
    input_names=['features'],   # Input tensor names
    output_names=['prediction'], # Output tensor names
    dynamic_axes={               # Variable batch size
        'features': {0: 'batch'},
        'prediction': {0: 'batch'}
    }
)

print("Model exported to simple_model.onnx")
```

### With labanalysis Modules

```python
import labanalysis as laban
import torch
import torch.nn as nn

# Model with FeaturesGenerator
class BiomechModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = laban.FeaturesGenerator(order=2)
        self.fc1 = nn.Linear(10, 32)  # Estimate feature count
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x is a dict
        x = self.features(x)
        x = torch.cat([v.unsqueeze(1) if v.ndim == 1 else v 
                       for v in x.values()], dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = BiomechModel()
model.eval()  # Set to evaluation mode

# Dummy input (dict)
dummy_input = {'velocity': torch.randn(1, 1)}

# Export
torch.onnx.export(
    model,
    dummy_input,
    "biomech_model.onnx",
    input_names=['velocity'],
    output_names=['force'],
    dynamic_axes={'velocity': {0: 'batch'}, 'force': {0: 'batch'}}
)
```

## ONNX Runtime Inference

### Loading and Running Models

```python
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession("simple_model.onnx")

# Check input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input name: {input_name}")   # 'features'
print(f"Output name: {output_name}") # 'prediction'

# Prepare input data (NumPy array)
input_data = np.random.randn(10, 5).astype(np.float32)

# Run inference
outputs = session.run(
    [output_name],           # Output names to compute
    {input_name: input_data} # Input dictionary
)

predictions = outputs[0]
print(predictions.shape)  # (10, 1)
```

### Performance Comparison

```python
import time
import torch
import onnxruntime as ort
import numpy as np

# Prepare test data
test_data = np.random.randn(1000, 5).astype(np.float32)

# PyTorch inference
model.eval()
with torch.no_grad():
    start = time.time()
    for i in range(100):
        _ = model(torch.from_numpy(test_data))
    pytorch_time = time.time() - start

# ONNX Runtime inference
session = ort.InferenceSession("simple_model.onnx")
input_name = session.get_inputs()[0].name

start = time.time()
for i in range(100):
    _ = session.run(None, {input_name: test_data})
onnx_time = time.time() - start

print(f"PyTorch: {pytorch_time:.3f}s")
print(f"ONNX Runtime: {onnx_time:.3f}s")
print(f"Speedup: {pytorch_time/onnx_time:.2f}x")

# Typical output:
# PyTorch: 0.523s
# ONNX Runtime: 0.087s
# Speedup: 6.01x
```

## Advanced Export

### Multi-Input Models

```python
class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 1)  # 3 + 2 + 1 inputs
    
    def forward(self, velocity, force, angle):
        # Multiple separate inputs
        x = torch.cat([velocity, force, angle], dim=1)
        return self.fc(x)

model = MultiInputModel()

# Dummy inputs (tuple)
dummy_inputs = (
    torch.randn(1, 3),  # velocity (3D)
    torch.randn(1, 2),  # force (2D)
    torch.randn(1, 1)   # angle (1D)
)

# Export
torch.onnx.export(
    model,
    dummy_inputs,
    "multi_input_model.onnx",
    input_names=['velocity', 'force', 'angle'],
    output_names=['output'],
    dynamic_axes={
        'velocity': {0: 'batch'},
        'force': {0: 'batch'},
        'angle': {0: 'batch'},
        'output': {0: 'batch'}
    }
)

# Inference
session = ort.InferenceSession("multi_input_model.onnx")
result = session.run(None, {
    'velocity': np.random.randn(5, 3).astype(np.float32),
    'force': np.random.randn(5, 2).astype(np.float32),
    'angle': np.random.randn(5, 1).astype(np.float32)
})
```

### Multi-Output Models

```python
class MultiOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(5, 32)
        self.head_power = nn.Linear(32, 1)
        self.head_efficiency = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        features = self.relu(self.shared(x))
        power = self.head_power(features)
        efficiency = self.head_efficiency(features)
        
        # Return tuple of outputs
        return power, efficiency

model = MultiOutputModel()
dummy_input = torch.randn(1, 5)

torch.onnx.export(
    model,
    dummy_input,
    "multi_output_model.onnx",
    input_names=['input'],
    output_names=['power', 'efficiency'],
    dynamic_axes={
        'input': {0: 'batch'},
        'power': {0: 'batch'},
        'efficiency': {0: 'batch'}
    }
)

# Inference returns list of outputs
session = ort.InferenceSession("multi_output_model.onnx")
power, efficiency = session.run(None, {
    'input': np.random.randn(10, 5).astype(np.float32)
})

print(power.shape)       # (10, 1)
print(efficiency.shape)  # (10, 1)
```

## Verification and Testing

### Verify ONNX Model

```python
import onnx

# Load ONNX model
onnx_model = onnx.load("simple_model.onnx")

# Check model is valid
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Print model graph
print(onnx.helper.printable_graph(onnx_model.graph))
```

### Compare Outputs (PyTorch vs ONNX)

```python
import torch
import onnxruntime as ort
import numpy as np

# Test input
test_input = torch.randn(5, 5)

# PyTorch prediction
model.eval()
with torch.no_grad():
    pytorch_output = model(test_input).numpy()

# ONNX prediction
session = ort.InferenceSession("simple_model.onnx")
onnx_output = session.run(None, {
    'features': test_input.numpy()
})[0]

# Compare
diff = np.abs(pytorch_output - onnx_output)
max_diff = diff.max()
mean_diff = diff.mean()

print(f"Max difference: {max_diff:.2e}")
print(f"Mean difference: {mean_diff:.2e}")

# Should be very small (< 1e-6)
assert max_diff < 1e-5, "ONNX output differs from PyTorch!"
print("✓ ONNX and PyTorch outputs match!")
```

## Practical Examples

### Force-Velocity Model Deployment

```python
import labanalysis as laban
import torch
import torch.nn as nn

# Train model
class HillModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
    
    def forward(self, velocity):
        x = self.relu(self.fc1(velocity))
        x = self.relu(self.fc2(x))
        force = self.softplus(self.fc3(x))  # Force > 0
        return force

model = HillModel()

# Train with labanalysis (skipped)
# trainer = laban.TorchTrainer(...)
# trainer.fit(model, dataset)

# Export to ONNX
model.eval()
dummy_velocity = torch.randn(1, 1)

torch.onnx.export(
    model,
    dummy_velocity,
    "hill_model.onnx",
    input_names=['velocity'],
    output_names=['force'],
    dynamic_axes={'velocity': {0: 'batch'}, 'force': {0: 'batch'}}
)

# Deploy with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("hill_model.onnx")

# Production inference (very fast!)
velocities = np.array([[1.0], [1.5], [2.0], [2.5]], dtype=np.float32)
forces = session.run(None, {'velocity': velocities})[0]

print("Velocity (m/s) → Force (N)")
for v, f in zip(velocities, forces):
    print(f"{v[0]:.1f} → {f[0]:.1f}")
```

### Jump Predictor for Mobile App

```python
# Train jump height predictor
class JumpPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        # x: [height, weight, age]
        out = self.network(x)
        return 0.2 + 0.6 * out  # Scale to [0.2, 0.8] meters

model = JumpPredictor()
# Train model...

# Export for mobile deployment
model.eval()
torch.onnx.export(
    model,
    torch.randn(1, 3),
    "jump_predictor.onnx",
    input_names=['athlete_data'],
    output_names=['jump_height'],
    opset_version=14,  # Ensure compatibility
    do_constant_folding=True
)

# Mobile app can now use ONNX Runtime mobile SDK
# to run inference on device
```

## Deployment Scenarios

### Web Deployment (ONNX.js)

```javascript
// JavaScript code for web browser
const onnx = require('onnxjs');

async function runModel() {
    // Load ONNX model
    const session = new onnx.InferenceSession();
    await session.loadModel('simple_model.onnx');
    
    // Prepare input
    const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const inputTensor = new onnx.Tensor(inputData, 'float32', [1, 5]);
    
    // Run inference
    const outputMap = await session.run([inputTensor]);
    const prediction = outputMap.values().next().value.data;
    
    console.log('Prediction:', prediction);
}
```

### C++ Deployment

```cpp
// C++ code using ONNX Runtime
#include <onnxruntime_cxx_api.h>

int main() {
    // Create ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    
    // Load model
    Ort::Session session(env, "simple_model.onnx", session_options);
    
    // Prepare input
    std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<int64_t> input_shape = {1, 5};
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());
    
    // Run inference
    const char* input_names[] = {"features"};
    const char* output_names[] = {"prediction"};
    
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1, output_names, 1);
    
    // Extract prediction
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "Prediction: " << output_data[0] << std::endl;
    
    return 0;
}
```

### Cloud Deployment (Azure ML)

```python
# Azure ML deployment
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

# Register ONNX model
ws = Workspace.from_config()
model = Model.register(
    workspace=ws,
    model_path="simple_model.onnx",
    model_name="biomech_model"
)

# Deploy as web service
inference_config = InferenceConfig(
    runtime="python",
    entry_script="score.py",
    conda_file="env.yml"
)

aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1
)

service = Model.deploy(
    workspace=ws,
    name="biomech-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
print(f"Service endpoint: {service.scoring_uri}")
```

## Troubleshooting

### Issue: Export fails with "symbolic shape" error

```python
# WRONG: Variable-length input not specified
torch.onnx.export(model, dummy_input, "model.onnx")

# RIGHT: Specify dynamic axes
torch.onnx.export(
    model, dummy_input, "model.onnx",
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

### Issue: ONNX output differs from PyTorch

```python
# Ensure model is in eval mode
model.eval()

# Disable dropout and batch norm updates
with torch.no_grad():
    torch.onnx.export(...)
```

### Issue: Unsupported operation in ONNX

```python
# Some PyTorch ops not supported in ONNX
# Check opset version compatibility

# Increase opset version
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=15  # Try higher version
)

# Or rewrite unsupported operations
# Example: replace torch.scatter with supported alternatives
```

## Best Practices

1. **Always verify exported model**
   ```python
   onnx.checker.check_model(onnx.load("model.onnx"))
   ```

2. **Test numerical equivalence**
   ```python
   assert np.allclose(pytorch_out, onnx_out, atol=1e-5)
   ```

3. **Use dynamic batch size**
   ```python
   dynamic_axes={'input': {0: 'batch'}}
   ```

4. **Set model to eval mode**
   ```python
   model.eval()
   ```

5. **Optimize with constant folding**
   ```python
   do_constant_folding=True
   ```

## See Also

- [Custom Models](custom-models.md) - Building PyTorch models
- [TorchTrainer](torch-trainer.md) - Training models
- [CPU Optimization Guide](../../advanced/CPU_OPTIMIZATION_GUIDE.md) - Training performance
- [API Reference: PyTorch](../../api-reference/modelling/pytorch.md) - Complete PyTorch API

**External Resources:**
- [ONNX Official Documentation](https://onnx.ai/onnx/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch ONNX Export Guide](https://pytorch.org/docs/stable/onnx.html)

---

**ONNX Deployment**: Export PyTorch models to ONNX format for production deployment, cross-platform inference, and hardware acceleration.
