# Tutorials

Complete end-to-end workflows showing real biomechanical analysis tasks from raw data to final results.

## Available Tutorials

### 1. [Jump Analysis](01-jump-analysis.md)

Complete countermovement jump (CMJ) analysis workflow.

**What you'll learn:**
- Load force platform data from TDF files
- Filter and process GRF signals
- Detect jump phases (unweighting, braking, propulsion, flight)
- Calculate jump metrics (height, power, RSI, force-time characteristics)
- Generate automated reports and visualizations

**Duration:** 15-20 minutes  
**Level:** Beginner  
**Data:** Force platform TDF file

### 2. [Gait Analysis](02-gait-analysis.md)

Walking and running analysis with gait cycle extraction.

**What you'll learn:**
- Load motion capture and force platform data
- Detect gait events (heel strike, toe-off)
- Extract individual gait cycles
- Calculate temporal-spatial parameters
- Analyze joint kinematics across gait cycle

**Duration:** 20-25 minutes  
**Level:** Intermediate  
**Data:** Motion capture + force platform TDF file

### 3. [Full Body Kinematics](03-full-body-kinematics.md)

Complete WholeBody model workflow with 86 properties.

**What you'll learn:**
- Set up WholeBody with 42 anatomical markers
- Access 36 joint angles
- Use computed properties (head center, foot planes)
- Transform between coordinate systems
- Export kinematic data

**Duration:** 25-30 minutes  
**Level:** Intermediate  
**Data:** Full body motion capture TDF file

### 4. [Strength Assessment](04-strength-assessment.md)

Isokinetic and isometric strength testing analysis.

**What you'll learn:**
- Load Biostrength machine data
- Analyze torque-angle and torque-velocity curves
- Calculate peak torque, power, work
- Compare bilateral symmetry
- Predict 1RM from submaximal loads

**Duration:** 15-20 minutes  
**Level:** Beginner  
**Data:** Biostrength data file

### 5. [Signal Processing Pipeline](05-signal-processing.md)

Complete signal processing workflow from raw to publication-ready.

**What you'll learn:**
- Handle missing data with interpolation
- Choose optimal filter parameters
- Apply multi-stage filtering
- Calculate derivatives correctly
- Validate processing quality

**Duration:** 20 minutes  
**Level:** Intermediate  
**Data:** Any TDF file with force/EMG data

### 6. [Custom Protocol](06-custom-protocol.md)

Create and extend TestProtocol for custom analysis.

**What you'll learn:**
- Inherit from TestProtocol
- Implement required methods
- Define custom results
- Add automated plotting
- Generate standardized reports

**Duration:** 30 minutes  
**Level:** Advanced  
**Data:** Custom test data

### 7. [Batch Processing](07-batch-processing.md)

Process multiple files efficiently.

**What you'll learn:**
- Load multiple TDF files
- Apply consistent processing
- Parallelize computations
- Aggregate results across participants
- Export summary statistics

**Duration:** 15 minutes  
**Level:** Intermediate  
**Data:** Multiple TDF files

### 8. [ML Modeling](08-ml-modeling.md)

PyTorch training and deployment workflow.

**What you'll learn:**
- Prepare biomechanical data for ML
- Build custom PyTorch models
- Use TorchTrainer for training
- Optimize with CPU settings (2-3x speedup)
- Export to ONNX for deployment

**Duration:** 30-40 minutes  
**Level:** Advanced  
**Data:** Training dataset

## Tutorial Format

Each tutorial includes:

1. **Scenario** - Real-world context and goals
2. **Prerequisites** - Required knowledge and data
3. **Step-by-step instructions** - Complete code with explanations
4. **Expected output** - What you should see at each step
5. **Complete script** - Full code for reference
6. **Next steps** - Related tutorials and guides

## Getting Started

### Choose Your Path

**New to labanalysis?**  
Start with tutorials 1-2 (Jump Analysis, Gait Analysis) to learn the basics.

**Experienced user?**  
Jump to tutorials 6-8 (Custom Protocol, Batch Processing, ML Modeling) for advanced topics.

**Specific task in mind?**  
Use the descriptions above to find the relevant tutorial.

### Required Setup

All tutorials assume:
- labanalysis installed ([installation guide](../getting-started/installation.md))
- Python 3.12+
- Basic Python knowledge
- Sample data files (or use tutorial examples)

### Sample Data

Tutorial sample data is available in the `test/assets/` directory of the repository:

```python
from pathlib import Path
import labanalysis as laban

# Typical data location
data_path = Path("test/assets/")
tdf_file = data_path / "example_jump.tdf"

record = laban.TimeseriesRecord.from_tdf(tdf_file)
```

## Related Resources

### Before Tutorials

- **[Quick Start](../getting-started/quick-start.md)** - 5-minute introduction
- **[Core Concepts](../getting-started/core-concepts.md)** - Understand the basics
- **[Your First Analysis](../getting-started/first-analysis.md)** - Guided walkthrough

### During Tutorials

- **[User Guide](../user-guide/README.md)** - Reference for specific tasks
- **[API Reference](../api-reference/README.md)** - Complete API documentation
- **[Examples](../examples/README.md)** - Quick code snippets

### After Tutorials

- **[Advanced Topics](../advanced/README.md)** - Optimization and extension
- **[Troubleshooting](../troubleshooting/README.md)** - Common issues
- **[Development](../development/README.md)** - Contributing guide

## Tutorial Progression

Recommended order for learning:

```
Beginner Path:
1. Jump Analysis (Tutorial 1)
   ↓
2. Gait Analysis (Tutorial 2)
   ↓
3. Strength Assessment (Tutorial 4)

Intermediate Path:
1. Signal Processing Pipeline (Tutorial 5)
   ↓
2. Full Body Kinematics (Tutorial 3)
   ↓
3. Batch Processing (Tutorial 7)

Advanced Path:
1. Custom Protocol (Tutorial 6)
   ↓
2. ML Modeling (Tutorial 8)
```

## Feedback

Found an issue in a tutorial? Have a suggestion for a new tutorial?

Contact: [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)

---

**Ready to start?** Pick a tutorial above and begin your labanalysis journey!
