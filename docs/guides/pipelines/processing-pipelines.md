# Signal Processing Pipelines

Guide to using ProcessingPipeline for automated signal preprocessing.

## Overview

ProcessingPipeline automates signal processing by applying sequences of functions to biomechanical data. It recursively processes all signals in nested Record structures.

**Key Features:**
- Type-specific processing (different functions for EMG, markers, force plates)
- Recursive application to nested structures
- In-place or copy-based processing
- Pre-configured defaults for common signal types

---

## Quick Start

### Using Default Pipeline

The simplest approach - use pre-configured defaults:

```python
import labanalysis as laban
from labanalysis.pipelines import get_default_processing_pipeline

# Load raw data
body = laban.WholeBody.from_tdf("motion.tdf", bodymass_kg=75)

# Get default pipeline
pipeline = get_default_processing_pipeline()

# Apply processing
pipeline(body, inplace=True)

# All signals now filtered and processed
```

**What the default pipeline does:**
- **EMG signals**: DC removal → 20-450 Hz bandpass → full-wave rectification → 200ms RMS envelope
- **Markers (Point3D)**: Gap filling (cubic spline) → 6 Hz lowpass filter
- **Force platforms**: Contact detection (30 N threshold) → gap filling → 30 Hz lowpass → moment update
- **Metabolic data**: 15-point moving average smoothing

---

## Default Processing Functions

### Individual Signal Types

Import and use specific defaults:

```python
from labanalysis.pipelines import (
    get_default_emgsignal_processing_func,
    get_default_point3d_processing_func,
    get_default_forceplatform_processing_func
)

# Process individual EMG signal
emg = laban.EMGSignal(data, time, 'mV', muscle_name='vastus_lateralis', side='left')
emg_processor = get_default_emgsignal_processing_func()
emg_processor(emg)

# Process marker
heel = laban.Point3D(positions, time, 'mm')
marker_processor = get_default_point3d_processing_func()
marker_processor(heel)

# Process force platform
fp = laban.ForcePlatform(...)
fp_processor = get_default_forceplatform_processing_func()
fp_processor(fp)
```

---

## Creating Custom Pipelines

### From Scratch

Build a pipeline with custom processing functions:

```python
from labanalysis import ProcessingPipeline
import labanalysis as laban

# Create empty pipeline
pipeline = ProcessingPipeline()

# Define custom EMG processing
def custom_emg_processing(emg):
    # Remove DC offset
    emg.remove_dc(inplace=True)
    
    # Bandpass filter (30-400 Hz instead of default 20-450 Hz)
    emg.bandpass(fcut_low=30, fcut_high=400, fsamp=1000, order=4, inplace=True)
    
    # 100ms RMS envelope (instead of default 200ms)
    emg.envelope(window_size=100, inplace=True)

# Add to pipeline
pipeline['EMGSignal'] = [custom_emg_processing]

# Define custom marker processing
def custom_marker_processing(marker):
    marker.fillna(inplace=True)
    # Higher cutoff frequency (10 Hz instead of 6 Hz)
    marker.apply(laban.butterworth_filt, fcut=10, fsamp=100, order=2, ftype='lowpass', inplace=True)

pipeline['Point3D'] = [custom_marker_processing]

# Apply to data
body = laban.WholeBody.from_tdf("motion.tdf", bodymass_kg=75)
pipeline(body, inplace=True)
```

### Modifying Defaults

Start with defaults and customize:

```python
from labanalysis.pipelines import get_default_processing_pipeline

# Get default pipeline
pipeline = get_default_processing_pipeline()

# Add extra processing for force platforms
def extra_fp_processing(fp):
    # Compute additional metrics
    fp.update_cop(inplace=True)
    print(f"Max vertical force: {fp.force['Fz'].max():.0f} N")

# Append to existing ForcePlatform processors
pipeline['ForcePlatform'].append(extra_fp_processing)

# Apply modified pipeline
jump = laban.SingleJump.from_tdf("jump.tdf", bodymass_kg=75)
pipeline(jump, inplace=True)
```

---

## Pipeline Operations

### Adding Processors

```python
pipeline = ProcessingPipeline()

# Add single processor
pipeline.add('Signal1D', my_signal_processor)

# Add multiple processors (applied in sequence)
pipeline['EMGSignal'] = [processor1, processor2, processor3]

# Append to existing processors
pipeline['Point3D'].append(additional_processor)
```

### Removing Processors

```python
# Remove specific processor
pipeline.remove('Signal1D', my_signal_processor)

# Remove all processors for a type
pipeline.pop('EMGSignal')

# Check what's in pipeline
emg_processors = pipeline.get('EMGSignal')
print(f"EMG has {len(emg_processors)} processors")
```

### Dict-like Interface

```python
# Access like dictionary
for signal_type in pipeline.keys():
    print(f"{signal_type}: {len(pipeline[signal_type])} processors")

# Iterate over all
for signal_type, processors in pipeline.items():
    print(f"{signal_type}: {processors}")
```

---

## In-Place vs Copy Processing

### In-Place Processing (Default)

Modifies the original data:

```python
pipeline = get_default_processing_pipeline()

jump = laban.SingleJump.from_tdf("jump.tdf", bodymass_kg=75)
original_max_force = jump.ground_reaction_force.module.max()

# Process in-place (modifies jump)
pipeline(jump, inplace=True)

new_max_force = jump.ground_reaction_force.module.max()
assert original_max_force != new_max_force  # Data changed
```

### Copy-Based Processing

Creates processed copy, keeps original unchanged:

```python
pipeline = get_default_processing_pipeline()

jump_raw = laban.SingleJump.from_tdf("jump.tdf", bodymass_kg=75)

# Process copy
jump_processed = pipeline(jump_raw, inplace=False)

# Original unchanged
assert jump_raw.ground_reaction_force is not jump_processed.ground_reaction_force
```

---

## Recursive Processing

Pipeline recursively processes nested Record structures:

```python
# WholeBody contains many nested signals:
# - 40+ Point3D markers
# - Multiple ForcePlatform objects
# - EMGSignal objects
# - Computed Signal3D velocities
# - All computed angles (Signal1D)

body = laban.WholeBody.from_tdf("motion.tdf", bodymass_kg=75)

pipeline = get_default_processing_pipeline()
pipeline(body, inplace=True)

# Pipeline automatically applied to:
# - All Point3D markers (gap fill + 6 Hz filter)
# - All ForcePlatform data (contact detection + 30 Hz filter)
# - All EMGSignal data (20-450 Hz bandpass + envelope)
# - Recursively through all nested structures
```

**How it works:**
1. Pipeline checks type of each item in Record
2. Applies matching processors
3. Recursively descends into nested Record objects
4. Processes all signals at all nesting levels

---

## Common Workflows

### Batch Processing Multiple Files

```python
from pathlib import Path
from labanalysis.pipelines import get_default_processing_pipeline

pipeline = get_default_processing_pipeline()

# Process all TDF files in directory
data_dir = Path("raw_data/")
for tdf_file in data_dir.glob("*.tdf"):
    # Load
    jump = laban.SingleJump.from_tdf(str(tdf_file), bodymass_kg=75)
    
    # Process
    pipeline(jump, inplace=True)
    
    # Save processed data
    output = f"processed/{tdf_file.stem}_processed.csv"
    jump.ground_reaction_force.to_dataframe().to_csv(output)
    
    print(f"Processed {tdf_file.name}")
```

### Protocol-Specific Processing

Different protocols may need different processing:

```python
def get_jump_test_pipeline():
    """Custom pipeline for jump tests."""
    pipeline = get_default_processing_pipeline()
    
    # Stricter force threshold for jumps
    def jump_fp_processing(fp):
        # ... custom contact detection
        pass
    
    pipeline['ForcePlatform'] = [jump_fp_processing]
    return pipeline

def get_balance_test_pipeline():
    """Custom pipeline for balance tests."""
    pipeline = get_default_processing_pipeline()
    
    # More aggressive smoothing for COP
    def balance_fp_processing(fp):
        # ... custom COP smoothing
        pass
    
    pipeline['ForcePlatform'] = [balance_fp_processing]
    return pipeline

# Use protocol-specific pipelines
jump_pipeline = get_jump_test_pipeline()
balance_pipeline = get_balance_test_pipeline()

jump = laban.SingleJump.from_tdf("jump.tdf", bodymass_kg=75)
jump_pipeline(jump, inplace=True)

balance = laban.UprightPosture.from_tdf("balance.tdf", bodymass_kg=75)
balance_pipeline(balance, inplace=True)
```

---

## Filter Parameters Reference

### Butterworth Filters

**EMG (Bandpass):**
```python
# Default: 20-450 Hz, 4th order
fcut_low=20, fcut_high=450, order=4, phase_corrected=True
```

**Markers (Lowpass):**
```python
# Default: 6 Hz, 4th order
fcut=6, order=4, phase_corrected=True
```

**Force Platforms (Lowpass):**
```python
# Default: 30 Hz, 4th order
fcut=30, order=4, phase_corrected=True
```

### Contact Detection

**Force Platform Threshold:**
```python
from labanalysis.constants import MINIMUM_CONTACT_FORCE_N
print(MINIMUM_CONTACT_FORCE_N)  # 30 N

# Forces below this threshold treated as no contact
```

### EMG Envelope

**RMS Window:**
```python
# Default: 200 ms moving window
window_size=200  # milliseconds
```

---

## Performance Tips

1. **Process once, analyze many times**: Apply pipeline once, then run multiple analyses on processed data

2. **Use in-place processing**: Faster and more memory-efficient when you don't need raw data

3. **Customize only what's needed**: Start with defaults, override only specific signal types

4. **Batch similar files**: Load-process-save in sequence for multiple files

---

## See Also

- [Pipelines API](../../api/pipelines.md) - Full API reference
- [Signal Processing](../signal-processing/overview.md) - Low-level processing functions
- [Records API](../../api/records/records.md) - Record data structures
- [Test Protocols](../test-protocols/overview.md) - Using pipelines in test workflows
