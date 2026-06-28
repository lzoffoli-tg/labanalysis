# Pipelines Module

The `pipelines` module provides configurable signal processing pipelines for automated data preprocessing.

## Overview

ProcessingPipeline applies sequences of signal processing functions to biomechanical data. The module includes default pipelines optimized for each signal type (EMG, force platforms, kinematic markers, metabolic data).

---

## ProcessingPipeline

Configurable pipeline for applying signal processing functions to Record objects.

**Module:** `labanalysis.pipelines`

**Description:**  
ProcessingPipeline is a dictionary-like container that maps data types to lists of processing functions. It recursively applies these functions to Record objects and their nested contents.

**Parameters:**
- `callable_dict` (dict or callable, optional): Dictionary mapping type names to callables, or single callable to apply to all types

**Methods:**
- `add(key, value)`: Add processing function for a data type
- `remove(key, value)`: Remove specific function
- `pop(key)`: Remove all functions for a data type
- `get(key)`: Get functions for a data type
- `apply(record, inplace=True)`: Apply pipeline to a Record
- `__call__(record, inplace=True)`: Shorthand for apply()

**Dict-like Access:**
- `pipeline[type_name] = [func1, func2]`: Set functions
- `pipeline[type_name]`: Get functions
- `pipeline.keys()`, `values()`, `items()`: Iterate

**Example:**
```python
import labanalysis as laban
from labanalysis import ProcessingPipeline

# Create empty pipeline
pipeline = ProcessingPipeline()

# Add processing for Signal1D
def my_signal_processor(signal):
    signal.fillna(inplace=True)
    signal.apply(laban.butterworth_filt, fcut=10, fsamp=100, order=4, ftype='lowpass', inplace=True)

pipeline.add('Signal1D', my_signal_processor)

# Add processing for ForcePlatform
def my_fp_processor(fp):
    fp.strip(inplace=True)
    fp.force[:, :] = laban.fillna(fp.force.to_numpy(), value=0)

pipeline['ForcePlatform'] = [my_fp_processor]

# Apply to record
record = laban.TimeseriesRecord()
# ... populate record ...
pipeline(record, inplace=True)
```

---

## Default Processing Functions

The module provides pre-configured processing functions for each signal type.

### get_default_emgsignal_processing_func()

Returns EMG signal processing function.

**Processing Steps:**
1. Remove DC offset (subtract mean)
2. Bandpass filter: 20-450 Hz (4th order Butterworth)
3. Full-wave rectification
4. RMS envelope: 200ms moving window

**Returns:** Callable function

**Example:**
```python
from labanalysis.pipelines import get_default_emgsignal_processing_func

emg_processor = get_default_emgsignal_processing_func()

# Apply to EMGSignal
emg = laban.EMGSignal(data, time, 'mV', muscle_name='vastus_lateralis', side='left')
emg_processor(emg)
```

---

### get_default_point3d_processing_func()

Returns 3D marker trajectory processing function.

**Processing Steps:**
1. Fill missing data (cubic spline interpolation)
2. Lowpass filter: 6 Hz (4th order Butterworth, phase-corrected)

**Returns:** Callable function

**Example:**
```python
from labanalysis.pipelines import get_default_point3d_processing_func

marker_processor = get_default_point3d_processing_func()

# Apply to Point3D
heel = laban.Point3D(positions, time, 'mm')
marker_processor(heel)
```

---

### get_default_signal1d_processing_func()

Returns 1D signal processing function (same as Point3D).

**Processing Steps:**
1. Fill missing data (cubic spline)
2. Lowpass filter: 6 Hz (4th order Butterworth)

**Returns:** Callable function

---

### get_default_signal3d_processing_func()

Returns 3D signal processing function (same as Point3D).

**Processing Steps:**
1. Fill missing data (cubic spline)
2. Lowpass filter: 6 Hz (4th order Butterworth)

**Returns:** Callable function

---

### get_default_forceplatform_processing_func()

Returns force platform processing function.

**Processing Steps:**
1. Contact detection: Set forces < 30 N to NaN
2. Strip NaN values from signal edges
3. Fill force NaNs with zeros
4. Fill position (COP) NaNs with cubic spline
5. Lowpass filter origin and force: 30 Hz (4th order Butterworth)
6. Update torque/moments from filtered data
7. Set moments to zero where vertical force < 30 N

**Returns:** Callable function

**Example:**
```python
from labanalysis.pipelines import get_default_forceplatform_processing_func

fp_processor = get_default_forceplatform_processing_func()

# Apply to ForcePlatform
fp = laban.ForcePlatform(...)
fp_processor(fp)
```

---

### get_default_metabolicrecord_processing_func()

Returns metabolic data processing function.

**Processing Steps:**
1. Apply 15-point moving average for breath-by-breath smoothing

**Returns:** Callable function

**Example:**
```python
from labanalysis.pipelines import get_default_metabolicrecord_processing_func

metabolic_processor = get_default_metabolicrecord_processing_func()

# Apply to MetabolicRecord
metabolic = laban.MetabolicRecord(...)
metabolic_processor(metabolic)
```

---

### get_default_processing_pipeline()

Returns complete pipeline with defaults for all signal types.

**Includes Processing For:**
- `EMGSignal`: 20-450 Hz bandpass, 200ms RMS envelope
- `Point3D`: Gap fill, 6 Hz lowpass
- `Signal1D`: Gap fill, 6 Hz lowpass  
- `Signal3D`: Gap fill, 6 Hz lowpass
- `ForcePlatform`: Contact detection, 30 Hz lowpass, moment update
- `MetabolicRecord`: 15-point moving average

**Returns:** ProcessingPipeline instance

**Example:**
```python
from labanalysis.pipelines import get_default_processing_pipeline

# Get default pipeline
pipeline = get_default_processing_pipeline()

# Load and process a jump
jump = laban.SingleJump.from_tdf("jump.tdf", bodymass_kg=75)
pipeline(jump, inplace=True)

# Or modify defaults
pipeline['EMGSignal'][0] = my_custom_emg_processor

# Apply modified pipeline
pipeline(jump, inplace=True)
```

---

## Custom Pipelines

### Creating from Scratch

```python
from labanalysis import ProcessingPipeline
import labanalysis as laban
import numpy as np

# Create custom pipeline
pipeline = ProcessingPipeline()

# Custom EMG processing with different parameters
def custom_emg(emg):
    emg.remove_dc(inplace=True)
    emg.bandpass(fcut_low=30, fcut_high=400, fsamp=1000, order=4, inplace=True)
    emg.envelope(window_size=100, inplace=True)  # 100ms instead of 200ms

pipeline['EMGSignal'] = [custom_emg]

# Custom marker processing (higher cutoff frequency)
def custom_marker(marker):
    marker.fillna(inplace=True)
    marker.apply(laban.butterworth_filt, fcut=10, fsamp=100, order=2, ftype='lowpass', inplace=True)

pipeline['Point3D'] = [custom_marker]
```

### Modifying Defaults

```python
from labanalysis.pipelines import get_default_processing_pipeline

# Start with defaults
pipeline = get_default_processing_pipeline()

# Add extra processing step
def extra_fp_processing(fp):
    # Additional force platform processing
    fp.update_cop(inplace=True)

# Append to existing ForcePlatform pipeline
pipeline['ForcePlatform'].append(extra_fp_processing)
```

### Applying to Nested Records

Pipelines recursively process nested Record structures:

```python
from labanalysis import WholeBody
from labanalysis.pipelines import get_default_processing_pipeline

# Load whole body data with many signals
body = WholeBody.from_tdf("motion.tdf", bodymass_kg=75)

# Pipeline automatically processes:
# - All Point3D markers (40+ markers)
# - All ForcePlatform objects
# - All EMGSignal objects
# - Nested TimeseriesRecord structures

pipeline = get_default_processing_pipeline()
pipeline(body, inplace=True)

# All signals now filtered and processed
```

---

## Filter Parameters

### Butterworth Filter Settings

**EMGSignal:**
- Bandpass: 20-450 Hz
- Order: 4
- Phase corrected: Yes

**Markers (Point3D, Signal1D, Signal3D):**
- Lowpass: 6 Hz
- Order: 4
- Phase corrected: Yes

**Force Platforms:**
- Lowpass: 30 Hz
- Order: 4
- Phase corrected: Yes

### Contact Detection Threshold

**ForcePlatform minimum contact force:**
- Threshold: 30 N
- Defined in: `labanalysis.constants.MINIMUM_CONTACT_FORCE_N`

### EMG Envelope Settings

**RMS window:**
- Default: 200 ms
- Customizable via envelope() method

---

## See Also

- [Signal Processing API](signalprocessing.md) - Low-level processing functions
- [Records API](records/records.md) - Record data structures
- [Pipelines Guide](../guides/pipelines/processing-pipelines.md) - Pipeline usage guide
