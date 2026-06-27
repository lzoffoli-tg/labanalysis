# Performance Optimization Tips

Best practices and techniques for optimizing labanalysis performance in production environments.

## Overview

This guide covers:

- **Memory optimization** for large datasets
- **Computational efficiency** patterns
- **I/O optimization** for file operations
- **Profiling techniques** to identify bottlenecks
- **Caching strategies** for repeated analyses

## Quick Reference

```python
# Memory: Use views instead of copies
marker_z = body.left_ankle["Z"]  # View (fast, no copy)

# Computation: Vectorize operations
velocities = np.diff(markers.to_numpy(), axis=0) / dt  # Vectorized

# I/O: Lazy loading
from pathlib import Path
files = Path("data/").glob("*.tdf")  # Generator, not list

# Caching: Store expensive results
@lru_cache(maxsize=128)
def expensive_calculation(params):
    return results
```

## Memory Optimization

### Use Views, Not Copies

```python
import labanalysis as laban

body = laban.WholeBody.from_tdf_file("motion.tdf")

# ❌ BAD: Creates copy (2× memory)
ankle_copy = body.left_ankle.copy()
z_values = ankle_copy["Z"].to_numpy()

# ✅ GOOD: Uses view (minimal memory)
ankle_view = body.left_ankle  # View
z_values = ankle_view["Z"].to_numpy()
```

**Memory savings**: For 100k samples × 30 markers = 3M floats ≈ 24 MB per copy avoided.

### Extract Only Needed Columns

```python
# ❌ BAD: Load full WholeBody (all 88 properties)
body = laban.WholeBody.from_tdf_file("long_trial.tdf")
ankle = body.left_ankle  # Still loads everything

# ✅ GOOD: Load only required markers
from labanalysis.io import read_tdf

# Read only specific markers
markers_dict = read_tdf(
    "long_trial.tdf",
    labels=["left_ankle_medial", "left_ankle_lateral"]
)

# Compute ankle center manually
left_ankle = (markers_dict["left_ankle_medial"] + 
              markers_dict["left_ankle_lateral"]) / 2
```

**Memory savings**: Loading 2 markers vs 30+ markers = ~15× reduction.

### Delete Unused Data

```python
import labanalysis as laban

# Load and process
body = laban.WholeBody.from_tdf_file("trial.tdf")
ankle_angle = body.left_ankle_flexionextension

# Extract what you need
angle_data = ankle_angle.to_numpy()
time = ankle_angle.index

# Delete large object
del body  # Free ~100 MB

# Continue with extracted data
import numpy as np
peak_angle = np.max(angle_data)
```

### Process in Chunks

```python
from pathlib import Path
import labanalysis as laban

def process_file_chunked(filepath, chunk_size=10000):
    """
    Process large file in chunks.
    
    Parameters
    ----------
    filepath : Path
        File to process
    chunk_size : int
        Samples per chunk
    """
    # Load full file (unavoidable for TDF format)
    body = laban.WholeBody.from_tdf_file(filepath)
    ankle = body.left_ankle
    
    n_samples = len(ankle)
    results = []
    
    # Process in chunks
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        
        # Extract chunk
        chunk = ankle[start:end]
        
        # Process chunk
        chunk_result = np.mean(chunk.to_numpy())
        results.append(chunk_result)
    
    # Clean up
    del body, ankle
    
    return np.mean(results)
```

## Computational Efficiency

### Vectorize Operations

```python
import numpy as np
import labanalysis as laban

# ❌ BAD: Loop over samples
distances = []
for i in range(len(marker1)):
    p1 = marker1.to_numpy()[i]
    p2 = marker2.to_numpy()[i]
    dist = np.sqrt(np.sum((p1 - p2)**2))
    distances.append(dist)

# ✅ GOOD: Vectorized
diff = marker1.to_numpy() - marker2.to_numpy()
distances = np.sqrt(np.sum(diff**2, axis=1))
```

**Speed**: Vectorized version is ~50-100× faster for 100k samples.

### Avoid Repeated Function Calls

```python
# ❌ BAD: Repeated expensive calculation
for trial in trials:
    body = laban.WholeBody.from_tdf_file(trial)
    ankle = body.left_ankle  # Recomputes every time
    knee = body.left_knee    # Recomputes every time

# ✅ GOOD: Cache results
for trial in trials:
    body = laban.WholeBody.from_tdf_file(trial)
    
    # Extract once
    ankle = body.left_ankle
    knee = body.left_knee
    
    # Reuse cached values
    distance = ankle - knee
    angle = calculate_angle(ankle, knee)
```

### Use NumPy Built-ins

```python
import numpy as np

# ❌ BAD: Manual implementation
peak_force = max(force_signal.to_numpy())
mean_force = sum(force_signal.to_numpy()) / len(force_signal)

# ✅ GOOD: NumPy built-ins (optimized C code)
peak_force = np.max(force_signal.to_numpy())
mean_force = np.mean(force_signal.to_numpy())
```

### Pre-allocate Arrays

```python
import numpy as np

n_trials = 1000

# ❌ BAD: Growing list
results = []
for i in range(n_trials):
    result = expensive_calculation(i)
    results.append(result)  # Reallocates memory

# ✅ GOOD: Pre-allocated array
results = np.zeros(n_trials)
for i in range(n_trials):
    results[i] = expensive_calculation(i)  # No reallocation
```

## I/O Optimization

### Lazy Loading with Generators

```python
from pathlib import Path

# ❌ BAD: Load all paths into memory
all_files = list(Path("data/").rglob("*.tdf"))  # 10k files = large list
for file in all_files:
    process(file)

# ✅ GOOD: Generator (minimal memory)
file_generator = Path("data/").rglob("*.tdf")  # Lazy iterator
for file in file_generator:
    process(file)
```

### Batch File Reading

```python
from pathlib import Path
import labanalysis as laban

def load_multiple_files(file_list):
    """Load multiple files efficiently."""
    # ❌ BAD: Load one-by-one
    # bodies = [laban.WholeBody.from_tdf_file(f) for f in file_list]
    
    # ✅ GOOD: Load with context manager (proper resource handling)
    bodies = []
    for filepath in file_list:
        try:
            body = laban.WholeBody.from_tdf_file(filepath)
            bodies.append(body)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            continue
    
    return bodies
```

### Cache Expensive File Reads

```python
from functools import lru_cache
import labanalysis as laban

@lru_cache(maxsize=10)
def load_reference_data(filepath):
    """
    Load reference data with caching.
    
    Caches last 10 files in memory.
    """
    return laban.WholeBody.from_tdf_file(filepath)

# First call: reads from disk
ref1 = load_reference_data("reference.tdf")  # ~500 ms

# Second call: returns cached result
ref2 = load_reference_data("reference.tdf")  # <1 ms
```

## Filtering Optimization

### Choose Appropriate Filter Order

```python
from labanalysis.signalprocessing import butterworth_filter

# ❌ SLOW: High order filter (expensive)
filtered_high = butterworth_filter(signal, frequency=10, order=8)

# ✅ FAST: Lower order (usually sufficient)
filtered_low = butterworth_filter(signal, frequency=10, order=4)
```

**Speed**: 4th order is ~2× faster than 8th order for minimal quality difference.

### Filter Once, Reuse

```python
from labanalysis.signalprocessing import butterworth_filter

# Load signal
ankle = body.left_ankle

# ❌ BAD: Filter each axis separately
x_filt = butterworth_filter(ankle["X"], frequency=6, order=4)
y_filt = butterworth_filter(ankle["Y"], frequency=6, order=4)
z_filt = butterworth_filter(ankle["Z"], frequency=6, order=4)

# ✅ GOOD: Filter once, extract axes
ankle_filt = butterworth_filter(ankle, frequency=6, order=4)
x_filt = ankle_filt["X"]
y_filt = ankle_filt["Y"]
z_filt = ankle_filt["Z"]
```

### Batch Filter Multiple Signals

```python
import numpy as np
from labanalysis.signalprocessing import butterworth_filter

# Multiple markers to filter
markers = [body.left_ankle, body.left_knee, body.left_hip]

# ❌ SLOW: Filter one-by-one
filtered = [butterworth_filter(m, frequency=6, order=4) for m in markers]

# ✅ FASTER: Combine, filter, split
# Stack all marker data
combined = np.hstack([m.to_numpy() for m in markers])

# Filter once (shared filter state)
from scipy.signal import butter, filtfilt
b, a = butter(4, 6 / (100 / 2), btype='low')  # Design once
filtered_combined = filtfilt(b, a, combined, axis=0)

# Split back
n_cols = markers[0].to_numpy().shape[1]
filtered = [
    laban.Point3D(
        data=filtered_combined[:, i*n_cols:(i+1)*n_cols],
        index=markers[i].index,
        columns=markers[i].columns,
        unit=markers[i].unit
    )
    for i in range(len(markers))
]
```

## Profiling and Benchmarking

### Time Critical Sections

```python
import time

# Simple timing
start = time.time()
result = expensive_operation()
elapsed = time.time() - start

print(f"Operation took {elapsed:.2f} seconds")
```

### Profile with cProfile

```python
import cProfile
import pstats
import labanalysis as laban

def analyze_jump(filepath):
    """Function to profile."""
    test = laban.SingleJump.from_tdf_file(filepath)
    return test.process()

# Profile function
profiler = cProfile.Profile()
profiler.enable()

result = analyze_jump("jump.tdf")

profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**Output**:
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.001    0.001    2.450    2.450 jumping.py:145(process)
      156    0.023    0.000    1.890    0.012 signalprocessing.py:45(butterworth_filter)
      312    1.245    0.004    1.867    0.006 filtfilt.py:23(filtfilt)
        ...
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    """Function to profile memory usage."""
    import labanalysis as laban
    
    body = laban.WholeBody.from_tdf_file("trial.tdf")  # ~100 MB
    ankle = body.left_ankle  # Minimal overhead
    
    result = ankle.to_numpy()  # ~10 MB copy
    
    return result

# Run and see memory usage line-by-line
memory_intensive_function()
```

### Line-by-Line Profiling

```python
# Install: pip install line_profiler
from line_profiler import LineProfiler

def critical_function():
    """Function to profile line-by-line."""
    # ... code ...
    pass

# Profile
profiler = LineProfiler()
profiler.add_function(critical_function)
profiler.enable()

critical_function()

profiler.disable()
profiler.print_stats()
```

## Caching Strategies

### Function Result Caching

```python
from functools import lru_cache
import labanalysis as laban

@lru_cache(maxsize=128)
def get_normative_data(test_type, age_group, gender):
    """
    Load normative data with caching.
    
    Caches 128 most recent lookups.
    """
    # Expensive database query or file read
    return normative_database.query(test_type, age_group, gender)

# First call: computes result
norms1 = get_normative_data('CMJ', '20-29', 'M')  # Slow

# Second call: returns cached result
norms2 = get_normative_data('CMJ', '20-29', 'M')  # Instant
```

### Property Caching

```python
class CachedJumpAnalysis:
    """Jump analysis with cached properties."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self._test = None
        self._results = None
    
    @property
    def test(self):
        """Lazy load test (cached)."""
        if self._test is None:
            import labanalysis as laban
            self._test = laban.SingleJump.from_tdf_file(self.filepath)
        return self._test
    
    @property
    def results(self):
        """Lazy compute results (cached)."""
        if self._results is None:
            self._results = self.test.process()
        return self._results
    
    @property
    def jump_height(self):
        """Access cached results."""
        return self.results.jump_height

# Usage
analysis = CachedJumpAnalysis("jump.tdf")

# First access: loads file and processes
height1 = analysis.jump_height  # Slow

# Subsequent access: uses cached results
height2 = analysis.jump_height  # Instant
```

### Disk Caching with joblib

```python
from joblib import Memory
import labanalysis as laban

# Create cache directory
cache = Memory('cache_dir', verbose=0)

@cache.cache
def process_jump(filepath):
    """
    Process jump with disk caching.
    
    Results cached to disk, persist across sessions.
    """
    test = laban.SingleJump.from_tdf_file(filepath)
    results = test.process()
    
    return {
        'jump_height': results.jump_height,
        'peak_force': results.peak_force
    }

# First call: processes and caches to disk
result1 = process_jump("athlete1_cmj.tdf")  # Slow

# Second call (even in new session): loads from disk cache
result2 = process_jump("athlete1_cmj.tdf")  # Fast
```

## Best Practices Summary

### Memory

1. **Use views** instead of copies
2. **Delete unused data** explicitly
3. **Load only needed markers** from files
4. **Process in chunks** for huge datasets

### Computation

1. **Vectorize** operations with NumPy
2. **Cache** expensive calculations
3. **Pre-allocate** arrays when size is known
4. **Use built-in functions** (NumPy, SciPy) over custom implementations

### I/O

1. **Use generators** for file lists
2. **Batch read** when possible
3. **Cache reference data** loaded multiple times
4. **Close files** properly (use context managers)

### Filtering

1. **Choose appropriate filter order** (4th usually sufficient)
2. **Filter once, reuse** instead of repeated filtering
3. **Batch filter** multiple signals together

### Profiling

1. **Profile before optimizing** (measure, don't guess)
2. **Focus on hotspots** (top 20% of time)
3. **Benchmark changes** to verify improvement
4. **Profile memory** for large dataset issues

## Performance Checklist

Before deploying to production, verify:

- [ ] Profiled critical code paths
- [ ] Vectorized loops where possible
- [ ] Using generators for large file lists
- [ ] Caching expensive, repeated calculations
- [ ] Deleting large objects when done
- [ ] Loading only needed data
- [ ] Using appropriate filter orders
- [ ] Batch processing when analyzing multiple files
- [ ] Memory usage stays within bounds
- [ ] No memory leaks in long-running processes

## See Also

- [Parallel Processing](parallel-processing.md) - Multiprocessing patterns
- [Unit Handling](unit-handling.md) - Efficient unit operations
- [CPU Optimization Guide](CPU_OPTIMIZATION_GUIDE.md) - Low-level optimization
- [Tutorial - Batch Processing](../tutorials/07-batch-processing.md) - Complete workflow

---

**Profile first, optimize second.** Focus on vectorization, caching, and memory management for biggest performance gains. Use parallel processing for batch operations.
