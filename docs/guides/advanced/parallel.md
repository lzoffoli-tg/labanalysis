# Parallel Processing and Batch Operations

Guide for processing multiple files efficiently using parallel processing, batch operations, and multiprocessing patterns.

## Overview

When analyzing multiple athletes, sessions, or test conditions, parallel processing dramatically reduces wall-clock time by utilizing multiple CPU cores. This guide covers:

- **Batch file processing** with concurrent I/O
- **Multiprocessing patterns** for CPU-intensive tasks
- **Progress tracking** for long-running operations
- **Error handling** in parallel contexts
- **Memory management** for large datasets

**When to use parallel processing**: Analyzing 10+ files, repetitive identical analyses, batch report generation.

## Quick Reference

```python
from multiprocessing import Pool
from pathlib import Path
import labanalysis as laban

def process_file(filepath):
    """Process single file."""
    body = laban.WholeBody.from_tdf_file(filepath)
    # Analysis here
    return results

# Parallel batch processing
files = list(Path("data/").glob("*.tdf"))
with Pool(processes=4) as pool:
    results = pool.map(process_file, files)
```

## Batch File Processing

### Basic Pattern: Sequential

```python
import labanalysis as laban
from pathlib import Path

def analyze_jump(filepath):
    """Analyze single jump test."""
    test = laban.SingleJump.from_tdf_file(filepath)
    results = test.process()
    
    return {
        'file': filepath.name,
        'jump_height': results.jump_height,
        'peak_force': results.peak_force,
        'rsi_modified': results.rsi_modified
    }

# Sequential processing
data_dir = Path("athlete_tests/")
jump_files = list(data_dir.glob("*_cmj.tdf"))

results = []
for file in jump_files:
    result = analyze_jump(file)
    results.append(result)

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(results)
```

**Problem**: With 100 files × 5 seconds each = 500 seconds total (8+ minutes).

### Parallel Pattern: multiprocessing.Pool

```python
from multiprocessing import Pool
from pathlib import Path
import labanalysis as laban

def analyze_jump(filepath):
    """
    Analyze single jump test.
    
    Must be picklable (no lambda, no nested functions).
    """
    test = laban.SingleJump.from_tdf_file(filepath)
    results = test.process()
    
    return {
        'file': filepath.name,
        'jump_height': results.jump_height,
        'peak_force': results.peak_force,
        'rsi_modified': results.rsi_modified
    }

# Parallel processing
data_dir = Path("athlete_tests/")
jump_files = list(data_dir.glob("*_cmj.tdf"))

# Use Pool with 4 worker processes
with Pool(processes=4) as pool:
    results = pool.map(analyze_jump, jump_files)

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(results)
```

**Speedup**: 100 files on 4 cores = ~125 seconds (4× faster).

### Progress Tracking with tqdm

```python
from multiprocessing import Pool
from tqdm import tqdm
import labanalysis as laban

def analyze_jump(filepath):
    """Analyze jump (same as above)."""
    test = laban.SingleJump.from_tdf_file(filepath)
    results = test.process()
    return {'file': filepath.name, 'jump_height': results.jump_height}

# Parallel with progress bar
jump_files = list(Path("data/").glob("*.tdf"))

with Pool(processes=4) as pool:
    results = list(tqdm(
        pool.imap(analyze_jump, jump_files),
        total=len(jump_files),
        desc="Processing jumps"
    ))

print(f"✓ Processed {len(results)} files")
```

**Output**:
```
Processing jumps: 100%|████████████| 100/100 [02:05<00:00,  1.25s/file]
✓ Processed 100 files
```

## Advanced Patterns

### Error Handling: Fail Gracefully

```python
from multiprocessing import Pool
from pathlib import Path
import labanalysis as laban

def safe_analyze_jump(filepath):
    """
    Analyze jump with error handling.
    
    Returns (filepath, result_or_error).
    """
    try:
        test = laban.SingleJump.from_tdf_file(filepath)
        results = test.process()
        
        return (filepath.name, {
            'status': 'success',
            'jump_height': results.jump_height,
            'peak_force': results.peak_force
        })
    
    except Exception as e:
        return (filepath.name, {
            'status': 'error',
            'error': str(e)
        })

# Process with error handling
jump_files = list(Path("data/").glob("*.tdf"))

with Pool(processes=4) as pool:
    results = pool.map(safe_analyze_jump, jump_files)

# Separate successes and failures
successes = [(f, r) for f, r in results if r['status'] == 'success']
failures = [(f, r) for f, r in results if r['status'] == 'error']

print(f"✓ Success: {len(successes)}/{len(results)}")
print(f"✗ Failed: {len(failures)}/{len(results)}")

# Log failures
if failures:
    import pandas as pd
    error_df = pd.DataFrame([
        {'file': f, 'error': r['error']} 
        for f, r in failures
    ])
    error_df.to_csv("processing_errors.csv", index=False)
    print("⚠ Error log saved to processing_errors.csv")
```

### Chunked Processing: Large File Lists

```python
from multiprocessing import Pool
from pathlib import Path
import labanalysis as laban

def process_chunk(files_chunk):
    """
    Process chunk of files.
    
    Returns list of results.
    """
    results = []
    for filepath in files_chunk:
        try:
            test = laban.SingleJump.from_tdf_file(filepath)
            res = test.process()
            results.append({
                'file': filepath.name,
                'jump_height': res.jump_height
            })
        except Exception as e:
            results.append({
                'file': filepath.name,
                'error': str(e)
            })
    
    return results

# Split into chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

jump_files = list(Path("data/").glob("*.tdf"))
file_chunks = list(chunks(jump_files, 25))  # 25 files per chunk

# Process chunks in parallel
with Pool(processes=4) as pool:
    chunk_results = pool.map(process_chunk, file_chunks)

# Flatten results
results = [item for chunk in chunk_results for item in chunk]
```

**Use case**: 1000+ files where `pool.map()` overhead becomes significant.

### Parameterized Processing

```python
from multiprocessing import Pool
from functools import partial
import labanalysis as laban

def analyze_jump_with_params(filepath, filter_freq=10, threshold=30):
    """
    Analyze jump with custom parameters.
    
    Parameters
    ----------
    filepath : Path
        File to process
    filter_freq : float
        Butterworth filter cutoff (Hz)
    threshold : float
        Force threshold for contact detection (N)
    """
    test = laban.SingleJump.from_tdf_file(filepath)
    # Custom processing with parameters
    results = test.process(
        force_filter_frequency=filter_freq,
        contact_threshold=threshold
    )
    
    return {
        'file': filepath.name,
        'jump_height': results.jump_height
    }

# Create partial function with fixed parameters
jump_files = list(Path("data/").glob("*.tdf"))

analyze_func = partial(
    analyze_jump_with_params,
    filter_freq=15,  # Custom filter
    threshold=50     # Custom threshold
)

with Pool(processes=4) as pool:
    results = pool.map(analyze_func, jump_files)
```

## CPU-Intensive Operations

### Filtering Large Datasets

```python
from multiprocessing import Pool
import numpy as np
import labanalysis as laban
from labanalysis.signalprocessing import butterworth_filter

def filter_body_markers(filepath):
    """
    Load and filter all body markers.
    
    CPU-intensive: multiple 3D signals, expensive filtering.
    """
    body = laban.WholeBody.from_tdf_file(filepath, labels="LABEL")
    
    filtered_markers = {}
    for label in body.labels:
        marker = body.get_point(label)
        
        # Filter each axis
        filtered = laban.Point3D(
            data=np.column_stack([
                butterworth_filter(marker["X"], frequency=6, order=4).to_numpy(),
                butterworth_filter(marker["Y"], frequency=6, order=4).to_numpy(),
                butterworth_filter(marker["Z"], frequency=6, order=4).to_numpy()
            ]),
            index=marker.index,
            columns=["X", "Y", "Z"],
            unit=marker.unit
        )
        
        filtered_markers[label] = filtered
    
    return filepath.name, filtered_markers

# Parallel filtering
motion_files = list(Path("motion_data/").glob("*.tdf"))

with Pool(processes=6) as pool:  # More cores for CPU-bound work
    results = pool.map(filter_body_markers, motion_files)

print(f"✓ Filtered {len(results)} motion files")
```

### Feature Extraction for ML

```python
from multiprocessing import Pool
import numpy as np
import labanalysis as laban

def extract_jump_features(filepath):
    """
    Extract features for machine learning.
    
    CPU-intensive: multiple derived signals, statistics.
    """
    test = laban.SingleJump.from_tdf_file(filepath)
    results = test.process()
    
    # Extract features
    features = {
        # Jump metrics
        'jump_height': results.jump_height,
        'flight_time': results.flight_time,
        'peak_force': results.peak_force,
        'peak_power': results.peak_power,
        
        # Phase durations
        'unweighting_duration': results.unweighting_duration,
        'braking_duration': results.braking_duration,
        'propulsive_duration': results.propulsive_duration,
        
        # Force-time characteristics
        'rfd_max': np.max(np.gradient(results.grf_z.to_numpy())),
        'impulse_total': np.trapz(results.grf_z.to_numpy(), results.grf_z.index),
        
        # Asymmetry (if dual force platforms)
        # Add more features as needed
    }
    
    return features

# Parallel feature extraction
jump_files = list(Path("training_data/").glob("*.tdf"))

with Pool(processes=8) as pool:
    feature_vectors = pool.map(extract_jump_features, jump_files)

# Convert to ML-ready format
import pandas as pd
X = pd.DataFrame(feature_vectors)
```

## Memory Management

### Lazy Loading Pattern

```python
from pathlib import Path
import labanalysis as laban

def get_file_paths():
    """
    Generator for file paths.
    
    Avoids loading all paths into memory at once.
    """
    for filepath in Path("large_dataset/").rglob("*.tdf"):
        yield filepath

def process_and_save(filepath):
    """
    Process file and save immediately.
    
    Doesn't return full results (saves memory).
    """
    test = laban.SingleJump.from_tdf_file(filepath)
    results = test.process()
    
    # Save immediately
    output_file = f"results/{filepath.stem}_results.csv"
    results.to_dataframe().to_csv(output_file, index=False)
    
    # Return only summary (lightweight)
    return {'file': filepath.name, 'status': 'saved', 'jump_height': results.jump_height}

# Process with generator
from multiprocessing import Pool

with Pool(processes=4) as pool:
    summaries = pool.imap(process_and_save, get_file_paths())
    
    # Consume generator (process files one-by-one)
    summary_list = list(summaries)

print(f"✓ Processed and saved {len(summary_list)} files")
```

### Limit Pool Size for Large Files

```python
from multiprocessing import Pool
from pathlib import Path
import labanalysis as laban

def process_large_motion_file(filepath):
    """Process large motion capture file (1M+ samples)."""
    body = laban.WholeBody.from_tdf_file(filepath)
    # Heavy processing
    return results

# Limit pool size to avoid memory overflow
large_files = list(Path("long_trials/").glob("*.tdf"))

# Use fewer processes for memory-intensive tasks
# Rule of thumb: processes = min(cores, available_RAM_GB / file_size_GB)
with Pool(processes=2) as pool:  # Only 2 concurrent large files
    results = pool.map(process_large_motion_file, large_files)
```

## Optimal Process Count

### CPU-Bound Tasks

```python
import multiprocessing as mp

# CPU-bound (filtering, feature extraction, computation)
optimal_processes = mp.cpu_count()  # Use all cores

# Leave some cores free for OS
optimal_processes = max(1, mp.cpu_count() - 1)
```

### I/O-Bound Tasks

```python
# I/O-bound (file reading, writing, network)
# Can use more processes than cores
optimal_processes = mp.cpu_count() * 2
```

### Memory-Constrained

```python
import psutil

# Estimate safe process count based on available memory
available_ram_gb = psutil.virtual_memory().available / (1024**3)
file_size_gb = 0.5  # Estimate per-file memory usage

optimal_processes = max(1, int(available_ram_gb / file_size_gb))
```

## Complete Batch Processing Example

```python
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import labanalysis as laban

def safe_process_jump(filepath):
    """Process jump with full error handling."""
    try:
        test = laban.SingleJump.from_tdf_file(filepath)
        results = test.process()
        
        return {
            'file': filepath.name,
            'status': 'success',
            'athlete': filepath.stem.split('_')[0],  # Parse athlete ID
            'date': filepath.stem.split('_')[1],     # Parse date
            'jump_height': results.jump_height,
            'peak_force': results.peak_force,
            'peak_power': results.peak_power,
            'rsi_modified': results.rsi_modified
        }
    
    except Exception as e:
        return {
            'file': filepath.name,
            'status': 'error',
            'error': str(e)
        }

def batch_process_jumps(data_dir, n_processes=4):
    """
    Batch process all jump tests in directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing jump test files
    n_processes : int
        Number of parallel processes
    
    Returns
    -------
    tuple
        (success_df, error_df)
    """
    data_dir = Path(data_dir)
    jump_files = sorted(data_dir.glob("*_cmj.tdf"))
    
    print(f"Found {len(jump_files)} jump files")
    print(f"Processing with {n_processes} workers...")
    
    # Parallel processing with progress bar
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(safe_process_jump, jump_files),
            total=len(jump_files),
            desc="Analyzing jumps"
        ))
    
    # Separate successes and errors
    successes = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']
    
    # Create DataFrames
    success_df = pd.DataFrame(successes)
    error_df = pd.DataFrame(errors) if errors else pd.DataFrame()
    
    # Report
    print(f"\n✓ Success: {len(successes)}/{len(results)}")
    if errors:
        print(f"✗ Errors: {len(errors)}/{len(results)}")
        error_df.to_csv(data_dir / "processing_errors.csv", index=False)
        print(f"  Error log: {data_dir / 'processing_errors.csv'}")
    
    return success_df, error_df

# Usage
if __name__ == "__main__":
    results_df, errors_df = batch_process_jumps(
        data_dir="athlete_data/season_2024/",
        n_processes=6
    )
    
    # Save results
    results_df.to_csv("jump_results_2024.csv", index=False)
    print(f"✓ Results saved to jump_results_2024.csv")
```

## Best Practices

### 1. Profile First

```python
import time

# Time sequential version
start = time.time()
results_seq = [process_file(f) for f in files[:10]]
seq_time = time.time() - start

# Time parallel version
start = time.time()
with Pool(4) as pool:
    results_par = pool.map(process_file, files[:10])
par_time = time.time() - start

print(f"Sequential: {seq_time:.2f}s")
print(f"Parallel (4): {par_time:.2f}s")
print(f"Speedup: {seq_time/par_time:.2f}×")
```

### 2. Use `if __name__ == "__main__":`

```python
# REQUIRED on Windows
if __name__ == "__main__":
    from multiprocessing import Pool
    
    files = list(Path("data/").glob("*.tdf"))
    
    with Pool(4) as pool:
        results = pool.map(process_file, files)
```

### 3. Keep Functions Picklable

```python
# ❌ BAD: Lambda (not picklable)
with Pool(4) as pool:
    results = pool.map(lambda f: process(f, param=10), files)

# ✅ GOOD: Use functools.partial
from functools import partial

process_with_param = partial(process, param=10)
with Pool(4) as pool:
    results = pool.map(process_with_param, files)
```

### 4. Clean Up Resources

```python
# ✅ Use context manager
with Pool(processes=4) as pool:
    results = pool.map(process_file, files)
# Pool automatically closed and joined

# ❌ Manual management (error-prone)
pool = Pool(processes=4)
results = pool.map(process_file, files)
pool.close()
pool.join()
```

## See Also

- [Performance Tips](performance-tips.md) - Optimization best practices
- [Unit Handling](unit-handling.md) - Efficient unit operations
- [Tutorial - Batch Processing](../tutorials/07-batch-processing.md) - Complete workflow
- [API Reference - Records](../api/records/records.md) - Record classes

---

**Use multiprocessing.Pool** for parallel batch processing of multiple test files. Profile to find optimal process count, handle errors gracefully, and track progress with tqdm.
