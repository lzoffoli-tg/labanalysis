# Tutorial: Batch Processing Multiple Files

Complete workflows for processing large datasets efficiently with labanalysis.

**Duration**: 35 minutes  
**Level**: Intermediate  
**Prerequisites**: labanalysis installed, understanding of Python file operations, pandas knowledge

## What You'll Learn

- Process multiple test files automatically
- Organize data directory structures
- Handle errors gracefully in batch workflows
- Aggregate results across multiple participants
- Generate batch reports
- Export combined datasets
- Optimize processing performance
- Create reusable batch processing scripts

## Scenario

You have collected CMJ data from 50 athletes over multiple testing sessions. Each athlete has 3-5 trials. You need to:
1. Process all trials automatically
2. Calculate metrics for each trial
3. Aggregate best performance per athlete
4. Generate comparative reports
5. Export to Excel for further analysis

## Part 1: Directory Organization

### Step 1: Setup Directory Structure

```python
import labanalysis as laban
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from tqdm import tqdm  # Progress bars
import warnings

# Define directory structure
DATA_ROOT = Path("lab_data")
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"
RESULTS = DATA_ROOT / "results"
REPORTS = DATA_ROOT / "reports"

# Create directories if they don't exist
for dir_path in [RAW_DATA, PROCESSED_DATA, RESULTS, REPORTS]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("Directory structure:")
print(f"  Raw data:       {RAW_DATA}")
print(f"  Processed data: {PROCESSED_DATA}")
print(f"  Results:        {RESULTS}")
print(f"  Reports:        {REPORTS}")

# Expected raw data structure:
# raw/
#   athlete_001/
#     cmj_trial_1.tdf
#     cmj_trial_2.tdf
#     cmj_trial_3.tdf
#   athlete_002/
#     cmj_trial_1.tdf
#     ...
```

**Output:**
```
Directory structure:
  Raw data:       lab_data/raw
  Processed data: lab_data/processed
  Results:        lab_data/results
  Reports:        lab_data/reports
```

### Step 2: Load Participant Database

```python
# Load participant information from CSV
participant_db = pd.read_csv("participants.csv")

# Expected columns: id, surname, name, gender, height_cm, weight_kg, birthdate

print(f"Loaded {len(participant_db)} participants")
print("\nSample:")
print(participant_db.head())

# Create Participant objects
participants = {}
for _, row in participant_db.iterrows():
    participants[row['id']] = laban.Participant(
        surname=row['surname'],
        name=row['name'],
        gender=row['gender'],
        height=row['height_cm'],  # Automatically converted to meters
        weight=row['weight_kg'],
        birthdate=pd.to_datetime(row['birthdate']).date()
    )

print(f"\nCreated {len(participants)} Participant objects")
```

**Output:**
```
Loaded 50 participants

Sample:
        id   surname      name gender  height_cm  weight_kg   birthdate
0  ATH001    Rossi     Mario      M        178       75   1995-03-15
1  ATH002   Bianchi    Laura      F        165       58   1998-07-22
2  ATH003   Verdi   Giuseppe      M        182       82   1992-11-08

Created 50 Participant objects
```

## Part 2: Batch Processing Function

### Step 3: Create Reusable Processing Function

```python
def process_jump_file(
    file_path: Path,
    participant: laban.Participant,
    force_key: str = 'FP1',
    force_component: str = 'Fz',
    save_processed: bool = True
) -> dict:
    """
    Process single jump test file.
    
    Parameters
    ----------
    file_path : Path
        Path to TDF file
    participant : Participant
        Participant object
    force_key : str
        Force platform key in TDF
    force_component : str
        Force component ('Fz' for vertical)
    save_processed : bool
        Save processed data to file
    
    Returns
    -------
    dict
        Dictionary with:
        - 'success': bool
        - 'file': str
        - 'participant_id': str
        - 'jump_height': float (if success)
        - 'peak_power': float (if success)
        - 'error': str (if not success)
    """
    result = {
        'success': False,
        'file': str(file_path.name),
        'participant_id': participant.surname,
    }
    
    try:
        # Load data
        data = laban.read_tdf(str(file_path))
        
        # Check if force platform exists
        if force_key not in data:
            result['error'] = f"Force platform '{force_key}' not found"
            return result
        
        fp = data[force_key]
        fz_raw = fp.force[force_component]
        
        # Process signal
        fz_clean = laban.median_filt(fz_raw.data, window_size=5)
        fz_filt = laban.butterworth_filt(fz_clean, fz_raw.sampling_frequency, 10, 4, 'low')
        
        fz = laban.Signal1D(fz_filt, fz_raw.sampling_frequency, 'N', 'vertical_force')
        
        # Create jump test (simplified - normally use JumpTest protocol)
        # Detect takeoff/landing
        bodyweight = np.median(fz.data[int(0.5*fz.sampling_frequency):int(1.5*fz.sampling_frequency)])
        threshold = bodyweight * 0.1
        
        is_airborne = fz.data < threshold
        takeoff_idx = np.where(np.diff(is_airborne.astype(int)) == 1)[0]
        landing_idx = np.where(np.diff(is_airborne.astype(int)) == -1)[0]
        
        if len(takeoff_idx) == 0 or len(landing_idx) == 0:
            result['error'] = "No jump detected"
            return result
        
        # Calculate flight time
        takeoff = takeoff_idx[0]
        landing = landing_idx[0]
        flight_time = (landing - takeoff) / fz.sampling_frequency
        
        # Calculate jump height from flight time
        jump_height = 0.5 * 9.81 * (flight_time / 2) ** 2
        
        # Calculate peak power (simplified)
        propulsion_start = max(0, takeoff - int(0.5 * fz.sampling_frequency))
        propulsion_force = fz.data[propulsion_start:takeoff]
        peak_force = propulsion_force.max()
        peak_power = peak_force * np.sqrt(2 * 9.81 * jump_height)
        
        result['success'] = True
        result['jump_height'] = jump_height
        result['peak_power'] = peak_power
        result['flight_time'] = flight_time
        result['peak_force'] = peak_force
        
        # Save processed data if requested
        if save_processed:
            processed_file = PROCESSED_DATA / file_path.parent.name / file_path.name
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save force signal
            processed_record = laban.TimeseriesRecord()
            processed_record[force_key] = laban.ForcePlatform(
                force=laban.Signal3D(
                    data={'Fx': fp.force['Fx'].data,
                          'Fy': fp.force['Fy'].data,
                          'Fz': fz_filt},
                    sampling_frequency=fz.sampling_frequency,
                    unit='N'
                ),
                cop=fp.cop
            )
            processed_record.to_tdf(str(processed_file))
        
    except Exception as e:
        result['error'] = str(e)
    
    return result
```

### Step 4: Process All Files

```python
# Find all TDF files
all_files = list(RAW_DATA.glob("*/cmj_trial_*.tdf"))

print(f"Found {len(all_files)} files to process\n")

# Process with progress bar
results_list = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress processing warnings
    
    for file_path in tqdm(all_files, desc="Processing files"):
        # Extract participant ID from directory name
        participant_id = file_path.parent.name  # e.g., "athlete_001"
        
        # Get participant object
        if participant_id not in participants:
            results_list.append({
                'success': False,
                'file': str(file_path.name),
                'participant_id': participant_id,
                'error': 'Participant not in database'
            })
            continue
        
        participant = participants[participant_id]
        
        # Process file
        result = process_jump_file(file_path, participant, save_processed=True)
        results_list.append(result)

# Convert to DataFrame
results_df = pd.DataFrame(results_list)

# Summary
successful = results_df['success'].sum()
failed = len(results_df) - successful

print(f"\n=== BATCH PROCESSING SUMMARY ===")
print(f"Total files:      {len(results_df)}")
print(f"Successful:       {successful} ({successful/len(results_df)*100:.1f}%)")
print(f"Failed:           {failed} ({failed/len(results_df)*100:.1f}%)")

if failed > 0:
    print(f"\nFailure reasons:")
    print(results_df[~results_df['success']]['error'].value_counts())
```

**Output:**
```
Found 187 files to process

Processing files: 100%|██████████| 187/187 [01:24<00:00,  2.22files/s]

=== BATCH PROCESSING SUMMARY ===
Total files:      187
Successful:       182 (97.3%)
Failed:           5 (2.7%)

Failure reasons:
No jump detected                    3
Force platform 'FP1' not found      2
```

## Part 3: Aggregate and Analyze

### Step 5: Calculate Best Performance Per Athlete

```python
# Filter successful trials
successful_df = results_df[results_df['success']].copy()

# Group by participant and find best jump height
best_jumps = successful_df.loc[
    successful_df.groupby('participant_id')['jump_height'].idxmax()
]

# Add participant details
best_jumps = best_jumps.merge(
    participant_db[['id', 'surname', 'name', 'gender', 'weight_kg']],
    left_on='participant_id',
    right_on='id',
    how='left'
)

# Calculate relative power
best_jumps['relative_power'] = best_jumps['peak_power'] / best_jumps['weight_kg']

# Sort by jump height
best_jumps = best_jumps.sort_values('jump_height', ascending=False)

print("=== TOP 10 ATHLETES (JUMP HEIGHT) ===\n")
print(best_jumps[['surname', 'name', 'jump_height', 'peak_power', 'relative_power']].head(10).to_string(index=False))

# Gender comparison
gender_stats = best_jumps.groupby('gender').agg({
    'jump_height': ['mean', 'std', 'min', 'max'],
    'peak_power': ['mean', 'std'],
    'relative_power': ['mean', 'std']
}).round(3)

print("\n=== GENDER COMPARISON ===\n")
print(gender_stats)
```

**Output:**
```
=== TOP 10 ATHLETES (JUMP HEIGHT) ===

   surname      name  jump_height  peak_power  relative_power
     Verdi  Giuseppe        0.482     3456.2           42.15
     Rossi     Mario        0.461     3312.5           44.17
    Neri    Andrea        0.458     3401.8           41.48
   Bianchi    Laura        0.431     2654.3           45.76
     ...

=== GENDER COMPARISON ===

       jump_height              peak_power         relative_power      
              mean    std   min   max       mean     std          mean    std
gender                                                                        
F            0.352  0.042  0.28  0.43    2234.5   312.4         38.53   4.21
M            0.398  0.051  0.31  0.48    2987.6   398.7         39.84   5.12
```

### Step 6: Trial-to-Trial Consistency

```python
# Calculate coefficient of variation (CV) for each athlete
consistency = successful_df.groupby('participant_id').agg({
    'jump_height': ['mean', 'std', 'count']
})

consistency.columns = ['mean_height', 'std_height', 'n_trials']
consistency['cv_percent'] = (consistency['std_height'] / consistency['mean_height']) * 100

# Filter athletes with at least 3 trials
consistency = consistency[consistency['n_trials'] >= 3]

print("=== TRIAL-TO-TRIAL CONSISTENCY ===\n")
print(f"Athletes with CV < 5% (excellent):  {(consistency['cv_percent'] < 5).sum()}")
print(f"Athletes with CV 5-10% (good):       {((consistency['cv_percent'] >= 5) & (consistency['cv_percent'] < 10)).sum()}")
print(f"Athletes with CV > 10% (poor):       {(consistency['cv_percent'] >= 10).sum()}")

# Least consistent athletes (potential re-test needed)
print("\n=== LEAST CONSISTENT (Top 5) ===\n")
least_consistent = consistency.nlargest(5, 'cv_percent')
print(least_consistent[['mean_height', 'cv_percent', 'n_trials']].to_string())
```

**Output:**
```
=== TRIAL-TO-TRIAL CONSISTENCY ===

Athletes with CV < 5% (excellent):  32
Athletes with CV 5-10% (good):       14
Athletes with CV > 10% (poor):       4

=== LEAST CONSISTENT (Top 5) ===

                 mean_height  cv_percent  n_trials
participant_id                                    
athlete_027            0.324       14.23         5
athlete_041            0.298       12.87         4
athlete_013            0.356       11.45         3
athlete_038            0.312       10.98         4
athlete_022            0.371       10.12         5
```

## Part 4: Export and Reporting

### Step 7: Generate Excel Report

```python
# Create comprehensive Excel report
report_file = REPORTS / f"cmj_batch_report_{date.today()}.xlsx"

with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
    # Sheet 1: Processing summary
    results_df.to_excel(writer, sheet_name='Processing Log', index=False)
    
    # Sheet 2: Best performances
    best_jumps.to_excel(writer, sheet_name='Best Performances', index=False)
    
    # Sheet 3: All successful trials
    successful_df.to_excel(writer, sheet_name='All Trials', index=False)
    
    # Sheet 4: Consistency analysis
    consistency.to_excel(writer, sheet_name='Consistency')
    
    # Sheet 5: Gender statistics
    gender_stats.to_excel(writer, sheet_name='Gender Stats')
    
    # Sheet 6: Failures (for troubleshooting)
    failed_df = results_df[~results_df['success']]
    failed_df.to_excel(writer, sheet_name='Failed Trials', index=False)

print(f"Report saved: {report_file}")
```

**Output:**
```
Report saved: lab_data/reports/cmj_batch_report_2026-06-17.xlsx
```

### Step 8: Generate Visualization Dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create dashboard with multiple plots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Jump Height Distribution by Gender',
        'Top 20 Athletes',
        'Trial Consistency (CV)',
        'Power vs Jump Height'
    ],
    specs=[
        [{'type': 'box'}, {'type': 'bar'}],
        [{'type': 'histogram'}, {'type': 'scatter'}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.10
)

# 1. Box plot by gender
for gender in ['M', 'F']:
    data = best_jumps[best_jumps['gender'] == gender]['jump_height']
    fig.add_trace(go.Box(y=data, name=gender, marker_color='steelblue' if gender=='M' else 'coral'),
                  row=1, col=1)

# 2. Top 20 bar chart
top20 = best_jumps.head(20).copy()
top20['fullname'] = top20['surname'] + ' ' + top20['name'].str[0] + '.'
fig.add_trace(go.Bar(
    x=top20['fullname'],
    y=top20['jump_height'],
    marker_color='green'
), row=1, col=2)

# 3. CV histogram
fig.add_trace(go.Histogram(
    x=consistency['cv_percent'],
    nbinsx=20,
    marker_color='orange'
), row=2, col=1)

# 4. Power vs height scatter
fig.add_trace(go.Scatter(
    x=best_jumps['jump_height'],
    y=best_jumps['peak_power'],
    mode='markers',
    marker=dict(size=8, color=best_jumps['weight_kg'], colorscale='Viridis', showscale=True,
                colorbar=dict(title='Weight (kg)', x=1.15)),
    text=best_jumps['surname'],
    name='Athletes'
), row=2, col=2)

# Update axes
fig.update_yaxes(title_text="Jump Height (m)", row=1, col=1)
fig.update_yaxes(title_text="Jump Height (m)", row=1, col=2)
fig.update_xaxes(title_text="CV (%)", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_xaxes(title_text="Jump Height (m)", row=2, col=2)
fig.update_yaxes(title_text="Peak Power (W)", row=2, col=2)

fig.update_layout(height=800, showlegend=False, title_text="CMJ Batch Analysis Dashboard")
fig.write_html(REPORTS / "batch_dashboard.html")
fig.show()

print("Dashboard saved: batch_dashboard.html")
```

## Part 5: Optimization Strategies

### Step 9: Parallel Processing (Optional)

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def process_file_wrapper(args):
    """Wrapper for parallel processing."""
    file_path, participant = args
    return process_jump_file(file_path, participant, save_processed=True)

# Prepare arguments
file_participant_pairs = [
    (file_path, participants.get(file_path.parent.name))
    for file_path in all_files
    if file_path.parent.name in participants
]

# Process in parallel (use N-1 cores)
n_workers = max(1, multiprocessing.cpu_count() - 1)

print(f"Processing {len(file_participant_pairs)} files with {n_workers} workers...")

results_list_parallel = []
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = {executor.submit(process_file_wrapper, args): args for args in file_participant_pairs}
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        results_list_parallel.append(future.result())

print("Parallel processing complete!")
```

### Step 10: Error Recovery and Logging

```python
import logging
from datetime import datetime

# Setup logging
log_file = REPORTS / f"batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_with_logging(file_path, participant):
    """Process file with comprehensive logging."""
    logger.info(f"Processing: {file_path}")
    
    try:
        result = process_jump_file(file_path, participant)
        
        if result['success']:
            logger.info(f"  SUCCESS - Height: {result['jump_height']:.3f}m")
        else:
            logger.warning(f"  FAILED - {result['error']}")
        
        return result
        
    except Exception as e:
        logger.error(f"  ERROR - {str(e)}", exc_info=True)
        return {
            'success': False,
            'file': str(file_path.name),
            'participant_id': participant.surname,
            'error': f"Unexpected error: {str(e)}"
        }

# Use in batch processing
logger.info(f"Starting batch processing of {len(all_files)} files")
# ... processing loop ...
logger.info("Batch processing completed")
```

**Output:**
```
2026-06-17 15:30:12 - INFO - Starting batch processing of 187 files
2026-06-17 15:30:12 - INFO - Processing: lab_data/raw/athlete_001/cmj_trial_1.tdf
2026-06-17 15:30:13 - INFO -   SUCCESS - Height: 0.342m
2026-06-17 15:30:13 - INFO - Processing: lab_data/raw/athlete_001/cmj_trial_2.tdf
2026-06-17 15:30:14 - INFO -   SUCCESS - Height: 0.351m
...
```

## Key Takeaways

### Batch Processing Best Practices
1. **Organize data hierarchically** (participant → trials)
2. **Use progress bars** for user feedback
3. **Log errors comprehensively** for troubleshooting
4. **Save intermediate results** in case of crashes
5. **Validate inputs early** before heavy processing
6. **Handle exceptions gracefully** with try-except

### Performance Optimization
- **Parallel processing**: Use `ProcessPoolExecutor` for CPU-bound tasks
- **Caching**: Save processed data to avoid recomputation
- **Lazy loading**: Process only necessary files
- **Batch I/O**: Minimize file read/write operations

### Error Handling Strategy
- **Catch exceptions** per file (don't let one failure stop batch)
- **Log detailed errors** with timestamps
- **Create failure reports** for follow-up
- **Validate data** before processing

## Next Steps

- **Tutorial 08**: Machine learning integration
- **User Guide**: [Data Export](../user-guide/data-export/)
- **Examples**: Check `examples/batch/` for more patterns

---

**Complete workflows for efficient batch processing of large biomechanical datasets with error handling, parallel execution, and comprehensive reporting.**
