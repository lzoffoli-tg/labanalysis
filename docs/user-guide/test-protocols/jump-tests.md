# Jump Tests

Complete guide to countermovement jump (CMJ), squat jump (SJ), and drop jump (DJ) protocols in labanalysis.

## Overview

Jump tests are fundamental assessments of lower-body power and neuromuscular function. labanalysis provides comprehensive support for:

- **Countermovement Jump (CMJ)**: Most common jump test with eccentric-concentric action
- **Squat Jump (SJ)**: Concentric-only jump from static position
- **Drop Jump (DJ)**: Plyometric jump from elevated platform
- **Repeated Jumps**: Multiple consecutive jumps for fatigue assessment

All protocols use force platform data to calculate:
- Jump height (flight time method)
- Peak power, average power
- Rate of force development (RFD)
- Reactive strength index (RSI, for DJ)
- Phase durations and impulses

## Quick Start

```python
import labanalysis as laban

# Create participant
participant = laban.Participant(
    name="John",
    surname="Doe",
    height=1.80,  # meters
    weight=75     # kg
)

# Load CMJ data from force platform
record = laban.TimeseriesRecord.from_tdf("cmj_trial.tdf")
fp = record['FP1']

# Create CMJ test
test = laban.JumpTest(
    participant=participant,
    protocol='CMJ',
    force_platform=fp
)

# Analyze
results = test.analyze()

# View results
print(f"Jump Height: {results.jump_height:.2f} m ({results.jump_height*100:.1f} cm)")
print(f"Peak Power: {results.peak_power:.1f} W ({results.peak_power_rel:.1f} W/kg)")
print(f"Flight Time: {results.flight_time*1000:.1f} ms")
```

**Output:**
```
Jump Height: 0.327 m (32.7 cm)
Peak Power: 3245.8 W (43.3 W/kg)
Flight Time: 458.2 ms
```

## Countermovement Jump (CMJ)

### Protocol

CMJ protocol with eccentric-concentric action:

1. **Quiet stand** (2-3 seconds): Establish bodyweight baseline
2. **Unweighting**: Rapid downward movement (countermovement)
3. **Propulsion**: Upward push-off against force platform
4. **Flight**: Ballistic phase with no ground contact
5. **Landing**: Ground contact and deceleration

### Analysis Workflow

```python
import labanalysis as laban
import numpy as np

# 1. Setup participant
participant = laban.Participant(
    name="Athlete",
    surname="001",
    height=1.75,
    weight=70,
    age=25,
    gender='M'
)

# 2. Load force platform data
record = laban.TimeseriesRecord.from_tdf("cmj.tdf")
fp = record['FP1']

# 3. Create and run test
test = laban.JumpTest(
    participant=participant,
    protocol='CMJ',
    force_platform=fp
)

results = test.analyze()

# 4. Access metrics
print("=== CMJ Results ===")
print(f"Jump Height: {results.jump_height*100:.1f} cm")
print(f"Flight Time: {results.flight_time*1000:.1f} ms")
print(f"Peak Force: {results.peak_force:.1f} N ({results.peak_force_rel:.2f} BW)")
print(f"Peak Power: {results.peak_power:.1f} W ({results.peak_power_rel:.1f} W/kg)")
print(f"Average Power: {results.avg_power:.1f} W ({results.avg_power_rel:.1f} W/kg)")
print(f"Peak RFD: {results.peak_rfd:.1f} N/s")
print(f"Impulse: {results.impulse:.1f} N·s")
print(f"\nPhase Durations:")
print(f"  Unweighting: {results.unweighting_duration*1000:.1f} ms")
print(f"  Propulsion: {results.propulsion_duration*1000:.1f} ms")
print(f"  Landing: {results.landing_duration*1000:.1f} ms")
```

**Output:**
```
=== CMJ Results ===
Jump Height: 32.7 cm
Flight Time: 458.2 ms
Peak Force: 1842.3 N (2.68 BW)
Peak Power: 3245.8 W (43.3 W/kg)
Average Power: 1823.5 W (24.3 W/kg)
Peak RFD: 8456.2 N/s

Phase Durations:
  Unweighting: 287.3 ms
  Propulsion: 312.5 ms
  Landing: 425.8 ms
```

### Manual Phase Detection

For custom analysis without JumpTest class:

```python
# Load and filter force
fz = fp.force['Fz']
freq = fz.sampling_frequency

fz_filtered = laban.butterworth_filt(
    fz.data,
    freq=freq,
    cut=10,
    order=4,
    filt_type='low'
)

# Calculate bodyweight (first 2 seconds)
bodyweight = fz_filtered[:int(2*freq)].mean()

# Detect phases using threshold (10% bodyweight)
threshold = bodyweight * 0.10
in_contact = fz_filtered > threshold

# Find transitions
transitions = np.diff(in_contact.astype(int))
takeoff_idx = np.where(transitions == -1)[0][0]  # Contact → flight
landing_idx = np.where(transitions == 1)[0][0]   # Flight → contact

# Find unweighting start (first time below 95% BW)
unweight_threshold = bodyweight * 0.95
unweight_start = np.where(fz_filtered[:takeoff_idx] < unweight_threshold)[0][0]

# Calculate metrics
flight_time = (landing_idx - takeoff_idx) / freq
jump_height = 0.5 * 9.81 * (flight_time / 2) ** 2

print(f"Takeoff: {takeoff_idx/freq:.3f} s")
print(f"Landing: {landing_idx/freq:.3f} s")
print(f"Flight time: {flight_time*1000:.1f} ms")
print(f"Jump height: {jump_height*100:.1f} cm")
```

### CMJ Metrics Interpretation

| Metric | Good (Male) | Elite (Male) | Good (Female) | Elite (Female) |
|--------|-------------|--------------|---------------|----------------|
| Jump Height | 35-45 cm | >50 cm | 25-35 cm | >40 cm |
| Peak Power | 40-50 W/kg | >60 W/kg | 30-40 W/kg | >50 W/kg |
| Peak Force | 2.0-2.5 BW | >3.0 BW | 1.8-2.2 BW | >2.5 BW |
| Peak RFD | 5000-8000 N/s | >10000 N/s | 4000-6000 N/s | >8000 N/s |

## Squat Jump (SJ)

### Protocol

SJ eliminates the countermovement to assess concentric-only power:

1. **Static squat position** (2 seconds): Hold at ~90° knee flexion
2. **Propulsion**: Maximal upward push without countermovement
3. **Flight**: Ballistic phase
4. **Landing**: Ground contact

### Analysis Workflow

```python
# Load SJ data
record = laban.TimeseriesRecord.from_tdf("sj.tdf")
fp = record['FP1']

# Create SJ test
test = laban.JumpTest(
    participant=participant,
    protocol='SJ',
    force_platform=fp
)

results = test.analyze()

print("=== SJ Results ===")
print(f"Jump Height: {results.jump_height*100:.1f} cm")
print(f"Peak Power: {results.peak_power:.1f} W ({results.peak_power_rel:.1f} W/kg)")

# Compare to CMJ (if available)
if cmj_results:
    eccentric_utilization = (cmj_results.jump_height - results.jump_height) / results.jump_height * 100
    print(f"\nEccentric Utilization Ratio: {eccentric_utilization:.1f}%")
```

**Expected:**
- SJ height typically 5-15% lower than CMJ
- Eccentric Utilization Ratio (EUR) = (CMJ - SJ) / SJ × 100
- EUR > 10% indicates good stretch-shortening cycle function

## Drop Jump (DJ)

### Protocol

DJ assesses reactive strength and plyometric ability:

1. **Drop** from elevated platform (20-40 cm)
2. **Landing**: Ground contact with eccentric loading
3. **Propulsion**: Minimal ground contact time
4. **Rebound flight**: Maximal height jump
5. **Final landing**: Ground contact

### Analysis Workflow

```python
# Load DJ data
record = laban.TimeseriesRecord.from_tdf("dj_30cm.tdf")
fp = record['FP1']

# Create DJ test (specify drop height)
test = laban.JumpTest(
    participant=participant,
    protocol='DJ',
    force_platform=fp,
    drop_height=0.30  # 30 cm drop
)

results = test.analyze()

print("=== Drop Jump Results (30 cm) ===")
print(f"Jump Height: {results.jump_height*100:.1f} cm")
print(f"Contact Time: {results.contact_time*1000:.1f} ms")
print(f"RSI: {results.rsi:.2f}")
print(f"Peak Force: {results.peak_force:.1f} N ({results.peak_force_rel:.2f} BW)")
```

**Output:**
```
=== Drop Jump Results (30 cm) ===
Jump Height: 28.5 cm
Contact Time: 245.3 ms
RSI: 1.16
Peak Force: 2345.7 N (3.41 BW)
```

### Reactive Strength Index (RSI)

RSI quantifies reactive ability:

```
RSI = Jump Height (m) / Contact Time (s)
```

```python
# Calculate RSI manually
jump_height_m = 0.285
contact_time_s = 0.245

rsi = jump_height_m / contact_time_s
print(f"RSI: {rsi:.2f}")  # Output: RSI: 1.16
```

**Interpretation:**
- RSI < 0.5: Poor reactive strength
- RSI 0.5-1.0: Moderate reactive strength
- RSI 1.0-2.0: Good reactive strength
- RSI > 2.0: Elite reactive strength

**Optimal drop height**: Maximize RSI, typically 20-40 cm for most athletes

### Drop Height Optimization

```python
# Test multiple drop heights
drop_heights = [0.20, 0.30, 0.40, 0.50]  # meters
results_list = []

for height in drop_heights:
    # Load corresponding file
    record = laban.TimeseriesRecord.from_tdf(f"dj_{int(height*100)}cm.tdf")
    fp = record['FP1']
    
    # Analyze
    test = laban.JumpTest(
        participant=participant,
        protocol='DJ',
        force_platform=fp,
        drop_height=height
    )
    results = test.analyze()
    results_list.append(results)
    
    print(f"Drop {int(height*100)} cm: RSI = {results.rsi:.2f}, "
          f"Height = {results.jump_height*100:.1f} cm, "
          f"Contact = {results.contact_time*1000:.1f} ms")

# Find optimal
optimal_idx = np.argmax([r.rsi for r in results_list])
optimal_height = drop_heights[optimal_idx]
print(f"\nOptimal drop height: {int(optimal_height*100)} cm")
```

**Output:**
```
Drop 20 cm: RSI = 0.98, Height = 26.3 cm, Contact = 268.4 ms
Drop 30 cm: RSI = 1.16, Height = 28.5 cm, Contact = 245.3 ms
Drop 40 cm: RSI = 1.08, Height = 27.2 cm, Contact = 251.8 ms
Drop 50 cm: RSI = 0.89, Height = 24.8 cm, Contact = 278.5 ms

Optimal drop height: 30 cm
```

## Repeated Jumps

### Protocol

Continuous jumping for fatigue assessment:

- **Duration**: 15-30 seconds continuous jumping
- **Instruction**: "Jump as high as possible with minimal ground contact"
- **Analysis**: Track jump height decline over time

### Analysis Workflow

```python
# Load repeated jump trial
record = laban.TimeseriesRecord.from_tdf("repeated_jumps.tdf")
fp = record['FP1']

# Create test
test = laban.JumpTest(
    participant=participant,
    protocol='RepeatedJumps',
    force_platform=fp,
    duration=15  # seconds
)

results = test.analyze()

print("=== Repeated Jumps (15s) ===")
print(f"Number of jumps: {results.n_jumps}")
print(f"Average height: {results.avg_jump_height*100:.1f} cm")
print(f"Best jump: {results.max_jump_height*100:.1f} cm")
print(f"Worst jump: {results.min_jump_height*100:.1f} cm")
print(f"Fatigue index: {results.fatigue_index:.1f}%")
print(f"Average contact time: {results.avg_contact_time*1000:.1f} ms")
print(f"Average RSI: {results.avg_rsi:.2f}")
```

**Output:**
```
=== Repeated Jumps (15s) ===
Number of jumps: 23
Average height: 28.3 cm
Best jump: 32.1 cm
Worst jump: 24.5 cm
Fatigue index: 23.7%
Average contact time: 287.5 ms
Average RSI: 0.98
```

### Fatigue Index Calculation

```python
# Fatigue index = decline from first to last jump
fatigue_index = (results.first_jump_height - results.last_jump_height) / results.first_jump_height * 100

# Alternative: Compare best 3 to worst 3
best_3 = np.mean(sorted(results.all_jump_heights, reverse=True)[:3])
worst_3 = np.mean(sorted(results.all_jump_heights)[:3])
fatigue_index_alt = (best_3 - worst_3) / best_3 * 100

print(f"Fatigue index (first vs last): {fatigue_index:.1f}%")
print(f"Fatigue index (best 3 vs worst 3): {fatigue_index_alt:.1f}%")
```

## Complete Workflow: Jump Battery

Test all three jump types in one session:

```python
import labanalysis as laban
import pandas as pd

# Setup participant
participant = laban.Participant(
    name="Athlete",
    surname="001",
    height=1.75,
    weight=70,
    age=25
)

# Protocols to test
protocols = ['SJ', 'CMJ', 'DJ']
results_dict = {}

# Analyze each jump type
for protocol in protocols:
    # Load data
    record = laban.TimeseriesRecord.from_tdf(f"{protocol.lower()}_trial.tdf")
    fp = record['FP1']
    
    # Create test
    test = laban.JumpTest(
        participant=participant,
        protocol=protocol,
        force_platform=fp,
        drop_height=0.30 if protocol == 'DJ' else None
    )
    
    # Analyze
    results = test.analyze()
    results_dict[protocol] = results
    
    print(f"\n{protocol} Results:")
    print(f"  Jump Height: {results.jump_height*100:.1f} cm")
    print(f"  Peak Power: {results.peak_power_rel:.1f} W/kg")

# Compare protocols
print("\n=== Protocol Comparison ===")
print(f"CMJ vs SJ height: {(results_dict['CMJ'].jump_height - results_dict['SJ'].jump_height)*100:.1f} cm")
print(f"Eccentric utilization: {((results_dict['CMJ'].jump_height - results_dict['SJ'].jump_height) / results_dict['SJ'].jump_height * 100):.1f}%")
print(f"DJ RSI: {results_dict['DJ'].rsi:.2f}")

# Export to Excel
df = pd.DataFrame({
    'Protocol': protocols,
    'Jump Height (cm)': [results_dict[p].jump_height*100 for p in protocols],
    'Peak Power (W/kg)': [results_dict[p].peak_power_rel for p in protocols],
    'Peak Force (BW)': [results_dict[p].peak_force_rel for p in protocols]
})

df.to_excel("jump_battery_results.xlsx", index=False)
print("\nResults exported to jump_battery_results.xlsx")
```

## Visualization

### Force-Time Curve

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load CMJ data
record = laban.TimeseriesRecord.from_tdf("cmj.tdf")
fp = record['FP1']
fz = fp.force['Fz']

# Filter
fz_filt = laban.butterworth_filt(fz.data, freq=fz.sampling_frequency, cut=10, order=4)
time = np.arange(len(fz_filt)) / fz.sampling_frequency

# Analyze to get phases
test = laban.JumpTest(participant=participant, protocol='CMJ', force_platform=fp)
results = test.analyze()

# Create figure
fig = go.Figure()

# Force trace
fig.add_trace(go.Scatter(
    x=time,
    y=fz_filt,
    mode='lines',
    name='Vertical Force',
    line=dict(color='blue', width=2)
))

# Mark phases
colors = {'unweighting': 'orange', 'propulsion': 'green', 'flight': 'red', 'landing': 'purple'}
for phase_name, color in colors.items():
    phase = getattr(results, f'{phase_name}_phase', None)
    if phase:
        start_idx, end_idx = phase
        fig.add_vrect(
            x0=time[start_idx],
            x1=time[end_idx],
            fillcolor=color,
            opacity=0.2,
            layer='below',
            line_width=0,
            annotation_text=phase_name.capitalize(),
            annotation_position='top left'
        )

# Bodyweight line
bodyweight = fz_filt[:int(2*fz.sampling_frequency)].mean()
fig.add_hline(y=bodyweight, line_dash='dash', line_color='gray', annotation_text='Bodyweight')

fig.update_layout(
    title='CMJ Force-Time Curve',
    xaxis_title='Time (s)',
    yaxis_title='Force (N)',
    hovermode='x unified',
    height=500
)

fig.show()
```

### Power-Time Curve

```python
# Calculate velocity and power
velocity = laban.winter_derivative1(fz_filt, freq=fz.sampling_frequency) / participant.weight
power = fz_filt * velocity

# Create subplot figure
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Force', 'Velocity', 'Power'),
    shared_xaxes=True,
    vertical_spacing=0.08
)

# Force
fig.add_trace(go.Scatter(x=time, y=fz_filt, name='Force', line=dict(color='blue')), row=1, col=1)

# Velocity
fig.add_trace(go.Scatter(x=time, y=velocity, name='Velocity', line=dict(color='green')), row=2, col=1)

# Power
fig.add_trace(go.Scatter(x=time, y=power, name='Power', line=dict(color='red')), row=3, col=1)

# Mark peak power
peak_power_idx = np.argmax(power)
fig.add_vline(x=time[peak_power_idx], line_dash='dash', line_color='red')

fig.update_yaxes(title_text='Force (N)', row=1, col=1)
fig.update_yaxes(title_text='Velocity (m/s)', row=2, col=1)
fig.update_yaxes(title_text='Power (W)', row=3, col=1)
fig.update_xaxes(title_text='Time (s)', row=3, col=1)

fig.update_layout(height=800, showlegend=False, title='CMJ Kinetics')
fig.show()
```

### Jump Height Comparison

```python
# Compare multiple athletes or trials
jump_data = {
    'Athlete 1': 0.327,
    'Athlete 2': 0.298,
    'Athlete 3': 0.412,
    'Athlete 4': 0.356,
    'Athlete 5': 0.289
}

fig = go.Figure(data=[
    go.Bar(
        x=list(jump_data.keys()),
        y=[h*100 for h in jump_data.values()],  # Convert to cm
        marker_color='steelblue'
    )
])

# Add normative band
fig.add_hrect(y0=35, y1=45, fillcolor='green', opacity=0.2, annotation_text='Good')

fig.update_layout(
    title='CMJ Jump Height Comparison',
    xaxis_title='Athlete',
    yaxis_title='Jump Height (cm)',
    showlegend=False
)

fig.show()
```

## Troubleshooting

### Phase Detection Fails

**Problem**: Cannot detect takeoff/landing

**Solution**: Adjust threshold
```python
# If bodyweight threshold too strict
threshold = bodyweight * 0.05  # Try 5% instead of 10%

# Or use absolute threshold
threshold = 20  # N
```

### Unrealistic Jump Heights

**Problem**: Jump height > 1 meter or < 0

**Cause**: Incorrect phase detection or force platform calibration

**Solution**: Validate phases visually
```python
# Plot force with detected phases
import matplotlib.pyplot as plt

plt.plot(fz_filtered)
plt.axvline(takeoff_idx, color='green', label='Takeoff')
plt.axvline(landing_idx, color='red', label='Landing')
plt.axhline(bodyweight, color='gray', linestyle='--', label='Bodyweight')
plt.legend()
plt.show()

# Check if phases make sense
print(f"Flight time: {(landing_idx - takeoff_idx) / freq * 1000:.1f} ms")
# Should be 200-600 ms for typical jumps
```

### Multiple Jumps in One File

**Problem**: File contains several jumps, not just one

**Solution**: Segment file or analyze first jump only
```python
# Find first takeoff
in_contact = fz_filtered > bodyweight * 0.10
transitions = np.diff(in_contact.astype(int))
all_takeoffs = np.where(transitions == -1)[0]

# Analyze only first jump
first_takeoff = all_takeoffs[0]
first_landing = np.where(transitions[first_takeoff:] == 1)[0][0] + first_takeoff

# Extract segment
jump_segment = fz_filtered[first_takeoff-1000:first_landing+1000]  # With padding
```

## See Also

- **[Tutorial: Jump Analysis](../../tutorials/01-jump-analysis.md)** - Complete CMJ workflow
- **[Force Platforms](../biomechanics/force-platforms.md)** - Force platform data guide
- **[Signal Processing](../signal-processing/README.md)** - Pre-processing techniques
- **[API Reference: Jump Tests](../../api-reference/protocols/jump-tests.md)** - Complete API

---

**References:**
- Linthorne NP (2001). Analysis of standing vertical jumps using a force platform. *Am J Phys* 69(11):1198-1204
- Moir GL (2008). Three different methods of calculating vertical jump height from force platform data. *Meas Phys Educ Exerc Sci* 12(4):207-218
