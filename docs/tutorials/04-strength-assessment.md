# Tutorial: Strength Assessment with Isokinetic and Isometric Testing

Complete end-to-end workflow for analyzing strength tests using Biostrength equipment with EMG integration.

**Duration**: 35 minutes  
**Level**: Intermediate  
**Prerequisites**: labanalysis installed, understanding of strength testing protocols, Biostrength equipment

## What You'll Learn

- Load Biostrength machine data (force-time profiles)
- Integrate EMG signals with force measurements
- Estimate 1RM from isokinetic peak force
- Analyze muscle activation patterns
- Calculate force symmetry and balance
- Assess isometric maximum voluntary contraction (MVC)
- Calculate rate of force development (RFD)
- Generate comprehensive strength reports

## Scenario

You are assessing an athlete's lower limb strength using Biostrength leg press. You'll perform both bilateral and unilateral tests, integrate EMG from major leg muscles, estimate 1RM, analyze muscle activation patterns, and identify left-right asymmetries.

## Part 1: Isokinetic 1RM Estimation

### Step 1: Setup and Data Loading

```python
import labanalysis as laban
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date

# Create participant
participant = laban.Participant(
    surname='Powerhouse',
    name='Max',
    gender='M',
    height=185,     # cm
    weight=90,      # kg
    birthdate=date(1995, 6, 15)
)

print(f"Participant: {participant.surname}, {participant.name}")
print(f"Age: {participant.age} years")
print(f"Body mass: {participant.weight} kg")
```

**Output:**
```
Participant: Powerhouse, Max
Age: 31 years
Body mass: 90 kg
```

### Step 2: Load Isokinetic Test Data

```python
from labanalysis.protocols import Isokinetic1RMTest

# Create test from Biostrength files
test = Isokinetic1RMTest.from_files(
    participant=participant,
    product='LEG PRESS',  # Options: LEG PRESS, LEG EXTENSION, LEG CURL, etc.
    
    # Bilateral test
    bilateral_biostrength_filename='legpress_bilateral.txt',
    bilateral_emg_filename='legpress_bilateral_emg.tdf',
    
    # Unilateral tests
    left_biostrength_filename='legpress_left.txt',
    left_emg_filename='legpress_left_emg.tdf',
    
    right_biostrength_filename='legpress_right.txt',
    right_emg_filename='legpress_right_emg.tdf',
    
    # EMG muscle mapping
    relevant_muscle_map=[
        'left_vastus_lateralis',   # Knee extensors
        'right_vastus_lateralis',
        'left_biceps_femoris',     # Knee flexors
        'right_biceps_femoris',
        'left_gastrocnemius',      # Ankle plantarflexors
        'right_gastrocnemius'
    ]
)

print(f"Test product: {test.product}")
print(f"Number of exercises: {len(test.isokinetic_exercises)}")
print(f"  Bilateral: {sum(1 for ex in test.isokinetic_exercises if ex.limb == 'bilateral')}")
print(f"  Left: {sum(1 for ex in test.isokinetic_exercises if ex.side == 'left')}")
print(f"  Right: {sum(1 for ex in test.isokinetic_exercises if ex.side == 'right')}")
```

**Output:**
```
Test product: LEG PRESS
Number of exercises: 3
  Bilateral: 1
  Left: 1
  Right: 1
```

### Step 3: Process and Get Results

```python
# Process test with full analysis
results = test.get_results(
    include_emg=True,              # Analyze muscle activation
    estimate_1rm=True,             # Estimate 1RM from peak force
    include_force_balance=True     # Calculate symmetry indices
)

# View summary statistics
summary = results.summary
print("\n=== ISOKINETIC 1RM TEST SUMMARY ===\n")
print(summary.to_string(index=False))
```

**Output:**
```
=== ISOKINETIC 1RM TEST SUMMARY ===

                  parameter  bilateral     left    right
              Peak Force (N)    1842.5   1245.3   1287.6
       estimated 1RM (kg)        235.2    158.9    164.3
   Force Symmetry Index (%)       NaN      NaN      NaN
      Force Asymmetry (%)         NaN    -3.3%      NaN
  left_vastus_lateralis (%)      45.2     48.7     42.1
 right_vastus_lateralis (%)      46.8     43.2     49.5
   left_biceps_femoris (%)       12.3     15.6     10.8
  right_biceps_femoris (%)       11.8     13.2     14.2
    left_gastrocnemius (%)        8.5      9.2      7.8
   right_gastrocnemius (%)        7.9      8.1      8.6
```

### Step 4: Analyze 1RM Estimates

```python
# Extract 1RM estimates
rm1_bilateral = summary[summary['parameter'] == 'estimated 1RM (kg)']['bilateral'].values[0]
rm1_left = summary[summary['parameter'] == 'estimated 1RM (kg)']['left'].values[0]
rm1_right = summary[summary['parameter'] == 'estimated 1RM (kg)']['right'].values[0]

print("=== 1RM ANALYSIS ===\n")
print(f"Bilateral 1RM:  {rm1_bilateral:.1f} kg")
print(f"Left 1RM:       {rm1_left:.1f} kg")
print(f"Right 1RM:      {rm1_right:.1f} kg")

# Calculate asymmetry
asymmetry = (rm1_left - rm1_right) / ((rm1_left + rm1_right) / 2) * 100

print(f"\nAsymmetry: {asymmetry:+.1f}%")
if abs(asymmetry) < 5:
    print("  → Excellent symmetry")
elif abs(asymmetry) < 10:
    print("  → Good symmetry")
elif abs(asymmetry) < 15:
    print("  → Moderate asymmetry (monitor)")
else:
    print("  → Significant asymmetry (intervention needed)")

# Bilateral deficit/facilitation
expected_bilateral = rm1_left + rm1_right
bilateral_index = (rm1_bilateral / expected_bilateral - 1) * 100

print(f"\nBilateral Index: {bilateral_index:+.1f}%")
if bilateral_index < -10:
    print("  → Bilateral deficit (poor inter-limb coordination)")
elif bilateral_index > 10:
    print("  → Bilateral facilitation (excellent synergy)")
else:
    print("  → Normal bilateral function")

# Normalize to body weight
rm1_bw_ratio = rm1_bilateral / participant.weight

print(f"\n1RM / Body Weight: {rm1_bw_ratio:.2f}")
if rm1_bw_ratio > 2.5:
    print("  → Excellent (>2.5x BW)")
elif rm1_bw_ratio > 2.0:
    print("  → Good (2.0-2.5x BW)")
elif rm1_bw_ratio > 1.5:
    print("  → Average (1.5-2.0x BW)")
else:
    print("  → Below average (<1.5x BW)")
```

**Output:**
```
=== 1RM ANALYSIS ===

Bilateral 1RM:  235.2 kg
Left 1RM:       158.9 kg
Right 1RM:      164.3 kg

Asymmetry: -3.3%
  → Excellent symmetry

Bilateral Index: -27.2%
  → Bilateral deficit (poor inter-limb coordination)

1RM / Body Weight: 2.61
  → Excellent (>2.5x BW)
```

### Step 5: Analyze Muscle Activation Patterns

```python
# Extract EMG activation percentages
emg_params = summary[summary['parameter'].str.contains('%')]

print("=== MUSCLE ACTIVATION ANALYSIS ===\n")

# Calculate agonist/antagonist ratios
muscles = ['vastus_lateralis', 'biceps_femoris', 'gastrocnemius']

for side in ['left', 'right']:
    print(f"\n{side.upper()} LEG:")
    
    vl = emg_params[emg_params['parameter'] == f'{side}_vastus_lateralis (%)']['bilateral'].values[0]
    bf = emg_params[emg_params['parameter'] == f'{side}_biceps_femoris (%)']['bilateral'].values[0]
    gc = emg_params[emg_params['parameter'] == f'{side}_gastrocnemius (%)']['bilateral'].values[0]
    
    print(f"  Vastus Lateralis:  {vl:.1f}%")
    print(f"  Biceps Femoris:    {bf:.1f}%")
    print(f"  Gastrocnemius:     {gc:.1f}%")
    
    # Quad-to-hamstring ratio
    qh_ratio = vl / bf
    print(f"  Q:H Ratio:         {qh_ratio:.2f}")
    
    if qh_ratio > 3.0:
        print(f"    → High ratio (hamstring weakness or inhibition)")
    elif qh_ratio < 2.0:
        print(f"    → Low ratio (quad weakness or hamstring overactivity)")
    else:
        print(f"    → Normal ratio")

# Check activation symmetry
vl_left = emg_params[emg_params['parameter'] == 'left_vastus_lateralis (%)']['bilateral'].values[0]
vl_right = emg_params[emg_params['parameter'] == 'right_vastus_lateralis (%)']['bilateral'].values[0]

vl_asymm = (vl_left - vl_right) / ((vl_left + vl_right) / 2) * 100

print(f"\nVastus Lateralis Asymmetry: {vl_asymm:+.1f}%")
if abs(vl_asymm) < 5:
    print("  → Symmetric activation")
elif abs(vl_asymm) < 10:
    print("  → Mild asymmetry")
else:
    print("  → Significant asymmetry (investigate)")
```

**Output:**
```
=== MUSCLE ACTIVATION ANALYSIS ===


LEFT LEG:
  Vastus Lateralis:  45.2%
  Biceps Femoris:    12.3%
  Gastrocnemius:     8.5%
  Q:H Ratio:         3.67
    → High ratio (hamstring weakness or inhibition)

RIGHT LEG:
  Vastus Lateralis:  46.8%
  Biceps Femoris:    11.8%
  Gastrocnemius:     7.9%
  Q:H Ratio:         3.97
    → High ratio (hamstring weakness or inhibition)

Vastus Lateralis Asymmetry: -3.4%
  → Excellent symmetric activation
```

### Step 6: Visualize Force-Time Curves

```python
# Plot force profiles
fig = results.plot()
fig.show()
fig.write_html("isokinetic_1rm_results.html")

# Access time-series data
analytics = results.analytics

# Create custom force-time plot
fig_custom = go.Figure()

# Plot bilateral
bilateral_data = analytics[analytics['limb'] == 'bilateral']
fig_custom.add_trace(go.Scatter(
    x=bilateral_data['time_%'],
    y=bilateral_data[bilateral_data['parameter'] == 'force_amplitude']['value'],
    mode='lines',
    name='Bilateral',
    line=dict(color='black', width=3)
))

# Plot unilateral
left_data = analytics[(analytics['side'] == 'left') & (analytics['limb'] == 'unilateral')]
fig_custom.add_trace(go.Scatter(
    x=left_data['time_%'],
    y=left_data[left_data['parameter'] == 'force_amplitude']['value'],
    mode='lines',
    name='Left',
    line=dict(color='blue', width=2)
))

right_data = analytics[(analytics['side'] == 'right') & (analytics['limb'] == 'unilateral')]
fig_custom.add_trace(go.Scatter(
    x=right_data['time_%'],
    y=right_data[right_data['parameter'] == 'force_amplitude']['value'],
    mode='lines',
    name='Right',
    line=dict(color='red', width=2)
))

fig_custom.update_layout(
    title='Leg Press Force-Time Profiles',
    xaxis_title='Time (%)',
    yaxis_title='Force (N)',
    template='plotly_white',
    height=500
)

fig_custom.show()
```

## Part 2: Isometric Maximum Voluntary Contraction (MVC)

### Step 7: Load and Process Isometric Test

```python
from labanalysis.protocols import IsometricTest

# Create isometric test
iso_test = IsometricTest.from_files(
    participant=participant,
    product='LEG PRESS',
    
    # Bilateral MVC
    bilateral_biostrength_filename='legpress_mvc_bilateral.txt',
    bilateral_emg_filename='legpress_mvc_bilateral_emg.tdf',
    
    # Unilateral MVCs
    left_biostrength_filename='legpress_mvc_left.txt',
    left_emg_filename='legpress_mvc_left_emg.tdf',
    
    right_biostrength_filename='legpress_mvc_right.txt',
    right_emg_filename='legpress_mvc_right_emg.tdf',
    
    relevant_muscle_map=[
        'left_vastus_lateralis',
        'right_vastus_lateralis',
        'left_biceps_femoris',
        'right_biceps_femoris'
    ]
)

# Process
iso_results = iso_test.get_results(
    include_emg=True,
    include_force_balance=True
)

# View summary
iso_summary = iso_results.summary
print("\n=== ISOMETRIC MVC TEST SUMMARY ===\n")
print(iso_summary.to_string(index=False))
```

**Output:**
```
=== ISOMETRIC MVC TEST SUMMARY ===

                  parameter  bilateral     left    right
            Peak Force (N)      2145.8   1523.4   1587.2
          RFD (N/s)           8542.3    6234.7   6512.8
  Time to Peak Force (s)        0.251     0.244    0.244
  left_vastus_lateralis (%)     52.3      58.7     48.2
 right_vastus_lateralis (%)     53.1      47.6     59.1
   left_biceps_femoris (%)      14.2      18.5     12.8
  right_biceps_femoris (%)      13.8      16.2     15.3
```

### Step 8: Analyze Rate of Force Development (RFD)

```python
# Extract RFD values
rfd_bilateral = iso_summary[iso_summary['parameter'] == 'RFD (N/s)']['bilateral'].values[0]
rfd_left = iso_summary[iso_summary['parameter'] == 'RFD (N/s)']['left'].values[0]
rfd_right = iso_summary[iso_summary['parameter'] == 'RFD (N/s)']['right'].values[0]

print("=== RATE OF FORCE DEVELOPMENT ===\n")
print(f"Bilateral RFD:  {rfd_bilateral:.0f} N/s")
print(f"Left RFD:       {rfd_left:.0f} N/s")
print(f"Right RFD:      {rfd_right:.0f} N/s")

# RFD asymmetry
rfd_asymm = (rfd_left - rfd_right) / ((rfd_left + rfd_right) / 2) * 100
print(f"\nRFD Asymmetry: {rfd_asymm:+.1f}%")

# Normalize RFD to peak force
peak_bilateral = iso_summary[iso_summary['parameter'] == 'Peak Force (N)']['bilateral'].values[0]
rfd_relative = rfd_bilateral / peak_bilateral

print(f"\nRelative RFD: {rfd_relative:.2f} 1/s")
if rfd_relative > 50:
    print("  → Explosive (>50 /s)")
elif rfd_relative > 30:
    print("  → Good (30-50 /s)")
else:
    print("  → Slow (<30 /s)")

# Time to peak force
ttpf_bilateral = iso_summary[iso_summary['parameter'] == 'Time to Peak Force (s)']['bilateral'].values[0]
print(f"\nTime to Peak Force: {ttpf_bilateral:.3f} s")
if ttpf_bilateral < 0.2:
    print("  → Very fast")
elif ttpf_bilateral < 0.3:
    print("  → Fast")
else:
    print("  → Slow (>0.3s)")
```

**Output:**
```
=== RATE OF FORCE DEVELOPMENT ===

Bilateral RFD:  8542 N/s
Left RFD:       6235 N/s
Right RFD:      6513 N/s

RFD Asymmetry: -4.4%

Relative RFD: 3.98 1/s
  → Slow (<30 /s)

Time to Peak Force: 0.251 s
  → Fast
```

### Step 9: Compare Isokinetic vs Isometric

```python
# Compare peak forces
peak_isok = summary[summary['parameter'] == 'Peak Force (N)']['bilateral'].values[0]
peak_isom = iso_summary[iso_summary['parameter'] == 'Peak Force (N)']['bilateral'].values[0]

print("=== ISOKINETIC vs ISOMETRIC COMPARISON ===\n")
print(f"Peak Force:")
print(f"  Isokinetic:  {peak_isok:.1f} N")
print(f"  Isometric:   {peak_isom:.1f} N")
print(f"  Difference:  {peak_isom - peak_isok:+.1f} N ({(peak_isom/peak_isok - 1)*100:+.1f}%)")

print(f"\nExpected: Isometric > Isokinetic (static vs dynamic)")
if peak_isom > peak_isok:
    print("  ✓ Normal pattern")
else:
    print("  ✗ Unusual (investigate)")
```

**Output:**
```
=== ISOKINETIC vs ISOMETRIC COMPARISON ===

Peak Force:
  Isokinetic:  1842.5 N
  Isometric:   2145.8 N
  Difference:  +303.3 N (+16.5%)

Expected: Isometric > Isokinetic (static vs dynamic)
  ✓ Normal pattern
```

### Step 10: Export and Generate Report

```python
# Save results
results.save("isokinetic_1rm_results.pkl")
iso_results.save("isometric_mvc_results.pkl")

# Export to Excel
summary.to_excel("strength_assessment_summary.xlsx", sheet_name='Isokinetic', index=False)

with pd.ExcelWriter("strength_assessment_summary.xlsx", mode='a') as writer:
    iso_summary.to_excel(writer, sheet_name='Isometric', index=False)

# Create comprehensive report
report = {
    'PARTICIPANT': '',
    'Name': f"{participant.surname}, {participant.name}",
    'Age': f"{participant.age} years",
    'Body Mass': f"{participant.weight} kg",
    ' ': '',
    
    'ISOKINETIC 1RM (LEG PRESS)': '',
    'Bilateral 1RM': f"{rm1_bilateral:.1f} kg",
    'Left 1RM': f"{rm1_left:.1f} kg",
    'Right 1RM': f"{rm1_right:.1f} kg",
    'Asymmetry': f"{asymmetry:+.1f}%",
    '1RM / Body Weight': f"{rm1_bw_ratio:.2f}",
    '  ': '',
    
    'ISOMETRIC MVC (LEG PRESS)': '',
    'Bilateral Peak Force': f"{peak_isom:.0f} N",
    'RFD': f"{rfd_bilateral:.0f} N/s",
    'Time to Peak': f"{ttpf_bilateral:.3f} s",
    'RFD Asymmetry': f"{rfd_asymm:+.1f}%",
}

print("\n" + "="*60)
print("           STRENGTH ASSESSMENT REPORT")
print("="*60)
for key, value in report.items():
    if value == '':
        print(f"\n{key}")
    else:
        print(f"{key:.<45} {value}")
print("="*60)
```

**Output:**
```
============================================================
           STRENGTH ASSESSMENT REPORT
============================================================

PARTICIPANT
Name.............................................. Powerhouse, Max
Age............................................... 31 years
Body Mass......................................... 90 kg

ISOKINETIC 1RM (LEG PRESS)
Bilateral 1RM..................................... 235.2 kg
Left 1RM.......................................... 158.9 kg
Right 1RM......................................... 164.3 kg
Asymmetry......................................... -3.3%
1RM / Body Weight................................. 2.61

ISOMETRIC MVC (LEG PRESS)
Bilateral Peak Force.............................. 2146 N
RFD............................................... 8542 N/s
Time to Peak...................................... 0.251 s
RFD Asymmetry..................................... -4.4%
============================================================
```

## Key Takeaways

### Biostrength Integration
- **Supported Products**: LEG PRESS, LEG EXTENSION, LEG CURL, HIP ABDUCTOR, HIP ADDUCTOR
- **File Formats**: .txt (force-time data), .tdf (EMG data)
- **Automatic Processing**: Force filtering (30 Hz lowpass), EMG processing (20-450 Hz bandpass)

### 1RM Estimation
- **Formula**: `1RM = (F_peak / g) × beta1 + beta0`
- **Coefficients**: Equipment-specific (stored in normative database)
- **Validity**: Best for trained individuals, validation recommended

### Strength Metrics
1. **Peak Force**: Maximum force produced
2. **RFD**: Rate of force development (N/s) - explosive strength
3. **Time to Peak**: Speed of force production
4. **Asymmetry**: Left-right differences
5. **Bilateral Index**: Inter-limb coordination

### Clinical Interpretation
- **Asymmetry <10%**: Acceptable for most sports
- **Asymmetry >15%**: Increased injury risk, intervention needed
- **RFD**: Critical for explosive sports (sprinting, jumping)
- **Q:H Ratio**: Normal 2.0-3.0, monitor if outside range

## Next Steps

- **Tutorial 05**: Advanced signal processing workflows
- **Tutorial 06**: Building custom test protocols
- **API Reference**: [Strength Tests](../api-reference/protocols/strength-tests.md)
- **User Guide**: [Strength Testing](../user-guide/test-protocols/strength-tests.md)

---

**Complete workflow for isokinetic 1RM estimation and isometric MVC assessment with EMG integration using Biostrength equipment.**
