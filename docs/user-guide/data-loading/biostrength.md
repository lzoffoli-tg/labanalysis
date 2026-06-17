# Biostrength Data Loading

Guide to loading data from Technogym Biostrength strength training equipment.

## Overview

Biostrength is Technogym's line of intelligent strength training machines that record motor position, load (torque), and timing data during exercise. labanalysis provides dedicated classes for each machine type with automatic conversion of raw motor data to biomechanically meaningful units (load in kgf, position in meters/degrees, speed, power).

**Supported Machines:**
- Chest Press
- Shoulder Press  
- Low Row
- Leg Press (standard and REV)
- Leg Extension (standard and REV)
- Leg Curl
- Adjustable Pulley REV

Each machine has specific calibration coefficients for spring correction, pulley radius, lever weight, and camme ratio.

## Quick Reference

```python
import labanalysis as laban

# Load data from TXT file
leg_press = laban.LegPress.from_txt_file("leg_press_trial.txt")

# Access biomechanical data
load_kgf = leg_press.load_kgf          # Load in kgf
position_m = leg_press.position_lever_m  # Position in meters
position_deg = leg_press.position_lever_deg  # Position in degrees
speed_ms = leg_press.speed_lever_ms     # Speed in m/s
power_w = leg_press.power_w             # Power in Watts

# Convert to DataFrame
df = leg_press.as_dataframe()
print(df.head())
#    Time (s)  Load (kgf)  Position (m)  Position (deg)  Speed (m/s)  Power (W)
# 0     0.00       45.23         0.123           45.6        0.000       0.0
# 1     0.01       46.12         0.125           46.2        0.020      23.1

# Extract time window
rep = leg_press.slice(start_time=2.0, stop_time=4.5)
```

## File Format

Biostrength machines export data as pipe-delimited text files (.txt or .csv):

```
Time|Counter|Position|Mode|Set|Torque|Speed|...
0.00|0|0.0|1|0|0.0|0.0|...
0.01|1|0.015|1|0|12.5|0.3|...
0.02|2|0.032|1|0|25.3|0.6|...
```

**Key Columns:**
- **Time**: Timestamp in seconds
- **Position**: Motor position in radians
- **Torque**: Motor torque in Nm (raw, requires calibration)

## Machine Classes

### Leg Press

```python
import labanalysis as laban

# Load data
leg_press = laban.LegPress.from_txt_file("data.txt")

# Machine-specific properties
print(f"Lever weight: {leg_press.lever_weight_kgf[0]:.1f} kgf")
print(f"Spring correction: {leg_press.spring_correction[0]:.2f}")

# Analyze 1RM
max_load = leg_press.load_kgf.max()
print(f"Peak load: {max_load:.1f} kgf")

# 1RM prediction (Brzycki coefficients)
reps = 8
predicted_1rm = leg_press.rm1_coefs[0] * max_load + leg_press.rm1_coefs[1]
print(f"Predicted 1RM ({reps} reps): {predicted_1rm:.1f} kgf")
```

**Calibration Parameters:**
- Pulley radius: 81.75 mm
- Lever weight: 9.0 + 0.17 × 85 kgf (body weight dependent)
- Spring correction: 1.0
- Camme ratio: 1.0

### Leg Extension (REV)

The REV version has adjustable roll position (1-21) affecting lever length and camme ratio:

```python
# Load with roll position
leg_ext = laban.LegExtensionREV.from_txt_file(
    "leg_extension.txt",
    roll_position=18  # Default position
)

# Roll position affects biomechanics
print(f"Roll position: {leg_ext.roll_position}")
print(f"Lever length: {leg_ext.lever_length_m[0]:.3f} m")
print(f"Camme ratio: {leg_ext.camme_ratio[0]:.3f}")
```

**Roll Position Effects:**
- Position 1: Longest lever (highest mechanical advantage)
- Position 21: Shortest lever (lowest mechanical advantage)
- Lever weight varies from 2.0 kgf (pos 1) to -0.5 kgf (pos 21)

### Chest Press

```python
chest_press = laban.ChestPress.from_txt_file("chest_press.txt")

# Analyze push phase
max_speed = chest_press.speed_lever_ms.max()
max_power = chest_press.power_w.max()

print(f"Peak speed: {max_speed:.2f} m/s")
print(f"Peak power: {max_power:.1f} W")
```

**Calibration Parameters:**
- Spring correction: 1.15
- Lever weight: -4.0 kgf
- Camme ratio: 0.74

### Other Machines

All machine classes follow the same interface:

```python
# Shoulder Press
shoulder = laban.ShoulderPress.from_txt_file("shoulder.txt")

# Low Row
row = laban.LowRow.from_txt_file("row.txt")

# Leg Curl
curl = laban.LegCurl.from_txt_file("curl.txt")

# Leg Extension (standard)
ext = laban.LegExtension.from_txt_file("extension.txt")

# Adjustable Pulley REV
pulley = laban.AdjustablePulleyREV.from_txt_file("pulley.txt")

# Leg Press REV
leg_press_rev = laban.LegPressREV.from_txt_file("leg_press_rev.txt")
```

## Common Workflows

### Analyze Single Repetition

```python
import labanalysis as laban
import numpy as np

# Load data
leg_press = laban.LegPress.from_txt_file("set.txt")

# Detect repetitions using position peaks
from labanalysis import find_peaks

# Find local maxima (extended positions)
peaks, _ = find_peaks(
    leg_press.position_lever_m,
    height=0.15,  # Minimum extension
    distance=100   # Minimum samples between reps
)

# Extract first repetition (peak to peak)
if len(peaks) >= 2:
    rep1 = leg_press.slice(
        start_time=leg_press.time_s[peaks[0]],
        stop_time=leg_press.time_s[peaks[1]]
    )
    
    # Analyze eccentric vs concentric
    # (Assuming velocity reversal at midpoint)
    mid_idx = len(rep1.time_s) // 2
    
    eccentric_power = rep1.power_w[:mid_idx].mean()
    concentric_power = rep1.power_w[mid_idx:].mean()
    
    print(f"Eccentric power: {eccentric_power:.1f} W")
    print(f"Concentric power: {concentric_power:.1f} W")
```

### Force-Velocity Profile

```python
import labanalysis as laban
import matplotlib.pyplot as plt

# Load multiple sets with different loads
sets = []
for file in ["set_20kg.txt", "set_40kg.txt", "set_60kg.txt", "set_80kg.txt"]:
    sets.append(laban.LegPress.from_txt_file(file))

# Extract peak values for each set
loads = []
velocities = []
powers = []

for s in sets:
    loads.append(s.load_kgf.max())
    velocities.append(s.speed_lever_ms.max())
    powers.append(s.power_w.max())

# Plot force-velocity relationship
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(velocities, loads)
plt.xlabel('Peak Velocity (m/s)')
plt.ylabel('Load (kgf)')
plt.title('Force-Velocity Profile')

plt.subplot(1, 2, 2)
plt.scatter(loads, powers)
plt.xlabel('Load (kgf)')
plt.ylabel('Peak Power (W)')
plt.title('Load-Power Profile')

plt.tight_layout()
plt.show()
```

### 1RM Prediction

```python
import labanalysis as laban

# Load test data (submaximal load)
test = laban.LegPress.from_txt_file("8rm_test.txt")

# Get peak load during test
test_load_kgf = test.load_kgf.max()
reps = 8  # Number of repetitions performed

# Predict 1RM using machine-specific coefficients
# leg_press.rm1_coefs = [0.65705, 9.17845]
predicted_1rm = test.rm1_coefs[0] * test_load_kgf + test.rm1_coefs[1]

print(f"Test load ({reps} reps): {test_load_kgf:.1f} kgf")
print(f"Predicted 1RM: {predicted_1rm:.1f} kgf")
```

### Export to DataFrame

```python
import labanalysis as laban

# Load data
chest_press = laban.ChestPress.from_txt_file("workout.txt")

# Convert to DataFrame
df = chest_press.as_dataframe()

# Available columns:
# - Time (s)
# - Load (kgf)
# - Position (m)
# - Position (deg)
# - Speed (m/s)
# - Speed (deg/s)
# - Power (W)

# Export to Excel
df.to_excel("chest_press_analysis.xlsx", index=False)

# Export to CSV
df.to_csv("chest_press_analysis.csv", index=False)
```

## Advanced Usage

### Custom Calibration

All machine classes can be instantiated with raw data and custom calibration:

```python
import numpy as np
import labanalysis as laban

# Raw motor data
time = np.linspace(0, 10, 1000)
position_rad = np.sin(time) * 0.5  # Simulated motion
torque_nm = np.random.randn(1000) * 10 + 50  # Simulated load

# Create machine with custom calibration
leg_press = laban.LegPress(
    time_s=time,
    motor_position_rad=position_rad,
    motor_load_nm=torque_nm
)

# Modify calibration coefficients (advanced)
leg_press._spring_correction = 1.1
leg_press._lever_weight_kgf = 10.0

# Recalculate biomechanical data
load = leg_press.load_kgf
position = leg_press.position_lever_m
```

### Accessing Raw Motor Data

```python
leg_press = laban.LegPress.from_txt_file("data.txt")

# Raw motor-level data
motor_position_rad = leg_press.position_motor_rad
motor_torque_nm = leg_press.torque_nm
motor_speed_rads = leg_press.speed_motor_rads

# Calibration coefficients
pulley_radius = leg_press.pulley_radius_m[0]
spring_corr = leg_press.spring_correction[0]
camme_ratio = leg_press.camme_ratio[0]

# Manual calculation
load_manual = (motor_torque_nm / spring_corr * camme_ratio / 
               pulley_radius / 9.80665)
```

## Troubleshooting

### Issue: "incorrect file" Error

```python
# Ensure file exists and has correct extension
import os
file_path = "data.txt"
assert os.path.exists(file_path), f"File not found: {file_path}"
assert file_path.endswith((".txt", ".csv")), "File must be .txt or .csv"

leg_press = laban.LegPress.from_txt_file(file_path)
```

### Issue: File Parsing Error

Biostrength files use pipe delimiter `|` and comma as decimal separator (European format):

```
Time|Position|Torque
0,00|0,015|12,5
0,01|0,032|25,3
```

The reader automatically handles comma-to-dot conversion. If parsing fails, check file format.

### Issue: Negative or Unrealistic Loads

Check lever weight calibration:

```python
leg_press = laban.LegPress.from_txt_file("data.txt")

# Check lever weight (includes body weight component)
print(f"Lever weight: {leg_press.lever_weight_kgf[0]:.1f} kgf")

# For leg press: 9.0 + 0.17 × body_weight
# If body weight is incorrect, loads will be offset
```

### Issue: Roll Position Ignored (Leg Extension REV)

```python
# WRONG: Default roll position used
leg_ext = laban.LegExtensionREV.from_txt_file("data.txt")

# RIGHT: Specify roll position
leg_ext = laban.LegExtensionREV.from_txt_file("data.txt", roll_position=15)
```

Roll position must be between 1 and 21.

## Machine Comparison Table

| Machine | Spring Corr. | Lever Weight (kgf) | Camme Ratio | 1RM Coefs |
|---------|--------------|-------------------|-------------|-----------|
| Leg Press | 1.00 | 9.0 + 0.17×BW | 1.00 | [0.657, 9.178] |
| Leg Extension | 0.79 | 1.0 | 0.738 | [0.735, 6.0] |
| Leg Extension REV | 1.00 | Variable (roll) | Variable (roll) | [0.735, 6.0] |
| Leg Curl | 0.79 | 7.0 | 0.598 | [0.697, 2.754] |
| Chest Press | 1.15 | -4.0 | 0.74 | [0.963, 2.845] |
| Shoulder Press | 1.00 | -1.2 | 0.794 | [0.862, 1.419] |
| Low Row | 1.00 | 5.0 | 0.64 | [0.696, 3.142] |
| Adj. Pulley REV | 1.00 | 0.01 | 0.25 | [1.0, 0.0] |

**BW** = Body Weight (kg)

## See Also

- [Strength Tests](../test-protocols/strength-tests.md) - Analyzing strength test protocols
- [Signal Processing: Peak Detection](../signal-processing/peak-detection.md) - Detecting repetitions
- [Equations: 1RM Prediction](../../api-reference/equations/strength.md) - Brzycki and other formulas
- [API Reference: Biostrength](../../api-reference/io/read.md#biostrength) - Complete API documentation

---

**Technogym Biostrength**: Intelligent strength training equipment with integrated data acquisition for biomechanical analysis and athlete profiling.
