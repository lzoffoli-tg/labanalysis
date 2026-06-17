# Biomechanical Equations Reference

Mathematical formulas and equations used in labanalysis for biomechanical calculations.

## Overview

This reference documents all mathematical equations implemented in labanalysis, including their derivations, assumptions, and literature sources.

## Kinematics

### Position, Velocity, Acceleration

**Velocity** (first derivative):
```
v(t) = dx/dt
```

**Discrete approximation** (Winter's method):
```
v[i] = (x[i+1] - x[i-1]) / (2 × Δt)
```

**Acceleration** (second derivative):
```
a(t) = dv/dt = d²x/dt²
```

**Discrete approximation**:
```
a[i] = (v[i+1] - v[i-1]) / (2 × Δt)
```

Where:
- x = position (m)
- v = velocity (m/s)
- a = acceleration (m/s²)
- Δt = time step (s)

**Reference**: Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement*.

### Distance and Displacement

**Euclidean Distance** (3D):
```
d = √((x₂ - x₁)² + (y₂ - y₁)² + (z₂ - z₁)²)
```

**Path Length** (cumulative distance):
```
L = Σᵢ √((xᵢ - xᵢ₋₁)² + (yᵢ - yᵢ₋₁)² + (zᵢ - zᵢ₋₁)²)
```

**Displacement** (straight-line distance from start to end):
```
D = √((x_final - x_initial)² + (y_final - y_initial)² + (z_final - z_initial)²)
```

### Angular Kinematics

**Angle Between Two Vectors**:
```
θ = arccos((v₁ · v₂) / (|v₁| × |v₂|))
```

**Angular Velocity**:
```
ω = dθ/dt
```

**Discrete approximation**:
```
ω[i] = (θ[i+1] - θ[i-1]) / (2 × Δt)
```

Units: rad/s or deg/s

## Kinetics

### Force and Moment

**Newton's Second Law**:
```
F = m × a
```

Where:
- F = force (N)
- m = mass (kg)
- a = acceleration (m/s²)

**Body Weight**:
```
BW = m × g
```

Where:
- BW = body weight (N)
- g = gravitational acceleration = 9.81 m/s²

**Moment (Torque)**:
```
M = r × F
```

Where:
- M = moment (N·m)
- r = moment arm (m)
- F = force (N)

### Impulse and Momentum

**Impulse**:
```
J = ∫ F dt ≈ Σ F[i] × Δt
```

**Impulse-Momentum Theorem**:
```
J = Δ(m × v) = m × Δv
```

Where:
- J = impulse (N·s)
- m = mass (kg)
- Δv = change in velocity (m/s)

**Discrete impulse** (trapezoidal rule):
```
J = Σᵢ ((F[i] + F[i+1]) / 2) × Δt
```

### Power and Work

**Mechanical Power**:
```
P = F × v
```

Where:
- P = power (W)
- F = force (N)
- v = velocity (m/s)

**Work**:
```
W = ∫ F · dx = F × d
```

Where:
- W = work (J)
- F = force (N)
- d = displacement (m)

**Power from Work**:
```
P = dW/dt
```

## Jump Analysis

### Jump Height from Flight Time

**Formula**:
```
h = g × t_flight² / 8
```

**Derivation**:

From projectile motion:
```
v_takeoff = g × t_flight / 2    (peak velocity at mid-flight)
h = v_takeoff² / (2g)            (kinematic equation)
h = (g × t_flight / 2)² / (2g)   (substitute)
h = g² × t_flight² / (4 × 2g)    (simplify)
h = g × t_flight² / 8            (final form)
```

Where:
- h = jump height (m)
- g = 9.81 m/s²
- t_flight = flight time (s)

**Reference**: Bosco, C., et al. (1983). *European Journal of Applied Physiology*, 50(2), 273-282.

### Jump Height from Velocity

**Formula**:
```
h = v_takeoff² / (2g)
```

**Derivation**:

From energy conservation:
```
KE_takeoff = PE_peak
½ m v² = m g h
h = v² / (2g)
```

Where:
- v_takeoff = takeoff velocity (m/s)
- g = 9.81 m/s²

### Rate of Force Development

**RFD** (force derivative):
```
RFD = ΔF / Δt
```

**Maximum RFD over sliding window**:
```
RFD_max = max((F[i + n] - F[i]) / (t[i + n] - t[i]))
```

Where:
- RFD = rate of force development (N/s)
- n = window size (samples)

**Reference**: Maffiuletti et al. (2016). *European Journal of Applied Physiology*, 116(6), 1091-1116.

### Reactive Strength Index

**RSI-modified** (for CMJ):
```
RSI_mod = Jump_Height / Time_to_Takeoff
```

Where:
- RSI_mod = reactive strength index modified (dimensionless or m/s)
- Jump_Height = vertical jump height (m)
- Time_to_Takeoff = time from movement initiation to takeoff (s)

**Reference**: Ebben & Petushek (2010). *Journal of Strength and Conditioning Research*, 24(8), 1983-1987.

### Force-Velocity Profile

**Linear F-V Relationship**:
```
F = F₀ - (F₀/V₀) × V
```

**Polynomial F-V Relationship** (2nd degree):
```
F = a × V² + b × V + c
```

**Maximal Power**:
```
P_max = F₀ × V₀ / 4
```

**F-V Imbalance**:
```
FV_imb = 100 × (F₀_actual - F₀_optimal) / F₀_optimal
```

Where:
- F₀ = maximal force at zero velocity (N)
- V₀ = maximal velocity at zero force (m/s)
- P_max = maximal power (W)

**Reference**: Samozino et al. (2012). *Scandinavian Journal of Medicine & Science in Sports*, 22(5), 648-658.

## Strength Equations

### 1RM Prediction

**Brzycki Equation**:
```
1RM = Weight / (1.0278 - 0.0278 × Reps)
```

Where:
- 1RM = one-repetition maximum (kg)
- Weight = load lifted (kg)
- Reps = number of repetitions performed

**Valid range**: 1-10 repetitions

**Reference**: Brzycki, M. (1993). Strength testing—predicting a one-rep max from reps-to-fatigue. *Journal of Physical Education, Recreation & Dance*, 64(1), 88-90.

**Epley Equation**:
```
1RM = Weight × (1 + 0.0333 × Reps)
```

**Valid range**: 1-10 repetitions

**Lander Equation**:
```
1RM = (100 × Weight) / (101.3 - 2.67123 × Reps)
```

**Valid range**: ≤ 10 repetitions

### Relative Strength

**Strength-to-Mass Ratio**:
```
Relative_Strength = Absolute_Strength / Body_Mass
```

Units: N/kg or kg/kg (for 1RM)

**Allometric Scaling** (accounts for body size):
```
Allometric_Strength = Absolute_Strength / Body_Mass^(2/3)
```

**Reference**: Jaric, S. (2003). *European Journal of Applied Physiology*, 89(2), 115-124.

## Cardiorespiratory Equations

### VO₂max Prediction

**ACSM Running Equation**:
```
VO₂ = (0.2 × Speed) + (0.9 × Speed × Grade) + 3.5
```

Where:
- VO₂ = oxygen consumption (mL/kg/min)
- Speed = running speed (m/min)
- Grade = treadmill incline (fraction, e.g., 0.10 for 10%)

**ACSM Cycling Equation**:
```
VO₂ = (10.8 × Watts / BM) + 7
```

Where:
- VO₂ = oxygen consumption (mL/kg/min)
- Watts = power output (W)
- BM = body mass (kg)

**Reference**: American College of Sports Medicine (2018). *ACSM's Guidelines for Exercise Testing and Prescription* (10th ed.).

### Heart Rate Reserve (Karvonen Method)

**Target Heart Rate**:
```
THR = ((HRmax - HRrest) × %Intensity) + HRrest
```

Where:
- THR = target heart rate (bpm)
- HRmax = maximal heart rate (bpm) ≈ 220 - Age
- HRrest = resting heart rate (bpm)
- %Intensity = desired intensity (0-1)

**Reference**: Karvonen et al. (1957). *Annales Medicinae Experimentalis et Biologiae Fenniae*, 35(3), 307-315.

### Energy Expenditure

**Metabolic Equivalent (MET)**:
```
MET = VO₂ / 3.5
```

**Energy Expenditure**:
```
kcal/min = (MET × 3.5 × BM) / 200
```

Where:
- MET = metabolic equivalent
- BM = body mass (kg)

## Balance and Postural Control

### Center of Pressure (COP)

**COP Position**:
```
COP_x = -M_y / F_z
COP_y = M_x / F_z
```

Where:
- COP = center of pressure (m)
- M = moment about axis (N·m)
- F_z = vertical ground reaction force (N)

**Reference**: Winter et al. (1996). *Journal of Neurophysiology*, 75(6), 2334-2343.

### Postural Sway Metrics

**RMS Sway**:
```
RMS = √(Σ(x_i - x_mean)² / N)
```

**Mean Velocity**:
```
v_mean = PathLength / TotalTime
```

**Path Length**:
```
L = Σᵢ √((x_i - x_{i-1})² + (y_i - y_{i-1})²)
```

**Sway Area** (95% confidence ellipse):
```
Area = π × 2.4477 × σ_x × σ_y × √(1 - ρ²)
```

Where:
- σ_x, σ_y = standard deviations of COP in x and y
- ρ = correlation coefficient between x and y

**Reference**: Prieto et al. (1996). *IEEE Transactions on Biomedical Engineering*, 43(9), 956-966.

**Romberg Quotient**:
```
RQ = Metric_EyesClosed / Metric_EyesOpen
```

Interpretation:
- RQ > 1: Visual dependence (normal)
- RQ ≈ 1: Reduced visual contribution

## Gait Analysis

### Spatiotemporal Parameters

**Stride Length**:
```
SL = v × t_stride
```

Where:
- SL = stride length (m)
- v = walking velocity (m/s)
- t_stride = stride time (s)

**Cadence**:
```
Cadence = 120 / t_stride
```

Units: steps/min (multiply by 2 for strides/min)

**Duty Factor**:
```
DF = t_contact / (t_contact + t_flight)
```

Where:
- DF = duty factor (0-1)
- t_contact = contact time (s)
- t_flight = flight time (s)

Interpretation:
- DF > 0.5: Walking (double support phase)
- DF < 0.5: Running (flight phase)

### Symmetry and Asymmetry

**Symmetry Index**:
```
SI = 100 × (Left - Right) / (0.5 × (Left + Right))
```

Interpretation:
- SI = 0: Perfect symmetry
- |SI| < 10%: Normal asymmetry
- |SI| ≥ 10%: Significant asymmetry

**Reference**: Robinson et al. (1987). *Journal of Manipulative and Physiological Therapeutics*, 10(4), 172-176.

**Gait Asymmetry Ratio** (GAR):
```
GAR = Left / Right
```

Interpretation:
- GAR = 1: Perfect symmetry
- GAR < 1: Left side weaker/shorter
- GAR > 1: Right side weaker/shorter

## Running Mechanics

### Vertical Oscillation

**Center of Mass Displacement**:
```
VO = max(z_COM) - min(z_COM)
```

Where:
- VO = vertical oscillation (m)
- z_COM = vertical position of center of mass (m)

Optimal range: 0.06-0.10 m

### Leg Stiffness

**Spring-Mass Model**:
```
k_leg = F_peak / Δy
```

Where:
- k_leg = leg stiffness (kN/m)
- F_peak = peak vertical GRF (N)
- Δy = vertical displacement of COM (m)

**Reference**: McMahon & Cheng (1990). *Journal of Biomechanics*, 23, 65-78.

### Ground Contact Time

**Contact Time**:
```
t_contact = time(Takeoff) - time(Landing)
```

Where threshold for landing/takeoff is typically 20-50 N or 10-20% of body weight.

## Signal Processing

### Butterworth Filter

**Transfer Function** (2nd order lowpass):
```
H(s) = ω_c² / (s² + √2 × ω_c × s + ω_c²)
```

Where:
- ω_c = cutoff angular frequency = 2π × f_c (rad/s)
- s = Laplace variable

**Cutoff Frequency Recommendations** (Winter 2009):
- Marker positions: 6-10 Hz
- Force platform: 10-50 Hz
- EMG envelope: 5-10 Hz

### Residual Analysis

**Residual**:
```
Residual = √(Σ(x_filtered - x_original)² / N)
```

Plot Residual vs. Cutoff Frequency to find optimal cutoff (at "knee" point).

### Power Spectral Density

**PSD Calculation** (Welch's method):

Uses FFT on overlapping segments with windowing.

**Median Frequency**:
```
f_median: ∫₀^f_median PSD(f) df = ∫_{f_median}^∞ PSD(f) df
```

Frequency at which half the signal power is below and half is above.

## Statistical Measures

### Normative Comparison

**Z-Score**:
```
z = (x - μ) / σ
```

Where:
- x = observed value
- μ = population mean
- σ = population standard deviation

Interpretation:
- |z| < 1: Within normal range (68%)
- |z| < 2: Slightly abnormal (95%)
- |z| ≥ 2: Significantly abnormal (> 95%)

### Coefficient of Variation

**CV**:
```
CV = (σ / μ) × 100
```

Units: %

Interpretation:
- CV < 10%: Low variability (high consistency)
- CV > 20%: High variability (low consistency)

### Correlation

**Pearson Correlation**:
```
r = Σ((x_i - x_mean) × (y_i - y_mean)) / √(Σ(x_i - x_mean)² × Σ(y_i - y_mean)²)
```

Range: -1 to +1
- r = +1: Perfect positive correlation
- r = 0: No correlation
- r = -1: Perfect negative correlation

## Unit Conversions

### Common Conversions

**Length**:
- 1 m = 100 cm = 1000 mm
- 1 inch = 0.0254 m
- 1 foot = 0.3048 m

**Force**:
- 1 N = 1 kg·m/s²
- 1 kN = 1000 N
- 1 lbf = 4.448 N

**Power**:
- 1 W = 1 J/s = 1 N·m/s
- 1 hp = 746 W

**Angle**:
- 1 rad = 57.2958 deg
- 1 deg = 0.0174533 rad
- π rad = 180 deg

## See Also

- [Biomechanics References](biomechanics.md) - Scientific literature
- [API Reference - Equations](../api-reference/equations/) - Implementation details
- [User Guide - Signal Processing](../user-guide/signal-processing/) - Practical applications

---

**Mathematical foundation for biomechanical analysis.** All equations are validated against peer-reviewed literature and implemented with appropriate assumptions.
