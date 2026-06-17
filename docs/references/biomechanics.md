# Biomechanics References

Scientific foundations and methodological references for biomechanical analysis in labanalysis.

## Overview

This document provides the scientific and methodological foundations for biomechanical calculations implemented in labanalysis, with references to primary literature.

## Motion Analysis

### Coordinate Systems

**ISB Recommendations**

Wu, G., et al. (2002). ISB recommendation on definitions of joint coordinate system of various joints for the reporting of human joint motion—Part I: ankle, hip, and spine. *Journal of Biomechanics*, 35(4), 543-548.

Wu, G., et al. (2005). ISB recommendation on definitions of joint coordinate systems of various joints for the reporting of human joint motion—Part II: shoulder, elbow, wrist and hand. *Journal of Biomechanics*, 38(5), 981-992.

**Key points**:
- Standard anatomical reference frames
- Joint coordinate systems for multi-segment models
- Convention for rotation sequences (flexion/extension, abduction/adduction, rotation)

### Marker-Based Kinematics

**Winter, D. A. (2009).** *Biomechanics and Motor Control of Human Movement* (4th ed.). Wiley.

**Relevant sections**:
- Chapter 2: Signal Processing (filtering, differentiation)
- Chapter 3: Anthropometry (segment parameters)
- Chapter 4: Kinematics (position, velocity, acceleration)

**Winter's Differentiation Method** (used in labanalysis):
```
velocity[i] = (position[i+1] - position[i-1]) / (2 × dt)
```

Advantages:
- Central difference (more accurate than forward/backward)
- Minimal noise amplification
- Suitable for biomechanical data

### Anthropometric Parameters

**De Leva, P. (1996).** Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. *Journal of Biomechanics*, 29(9), 1223-1230.

**Used for**:
- Segment mass distribution
- Center of mass location
- Moments of inertia
- Body segment parameters in WholeBody class

**Example parameters** (from De Leva 1996):
- Pelvis mass: 14.2% of total body mass
- Thigh mass: 14.16% of total body mass
- Shank mass: 4.33% of total body mass
- Foot mass: 1.37% of total body mass

## Gait Analysis

### Spatiotemporal Parameters

**Perry, J., & Burnfield, J. M. (2010).** *Gait Analysis: Normal and Pathological Function* (2nd ed.). SLACK Incorporated.

**Standard gait metrics**:
- Stride length: Distance between successive heel strikes of same foot
- Step length: Distance between heel strikes of opposite feet
- Cadence: Steps per minute
- Stance time: Duration of foot contact
- Swing time: Duration of foot off ground
- Double support: Both feet on ground simultaneously

### Gait Cycle Normalization

**Standard approach** (used in labanalysis):
- Gait cycle = 0-100% (heel strike to next heel strike)
- Stance phase = 0-60%
- Swing phase = 60-100%

**Reference**: Vaughan, C. L., Davis, B. L., & O'Connor, J. C. (1992). *Dynamics of Human Gait*. Kiboho Publishers.

## Force Platform Analysis

### Ground Reaction Forces

**Winter, D. A. (2009).** *Biomechanics and Motor Control of Human Movement* (4th ed.). Chapter 5: Kinetics.

**GRF Components**:
- Vertical force (Fz): Largest component (typically 1-3× body weight in gait)
- Anterior-posterior force (Fx): Braking and propulsion
- Mediolateral force (Fy): Balance and stability

### Center of Pressure

**COP Calculation**:
```
COP_x = -M_y / F_z
COP_y = M_x / F_z
```

**Reference**: Winter, D. A., et al. (1996). Unified theory regarding A/P and M/L balance in quiet stance. *Journal of Neurophysiology*, 75(6), 2334-2343.

### Balance Assessment

**Postural Sway Metrics**:

Prieto, T. E., et al. (1996). Measures of postural steadiness: differences between healthy young and elderly adults. *IEEE Transactions on Biomedical Engineering*, 43(9), 956-966.

**Common metrics**:
- RMS sway: Root mean square of COP displacement
- Mean velocity: Average COP velocity
- Path length: Total COP trajectory length
- Sway area: 95% confidence ellipse area

**Romberg Quotient**:
```
RQ = Metric_EyesClosed / Metric_EyesOpen
```

**Interpretation**:
- RQ > 1: Greater sway with eyes closed (normal, indicates visual dependence)
- RQ ≈ 1: No difference (may indicate vestibular or proprioceptive issues)

**Reference**: Scoppa, F., et al. (2013). Clinical stabilometry standardization: basic definitions–acquisition interval–sampling frequency. *Gait & Posture*, 37(2), 290-292.

## Jump Analysis

### Flight Time Method

**Jump Height Calculation**:
```
h = g × t_flight² / 8
```

Where:
- h = jump height (m)
- g = gravitational acceleration (9.81 m/s²)
- t_flight = flight time (s)

**Reference**: Bosco, C., et al. (1983). A simple method for measurement of mechanical power in jumping. *European Journal of Applied Physiology*, 50(2), 273-282.

### Force-Velocity Profile

**Samozino, P., et al. (2012).** A simple method for measuring power, force, velocity properties, and mechanical effectiveness in sprint running. *Scandinavian Journal of Medicine & Science in Sports*, 22(5), 648-658.

**F-V Relationship**:
```
F = F₀ - (F₀/V₀) × V
```

**Optimal F-V Profile**:
- F₀: Theoretical maximal force (velocity = 0)
- V₀: Theoretical maximal velocity (force = 0)
- P_max: Maximal power = F₀ × V₀ / 4

**FV Imbalance**:
```
FV_imbalance = 100 × (F₀_actual - F₀_optimal) / F₀_optimal
```

### Reactive Strength Index

**RSI-modified** (countermovement jump):
```
RSI_mod = Jump_Height / Time_to_Takeoff
```

**Reference**: Ebben, W. P., & Petushek, E. J. (2010). Using the reactive strength index modified to evaluate plyometric performance. *Journal of Strength and Conditioning Research*, 24(8), 1983-1987.

## EMG Analysis

### Signal Processing

**Filtering Recommendations**:

De Luca, C. J., et al. (2010). Filtering the surface EMG signal: Movement artifact and baseline noise contamination. *Journal of Biomechanics*, 43(8), 1573-1579.

**Recommendations**:
- Bandpass filter: 20-450 Hz (remove motion artifact and ECG)
- Notch filter: 50/60 Hz (remove powerline interference)
- RMS envelope: 50-100 ms moving window

### Activation Detection

**Threshold-Based Detection**:
```
Threshold = Mean_baseline + 3 × SD_baseline
```

**Reference**: Hodges, P. W., & Bui, B. H. (1996). A comparison of computer-based methods for the determination of onset of muscle contraction using electromyography. *Electroencephalography and Clinical Neurophysiology*, 101(6), 511-519.

### Muscle Fatigue

**Median Frequency Analysis**:

Median frequency decreases with muscle fatigue.

**Reference**: Arendt-Nielsen, L., & Mills, K. R. (1985). The relationship between mean power frequency of the EMG spectrum and muscle fibre conduction velocity. *Electroencephalography and Clinical Neurophysiology*, 60(2), 130-134.

## Signal Processing

### Butterworth Filtering

**Cutoff Frequency Selection**:

Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement* (4th ed.). Chapter 2.

**Recommendations**:
- Marker positions: 6-10 Hz
- Force platform: 10-50 Hz
- EMG envelope: 5-10 Hz

**Filter Order**: 4th order (balance between attenuation and phase lag)

### Residual Analysis

**Determining Optimal Cutoff Frequency**:

Winter, D. A. (2009). Chapter 2: Signal Processing.

**Residual calculation**:
```
Residual = sqrt(sum((filtered - original)²) / N)
```

Plot residual vs. cutoff frequency to find "knee" point.

### Derivative Estimation

**Winter's Method** (central finite difference with spacing of 2):

Used in labanalysis for velocity and acceleration calculation.

**Advantages**:
- Central difference (symmetric)
- Minimal noise amplification
- No phase shift

## Running Biomechanics

### Spatiotemporal Parameters

**Contact Time and Flight Time**:

Cavagna, G. A., et al. (1976). The mechanics of walking and running. *Proceedings of the Royal Society of London B*, 193(1113), 347-367.

**Duty Factor**:
```
Duty_Factor = Contact_Time / (Contact_Time + Flight_Time)
```

- Walking: DF > 0.5 (double support phase)
- Running: DF < 0.5 (flight phase)

### Vertical Oscillation

**Definition**: Vertical displacement of center of mass during running stride.

**Optimal Range**: 6-10 cm (lower = more economical)

**Reference**: McMahon, T. A., & Cheng, G. C. (1990). The mechanics of running: how does stiffness couple with speed? *Journal of Biomechanics*, 23, 65-78.

## Statistical Methods

### Normative Data Comparison

**Z-Score Calculation**:
```
z = (value - mean_norm) / SD_norm
```

**Interpretation**:
- |z| < 1: Within normal range (68th percentile)
- |z| < 2: Slightly abnormal (95th percentile)
- |z| ≥ 2: Significantly abnormal

### Bilateral Asymmetry

**Symmetry Index**:
```
SI = 100 × (Left - Right) / (0.5 × (Left + Right))
```

**Reference**: Robinson, R. O., et al. (1987). Use of force platform variables to quantify the effects of chiropractic manipulation on gait symmetry. *Journal of Manipulative and Physiological Therapeutics*, 10(4), 172-176.

## Equipment Specifications

### Force Platforms

**Sampling Frequency**:
- Typical: 1000 Hz
- Minimum: 100 Hz for gait, 500 Hz for jumping

**Natural Frequency**: > 200 Hz (avoid resonance effects)

**Reference**: Bobbert, M. F., & Schamhardt, H. C. (1990). Accuracy of determining the point of force application with piezoelectric force plates. *Journal of Biomechanics*, 23(7), 705-710.

### Motion Capture

**Marker Placement**:

Cappozzo, A., et al. (1995). Position and orientation in space of bones during movement: anatomical frame definition and determination. *Clinical Biomechanics*, 10(4), 171-178.

**Sampling Frequency**:
- Gait: 100-200 Hz
- Running/jumping: 200-500 Hz
- High-speed movements: > 500 Hz

## Software Validation

**Best Practices**:

Knudson, D., & Morrison, C. (2002). *Qualitative Analysis of Human Movement* (2nd ed.). Human Kinetics.

**Validation checklist**:
- Compare against gold standard (published data)
- Test with synthetic data (known outcomes)
- Cross-validate with commercial software
- Document accuracy and precision

## Additional Resources

### Textbooks

1. **Winter, D. A. (2009).** *Biomechanics and Motor Control of Human Movement* (4th ed.). Wiley.
   - Comprehensive biomechanics reference
   - Signal processing methods
   - Kinematic and kinetic analysis

2. **Robertson, D. G. E., et al. (2013).** *Research Methods in Biomechanics* (2nd ed.). Human Kinetics.
   - Experimental design
   - Data collection protocols
   - Analysis techniques

3. **Zatsiorsky, V. M., & Prilutsky, B. I. (2012).** *Biomechanics of Skeletal Muscles*. Human Kinetics.
   - Muscle mechanics
   - Force production
   - Motor control

### Online Resources

**ISB (International Society of Biomechanics)**:
- https://isbweb.org/
- Standards and recommendations
- Software resources

**ASB (American Society of Biomechanics)**:
- https://www.asbweb.org/
- Educational resources
- Conference proceedings

## Citation Guidelines

When publishing work using labanalysis, cite relevant methodological papers:

**For joint angles**:
```
Wu, G., et al. (2002, 2005). ISB recommendations on joint coordinate systems.
```

**For marker data processing**:
```
Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement.
```

**For anthropometric parameters**:
```
De Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.
```

**For jump analysis**:
```
Bosco, C., et al. (1983). A simple method for measurement of mechanical power in jumping.
```

**For force-velocity profiling**:
```
Samozino, P., et al. (2012). A simple method for measuring force-velocity-power profile.
```

---

**Scientific foundation matters.** labanalysis implements validated biomechanical methods from peer-reviewed literature. Cite original sources when publishing results.
