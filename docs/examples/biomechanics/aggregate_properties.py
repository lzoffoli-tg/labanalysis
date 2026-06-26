"""
Example demonstrating the use of segment_lengths and joint_angles aggregate properties.

This script shows how to use the two new WholeBody properties that combine all
segment dimensions and all joint angles into single Timeseries objects.
"""

import numpy as np
import labanalysis as laban

# Create a simple WholeBody with minimal markers
n_frames = 10
time_index = list(range(n_frames))

# Pelvis markers
left_asis = laban.Point3D(
    data=np.array([[-0.10, 0.90, 0.00]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
right_asis = laban.Point3D(
    data=np.array([[0.10, 0.90, 0.00]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
left_psis = laban.Point3D(
    data=np.array([[-0.08, 0.85, -0.15]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
right_psis = laban.Point3D(
    data=np.array([[0.08, 0.85, -0.15]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)

# Hip markers
left_throcanter = laban.Point3D(
    data=np.array([[-0.15, 0.85, 0.00]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
right_throcanter = laban.Point3D(
    data=np.array([[0.15, 0.85, 0.00]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)

# Knee markers
left_knee_lat = laban.Point3D(
    data=np.array([[-0.18, 0.50, 0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
left_knee_med = laban.Point3D(
    data=np.array([[-0.12, 0.50, 0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
right_knee_lat = laban.Point3D(
    data=np.array([[0.18, 0.50, 0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
right_knee_med = laban.Point3D(
    data=np.array([[0.12, 0.50, 0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)

# Ankle markers
left_ankle_lat = laban.Point3D(
    data=np.array([[-0.15, 0.08, 0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
left_ankle_med = laban.Point3D(
    data=np.array([[-0.10, 0.08, 0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
right_ankle_lat = laban.Point3D(
    data=np.array([[0.15, 0.08, 0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
right_ankle_med = laban.Point3D(
    data=np.array([[0.10, 0.08, 0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)

# Neck/trunk markers
c7 = laban.Point3D(
    data=np.array([[0.00, 1.40, 0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
sc = laban.Point3D(
    data=np.array([[0.00, 1.42, -0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
t5 = laban.Point3D(
    data=np.array([[0.00, 1.10, -0.08]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)
l2 = laban.Point3D(
    data=np.array([[0.00, 0.95, -0.05]] * n_frames),
    index=time_index,
    columns=["X", "Y", "Z"],
)

# Create WholeBody
body = laban.WholeBody(
    left_asis=left_asis, right_asis=right_asis,
    left_psis=left_psis, right_psis=right_psis,
    left_throcanter=left_throcanter, right_throcanter=right_throcanter,
    left_knee_lateral=left_knee_lat, left_knee_medial=left_knee_med,
    right_knee_lateral=right_knee_lat, right_knee_medial=right_knee_med,
    left_ankle_lateral=left_ankle_lat, left_ankle_medial=left_ankle_med,
    right_ankle_lateral=right_ankle_lat, right_ankle_medial=right_ankle_med,
    c7=c7, sc=sc, t5=t5, l2=l2,
)

print("=" * 80)
print("SEGMENT LENGTHS AND WIDTHS")
print("=" * 80)

# Get all segment lengths in a single Timeseries
lengths = body.segment_lengths

print(f"\nShape: {lengths.shape}")
print(f"Unit: {lengths.unit}")
print(f"\nAvailable dimensions ({len(lengths.columns)} total):")
for col in lengths.columns:
    mean_value = np.mean(lengths[col].to_numpy())
    print(f"  - {col:30s}: {mean_value:.4f} {lengths.unit}")

# Convert to DataFrame for easy viewing
df_lengths = lengths.to_dataframe()
print(f"\nFirst 5 frames:")
print(df_lengths.head())

print("\n" + "=" * 80)
print("JOINT ANGLES")
print("=" * 80)

# Get all joint angles in a single Timeseries
angles = body.joint_angles

print(f"\nShape: {angles.shape}")
print(f"Unit: {angles.unit}")
print(f"\nAvailable angles ({len(angles.columns)} total):")
for col in angles.columns:
    mean_value = np.mean(angles[col].to_numpy())
    print(f"  - {col:45s}: {mean_value:7.2f} {angles.unit}")

# Convert to DataFrame for easy viewing
df_angles = angles.to_dataframe()
print(f"\nFirst 5 frames:")
print(df_angles.head())

print("\n" + "=" * 80)
print("ACCESSING INDIVIDUAL COLUMNS")
print("=" * 80)

# Access specific columns
left_thigh = lengths['left_thigh_length']
print(f"\nLeft thigh length:")
print(f"  Mean: {np.mean(left_thigh.to_numpy()):.4f} {left_thigh.unit}")
print(f"  Std:  {np.std(left_thigh.to_numpy()):.4f} {left_thigh.unit}")

left_knee_angle = angles['left_knee_flexionextension']
print(f"\nLeft knee flexion/extension:")
print(f"  Mean: {np.mean(left_knee_angle.to_numpy()):.2f} {left_knee_angle.unit}")
print(f"  Range: [{np.min(left_knee_angle.to_numpy()):.2f}, {np.max(left_knee_angle.to_numpy()):.2f}] {left_knee_angle.unit}")

print("\n" + "=" * 80)
print("EXPORTING TO PANDAS")
print("=" * 80)

# Both properties return Timeseries, which can be easily converted to pandas
import pandas as pd

# Combine lengths and angles into a single DataFrame
df_combined = pd.concat([
    lengths.to_dataframe(),
    angles.to_dataframe()
], axis=1)

print(f"\nCombined DataFrame shape: {df_combined.shape}")
print(f"Total columns: {len(df_combined.columns)}")
print(f"  - Segment dimensions: {len(lengths.columns)}")
print(f"  - Joint angles: {len(angles.columns)}")

# Save to CSV (optional)
# df_combined.to_csv('biomechanical_data.csv')

print("\n" + "=" * 80)
print("Example completed successfully!")
print("=" * 80)
