"""
Synthetic data generators for jump testing.

This module provides utilities to generate synthetic force platform signals
for testing JumpTest protocols with SingleJump, DropJump, and RepeatedJumps.
"""

import numpy as np

from labanalysis.constants import G, MINIMUM_CONTACT_FORCE_N
from labanalysis.records import ForcePlatform
from labanalysis.timeseries import Point3D, Signal3D


def generate_kinematic_markers(
    time,
    side="bilateral",
    contact_samples=250,
    flight_samples=300,
    pre_jump_samples=500,
    fsamp=1000.0
):
    """
    Generate synthetic kinematic markers for jump analysis.

    Creates realistic marker positions for lower limb joints during jump phases.
    Markers move through squat → propulsion → flight → landing phases.

    Parameters
    ----------
    time : np.ndarray
        Time vector for the signal
    side : str
        "left", "right", or "bilateral"
    contact_samples : int
        Number of samples in contact phase
    flight_samples : int
        Number of samples in flight phase
    pre_jump_samples : int
        Number of samples before jump
    fsamp : float
        Sampling frequency in Hz

    Returns
    -------
    dict
        Dictionary of Point3D markers for hip, knee, ankle
    """
    n_samples = len(time)
    markers = {}

    # Anatomical heights (approximate for 178cm person)
    hip_height = 0.90  # m
    knee_height = 0.50  # m
    ankle_height = 0.08  # m

    # Lateral offsets
    hip_width = 0.15  # m (pelvis width)
    knee_width = 0.10  # m
    ankle_width = 0.08  # m

    # Generate movement pattern (vertical displacement during jump)
    # Phase indices
    idx_contact_start = pre_jump_samples
    idx_contact_end = idx_contact_start + contact_samples
    idx_flight_end = idx_contact_end + flight_samples

    # Vertical displacement pattern
    squat_depth = 0.30  # 30cm squat
    jump_height = 0.35  # 35cm jump apex

    vertical_displacement = np.zeros(n_samples)

    # Pre-jump: standing height
    vertical_displacement[:idx_contact_start] = 0

    # Contact phase: squat down then propulsion up
    t_contact = np.linspace(0, 1, contact_samples)
    # First half: squat down, second half: push up
    squat_pattern = -squat_depth * np.sin(np.pi * t_contact) ** 2
    vertical_displacement[idx_contact_start:idx_contact_end] = squat_pattern

    # Flight phase: parabolic trajectory
    if idx_flight_end <= n_samples:
        t_flight = np.linspace(0, 1, flight_samples)
        flight_pattern = jump_height * (4 * t_flight * (1 - t_flight))
        vertical_displacement[idx_contact_end:idx_flight_end] = flight_pattern

    # Landing and post-landing: back to standing
    if idx_flight_end < n_samples:
        vertical_displacement[idx_flight_end:] = 0

    # Add small random noise
    noise = np.random.normal(0, 0.002, n_samples)  # 2mm noise

    # Generate markers based on side
    sides_to_generate = []
    if side == "bilateral":
        sides_to_generate = ["left", "right"]
    else:
        sides_to_generate = [side]

    for marker_side in sides_to_generate:
        lateral_sign = -1 if marker_side == "left" else 1

        # Hip markers (trochanter, ASIS, PSIS)
        trochanter_data = np.column_stack([
            np.full(n_samples, lateral_sign * hip_width) + noise,  # X (lateral)
            hip_height + vertical_displacement + noise,  # Y (vertical)
            np.zeros(n_samples) + noise  # Z (anteroposterior)
        ])
        markers[f"{marker_side}_trochanter"] = Point3D(
            data=trochanter_data, index=time, unit="m"
        )

        # ASIS (anterior superior iliac spine) - slightly forward and higher
        asis_data = np.column_stack([
            np.full(n_samples, lateral_sign * hip_width * 0.8) + noise,
            hip_height + 0.05 + vertical_displacement + noise,
            np.full(n_samples, 0.10) + noise  # 10cm forward
        ])
        markers[f"{marker_side}_asis"] = Point3D(
            data=asis_data, index=time, unit="m"
        )

        # PSIS (posterior superior iliac spine) - slightly backward
        psis_data = np.column_stack([
            np.full(n_samples, lateral_sign * hip_width * 0.7) + noise,
            hip_height + 0.02 + vertical_displacement + noise,
            np.full(n_samples, -0.10) + noise  # 10cm backward
        ])
        markers[f"{marker_side}_psis"] = Point3D(
            data=psis_data, index=time, unit="m"
        )

        # Knee markers (medial and lateral condyles)
        knee_data_lateral = np.column_stack([
            np.full(n_samples, lateral_sign * knee_width) + noise,
            knee_height + vertical_displacement * 0.7 + noise,  # Knee bends less than hip
            np.zeros(n_samples) + noise
        ])
        markers[f"{marker_side}_knee_lateral"] = Point3D(
            data=knee_data_lateral, index=time, unit="m"
        )

        knee_data_medial = np.column_stack([
            np.full(n_samples, lateral_sign * knee_width * 0.6) + noise,
            knee_height + vertical_displacement * 0.7 + noise,
            np.zeros(n_samples) + noise
        ])
        markers[f"{marker_side}_knee_medial"] = Point3D(
            data=knee_data_medial, index=time, unit="m"
        )

        # Ankle markers (medial and lateral malleoli)
        ankle_data_lateral = np.column_stack([
            np.full(n_samples, lateral_sign * ankle_width) + noise,
            ankle_height + vertical_displacement * 0.3 + noise,  # Ankle moves little
            np.zeros(n_samples) + noise
        ])
        markers[f"{marker_side}_ankle_lateral"] = Point3D(
            data=ankle_data_lateral, index=time, unit="m"
        )

        ankle_data_medial = np.column_stack([
            np.full(n_samples, lateral_sign * ankle_width * 0.5) + noise,
            ankle_height + vertical_displacement * 0.3 + noise,
            np.zeros(n_samples) + noise
        ])
        markers[f"{marker_side}_ankle_medial"] = Point3D(
            data=ankle_data_medial, index=time, unit="m"
        )

        # Foot markers for ankle calculations
        heel_data = np.column_stack([
            np.full(n_samples, lateral_sign * ankle_width * 0.5) + noise,
            np.full(n_samples, 0.02) + noise,  # 2cm off ground
            np.full(n_samples, -0.10) + noise  # 10cm behind ankle
        ])
        markers[f"{marker_side}_heel"] = Point3D(
            data=heel_data, index=time, unit="m"
        )

        toe_data = np.column_stack([
            np.full(n_samples, lateral_sign * ankle_width * 0.5) + noise,
            np.full(n_samples, 0.01) + noise,
            np.full(n_samples, 0.15) + noise  # 15cm forward
        ])
        markers[f"{marker_side}_toe"] = Point3D(
            data=toe_data, index=time, unit="m"
        )

        # Metatarsal heads
        markers[f"{marker_side}_first_metatarsal_head"] = Point3D(
            data=np.column_stack([
                np.full(n_samples, lateral_sign * 0.02) + noise,
                np.full(n_samples, 0.01) + noise,
                np.full(n_samples, 0.12) + noise
            ]),
            index=time, unit="m"
        )

        markers[f"{marker_side}_fifth_metatarsal_head"] = Point3D(
            data=np.column_stack([
                np.full(n_samples, lateral_sign * 0.06) + noise,
                np.full(n_samples, 0.01) + noise,
                np.full(n_samples, 0.10) + noise
            ]),
            index=time, unit="m"
        )

    return markers


def generate_jump_force_signal(
    bodymass_kg=75.0,
    jump_height_m=0.30,
    contact_time_s=0.300,
    flight_time_s=None,
    fsamp=1000.0,
    noise_level=10.0,
    pre_jump_duration_s=0.5,
    post_landing_duration_s=0.5,
):
    """
    Generate synthetic vertical ground reaction force for a single jump.

    Creates a realistic force profile including:
    - Pre-jump standing phase (bodyweight)
    - Counter-movement phase (if CMJ)
    - Propulsion phase (increasing force to takeoff)
    - Flight phase (zero force)
    - Landing phase (impact and stabilization)
    - Post-landing standing phase (bodyweight)

    Parameters
    ----------
    bodymass_kg : float, optional
        Subject body mass in kg (default: 75.0)
    jump_height_m : float, optional
        Jump height in meters (default: 0.30)
    contact_time_s : float, optional
        Ground contact time during propulsion in seconds (default: 0.300)
    flight_time_s : float, optional
        Flight time in seconds. If None, calculated from jump_height_m
        using flight_time = 2 * sqrt(2 * height / g)
    fsamp : float, optional
        Sampling frequency in Hz (default: 1000.0)
    noise_level : float, optional
        Standard deviation of Gaussian noise in N (default: 10.0)
    pre_jump_duration_s : float, optional
        Duration of standing phase before jump (default: 0.5)
    post_landing_duration_s : float, optional
        Duration of standing phase after landing (default: 0.5)

    Returns
    -------
    Signal3D
        3D force signal (X, Y, Z) in Newtons with Z = vertical axis
    Point3D
        3D center of pressure (constant at origin)
    float
        Total duration of the signal in seconds
    """
    # Calculate flight time from jump height if not provided
    if flight_time_s is None:
        flight_time_s = 2 * np.sqrt(2 * jump_height_m / G)

    bodyweight_N = bodymass_kg * G

    # Phase durations
    pre_jump_samples = int(pre_jump_duration_s * fsamp)
    contact_samples = int(contact_time_s * fsamp)
    flight_samples = int(flight_time_s * fsamp)
    landing_samples = int(contact_time_s * fsamp)  # Same as takeoff
    post_landing_samples = int(post_landing_duration_s * fsamp)

    total_samples = (
        pre_jump_samples
        + contact_samples
        + flight_samples
        + landing_samples
        + post_landing_samples
    )

    # Build time vector
    time = np.arange(total_samples) / fsamp

    # Initialize force components (X=lateral, Y=vertical, Z=anteroposterior)
    # Convention: Y is vertical axis
    force_x = np.zeros(total_samples)
    force_y = np.zeros(total_samples)
    force_z = np.zeros(total_samples)

    # Phase 1: Pre-jump standing (bodyweight)
    idx_start = 0
    idx_end = pre_jump_samples
    force_y[idx_start:idx_end] = bodyweight_N

    # Phase 2: Propulsion phase (takeoff)
    # Model as quadratic increase from bodyweight to peak force
    # Peak force calculated from impulse-momentum theorem
    # Impulse = m * takeoff_velocity = Integral(F - mg) dt
    takeoff_velocity = np.sqrt(2 * G * jump_height_m)
    impulse_required = bodymass_kg * takeoff_velocity

    # Average net force during propulsion
    avg_net_force = impulse_required / contact_time_s
    peak_force = bodyweight_N + 2.5 * avg_net_force  # Increased multiplier for realistic force

    idx_start = idx_end
    idx_end = idx_start + contact_samples
    t_norm = np.linspace(0, 1, contact_samples)
    # Create more realistic force profile with higher peak
    # Using sin^2 profile for smoother transition
    force_profile = np.sin(np.pi * t_norm) ** 2
    force_y[idx_start:idx_end] = bodyweight_N + force_profile * (peak_force - bodyweight_N) * 1.5

    # Phase 3: Flight phase (zero force)
    idx_start = idx_end
    idx_end = idx_start + flight_samples
    force_y[idx_start:idx_end] = 0

    # Phase 4: Landing phase (impact force)
    # Landing impact typically higher than takeoff force
    landing_peak_force = peak_force * 1.5
    idx_start = idx_end
    idx_end = idx_start + landing_samples
    t_norm = np.linspace(0, 1, landing_samples)
    # Sharp peak at landing, then decay
    landing_profile = np.exp(-5 * t_norm) * (1 - t_norm * 0.8)
    force_y[idx_start:idx_end] = bodyweight_N + landing_profile * (landing_peak_force - bodyweight_N)

    # Phase 5: Post-landing standing (bodyweight)
    idx_start = idx_end
    idx_end = idx_start + post_landing_samples
    force_y[idx_start:idx_end] = bodyweight_N

    # Add small lateral and anteroposterior forces (noise)
    force_x += np.random.normal(0, noise_level * 0.1, total_samples)
    force_z += np.random.normal(0, noise_level * 0.1, total_samples)

    # Add vertical noise
    force_y += np.random.normal(0, noise_level, total_samples)

    # Ensure no negative forces
    force_y = np.maximum(force_y, 0)

    # Create force signal (X, Y, Z) where Y is vertical
    force_data = np.column_stack([force_x, force_y, force_z])
    force_signal = Signal3D(
        data=force_data,
        index=time,
        columns=["X", "Y", "Z"],
        unit="N"
    )

    # Create center of pressure (constant at origin for simplicity)
    cop_data = np.zeros((total_samples, 3))
    cop = Point3D(
        data=cop_data,
        index=time,
        columns=["X", "Y", "Z"],
        unit="m"
    )

    # Create torque signal (minimal for synthetic data)
    torque_data = np.zeros((total_samples, 3))
    torque_data += np.random.normal(0, 0.1, (total_samples, 3))  # Small random torque
    torque = Signal3D(
        data=torque_data,
        index=time,
        columns=["X", "Y", "Z"],
        unit="N*m"
    )

    duration = time[-1]

    return force_signal, cop, torque, duration


def generate_squat_jump_force_platform(
    bodymass_kg=75.0,
    jump_height_m=0.30,
    side="bilateral",
    fsamp=1000.0,
    include_kinematics=False
):
    """
    Generate synthetic ForcePlatform for a squat jump (SJ).

    Squat jump has no counter-movement - starts from static semi-squat.

    Parameters
    ----------
    bodymass_kg : float
        Body mass in kg
    jump_height_m : float
        Target jump height in meters
    side : str
        "left", "right", or "bilateral"
    fsamp : float
        Sampling frequency in Hz
    include_kinematics : bool
        If True, returns tuple (ForcePlatform, kinematic_markers_dict)

    Returns
    -------
    ForcePlatform or tuple
        Force platform with realistic SJ force profile.
        If include_kinematics=True, returns (ForcePlatform, dict of markers)
    """
    force, cop, torque, duration = generate_jump_force_signal(
        bodymass_kg=bodymass_kg,
        jump_height_m=jump_height_m,
        contact_time_s=0.250,  # SJ typically shorter contact time
        fsamp=fsamp,
        noise_level=8.0,
    )

    # Offset COP based on side to avoid identical positions for bilateral jumps
    if side == "left":
        cop[:, "X"] = cop.to_numpy()[:, 0] - 0.15  # 15cm to the left (negative X)
    elif side == "right":
        cop[:, "X"] = cop.to_numpy()[:, 0] + 0.15  # 15cm to the right (positive X)

    fp = ForcePlatform(
        origin=cop,
        force=force,
        torque=torque
    )

    if include_kinematics:
        markers = generate_kinematic_markers(
            time=cop.index,
            side=side,
            contact_samples=int(0.250 * fsamp),
            flight_samples=int(2 * np.sqrt(2 * jump_height_m / G) * fsamp),
            pre_jump_samples=int(0.5 * fsamp),
            fsamp=fsamp
        )
        return fp, markers

    return fp


def generate_cmj_force_platform(
    bodymass_kg=75.0,
    jump_height_m=0.35,
    side="bilateral",
    fsamp=1000.0,
    include_kinematics=False
):
    """
    Generate synthetic ForcePlatform for a counter-movement jump (CMJ).

    CMJ includes counter-movement phase before propulsion.

    Parameters
    ----------
    bodymass_kg : float
        Body mass in kg
    jump_height_m : float
        Target jump height in meters
    side : str
        "left", "right", or "bilateral"
    fsamp : float
        Sampling frequency in Hz
    include_kinematics : bool
        If True, returns tuple (ForcePlatform, kinematic_markers_dict)

    Returns
    -------
    ForcePlatform or tuple
        Force platform with realistic CMJ force profile.
        If include_kinematics=True, returns (ForcePlatform, dict of markers)
    """
    force, cop, torque, duration = generate_jump_force_signal(
        bodymass_kg=bodymass_kg,
        jump_height_m=jump_height_m,
        contact_time_s=0.350,  # CMJ typically longer contact time
        fsamp=fsamp,
        noise_level=10.0,
    )

    # Offset COP based on side to avoid identical positions for bilateral jumps
    if side == "left":
        cop[:, "X"] = cop.to_numpy()[:, 0] - 0.15  # 15cm to the left (negative X)
    elif side == "right":
        cop[:, "X"] = cop.to_numpy()[:, 0] + 0.15  # 15cm to the right (positive X)

    fp = ForcePlatform(
        origin=cop,
        force=force,
        torque=torque
    )

    if include_kinematics:
        markers = generate_kinematic_markers(
            time=cop.index,
            side=side,
            contact_samples=int(0.350 * fsamp),
            flight_samples=int(2 * np.sqrt(2 * jump_height_m / G) * fsamp),
            pre_jump_samples=int(0.5 * fsamp),
            fsamp=fsamp
        )
        return fp, markers

    return fp


def generate_drop_jump_force_platform(
    bodymass_kg=75.0,
    drop_height_cm=40,
    jump_height_m=0.25,
    side="bilateral",
    fsamp=1000.0,
    include_kinematics=False
):
    """
    Generate synthetic ForcePlatform for a drop jump (DJ).

    Drop jump includes landing from height followed by immediate rebound jump.

    Parameters
    ----------
    bodymass_kg : float
        Body mass in kg
    drop_height_cm : int
        Drop height in centimeters
    jump_height_m : float
        Rebound jump height in meters
    side : str
        "left", "right", or "bilateral"
    fsamp : float
        Sampling frequency in Hz
    include_kinematics : bool
        If True, returns tuple (ForcePlatform, kinematic_markers_dict)

    Returns
    -------
    ForcePlatform or tuple
        Force platform with realistic DJ force profile.
        If include_kinematics=True, returns (ForcePlatform, dict of markers)
    """
    # DJ has very short ground contact time (reactive)
    contact_time_s = 0.200

    force, cop, torque, duration = generate_jump_force_signal(
        bodymass_kg=bodymass_kg,
        jump_height_m=jump_height_m,
        contact_time_s=contact_time_s,
        fsamp=fsamp,
        noise_level=15.0,
        pre_jump_duration_s=0.2,  # Shorter pre-jump for DJ
    )

    # Offset COP based on side to avoid identical positions for bilateral jumps
    if side == "left":
        cop[:, "X"] = cop.to_numpy()[:, 0] - 0.15  # 15cm to the left (negative X)
    elif side == "right":
        cop[:, "X"] = cop.to_numpy()[:, 0] + 0.15  # 15cm to the right (positive X)

    fp = ForcePlatform(
        origin=cop,
        force=force,
        torque=torque
    )

    if include_kinematics:
        markers = generate_kinematic_markers(
            time=cop.index,
            side=side,
            contact_samples=int(contact_time_s * fsamp),
            flight_samples=int(2 * np.sqrt(2 * jump_height_m / G) * fsamp),
            pre_jump_samples=int(0.2 * fsamp),
            fsamp=fsamp
        )
        return fp, markers

    return fp


__all__ = [
    "generate_jump_force_signal",
    "generate_squat_jump_force_platform",
    "generate_cmj_force_platform",
    "generate_drop_jump_force_platform",
]
