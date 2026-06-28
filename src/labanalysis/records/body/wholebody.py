"""WholeBody class - Full body biomechanical model."""

from ._base import WholeBodyBase
from ._helpers import HelpersMixin
from ._aggregation import AggregationMixin
from .joint_centers import JointCentersMixin
from .anthropometry import AnthropometryMixin
from .angles import AngularMeasuresMixin

__all__ = ["WholeBody"]


class WholeBody(
    WholeBodyBase,
    HelpersMixin,
    JointCentersMixin,
    AnthropometryMixin,
    AngularMeasuresMixin,
    AggregationMixin,
):
    """
    Full body biomechanical model with 42+ anatomical markers.

    Provides comprehensive biomechanical analysis including:
    - Joint centers and reference frames (ankle, knee, hip, shoulder, elbow, wrist, neck, head)
    - Anthropometric measurements (segment lengths, widths, heights)
    - Joint angles (flexion/extension, abduction/adduction, rotation for all major joints)
    - Pelvis and trunk kinematics (tilt, rotation in global and local frames)
    - Spine curvature (lumbar lordosis, dorsal kyphosis)

    The class uses a modular mixin composition pattern where functionality is organized
    into logical categories:
    - WholeBodyBase: Core initialization and marker management
    - HelpersMixin: Geometric calculations (planes, projections, De Leva regression)
    - JointCentersMixin: All joint center and reference frame properties
    - AnthropometryMixin: All segment length and width measurements
    - AngularMeasuresMixin: All joint angle calculations
    - AggregationMixin: Methods to combine properties (segment_lengths, joint_angles, copy)

    Parameters
    ----------
    All anatomical markers are optional Point3D instances:

    Pelvis markers:
        left_asis, right_asis : Point3D, optional
            Anterior superior iliac spine markers
        left_psis, right_psis : Point3D, optional
            Posterior superior iliac spine markers
        left_trochanter, right_trochanter : Point3D, optional
            Greater trochanter markers

    Lower limb markers:
        left_knee_lateral, right_knee_lateral : Point3D, optional
            Lateral knee markers (lateral epicondyle of femur)
        left_knee_medial, right_knee_medial : Point3D, optional
            Medial knee markers (medial epicondyle of femur)
        left_ankle_lateral, right_ankle_lateral : Point3D, optional
            Lateral ankle markers (lateral malleolus)
        left_ankle_medial, right_ankle_medial : Point3D, optional
            Medial ankle markers (medial malleolus)
        left_heel, right_heel : Point3D, optional
            Heel markers (calcaneus)
        left_toe, right_toe : Point3D, optional
            Toe markers (2nd metatarsal head or equivalent)
        left_fifth_met, right_fifth_met : Point3D, optional
            Fifth metatarsal head markers
        left_first_met, right_first_met : Point3D, optional
            First metatarsal head markers

    Upper limb markers:
        left_acromion, right_acromion : Point3D, optional
            Acromion process markers
        left_shoulder_anterior, right_shoulder_anterior : Point3D, optional
            Anterior shoulder markers
        left_shoulder_posterior, right_shoulder_posterior : Point3D, optional
            Posterior shoulder markers
        left_elbow_lateral, right_elbow_lateral : Point3D, optional
            Lateral elbow markers (lateral epicondyle of humerus)
        left_elbow_medial, right_elbow_medial : Point3D, optional
            Medial elbow markers (medial epicondyle of humerus)
        left_wrist_radial, right_wrist_radial : Point3D, optional
            Radial styloid process markers
        left_wrist_ulnar, right_wrist_ulnar : Point3D, optional
            Ulnar styloid process markers

    Axial markers:
        vertex : Point3D, optional
            Top of the head marker
        c7 : Point3D, optional
            7th cervical vertebra marker
        sternum : Point3D, optional
            Sternum marker (jugular notch or xiphoid process)

    EMG signals (optional):
        left_rectus_femoris, right_rectus_femoris : EMGSignal, optional
        left_vastus_lateralis, right_vastus_lateralis : EMGSignal, optional
        left_vastus_medialis, right_vastus_medialis : EMGSignal, optional
        left_biceps_femoris, right_biceps_femoris : EMGSignal, optional
        left_semitendinosus, right_semitendinosus : EMGSignal, optional
        left_tibialis_anterior, right_tibialis_anterior : EMGSignal, optional
        left_gastrocnemius_medialis, right_gastrocnemius_medialis : EMGSignal, optional
        left_gastrocnemius_lateralis, right_gastrocnemius_lateralis : EMGSignal, optional
        left_soleus, right_soleus : EMGSignal, optional

    Force platforms (optional):
        ground_reaction_force : ForcePlatform, optional
            Ground reaction force data

    Examples
    --------
    Create from individual markers:

    >>> left_asis = Point3D(...)
    >>> right_asis = Point3D(...)
    >>> body = WholeBody(left_asis=left_asis, right_asis=right_asis, ...)

    Load from TDF file:

    >>> body = WholeBody.from_tdf(
    ...     "capture.tdf",
    ...     left_asis="L_ASIS",
    ...     right_asis="R_ASIS",
    ...     left_knee_lateral="L_KNEE_LAT",
    ...     # ... other marker labels ...
    ... )

    Access computed properties:

    >>> # Joint centers
    >>> left_ankle_center = body.left_ankle
    >>> pelvis_center = body.pelvis_center
    >>>
    >>> # Anthropometry
    >>> left_leg_len = body.left_leg_length
    >>> shoulder_w = body.shoulder_width
    >>>
    >>> # Joint angles
    >>> left_knee_angle = body.left_knee_flexionextension
    >>> pelvis_tilt = body.pelvis_anteroposterior_tilt_global
    >>>
    >>> # Aggregate all measurements
    >>> all_lengths = body.segment_lengths()
    >>> all_angles = body.joint_angles

    Notes
    -----
    - Properties gracefully degrade when markers are missing: they return Signal1D/Point3D
      filled with NaN instead of raising errors
    - Reference frames follow ISB recommendations where applicable
    - Angles use anatomical conventions (flexion/extension, abduction/adduction, etc.)
    - The class supports deep copying via the `copy()` method

    See Also
    --------
    TimeseriesRecord : Base class for timeseries data
    Point3D : 3D point timeseries
    Signal1D : 1D signal timeseries
    ReferenceFrame : 3D reference frame with orientation
    """

    # List of angular measure properties for introspection
    # Automatically extracted from AngularMeasuresMixin via introspection
    _angular_measures = [
        name for name in dir(AngularMeasuresMixin)
        if isinstance(getattr(AngularMeasuresMixin, name, None), property)
        and not name.startswith('_')  # Exclude private properties like _left_foot_plane
        and not name.endswith('_referenceframe')  # Exclude reference frames (not angles)
    ]
