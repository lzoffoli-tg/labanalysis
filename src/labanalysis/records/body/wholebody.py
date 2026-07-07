"""WholeBody class - Full body biomechanical model."""

import numpy as np

from ...timeseries.emgsignal import EMGSignal
from ...timeseries.point3d import Point3D
from ...timeseries.signal1d import Signal1D
from ...timeseries.signal3d import Signal3D
from ...timeseries.timeseries import Timeseries
from ..forceplatform import ForcePlatform
from ..record import Record
from ..timeseriesrecord import TimeseriesRecord
from .bodies.head import Head
from .bodies.left_ankle import LeftAnkle
from .bodies.left_elbow import LeftElbow
from .bodies.left_foot import LeftFoot
from .bodies.left_hip import LeftHip
from .bodies.left_knee import LeftKnee
from .bodies.left_shoulder import LeftShoulder
from .bodies.neck import Neck
from .bodies.pelvis import Pelvis
from .bodies.right_ankle import RightAnkle
from .bodies.right_elbow import RightElbow
from .bodies.right_foot import RightFoot
from .bodies.right_hip import RightHip
from .bodies.right_knee import RightKnee
from .bodies.right_shoulder import RightShoulder
from .bodies.trunk import Trunk

__all__ = ["WholeBody"]


class WholeBody(TimeseriesRecord):
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

    def __init__(
        self,
        left_hand_ground_reaction_force: ForcePlatform | None = None,
        right_hand_ground_reaction_force: ForcePlatform | None = None,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        left_heel: Point3D | None = None,
        right_heel: Point3D | None = None,
        left_toe: Point3D | None = None,
        right_toe: Point3D | None = None,
        left_first_metatarsal_head: Point3D | None = None,
        left_fifth_metatarsal_head: Point3D | None = None,
        right_first_metatarsal_head: Point3D | None = None,
        right_fifth_metatarsal_head: Point3D | None = None,
        left_ankle_medial: Point3D | None = None,
        left_ankle_lateral: Point3D | None = None,
        right_ankle_medial: Point3D | None = None,
        right_ankle_lateral: Point3D | None = None,
        left_knee_medial: Point3D | None = None,
        left_knee_lateral: Point3D | None = None,
        right_knee_medial: Point3D | None = None,
        right_knee_lateral: Point3D | None = None,
        right_trochanter: Point3D | None = None,
        left_trochanter: Point3D | None = None,
        left_asis: Point3D | None = None,
        right_asis: Point3D | None = None,
        left_psis: Point3D | None = None,
        right_psis: Point3D | None = None,
        left_shoulder_anterior: Point3D | None = None,
        left_shoulder_posterior: Point3D | None = None,
        left_acromion: Point3D | None = None,
        right_shoulder_anterior: Point3D | None = None,
        right_shoulder_posterior: Point3D | None = None,
        right_acromion: Point3D | None = None,
        left_elbow_medial: Point3D | None = None,
        left_elbow_lateral: Point3D | None = None,
        right_elbow_medial: Point3D | None = None,
        right_elbow_lateral: Point3D | None = None,
        left_wrist_medial: Point3D | None = None,
        left_wrist_lateral: Point3D | None = None,
        right_wrist_medial: Point3D | None = None,
        right_wrist_lateral: Point3D | None = None,
        s2: Point3D | None = None,
        l2: Point3D | None = None,
        c7: Point3D | None = None,
        t5: Point3D | None = None,
        sc: Point3D | None = None,
        head_anterior: Point3D | None = None,
        head_posterior: Point3D | None = None,
        head_left: Point3D | None = None,
        head_right: Point3D | None = None,
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        forces: dict[str, ForcePlatform | None] = dict(
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
        )
        points: dict[str, Point3D | None] = dict(
            left_heel=left_heel,
            right_heel=right_heel,
            left_toe=left_toe,
            right_toe=right_toe,
            left_first_metatarsal_head=left_first_metatarsal_head,
            left_fifth_metatarsal_head=left_fifth_metatarsal_head,
            right_first_metatarsal_head=right_first_metatarsal_head,
            right_fifth_metatarsal_head=right_fifth_metatarsal_head,
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral,
            right_ankle_medial=right_ankle_medial,
            right_ankle_lateral=right_ankle_lateral,
            left_knee_medial=left_knee_medial,
            left_knee_lateral=left_knee_lateral,
            right_knee_medial=right_knee_medial,
            right_knee_lateral=right_knee_lateral,
            left_trochanter=left_trochanter,
            right_trochanter=right_trochanter,
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
            left_shoulder_anterior=left_shoulder_anterior,
            left_shoulder_posterior=left_shoulder_posterior,
            left_acromion=left_acromion,
            right_shoulder_anterior=right_shoulder_anterior,
            right_shoulder_posterior=right_shoulder_posterior,
            right_acromion=right_acromion,
            left_elbow_medial=left_elbow_medial,
            left_elbow_lateral=left_elbow_lateral,
            right_elbow_medial=right_elbow_medial,
            right_elbow_lateral=right_elbow_lateral,
            left_wrist_medial=left_wrist_medial,
            left_wrist_lateral=left_wrist_lateral,
            right_wrist_medial=right_wrist_medial,
            right_wrist_lateral=right_wrist_lateral,
            s2=s2,
            c7=c7,
            t5=t5,
            sc=sc,
            l2=l2,
            head_anterior=head_anterior,
            head_posterior=head_posterior,
            head_left=head_left,
            head_right=head_right,
        )
        i = 0
        keys = list(forces.keys())
        for key in keys:
            if forces[key] is None:
                forces.pop(key)
                continue
            if not isinstance(forces[key], ForcePlatform):
                raise ValueError(f"{i} must be a ForcePlatform object.")
        keys = list(points.keys())
        for v in keys:
            if points[v] is None:
                points.pop(v)
                continue
            if not isinstance(points[v], Point3D):
                raise ValueError(f"{i} must be a Point3D object.")
        extras = extra_signals.copy()
        for i, v in extras.items():
            if not isinstance(v, (Timeseries, Record)):
                raise ValueError(f"{i} must be a Timeseries or Record object.")

        super().__init__(**points, **forces, **extras)  # type: ignore

    @classmethod
    def from_tdf(
        cls,
        filename: str,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        left_heel: str | None = None,
        right_heel: str | None = None,
        left_toe: str | None = None,
        right_toe: str | None = None,
        left_first_metatarsal_head: str | None = None,
        left_fifth_metatarsal_head: str | None = None,
        right_first_metatarsal_head: str | None = None,
        right_fifth_metatarsal_head: str | None = None,
        left_ankle_medial: str | None = None,
        left_ankle_lateral: str | None = None,
        right_ankle_medial: str | None = None,
        right_ankle_lateral: str | None = None,
        left_knee_medial: str | None = None,
        left_knee_lateral: str | None = None,
        right_knee_medial: str | None = None,
        right_knee_lateral: str | None = None,
        right_trochanter: str | None = None,
        left_trochanter: str | None = None,
        left_asis: str | None = None,
        right_asis: str | None = None,
        left_psis: str | None = None,
        right_psis: str | None = None,
        left_shoulder_anterior: str | None = None,
        left_shoulder_posterior: str | None = None,
        left_acromion: str | None = None,
        right_shoulder_anterior: str | None = None,
        right_shoulder_posterior: str | None = None,
        right_acromion: str | None = None,
        left_elbow_medial: str | None = None,
        left_elbow_lateral: str | None = None,
        right_elbow_medial: str | None = None,
        right_elbow_lateral: str | None = None,
        left_wrist_medial: str | None = None,
        left_wrist_lateral: str | None = None,
        right_wrist_medial: str | None = None,
        right_wrist_lateral: str | None = None,
        s2: str | None = None,
        l2: str | None = None,
        c7: str | None = None,
        t5: str | None = None,
        sc: str | None = None,
        head_anterior: str | None = None,
        head_posterior: str | None = None,
        head_left: str | None = None,
        head_right: str | None = None,
    ):
        """
        Create a WholeBody from a TDF file.

        Parameters
        ----------
        filename : str
            Path to TDF file
        left_hand_ground_reaction_force : str, optional
            Label for left hand GRF in TDF
        right_hand_ground_reaction_force : str, optional
            Label for right hand GRF in TDF
        left_foot_ground_reaction_force : str, optional
            Label for left foot GRF in TDF
        right_foot_ground_reaction_force : str, optional
            Label for right foot GRF in TDF
        left_heel : str, optional
            Label for left heel marker in TDF
        right_heel : str, optional
            Label for right heel marker in TDF
        left_toe : str, optional
            Label for left toe marker in TDF
        right_toe : str, optional
            Label for right toe marker in TDF
        left_first_metatarsal_head : str, optional
            Label for left first metatarsal head marker in TDF
        left_fifth_metatarsal_head : str, optional
            Label for left fifth metatarsal head marker in TDF
        right_first_metatarsal_head : str, optional
            Label for right first metatarsal head marker in TDF
        right_fifth_metatarsal_head : str, optional
            Label for right fifth metatarsal head marker in TDF
        left_ankle_medial : str, optional
            Label for left medial malleolus marker in TDF
        left_ankle_lateral : str, optional
            Label for left lateral malleolus marker in TDF
        right_ankle_medial : str, optional
            Label for right medial malleolus marker in TDF
        right_ankle_lateral : str, optional
            Label for right lateral malleolus marker in TDF
        left_knee_medial : str, optional
            Label for left medial femoral epicondyle marker in TDF
        left_knee_lateral : str, optional
            Label for left lateral femoral epicondyle marker in TDF
        right_knee_medial : str, optional
            Label for right medial femoral epicondyle marker in TDF
        right_knee_lateral : str, optional
            Label for right lateral femoral epicondyle marker in TDF
        left_trochanter : str, optional
            Label for left greater trochanter marker in TDF
        right_trochanter : str, optional
            Label for right greater trochanter marker in TDF
        left_asis : str, optional
            Label for left ASIS marker in TDF
        right_asis : str, optional
            Label for right ASIS marker in TDF
        left_psis : str, optional
            Label for left PSIS marker in TDF
        right_psis : str, optional
            Label for right PSIS marker in TDF
        left_shoulder_anterior : str, optional
            Label for left anterior shoulder marker in TDF
        left_shoulder_posterior : str, optional
            Label for left posterior shoulder marker in TDF
        left_acromion : str, optional
            Label for left acromion marker in TDF
        right_shoulder_anterior : str, optional
            Label for right anterior shoulder marker in TDF
        right_shoulder_posterior : str, optional
            Label for right posterior shoulder marker in TDF
        right_acromion : str, optional
            Label for right acromion marker in TDF
        left_elbow_medial : str, optional
            Label for left medial epicondyle marker in TDF
        left_elbow_lateral : str, optional
            Label for left lateral epicondyle marker in TDF
        right_elbow_medial : str, optional
            Label for right medial epicondyle marker in TDF
        right_elbow_lateral : str, optional
            Label for right lateral epicondyle marker in TDF
        left_wrist_medial : str, optional
            Label for left medial wrist marker in TDF
        left_wrist_lateral : str, optional
            Label for left lateral wrist marker in TDF
        right_wrist_medial : str, optional
            Label for right medial wrist marker in TDF
        right_wrist_lateral : str, optional
            Label for right lateral wrist marker in TDF
        s2 : str, optional
            Label for second sacral vertebra marker in TDF
        l2 : str, optional
            Label for second lumbar vertebra marker in TDF
        c7 : str, optional
            Label for seventh cervical vertebra marker in TDF
        t5 : str, optional
            Label for fifth thoracic vertebra marker in TDF
        sc : str, optional
            Label for sternoclavicular joint marker in TDF
        head_anterior : str, optional
            Label for anterior cranium marker in TDF
        head_posterior : str, optional
            Label for posterior cranium marker in TDF
        head_left : str, optional
            Label for left cranium marker in TDF
        head_right : str, optional
            Label for right cranium marker in TDF

        Returns
        -------
        WholeBody
            Instance created from TDF data
        """
        tdf = TimeseriesRecord.from_tdf(filename)
        points = {
            "left_heel": left_heel,
            "right_heel": right_heel,
            "left_toe": left_toe,
            "right_toe": right_toe,
            "left_first_metatarsal_head": left_first_metatarsal_head,
            "left_fifth_metatarsal_head": left_fifth_metatarsal_head,
            "right_first_metatarsal_head": right_first_metatarsal_head,
            "right_fifth_metatarsal_head": right_fifth_metatarsal_head,
            "left_ankle_medial": left_ankle_medial,
            "left_ankle_lateral": left_ankle_lateral,
            "right_ankle_medial": right_ankle_medial,
            "right_ankle_lateral": right_ankle_lateral,
            "left_knee_medial": left_knee_medial,
            "left_knee_lateral": left_knee_lateral,
            "right_knee_medial": right_knee_medial,
            "right_knee_lateral": right_knee_lateral,
            "right_trochanter": right_trochanter,
            "left_trochanter": left_trochanter,
            "left_asis": left_asis,
            "right_asis": right_asis,
            "left_psis": left_psis,
            "right_psis": right_psis,
            "left_shoulder_anterior": left_shoulder_anterior,
            "left_shoulder_posterior": left_shoulder_posterior,
            "left_acromion": left_acromion,
            "right_shoulder_anterior": right_shoulder_anterior,
            "right_shoulder_posterior": right_shoulder_posterior,
            "right_acromion": right_acromion,
            "left_elbow_medial": left_elbow_medial,
            "left_elbow_lateral": left_elbow_lateral,
            "right_elbow_medial": right_elbow_medial,
            "right_elbow_lateral": right_elbow_lateral,
            "left_wrist_medial": left_wrist_medial,
            "left_wrist_lateral": left_wrist_lateral,
            "right_wrist_medial": right_wrist_medial,
            "right_wrist_lateral": right_wrist_lateral,
            "s2": s2,
            "c7": c7,
            "l2": l2,
            "t5": t5,
            "sc": sc,
            "head_anterior": head_anterior,
            "head_posterior": head_posterior,
            "head_left": head_left,
            "head_right": head_right,
        }
        forces = {
            "left_hand_ground_reaction_force": left_hand_ground_reaction_force,
            "right_hand_ground_reaction_force": right_hand_ground_reaction_force,
            "left_foot_ground_reaction_force": left_foot_ground_reaction_force,
            "right_foot_ground_reaction_force": right_foot_ground_reaction_force,
        }
        out = cls()
        keys = list(points.keys())
        for key in keys:
            val = points[key]
            if val is None:
                points.pop(key)
        points_vals = np.array(list(points.values()))
        points_keys = np.array(list(points.keys()))
        keys = list(forces.keys())
        for key in keys:
            val = forces[key]
            if val is None:
                forces.pop(key)
        forceplatforms_vals = np.array(list(forces.values()))
        forceplatforms_keys = np.array(list(forces.keys()))
        for key, val in tdf.items():
            if key in points_vals:
                if not isinstance(val, Point3D):
                    raise ValueError("key must be a Point3D")
                idx = np.where(points_vals == key)[0][0]
                out[points_keys[idx]] = val

            elif key in forceplatforms_vals:
                if not isinstance(val, ForcePlatform):
                    raise ValueError("key must be a ForcePlatform")
                idx = np.where(forceplatforms_vals == key)[0][0]
                out[forceplatforms_keys[idx]] = val

            elif isinstance(val, (Timeseries, Record)):
                out[key] = val

            else:
                raise ValueError(f"{key} must be a Record or Timeseries")

        return out

    @property
    def left_hand_ground_reaction_force(self):
        """left hand ground reaction force"""
        if "left_hand_ground_reaction_force" in self.keys():
            out: ForcePlatform = self["left_hand_ground_reaction_force"]  # type: ignore
            return out
        return None

    @property
    def right_hand_ground_reaction_force(self):
        """right hand ground reaction force"""
        if "right_hand_ground_reaction_force" in self.keys():
            out: ForcePlatform = self["right_hand_ground_reaction_force"]  # type: ignore
            return out
        return None

    @property
    def left_foot_ground_reaction_force(self):
        """left foot ground reaction force"""
        if "left_foot_ground_reaction_force" in self.keys():
            out: ForcePlatform = self["left_foot_ground_reaction_force"]  # type: ignore
            return out
        return None

    @property
    def right_foot_ground_reaction_force(self):
        """right foot ground reaction force"""
        if "right_foot_ground_reaction_force" in self.keys():
            out: ForcePlatform = self["right_foot_ground_reaction_force"]  # type: ignore
            return out
        return None

    @property
    def left_heel(self):
        """left heel point"""
        if "left_heel" in self.keys():
            out: Point3D = self["left_heel"]  # type: ignore
            return out
        return None

    @property
    def right_heel(self):
        """right heel point"""
        if "right_heel" in self.keys():
            out: Point3D = self["right_heel"]  # type: ignore
            return out
        return None

    @property
    def left_toe(self):
        """left toe point"""
        if "left_toe" in self.keys():
            out: Point3D = self["left_toe"]  # type: ignore
            return out
        return None

    @property
    def right_toe(self):
        """right toe point"""
        if "right_toe" in self.keys():
            out: Point3D = self["right_toe"]  # type: ignore
            return out
        return None

    @property
    def left_first_metatarsal_head(self):
        """left first metatarsal head point"""
        if "left_first_metatarsal_head" in self.keys():
            out: Point3D = self["left_first_metatarsal_head"]  # type: ignore
            return out
        return None

    @property
    def right_first_metatarsal_head(self):
        """right first metatarsal head point"""
        if "right_first_metatarsal_head" in self.keys():
            out: Point3D = self["right_first_metatarsal_head"]  # type: ignore
            return out
        return None

    @property
    def left_fifth_metatarsal_head(self):
        """left fifth metatarsal head point"""
        if "left_fifth_metatarsal_head" in self.keys():
            out: Point3D = self["left_fifth_metatarsal_head"]  # type: ignore
            return out
        return None

    @property
    def right_fifth_metatarsal_head(self):
        """right fifth metatarsal head point"""
        if "right_fifth_metatarsal_head" in self.keys():
            out: Point3D = self["right_fifth_metatarsal_head"]  # type: ignore
            return out
        return None

    @property
    def left_ankle_medial(self):
        """left ankle medial point"""
        if "left_ankle_medial" in self.keys():
            out: Point3D = self["left_ankle_medial"]  # type: ignore
            return out
        return None

    @property
    def left_ankle_lateral(self):
        """left ankle lateral point"""
        if "left_ankle_lateral" in self.keys():
            out: Point3D = self["left_ankle_lateral"]  # type: ignore
            return out
        return None

    @property
    def right_ankle_medial(self):
        """right ankle medial point"""
        if "right_ankle_medial" in self.keys():
            out: Point3D = self["right_ankle_medial"]  # type: ignore
            return out
        return None

    @property
    def right_ankle_lateral(self):
        """right ankle lateral point"""
        if "right_ankle_lateral" in self.keys():
            out: Point3D = self["right_ankle_lateral"]  # type: ignore
            return out
        return None

    @property
    def left_knee_medial(self):
        """left knee medial point"""
        if "left_knee_medial" in self.keys():
            out: Point3D = self["left_knee_medial"]  # type: ignore
            return out
        return None

    @property
    def left_knee_lateral(self):
        """left knee lateral point"""
        if "left_knee_lateral" in self.keys():
            out: Point3D = self["left_knee_lateral"]  # type: ignore
            return out
        return None

    @property
    def right_knee_medial(self):
        """right knee medial point"""
        if "right_knee_medial" in self.keys():
            out: Point3D = self["right_knee_medial"]  # type: ignore
            return out
        return None

    @property
    def right_knee_lateral(self):
        """right knee lateral point"""
        if "right_knee_lateral" in self.keys():
            out: Point3D = self["right_knee_lateral"]  # type: ignore
            return out
        return None

    @property
    def right_trochanter(self):
        """right trochanter point"""
        if "right_trochanter" in self.keys():
            out: Point3D = self["right_trochanter"]  # type: ignore
            return out
        return None

    @property
    def left_trochanter(self):
        """left trochanter point"""
        if "left_trochanter" in self.keys():
            out: Point3D = self["left_trochanter"]  # type: ignore
            return out
        return None

    @property
    def left_asis(self):
        """left asis point"""
        if "left_asis" in self.keys():
            out: Point3D = self["left_asis"]  # type: ignore
            return out
        return None

    @property
    def right_asis(self):
        """right asis point"""
        if "right_asis" in self.keys():
            out: Point3D = self["right_asis"]  # type: ignore
            return out
        return None

    @property
    def left_psis(self):
        """left psis point"""
        if "left_psis" in self.keys():
            out: Point3D = self["left_psis"]  # type: ignore
            return out
        return None

    @property
    def right_psis(self):
        """right psis point"""
        if "right_psis" in self.keys():
            out: Point3D = self["right_psis"]  # type: ignore
            return out
        return None

    @property
    def left_shoulder_anterior(self):
        """left shoulder anterior point"""
        if "left_shoulder_anterior" in self.keys():
            out: Point3D = self["left_shoulder_anterior"]  # type: ignore
            return out
        return None

    @property
    def left_shoulder_posterior(self):
        """left shoulder posterior point"""
        if "left_shoulder_posterior" in self.keys():
            out: Point3D = self["left_shoulder_posterior"]  # type: ignore
            return out
        return None

    @property
    def left_acromion(self):
        """left acromion point"""
        if "left_acromion" in self.keys():
            out: Point3D = self["left_acromion"]  # type: ignore
            return out
        return None

    @property
    def right_shoulder_anterior(self):
        """right shoulder anterior point"""
        if "right_shoulder_anterior" in self.keys():
            out: Point3D = self["right_shoulder_anterior"]  # type: ignore
            return out
        return None

    @property
    def right_shoulder_posterior(self):
        """right shoulder posterior point"""
        if "right_shoulder_posterior" in self.keys():
            out: Point3D = self["right_shoulder_posterior"]  # type: ignore
            return out
        return None

    @property
    def right_acromion(self):
        """right acromion point"""
        if "right_acromion" in self.keys():
            out: Point3D = self["right_acromion"]  # type: ignore
            return out
        return None

    @property
    def left_elbow_medial(self):
        """left elbow medial point"""
        if "left_elbow_medial" in self.keys():
            out: Point3D = self["left_elbow_medial"]  # type: ignore
            return out
        return None

    @property
    def left_elbow_lateral(self):
        """left elbow lateral point"""
        if "left_elbow_lateral" in self.keys():
            out: Point3D = self["left_elbow_lateral"]  # type: ignore
            return out
        return None

    @property
    def right_elbow_medial(self):
        """right elbow medial point"""
        if "right_elbow_medial" in self.keys():
            out: Point3D = self["right_elbow_medial"]  # type: ignore
            return out
        return None

    @property
    def right_elbow_lateral(self):
        """right elbow lateral point"""
        if "right_elbow_lateral" in self.keys():
            out: Point3D = self["right_elbow_lateral"]  # type: ignore
            return out
        return None

    @property
    def left_wrist_medial(self):
        """left wrist medial point"""
        if "left_wrist_medial" in self.keys():
            out: Point3D = self["left_wrist_medial"]  # type: ignore
            return out
        return None

    @property
    def left_wrist_lateral(self):
        """left wrist lateral point"""
        if "left_wrist_lateral" in self.keys():
            out: Point3D = self["left_wrist_lateral"]  # type: ignore
            return out
        return None

    @property
    def right_wrist_medial(self):
        """right wrist medial point"""
        if "right_wrist_medial" in self.keys():
            out: Point3D = self["right_wrist_medial"]  # type: ignore
            return out
        return None

    @property
    def right_wrist_lateral(self):
        """right wrist lateral point"""
        if "right_wrist_lateral" in self.keys():
            out: Point3D = self["right_wrist_lateral"]  # type: ignore
            return out
        return None

    @property
    def s2(self):
        """s2 point"""
        if "s2" in self.keys():
            out: Point3D = self["s2"]  # type: ignore
            return out
        return None

    @property
    def l2(self):
        """l2 point"""
        if "l2" in self.keys():
            out: Point3D = self["l2"]  # type: ignore
            return out
        return None

    @property
    def c7(self):
        """c7 point"""
        if "c7" in self.keys():
            out: Point3D = self["c7"]  # type: ignore
            return out
        return None

    @property
    def t5(self):
        """t5 point"""
        if "t5" in self.keys():
            out: Point3D = self["t5"]  # type: ignore
            return out
        return None

    @property
    def sc(self):
        """sc point"""
        if "sc" in self.keys():
            out: Point3D = self["sc"]  # type: ignore
            return out
        return None

    @property
    def head_anterior(self):
        """head anterior point"""
        if "head_anterior" in self.keys():
            out: Point3D = self["head_anterior"]  # type: ignore
            return out
        return None

    @property
    def head_posterior(self):
        """head posterior point"""
        if "head_posterior" in self.keys():
            out: Point3D = self["head_posterior"]  # type: ignore
            return out
        return None

    @property
    def head_left(self):
        """head left point"""
        if "head_left" in self.keys():
            out: Point3D = self["head_left"]  # type: ignore
            return out
        return None

    @property
    def head_right(self):
        """head right point"""
        if "head_right" in self.keys():
            out: Point3D = self["head_right"]  # type: ignore
            return out
        return None

    @property
    def left_foot(self):
        """left foot plane"""
        if (
            self.left_toe is not None
            and self.left_heel is not None
            and self.left_fifth_metatarsal_head is not None
        ):
            return LeftFoot(
                self.left_toe,
                self.left_heel,
                self.left_fifth_metatarsal_head,
                self.left_first_metatarsal_head,
            )
        return None

    @property
    def right_foot(self):
        """right foot plane"""
        if (
            self.right_toe is not None
            and self.right_heel is not None
            and self.right_fifth_metatarsal_head is not None
        ):
            return RightFoot(
                self.right_toe,
                self.right_heel,
                self.right_fifth_metatarsal_head,
                self.right_first_metatarsal_head,
            )
        return None

    @property
    def left_ankle(self):
        """left ankle joint"""
        if (
            self.left_ankle_lateral is not None
            and self.left_ankle_medial is not None
            and self.left_knee_lateral is not None
            and self.left_knee_medial is not None
            and self.left_foot is not None
        ):
            return LeftAnkle(
                self.left_ankle_lateral,
                self.left_ankle_medial,
                self.left_knee_lateral,
                self.left_knee_medial,
                self.left_foot,
            )
        return None

    @property
    def right_ankle(self):
        """right ankle joint"""
        if (
            self.right_ankle_lateral is not None
            and self.right_ankle_medial is not None
            and self.right_knee_lateral is not None
            and self.right_knee_medial is not None
            and self.right_foot is not None
        ):
            return RightAnkle(
                self.right_ankle_lateral,
                self.right_ankle_medial,
                self.right_knee_lateral,
                self.right_knee_medial,
                self.right_foot,
            )
        return None

    @property
    def pelvis(self):
        """pelvis plane"""
        if (
            self.left_asis is not None
            and self.left_psis is not None
            and self.right_asis is not None
            and self.right_psis is not None
        ):
            return Pelvis(
                self.left_asis,
                self.right_asis,
                self.left_psis,
                self.right_psis,
            )
        return None

    @property
    def left_hip(self):
        """left hip joint"""
        if (
            self.pelvis is not None
            and self.left_knee_lateral is not None
            and self.left_knee_medial is not None
        ):
            return LeftHip(
                self.pelvis,
                self.left_knee_lateral,
                self.left_knee_medial,
                self.left_trochanter,
            )
        return None

    @property
    def right_hip(self):
        """right hip joint"""
        if (
            self.pelvis is not None
            and self.right_knee_lateral is not None
            and self.right_knee_medial is not None
        ):
            return RightHip(
                self.pelvis,
                self.right_knee_lateral,
                self.right_knee_medial,
                self.right_trochanter,
            )
        return None

    @property
    def left_knee(self):
        """left knee joint"""
        if (
            self.left_hip is not None
            and self.left_ankle is not None
            and self.left_knee_lateral is not None
            and self.left_knee_medial is not None
        ):
            return LeftKnee(
                self.left_hip,
                self.left_ankle,
                self.left_knee_lateral,
                self.left_knee_medial,
            )
        return None

    @property
    def right_knee(self):
        """right knee joint"""
        if (
            self.right_hip is not None
            and self.right_ankle is not None
            and self.right_knee_lateral is not None
            and self.right_knee_medial is not None
        ):
            return RightKnee(
                self.right_hip,
                self.right_ankle,
                self.right_knee_lateral,
                self.right_knee_medial,
            )
        return None

    @property
    def trunk(self):
        """trunk joint"""
        if (
            self.c7 is not None
            and self.sc is not None
            and self.sx is not None
            and self.l2 is not None
            and self.t5 is not None
            and self.pelvis_plane is not None
            and self.left_hip_joint is not None
            and self.right_hip_joint is not None
        ):
            return Trunk(
                self.c7,
                self.sc,
                self.l2,
                self.t5,
                self.pelvis_plane,
                self.left_hip_joint,
                self.right_hip_joint,
            )
        return None

    @property
    def head(self):
        """head plane"""
        if (
            self.head_anterior is not None
            and self.head_posterior is not None
            and self.head_left is not None
            and self.head_right is not None
        ):
            return Head(
                self.head_anterior,
                self.head_posterior,
                self.head_left,
                self.head_right,
            )
        return None

    @property
    def neck(self):
        """neck joint"""
        if (
            self.c7 is not None
            and self.sc is not None
            and self.pelvis_plane is not None
            and self.head_plane is not None
            and self.left_shoulder is not None
            and self.right_shoulder is not None
        ):
            return Neck(
                self.c7,
                self.sc,
                self.pelvis_plane,
                self.head_plane,
                self.left_shoulder,
                self.right_shoulder,
            )
        return None

    @property
    def left_shoulder(self):
        """left shoulder joint"""
        if (
            self.c7 is not None
            and self.sc is not None
            and self.pelvis_plane is not None
            and self.left_acromion is not None
            and self.left_elbow_lateral is not None
            and self.left_elbow_medial is not None
        ):
            return LeftShoulder(
                self.c7,
                self.sc,
                self.pelvis_plane,
                self.left_acromion,
                self.left_elbow_lateral,
                self.left_elbow_lateral,
            )
        return None

    @property
    def right_shoulder(self):
        """right shoulder joint"""
        if (
            self.c7 is not None
            and self.sc is not None
            and self.pelvis_plane is not None
            and self.right_acromion is not None
            and self.right_elbow_lateral is not None
            and self.right_elbow_medial is not None
        ):
            return RightShoulder(
                self.c7,
                self.sc,
                self.pelvis_plane,
                self.right_acromion,
                self.right_elbow_lateral,
                self.right_elbow_lateral,
            )
        return None

    @property
    def left_elbow(self):
        """left elbow joint"""
        if (
            self.left_shoulder_joint is not None
            and self.left_elbow_lateral is not None
            and self.left_elbow_medial is not None
            and self.left_wrist_lateral is not None
            and self.left_wrist_medial is not None
        ):
            return LeftElbow(
                self.left_shoulder_joint,
                self.left_elbow_lateral,
                self.left_elbow_medial,
                self.left_wrist_lateral,
                self.left_wrist_medial,
            )
        return None

    @property
    def right_elbow(self):
        """right elbow joint"""
        if (
            self.right_shoulder_joint is not None
            and self.right_elbow_lateral is not None
            and self.right_elbow_medial is not None
            and self.right_wrist_lateral is not None
            and self.right_wrist_medial is not None
        ):
            return RightElbow(
                self.right_shoulder_joint,
                self.right_elbow_lateral,
                self.right_elbow_medial,
                self.right_wrist_lateral,
                self.right_wrist_medial,
            )
        return None
