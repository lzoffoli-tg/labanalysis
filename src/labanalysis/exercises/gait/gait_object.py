"""Base class for gait analysis."""

from typing import Literal
import warnings

from ...constants import *
from ...signalprocessing import *
from ...timeseries import *
from ...records.body import WholeBody
from ...records.forceplatform import ForcePlatform

__all__ = ["GaitObject"]


class GaitObject(WholeBody):
    """
    Base class for gait analysis with kinetic and kinematic cycle detection.

    GaitObject extends WholeBody to provide specialized functionality for gait
    analysis, including support for multiple cycle detection algorithms, ground
    reaction force tracking, and gait-specific anatomical landmarks.

    The class supports two cycle detection algorithms:
    - 'kinetics': Uses force platform data (ground reaction forces) to detect
      foot contact events and gait cycles.
    - 'kinematics': Uses marker trajectories (heel and toe positions) to detect
      foot contact events and gait cycles based on vertical position thresholds.

    The algorithm selection is automatic based on available data, with fallback
    logic if the preferred algorithm cannot be used.

    Parameters
    ----------
    algorithm : {'kinematics', 'kinetics'}
        Cycle detection algorithm to use.
    ground_reaction_force_threshold : float or int, optional
        Minimum ground reaction force (in Newtons) for contact detection when
        using kinetics algorithm. Default is DEFAULT_MINIMUM_CONTACT_GRF_N.
    height_threshold : float or int, optional
        Maximum vertical height (as percentage) for contact detection when
        using kinematics algorithm. Default is DEFAULT_MINIMUM_HEIGHT_PERCENTAGE.
    left_hand_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for left hand contact.
    right_hand_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for right hand contact.
    left_foot_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for left foot contact.
    right_foot_ground_reaction_force : ForcePlatform or None, optional
        Force platform data for right foot contact.
    left_heel : Point3D or None, optional
        Left heel marker trajectory.
    right_heel : Point3D or None, optional
        Right heel marker trajectory.
    left_toe : Point3D or None, optional
        Left toe marker trajectory.
    right_toe : Point3D or None, optional
        Right toe marker trajectory.
    left_first_metatarsal_head : Point3D or None
        Left first metatarsal head marker.
    left_fifth_metatarsal_head : Point3D or None
        Left fifth metatarsal head marker.
    right_first_metatarsal_head : Point3D or None
        Right first metatarsal head marker.
    right_fifth_metatarsal_head : Point3D or None, optional
        Left metatarsal head marker trajectory.
    right_metatarsal_head : Point3D or None, optional
        Right metatarsal head marker trajectory.
    left_acromion : Point3D or None, optional
        Left acromion marker (shoulder tip) trajectory.
    right_acromion : Point3D or None, optional
        Right acromion marker (shoulder tip) trajectory.
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals (e.g., joint angles, EMG channels, other markers).

    Notes
    -----
    This class inherits all 42 anatomical markers from WholeBody (38 markers + 4 force platforms).
    Only the most commonly used markers for gait analysis are listed above. See WholeBody
    documentation for the complete list of available anatomical markers including ankle, knee,
    hip, shoulder, elbow, wrist markers, and spinal markers (s2, l2, c7, sc).

    Attributes
    ----------
    algorithm : str
        The selected cycle detection algorithm ('kinetics' or 'kinematics').
    ground_reaction_force_threshold : float
        Ground reaction force threshold for contact detection (Newtons).
    height_threshold : float
        Height threshold for contact detection (percentage).

    Notes
    -----
    Algorithm selection follows these rules:
    1. If 'kinetics' is requested but no force platform data is available,
       automatically falls back to 'kinematics' (with warning).
    2. If 'kinematics' is requested but marker data is incomplete,
       automatically falls back to 'kinetics' (with warning).
    3. If neither algorithm can be satisfied, raises ValueError.

    The kinematics algorithm requires all four markers: left_heel, right_heel,
    left_toe, and right_toe. The kinetics algorithm requires at least one
    ForcePlatform object providing ground reaction force data.

    See Also
    --------
    WholeBody : Parent class providing biomechanical body model.
    GaitCycle : Represents a single gait cycle.
    GaitExercise : Represents a sequence of gait cycles.
    """

    _algorithm: Literal["kinetics", "kinematics"]
    _grf_threshold: float
    _height_threshold: float

    # * constructor

    def __init__(
        self,
        speed: float | int,
        grade: float | int,
        algorithm: Literal["kinematics", "kinetics"] = "kinetics",
        ground_reaction_force_threshold: float | int = DEFAULT_MINIMUM_CONTACT_GRF_N,
        height_threshold: float | int = DEFAULT_MINIMUM_HEIGHT_PERCENTAGE,
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
        sc: Point3D | None = None,  # sternoclavicular joint
        head_anterior: Point3D | None = None,
        head_posterior: Point3D | None = None,
        head_left: Point3D | None = None,
        head_right: Point3D | None = None,
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a GaitObject.

        Parameters
        ----------
        speed : float or int
            Gait speed value.
        grade : float or int
            Gait grade (incline) value.
        algorithm : {'kinematics', 'kinetics'}
            Cycle detection algorithm to use.
        ground_reaction_force_threshold : float or int, optional
            Minimum ground reaction force (in Newtons) for contact detection when
            using kinetics algorithm. Default is DEFAULT_MINIMUM_CONTACT_GRF_N.
        height_threshold : float or int, optional
            Maximum vertical height (as percentage) for contact detection when
            using kinematics algorithm. Default is DEFAULT_MINIMUM_HEIGHT_PERCENTAGE.
        left_hand_ground_reaction_force : ForcePlatform or None, optional
            Force platform data for left hand contact.
        right_hand_ground_reaction_force : ForcePlatform or None, optional
            Force platform data for right hand contact.
        left_foot_ground_reaction_force : ForcePlatform or None, optional
            Force platform data for left foot contact.
        right_foot_ground_reaction_force : ForcePlatform or None, optional
            Force platform data for right foot contact.
        left_heel : Point3D or None, optional
            Left heel marker trajectory.
        right_heel : Point3D or None, optional
            Right heel marker trajectory.
        left_toe : Point3D or None, optional
            Left toe marker trajectory.
        right_toe : Point3D or None, optional
            Right toe marker trajectory.
        left_first_metatarsal_head : Point3D or None, optional
            Left first metatarsal head marker.
        left_fifth_metatarsal_head : Point3D or None, optional
            Left fifth metatarsal head marker.
        right_first_metatarsal_head : Point3D or None, optional
            Right first metatarsal head marker.
        right_fifth_metatarsal_head : Point3D or None, optional
            Right fifth metatarsal head marker.
        left_ankle_medial : Point3D or None, optional
            Left ankle medial malleolus marker.
        left_ankle_lateral : Point3D or None, optional
            Left ankle lateral malleolus marker.
        right_ankle_medial : Point3D or None, optional
            Right ankle medial malleolus marker.
        right_ankle_lateral : Point3D or None, optional
            Right ankle lateral malleolus marker.
        left_knee_medial : Point3D or None, optional
            Left knee medial epicondyle marker.
        left_knee_lateral : Point3D or None, optional
            Left knee lateral epicondyle marker.
        right_knee_medial : Point3D or None, optional
            Right knee medial epicondyle marker.
        right_knee_lateral : Point3D or None, optional
            Right knee lateral epicondyle marker.
        left_trochanter : Point3D or None, optional
            Left greater trochanter marker.
        right_trochanter : Point3D or None, optional
            Right greater trochanter marker.
        left_asis : Point3D or None, optional
            Left anterior superior iliac spine marker.
        right_asis : Point3D or None, optional
            Right anterior superior iliac spine marker.
        left_psis : Point3D or None, optional
            Left posterior superior iliac spine marker.
        right_psis : Point3D or None, optional
            Right posterior superior iliac spine marker.
        left_shoulder_anterior : Point3D or None, optional
            Left shoulder anterior marker.
        left_shoulder_posterior : Point3D or None, optional
            Left shoulder posterior marker.
        left_acromion : Point3D or None, optional
            Left acromion (shoulder tip) marker.
        right_shoulder_anterior : Point3D or None, optional
            Right shoulder anterior marker.
        right_shoulder_posterior : Point3D or None, optional
            Right shoulder posterior marker.
        right_acromion : Point3D or None, optional
            Right acromion (shoulder tip) marker.
        left_elbow_medial : Point3D or None, optional
            Left elbow medial epicondyle marker.
        left_elbow_lateral : Point3D or None, optional
            Left elbow lateral epicondyle marker.
        right_elbow_medial : Point3D or None, optional
            Right elbow medial epicondyle marker.
        right_elbow_lateral : Point3D or None, optional
            Right elbow lateral epicondyle marker.
        left_wrist_medial : Point3D or None, optional
            Left wrist medial marker.
        left_wrist_lateral : Point3D or None, optional
            Left wrist lateral marker.
        right_wrist_medial : Point3D or None, optional
            Right wrist medial marker.
        right_wrist_lateral : Point3D or None, optional
            Right wrist lateral marker.
        s2 : Point3D or None, optional
            Second sacral vertebra marker.
        l2 : Point3D or None, optional
            Second lumbar vertebra marker.
        c7 : Point3D or None, optional
            Seventh cervical vertebra marker.
        t5 : Point3D or None, optional
            Fifth thoracic vertebra marker.
        sc : Point3D or None, optional
            Sternoclavicular joint marker.
        head_anterior : Point3D or None, optional
            Head anterior marker.
        head_posterior : Point3D or None, optional
            Head posterior marker.
        head_left : Point3D or None, optional
            Head left side marker.
        head_right : Point3D or None, optional
            Head right side marker.
        **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
            Additional signals (e.g., joint angles, EMG channels, other markers).
        """
        signals = {
            **extra_signals,
            **dict(
                left_hand_ground_reaction_force=left_hand_ground_reaction_force,
                right_hand_ground_reaction_force=right_hand_ground_reaction_force,
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
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
            ),
        }
        super().__init__(**{i: v for i, v in signals.items() if v is not None})  # type: ignore

        # set the thresholds
        self.set_height_threshold(height_threshold)
        self.set_grf_threshold(ground_reaction_force_threshold)

        # set the algorithm
        self.set_algorithm(algorithm)

        # set the speed and grade properties
        self.set_speed(speed)
        self.set_grade(grade)

    def set_speed(self, speed: float | int):
        """
        Set the gait speed.

        Parameters
        ----------
        speed : float or int
            Gait speed value.

        Raises
        ------
        ValueError
            If speed is not a float or int.
        """
        if not isinstance(speed, (int, float)):
            raise ValueError("'speed' must be a float or int")
        self._speed = float(speed)

    @property
    def speed(self):
        """
        Get the gait speed.
        """
        return self._speed

    @property
    def pace(self):
        """
        Get the pace of the cycle in minutes per kilometer.
        """
        return 60 / self.speed

    @property
    def grade(self):
        """
        Get the gait grade (incline).
        """
        return self._grade

    def set_grade(self, grade: float | int):
        """
        Set the gait grade (incline).

        Parameters
        ----------
        grade : float or int
            Gait grade (incline) value.

        Raises
        ------
        ValueError
            If grade is not a float or int.
        """
        if not isinstance(grade, (int, float)):
            raise ValueError("'grade' must be a float or int")
        self._grade = float(grade)

    @property
    def algorithm(self):
        """
        Get the selected cycle detection algorithm.
        """
        return self._algorithm

    @property
    def ground_reaction_force_threshold(self):
        """
        Get the ground reaction force threshold (in Newtons).
        """
        return self._grf_threshold

    @property
    def height_threshold(self):
        """
        Get the height threshold (as percentage).
        """
        return self._height_threshold

    def set_grf_threshold(self, threshold: float | int):
        """
        Set the ground reaction force threshold.

        Parameters
        ----------
        threshold : float or int
            Threshold value in Newtons.

        Raises
        ------
        ValueError
            If threshold is not a float or int.
        """
        if not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a float or int")
        self._grf_threshold = float(threshold)

    def set_height_threshold(self, threshold: float | int):
        """
        Set the height threshold.

        Parameters
        ----------
        threshold : float or int
            Threshold value as percentage.

        Raises
        ------
        ValueError
            If threshold is not a float or int.
        """
        if not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a float or int")
        self._height_threshold = float(threshold)

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """
        Set the gait cycle detection algorithm.

        Automatically falls back to an alternative algorithm if the requested one
        cannot be used based on available data. The kinetics algorithm requires
        force platform data, while the kinematics algorithm requires heel and toe
        marker trajectories for both feet.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}
            Requested cycle detection algorithm.

        Raises
        ------
        ValueError
            If algorithm is not 'kinematics' or 'kinetics', or if neither
            algorithm can be used based on available data.

        Warns
        -----
        UserWarning
            If the requested algorithm cannot be used and fallback occurs.
        """
        algorithms = ["kinematics", "kinetics"]
        if not isinstance(algorithm, str) or algorithm not in algorithms:
            msg = "'algorithm' must be any between 'kinematics' or 'kinetics'."
            raise ValueError(msg)
        algo = algorithm
        if (
            algo == "kinetics"
            and self.resultant_force is None
            and all(
                [
                    self.left_heel is not None,
                    self.left_toe is not None,
                    self.right_heel is not None,
                    self.right_toe is not None,
                ]
            )
        ):
            msg = f"'forceplatforms data' not found. The 'algorithm' option"
            msg += " has been set to 'kinematics'."
            warnings.warn(msg)
            algo = "kinematics"
        elif (
            algo == "kinematics"
            and self.resultant_force is not None
            and not all(
                [
                    self.left_heel is not None,
                    self.left_toe is not None,
                    self.right_heel is not None,
                    self.right_toe is not None,
                ]
            )
        ):
            msg = f"Not all left_heel, right_heel, left_toe and right_toe"
            msg += " markers have been found to run the 'kinematics' algorithm."
            msg += " The 'kinetics' algorithm has therefore been selected."
            warnings.warn(msg)
            algo = "kinetics"
        elif self.resultant_force is None and any(
            [
                self.left_heel is None,
                self.left_toe is None,
                self.right_heel is None,
                self.right_toe is None,
            ]
        ):
            msg = "Neither ground reaction force nor left_heel, right_heel, "
            msg += "left_toe and right_toe markers have been found."
            msg += " Therefore none of the available algorithms can be used."
            raise ValueError(msg)

        self._algorithm = algo
