"""Base class for gait analysis."""

from typing import Literal

from ...constants import *
from ...signalprocessing import *
from ...timeseries import *
from ..body import WholeBody
from ..forceplatform import ForcePlatform


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
        algorithm: Literal["kinematics", "kinetics"],
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
        algorithm : {'kinematics', 'kinetics'}
            The cycle detection algorithm to use.
        left_heel : Point3D or None, optional
            Left heel marker data.
        right_heel : Point3D or None, optional
            Right heel marker data.
        left_toe : Point3D or None, optional
            Left toe marker data.
        right_toe : Point3D or None, optional
            Right toe marker data.
        left_first_metatarsal_head : Point3D or None
        Left first metatarsal head marker.
    left_fifth_metatarsal_head : Point3D or None
        Left fifth metatarsal head marker.
    right_first_metatarsal_head : Point3D or None
        Right first metatarsal head marker.
    right_fifth_metatarsal_head : Point3D or None, optional
            Left metatarsal head marker data.
        right_metatarsal_head : Point3D or None, optional
            Right metatarsal head marker data.
        ground_reaction_force : ForcePlatform or None
            Ground reaction force data.
        ground_reaction_force_threshold : float or int, optional
            Minimum ground reaction force for contact detection.
        height_threshold : float or int, optional
            Maximum vertical height for contact detection.
        vertical_axis : {'X', 'Y', 'Z'}, optional
            The vertical axis.
        antpos_axis : {'X', 'Y', 'Z'}, optional
            The anterior-posterior axis.
        **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
            Additional signals to include.
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

    @property
    def algorithm(self):
        """
        Get the selected cycle detection algorithm.

        Returns
        -------
        str
            The algorithm label.
        """
        return self._algorithm

    @property
    def ground_reaction_force_threshold(self):
        """
        Get the ground reaction force threshold.

        Returns
        -------
        float
        """
        return self._grf_threshold

    @property
    def height_threshold(self):
        """
        Get the height threshold.

        Returns
        -------
        float
        """
        return self._height_threshold

    def set_grf_threshold(self, threshold: float | int):
        """
        Set the ground reaction force threshold.

        Parameters
        ----------
        threshold : float or int
            Threshold value.
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
            Threshold value.
        """
        if not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a float or int")
        self._height_threshold = float(threshold)

    def set_algorithm(self, algorithm: Literal["kinematics", "kinetics"]):
        """
        Set the gait cycle detection algorithm.

        Parameters
        ----------
        algorithm : {'kinematics', 'kinetics'}
            Algorithm label.
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


