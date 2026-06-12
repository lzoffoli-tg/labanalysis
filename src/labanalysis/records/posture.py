"""repeatedjump module"""

#! IMPORTS

from .bodies import WholeBody
from .records import ForcePlatform
from .timeseries import EMGSignal, Point3D, Signal1D, Signal3D

__all__ = ["UprightPosture", "PronePosture"]


#! CLASSES


class UprightPosture(WholeBody):
    """
    Represents an upright posture stance for balance and stability analysis.

    This class extends WholeBody for analyzing upright standing postures, typically
    used in balance tests and postural stability assessments.

    Parameters
    ----------
    left_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for left foot contact.
    right_foot_ground_reaction_force : ForcePlatform, optional
        Force platform data for right foot contact.
    left_acromion : Point3D, optional
        Left acromion marker (shoulder tip).
    right_acromion : Point3D, optional
        Right acromion marker (shoulder tip).
    **signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals (anatomical markers, EMG, etc.).

    Notes
    -----
    This class inherits all 42 anatomical markers and force platforms from WholeBody.
    At least one foot force platform (left or right) must be provided.

    See Also
    --------
    WholeBody : Parent class with biomechanical body model.
    PronePosture : Represents a prone (plank) posture stance.
    """

    @property
    def side(self):
        """
        Returns which side(s) have force data.

        Returns
        -------
        str
            "bilateral", "left", or "right".
        """
        left_foot = self.get("left_foot_ground_reaction_force")
        right_foot = self.get("right_foot_ground_reaction_force")
        if left_foot is not None and right_foot is not None:
            return "bilateral"
        if left_foot is not None:
            return "left"
        if right_foot is not None:
            return "right"
        raise ValueError("both left_foot and right_foot are None")

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
        left_metatarsal_head: Point3D | None = None,
        right_metatarsal_head: Point3D | None = None,
        left_ankle_medial: Point3D | None = None,
        left_ankle_lateral: Point3D | None = None,
        right_ankle_medial: Point3D | None = None,
        right_ankle_lateral: Point3D | None = None,
        left_knee_medial: Point3D | None = None,
        left_knee_lateral: Point3D | None = None,
        right_knee_medial: Point3D | None = None,
        right_knee_lateral: Point3D | None = None,
        right_throcanter: Point3D | None = None,
        left_throcanter: Point3D | None = None,
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
        sc: Point3D | None = None,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        all_signals = {
            **signals,
            **dict(
                left_hand_ground_reaction_force=left_hand_ground_reaction_force,
                right_hand_ground_reaction_force=right_hand_ground_reaction_force,
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                left_heel=left_heel,
                right_heel=right_heel,
                left_toe=left_toe,
                right_toe=right_toe,
                left_metatarsal_head=left_metatarsal_head,
                right_metatarsal_head=right_metatarsal_head,
                left_ankle_medial=left_ankle_medial,
                left_ankle_lateral=left_ankle_lateral,
                right_ankle_medial=right_ankle_medial,
                right_ankle_lateral=right_ankle_lateral,
                left_knee_medial=left_knee_medial,
                left_knee_lateral=left_knee_lateral,
                right_knee_medial=right_knee_medial,
                right_knee_lateral=right_knee_lateral,
                left_throcanter=left_throcanter,
                right_throcanter=right_throcanter,
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
                l2=l2,
                c7=c7,
                sc=sc,
            ),
        }
        # Check that at least one foot force platform is provided
        if left_foot_ground_reaction_force is None and right_foot_ground_reaction_force is None:
            raise ValueError(
                "at least one of 'left_foot_ground_reaction_force' or "
                "'right_foot_ground_reaction_force' must be provided."
            )
        super().__init__(**{i: v for i, v in all_signals.items() if v is not None})

    @classmethod
    def from_tdf(
        cls,
        file: str,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_foot_ground_reaction_force: str | None = "left_foot",
        right_foot_ground_reaction_force: str | None = "right_foot",
        left_heel: str | None = None,
        right_heel: str | None = None,
        left_toe: str | None = None,
        right_toe: str | None = None,
        left_metatarsal_head: str | None = None,
        right_metatarsal_head: str | None = None,
        left_ankle_medial: str | None = None,
        left_ankle_lateral: str | None = None,
        right_ankle_medial: str | None = None,
        right_ankle_lateral: str | None = None,
        left_knee_medial: str | None = None,
        left_knee_lateral: str | None = None,
        right_knee_medial: str | None = None,
        right_knee_lateral: str | None = None,
        right_throcanter: str | None = None,
        left_throcanter: str | None = None,
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
        sc: str | None = None,
    ):
        """
        Create a Jump object from a TDF file.

        Parameters
        ----------
        file : str
            Path to the TDF file.
        bodymass_kg : float or int
            The subject's body mass in kilograms.
        vertical_axis : str, optional
            Name of the vertical axis in the force data.
        anteroposterior_axis : str, optional
            Name of the anteroposterior axis in the force data.
        left_foot_ground_reaction_force : str or None, optional
            Key for left foot force data.
        right_foot_ground_reaction_force : str or None, optional
            Key for right foot force data.

        Returns
        -------
        Jump
            A Jump object created from the TDF file.
        """
        if left_foot_ground_reaction_force is None and right_foot_ground_reaction_force is None:
            raise ValueError(
                "at least one of left_foot_ground_reaction_force or "
                "right_foot_ground_reaction_force must be provided."
            )
        record = WholeBody.from_tdf(
            file,
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            left_heel=left_heel,
            right_heel=right_heel,
            left_toe=left_toe,
            right_toe=right_toe,
            left_metatarsal_head=left_metatarsal_head,
            right_metatarsal_head=right_metatarsal_head,
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral,
            right_ankle_medial=right_ankle_medial,
            right_ankle_lateral=right_ankle_lateral,
            left_knee_medial=left_knee_medial,
            left_knee_lateral=left_knee_lateral,
            right_knee_medial=right_knee_medial,
            right_knee_lateral=right_knee_lateral,
            left_throcanter=left_throcanter,
            right_throcanter=right_throcanter,
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
            l2=l2,
            c7=c7,
            sc=sc,
        )
        return cls(**record._data)  # type: ignore

    def copy(self):
        return UprightPosture(**self._data)  # type: ignore


class PronePosture(WholeBody):
    """
    Represents a prone (plank) posture stance for core stability analysis.

    This class extends WholeBody for analyzing prone/plank positions, typically
    used in core stability and endurance tests.

    Parameters
    ----------
    left_foot_ground_reaction_force : ForcePlatform
        Force platform data for left foot contact (required).
    right_foot_ground_reaction_force : ForcePlatform
        Force platform data for right foot contact (required).
    left_hand_ground_reaction_force : ForcePlatform
        Force platform data for left hand contact (required).
    right_hand_ground_reaction_force : ForcePlatform
        Force platform data for right hand contact (required).
    left_acromion : Point3D, optional
        Left acromion marker (shoulder tip).
    right_acromion : Point3D, optional
        Right acromion marker (shoulder tip).
    **signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals (anatomical markers, EMG, etc.).

    Notes
    -----
    This class inherits all 42 anatomical markers and force platforms from WholeBody.
    All four force platforms (both feet and both hands) are required for prone posture analysis.

    See Also
    --------
    WholeBody : Parent class with biomechanical body model.
    UprightPosture : Represents an upright standing posture.
    """

    def __init__(
        self,
        left_foot_ground_reaction_force: ForcePlatform,
        right_foot_ground_reaction_force: ForcePlatform,
        left_hand_ground_reaction_force: ForcePlatform,
        right_hand_ground_reaction_force: ForcePlatform,
        left_heel: Point3D | None = None,
        right_heel: Point3D | None = None,
        left_toe: Point3D | None = None,
        right_toe: Point3D | None = None,
        left_metatarsal_head: Point3D | None = None,
        right_metatarsal_head: Point3D | None = None,
        left_ankle_medial: Point3D | None = None,
        left_ankle_lateral: Point3D | None = None,
        right_ankle_medial: Point3D | None = None,
        right_ankle_lateral: Point3D | None = None,
        left_knee_medial: Point3D | None = None,
        left_knee_lateral: Point3D | None = None,
        right_knee_medial: Point3D | None = None,
        right_knee_lateral: Point3D | None = None,
        right_throcanter: Point3D | None = None,
        left_throcanter: Point3D | None = None,
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
        sc: Point3D | None = None,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        # Validate required force platforms
        for name, force in [
            ("left_foot_ground_reaction_force", left_foot_ground_reaction_force),
            ("right_foot_ground_reaction_force", right_foot_ground_reaction_force),
            ("left_hand_ground_reaction_force", left_hand_ground_reaction_force),
            ("right_hand_ground_reaction_force", right_hand_ground_reaction_force),
        ]:
            if not isinstance(force, ForcePlatform):
                raise ValueError(f"{name} must be a ForcePlatform instance")

        all_signals = {
            **signals,
            **dict(
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                left_hand_ground_reaction_force=left_hand_ground_reaction_force,
                right_hand_ground_reaction_force=right_hand_ground_reaction_force,
                left_heel=left_heel,
                right_heel=right_heel,
                left_toe=left_toe,
                right_toe=right_toe,
                left_metatarsal_head=left_metatarsal_head,
                right_metatarsal_head=right_metatarsal_head,
                left_ankle_medial=left_ankle_medial,
                left_ankle_lateral=left_ankle_lateral,
                right_ankle_medial=right_ankle_medial,
                right_ankle_lateral=right_ankle_lateral,
                left_knee_medial=left_knee_medial,
                left_knee_lateral=left_knee_lateral,
                right_knee_medial=right_knee_medial,
                right_knee_lateral=right_knee_lateral,
                left_throcanter=left_throcanter,
                right_throcanter=right_throcanter,
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
                l2=l2,
                c7=c7,
                sc=sc,
            ),
        }
        super().__init__(**{i: v for i, v in all_signals.items() if v is not None})

    @classmethod
    def from_tdf(
        cls,
        file: str,
        left_foot_ground_reaction_force: str = "left_foot",
        right_foot_ground_reaction_force: str = "right_foot",
        left_hand_ground_reaction_force: str = "left_hand",
        right_hand_ground_reaction_force: str = "right_hand",
        left_heel: str | None = None,
        right_heel: str | None = None,
        left_toe: str | None = None,
        right_toe: str | None = None,
        left_metatarsal_head: str | None = None,
        right_metatarsal_head: str | None = None,
        left_ankle_medial: str | None = None,
        left_ankle_lateral: str | None = None,
        right_ankle_medial: str | None = None,
        right_ankle_lateral: str | None = None,
        left_knee_medial: str | None = None,
        left_knee_lateral: str | None = None,
        right_knee_medial: str | None = None,
        right_knee_lateral: str | None = None,
        right_throcanter: str | None = None,
        left_throcanter: str | None = None,
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
        sc: str | None = None,
    ):
        """read from file"""
        record = WholeBody.from_tdf(
            file,
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            left_heel=left_heel,
            right_heel=right_heel,
            left_toe=left_toe,
            right_toe=right_toe,
            left_metatarsal_head=left_metatarsal_head,
            right_metatarsal_head=right_metatarsal_head,
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral,
            right_ankle_medial=right_ankle_medial,
            right_ankle_lateral=right_ankle_lateral,
            left_knee_medial=left_knee_medial,
            left_knee_lateral=left_knee_lateral,
            right_knee_medial=right_knee_medial,
            right_knee_lateral=right_knee_lateral,
            left_throcanter=left_throcanter,
            right_throcanter=right_throcanter,
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
            l2=l2,
            c7=c7,
            sc=sc,
        )
        labels = {
            "left_foot_ground_reaction_force": left_foot_ground_reaction_force,
            "right_foot_ground_reaction_force": right_foot_ground_reaction_force,
            "left_hand_ground_reaction_force": left_hand_ground_reaction_force,
            "right_hand_ground_reaction_force": right_hand_ground_reaction_force,
        }
        for key in labels.keys():
            if record.get(key) is None:
                raise ValueError(f"{key} not found within the provided file.")
        return cls(**record._data)  # type: ignore

    def copy(self):
        return PronePosture(**self._data)  # type: ignore
