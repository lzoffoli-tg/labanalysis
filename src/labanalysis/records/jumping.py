"""repeatedjump module"""

#! IMPORTS

import numpy as np

from ..constants import MINIMUM_CONTACT_FORCE_N, MINIMUM_FLIGHT_TIME_S
from ..signalprocessing import continuous_batches, fillna, butterworth_filt
from .bodies import WholeBody
from .records import *
from .timeseries import *

__all__ = ["RepeatedJumps", "SingleJump", "DropJump"]


#! CLASSES


class SingleJump(WholeBody):
    """
    Represents a single jump trial, providing methods and properties to analyze
    phases, forces, and performance metrics of the jump.

    Parameters
    ----------
    bodymass_kg : float
        The subject's body mass in kilograms (required).
    left_foot_ground_reaction_force : ForcePlatform, optional
        ForcePlatform object for the left foot.
    right_foot_ground_reaction_force : ForcePlatform, optional
        ForcePlatform object for the right foot.
    left_acromion : Point3D, optional
        Left acromion marker (shoulder tip).
    right_acromion : Point3D, optional
        Right acromion marker (shoulder tip).
    straight_legs : bool, optional
        Whether the jump was performed with straight legs (default False).
    free_hands : bool, optional
        Whether hands were free during the jump (default False).
    **signals : Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
        Additional signals (anatomical markers, EMG, etc.).

    Notes
    -----
    This class inherits all 42 anatomical markers and force platforms from WholeBody.
    At least one foot force platform (left or right) must be provided.

    Attributes
    ----------
    _bodymass_kg : float
        The subject's body mass in kilograms.
    _vertical_axis : str
        Name of the vertical axis.
    _antpos_axis : str
        Name of the anteroposterior axis.

    Properties
    ----------
    vertical_axis : str
        The vertical axis label.
    anteroposterior_axis : str
        The anteroposterior axis label.
    lateral_axis : str
        The lateral axis label.
    vertical_force : np.ndarray
        The mean vertical ground reaction force across both feet.
    side : str
        "bilateral", "left", or "right" depending on available force data.
    bodymass_kg : float
        The subject's body mass in kilograms.
    eccentric_phase : TimeseriesRecord
        Data for the eccentric phase of the jump.
    concentric_phase : TimeseriesRecord
        Data for the concentric phase of the jump.
    flight_phase : TimeseriesRecord
        Data for the flight phase of the jump.
    contact_time_s : float
        Duration of the contact phase (s).
    flight_time_s : float
        Duration of the flight phase (s).
    takeoff_velocity_ms : float
        Takeoff velocity at the end of the concentric phase (m/s).
    elevation_cm : float
        Jump elevation (cm) calculated from flight time.
    muscle_coordination_and_balance : pd.DataFrame
        Coordination and balance metrics from EMG signals (if available).
    force_coordination_and_balance : pd.DataFrame
        Coordination and balance metrics from force signals.
    output_metrics : pd.DataFrame
        Summary metrics for the jump.

    Methods
    -------
    __init__(...)
        Initialize a Jump object.
    from_tdf(...)
        Create a Jump object from a TDF file.
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

    @property
    def bodymass_kg(self):
        """
        Returns the subject's body mass in kilograms.

        Returns
        -------
        float
            Body mass in kg.
        """
        return self._bodymass_kg

    def set_bodymass_kg(self, bodymass_kg: float):
        if not isinstance(bodymass_kg, (float, int)) or bodymass_kg <= 0:
            raise ValueError("bodymass_kg must be a float or int > 0.")
        self._bodymass_kg = bodymass_kg

    @property
    def straight_legs(self):
        return self._straight_legs

    def set_straight_legs(self, straight: bool):
        if not isinstance(straight, bool):
            raise ValueError("straight must be True or False.")
        self._straight_legs = straight

    @property
    def free_hands(self):
        return self._free_hands

    def set_free_hands(self, free: bool):
        if not isinstance(free, bool):
            raise ValueError("free must be True or False.")
        self._free_hands = free

    @property
    def contact_phase(self):
        """
        Returns the concentric phase of the jump.

        Returns
        -------
        TimeseriesRecord
            Data for the concentric phase.

        Procedure
        ---------
            1. get the longest countinuous batch with positive acceleration
            of S2 occurring before con_end.
            2. define 'con_start' as the last local minima in the vertical grf
            occurring before the beginning of the batch defined in 2.
            3. define 'con_end' as the end of the concentric phase as the time
            instant immediately before the flight phase. Please look at the
            concentric_phase documentation to have a detailed view about how
            it is detected.
        """
        # get the longest batch with grf lower than 30N
        vgrf = self.resultant_force.copy()
        grfy = vgrf.force[self.vertical_axis].to_numpy().flatten()
        grft = vgrf.index
        batches = continuous_batches(
            (grfy > MINIMUM_CONTACT_FORCE_N) & (grft < self.flight_phase.index[0])
        )
        if len(batches) == 0:
            raise RuntimeError("No contact phase was found.")
        batch = batches[-1]
        start = grft[batch[0]]
        stop = grft[batch[-1]]

        signals = {k: v.copy().loc[start:stop, :] for k, v in self.items()}
        return WholeBody(**signals)  # type: ignore

    @property
    def flight_phase(self):
        """
        Returns the flight phase of the jump.

        Returns
        -------
        TimeseriesRecord
            Data for the flight phase.

        Procedure
        ---------
            1. get the longest batch with grf lower than 30N.
            2. define 'flight_start' as the first local minima occurring after
            the start of the detected batch.
            3. define 'flight_end' as the last local minima occurring before the
            end of the detected batch.
        """

        # get the longest batch with grf lower than 30N
        vgrf = self.resultant_force.copy()
        grfy = vgrf.force.copy()[self.vertical_axis].to_numpy().flatten()
        grfy = fillna(arr=grfy, value=0).flatten()  # type: ignore
        grft = self.index
        batches = continuous_batches(grfy <= MINIMUM_CONTACT_FORCE_N)
        msg = "No flight phase found."
        if len(batches) == 0:
            raise RuntimeError("No flight phase was found.")
        batch = batches[np.argmax([len(i) for i in batches])]

        # check the length of the batch is at minimum 2 samples
        if len(batch) < 2:
            raise RuntimeError(msg)

        # # get the time samples corresponding to the start and end of each
        # batch
        start = float(np.round(grft[batch[0]], 3))
        stop = float(np.round(grft[batch[-1]], 3))

        # return a slice of the available data
        signals = {k: v.copy().loc[start:stop, :] for k, v in self.items()}
        return WholeBody(**signals)  # type: ignore

    def __init__(
        self,
        bodymass_kg: float,
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
        straight_legs: bool = False,
        free_hands: bool = False,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """Initialize a Jump object."""
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
                t5=t5,
                sc=sc,
                head_anterior=head_anterior,
                head_posterior=head_posterior,
                head_left=head_left,
                head_right=head_right,
            ),
        }
        # Check that at least one foot force platform is provided
        if left_foot_ground_reaction_force is None and right_foot_ground_reaction_force is None:
            raise ValueError(
                "at least one of 'left_foot_ground_reaction_force' or "
                "'right_foot_ground_reaction_force' must be ForcePlatform instances."
            )
        super().__init__(**{i: v for i, v in all_signals.items() if v is not None})
        self.set_bodymass_kg(bodymass_kg)
        self.set_straight_legs(straight_legs)
        self.set_free_hands(free_hands)

    @classmethod
    def from_tdf(
        cls,
        file: str,
        bodymass_kg: float | int,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_foot_ground_reaction_force: str | None = "left_foot",
        right_foot_ground_reaction_force: str | None = "right_foot",
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
        straight_legs: bool = False,
        free_hands: bool = False,
    ):
        """Create a Jump object from a TDF file."""
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
            t5=t5,
            sc=sc,
            head_anterior=head_anterior,
            head_posterior=head_posterior,
            head_left=head_left,
            head_right=head_right,
        )
        return cls(
            bodymass_kg=bodymass_kg,
            straight_legs=straight_legs,
            free_hands=free_hands,
            **record._data,  # type: ignore
        )

    def copy(self):
        return SingleJump(
            self.bodymass_kg,
            straight_legs=self.straight_legs,
            free_hands=self.free_hands,
            **{i: v.copy() for i, v in self.items()},  # type: ignore
        )


class DropJump(SingleJump):
    """
    Represents a single jump trial, providing methods and properties to analyze
    phases, forces, and performance metrics of the jump.

    Parameters
    ----------
    bodymass_kg : float
        The subject's body mass in kilograms.
    left_foot_ground_reaction_force : ForcePlatform, optional
        ForcePlatform object for the left foot.
    right_foot_ground_reaction_force : ForcePlatform, optional
        ForcePlatform object for the right foot.
    vertical_axis : str, optional
        Name of the vertical axis in the force data (default "Y").
    anteroposterior_axis : str, optional
        Name of the anteroposterior axis in the force data (default "X").
    **signals : Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
        Additional signals to include in the record.

    Attributes
    ----------
    _bodymass_kg : float
        The subject's body mass in kilograms.
    _vertical_axis : str
        Name of the vertical axis.
    _antpos_axis : str
        Name of the anteroposterior axis.

    Properties
    ----------
    vertical_axis : str
        The vertical axis label.
    anteroposterior_axis : str
        The anteroposterior axis label.
    lateral_axis : str
        The lateral axis label.
    vertical_force : np.ndarray
        The mean vertical ground reaction force across both feet.
    side : str
        "bilateral", "left", or "right" depending on available force data.
    bodymass_kg : float
        The subject's body mass in kilograms.
    eccentric_phase : TimeseriesRecord
        Data for the eccentric phase of the jump.
    concentric_phase : TimeseriesRecord
        Data for the concentric phase of the jump.
    flight_phase : TimeseriesRecord
        Data for the flight phase of the jump.
    contact_time_s : float
        Duration of the contact phase (s).
    flight_time_s : float
        Duration of the flight phase (s).
    takeoff_velocity_ms : float
        Takeoff velocity at the end of the concentric phase (m/s).
    elevation_cm : float
        Jump elevation (cm) calculated from flight time.
    muscle_coordination_and_balance : pd.DataFrame
        Coordination and balance metrics from EMG signals (if available).
    force_coordination_and_balance : pd.DataFrame
        Coordination and balance metrics from force signals.
    output_metrics : pd.DataFrame
        Summary metrics for the jump.

    Methods
    -------
    __init__(...)
        Initialize a Jump object.
    from_tdf(...)
        Create a Jump object from a TDF file.
    """

    @property
    def landing_phase(self):
        """
        Returns the landing phase of the drop jump.

        Procedure
        ---------
            1. get the batch with grf lower than 30N occurring before the contact phase.
        """

        # # get the time samples corresponding to the start and end of each
        # batch
        start = float(round(self.index[0], 3))
        contact_time_start = float(round(self.contact_phase.index[0], 3))
        stop = float(
            np.round(self.index[np.where(self.index < contact_time_start)[0][-1]], 3)
        )

        # return the landing phase
        signals = {k: v.copy().loc[start:stop, :] for k, v in self.items()}
        return WholeBody(**signals)  # type: ignore

    @property
    def flight_phase(self):
        """
        Returns the flight phase of the jump.

        Returns
        -------
        TimeseriesRecord
            Data for the flight phase.

        Procedure
        ---------
            1. get the longest batch with grf lower than 30N.
            2. define 'flight_start' as the first local minima occurring after
            the start of the detected batch.
            3. define 'flight_end' as the last local minima occurring before the
            end of the detected batch.
        """

        # get vertical force
        vgrf = self.resultant_force.copy()
        grfy = vgrf.force.copy()[self.vertical_axis].to_numpy().flatten()
        grft = self.index

        # get contact phases
        contact_batches = continuous_batches(grfy > MINIMUM_CONTACT_FORCE_N)

        # if the jump starts with a contact phase (i.e. the box is on the
        # force platforms) ignore the first contact
        if contact_batches[0][0] == 0:
            contact_batches = contact_batches[1:]

        # ensure that a minimum of 2 contact phases
        if len(contact_batches) < 2:
            raise RuntimeError("No flight phase found.")

        # get the second flight phase
        start = float(round(grft[contact_batches[0][-1] + 1], 3))
        stop = float(round(grft[contact_batches[1][0] - 1], 3))

        # return the landing phase
        signals = {k: v.copy().loc[start:stop, :] for k, v in self.items()}
        return WholeBody(**signals)  # type: ignore

    def __init__(
        self,
        box_height_cm: float,
        bodymass_kg: float,
        free_hands: bool = False,
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
        t5: Point3D | None = None,
        sc: Point3D | None = None,
        head_anterior: Point3D | None = None,
        head_posterior: Point3D | None = None,
        head_left: Point3D | None = None,
        head_right: Point3D | None = None,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """Initialize a DropJump object."""
        super().__init__(
            bodymass_kg=bodymass_kg,
            free_hands=free_hands,
            straight_legs=False,
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
            t5=t5,
            sc=sc,
            head_anterior=head_anterior,
            head_posterior=head_posterior,
            head_left=head_left,
            head_right=head_right,
            **signals,  # type: ignore
        )
        self.set_box_height_cm(box_height_cm)

    def set_box_height_cm(self, box_height_cm: float):
        """
        Set the box height in centimeters.
        """
        # check box height
        if not isinstance(box_height_cm, (float, int)):
            raise ValueError("box_height_cm must be a float or int")
        self._box_height_cm = float(box_height_cm)

    @property
    def box_height_cm(self):
        """
        Returns the box height in centimeters.
        """
        return self._box_height_cm

    @classmethod
    def from_tdf(
        cls,
        file: str,
        box_height_cm: float,
        bodymass_kg: float | int,
        free_hands: bool = False,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_foot_ground_reaction_force: str | None = "left_foot",
        right_foot_ground_reaction_force: str | None = "right_foot",
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
        t5: str | None = None,
        sc: str | None = None,
        head_anterior: str | None = None,
        head_posterior: str | None = None,
        head_left: str | None = None,
        head_right: str | None = None,
    ):
        """Create a DropJump object from a TDF file."""
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
            t5=t5,
            sc=sc,
            head_anterior=head_anterior,
            head_posterior=head_posterior,
            head_left=head_left,
            head_right=head_right,
        )
        return cls(
            box_height_cm=box_height_cm,
            bodymass_kg=bodymass_kg,
            free_hands=free_hands,
            **record._data,  # type: ignore
        )

    def copy(self):
        return DropJump(
            box_height_cm=self.box_height_cm,
            bodymass_kg=self.bodymass_kg,
            free_hands=self.free_hands,
            **{i: v.copy() for i, v in self.items()},  # type: ignore
        )


class RepeatedJumps(WholeBody):

    @property
    def bodymass_kg(self):
        """
        Returns the subject's body mass in kilograms.

        Returns
        -------
        float
            Body mass in kg.
        """
        return self._bodymass_kg

    def set_bodymass_kg(self, bodymass_kg: float):
        if not isinstance(bodymass_kg, (float, int)) or bodymass_kg <= 0:
            raise ValueError("bodymass_kg must be a float or int > 0.")
        self._bodymass_kg = bodymass_kg

    @property
    def excluded_jumps(self):
        return self._excluded_jumps

    def set_excluded_jumps(self, jumps: list[int]):
        if not isinstance(jumps, list) or not all([isinstance(i, int) for i in jumps]):
            raise ValueError("jumps must be a list of int")
        self._excluded_jumps = jumps

    @property
    def straight_legs(self):
        return self._straight_legs

    def set_straight_legs(self, straight: bool):
        if not isinstance(straight, bool):
            raise ValueError("straight must be True or False.")
        self._straight_legs = straight

    @property
    def free_hands(self):
        return self._free_hands

    def set_free_hands(self, free: bool):
        if not isinstance(free, bool):
            raise ValueError("free must be True or False.")
        self._free_hands = free

    def __init__(
        self,
        bodymass_kg: float,
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
        exclude_jumps: list[int] = [0, -1],
        straight_legs: bool = False,
        free_hands: bool = False,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """Initialize a RepeatedJumps object."""
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
                t5=t5,
                sc=sc,
                head_anterior=head_anterior,
                head_posterior=head_posterior,
                head_left=head_left,
                head_right=head_right,
            ),
        }
        if left_foot_ground_reaction_force is None and right_foot_ground_reaction_force is None:
            raise ValueError(
                "at least one of 'left_foot_ground_reaction_force' or "
                "'right_foot_ground_reaction_force' must be ForcePlatform instances."
            )
        super().__init__(**{i: v for i, v in all_signals.items() if v is not None})
        self.set_bodymass_kg(bodymass_kg)
        self.set_excluded_jumps(exclude_jumps)
        self.set_straight_legs(straight_legs)
        self.set_free_hands(free_hands)

    @classmethod
    def from_tdf(
        cls,
        file: str,
        bodymass_kg: float | int,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_foot_ground_reaction_force: str | None = "left_foot",
        right_foot_ground_reaction_force: str | None = "right_foot",
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
        exclude_jumps: list[int] = [],
        straight_legs: bool = False,
        free_hands: bool = False,
    ):
        """Create a RepeatedJumps object from a TDF file."""
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
            t5=t5,
            sc=sc,
            head_anterior=head_anterior,
            head_posterior=head_posterior,
            head_left=head_left,
            head_right=head_right,
        )
        return cls(
            bodymass_kg=bodymass_kg,
            exclude_jumps=exclude_jumps,
            straight_legs=straight_legs,
            free_hands=free_hands,
            **record._data,  # type: ignore
        )

    def copy(self):
        return RepeatedJumps(
            bodymass_kg=self.bodymass_kg,
            free_hands=self.free_hands,
            exclude_jumps=self.excluded_jumps,
            straight_legs=self.straight_legs,
            **{i: v.copy() for i, v in self.items()},  # type: ignore
        )

    @property
    def jumps(self):
        vgrf = self.resultant_force.copy()
        time = vgrf.index
        vgrf = vgrf.force[self.vertical_axis].to_numpy().flatten()
        vgrf = fillna(arr=vgrf, value=0).flatten()  # type: ignore
        fsamp = float(1 / np.mean(np.diff(time)))
        vgrf = butterworth_filt(
            arr=vgrf,
            fsamp=fsamp,
            fcut=50.0,
            order=4,
            ftype="lowpass",
            phase_corrected=True,
        )

        # get the batches with grf lower than 30N (i.e flight phases)
        flight_batches = continuous_batches(vgrf <= float(MINIMUM_CONTACT_FORCE_N))

        # remove those batches resulting in too short flight phases
        # (i.e. ~0.2s flight time)
        fsamp = 1 / np.mean(np.diff(time))
        min_samples = int(round(MINIMUM_FLIGHT_TIME_S * fsamp))
        flight_batches = [i for i in flight_batches if len(i) >= min_samples]

        # ensure that the first jump does not start with a flight
        if flight_batches[0][0] == 0:
            flight_batches = flight_batches[1:]

        # ensure that the last jump does not end in flight
        if flight_batches[-1][-1] == len(vgrf) - 1:
            flight_batches = flight_batches[:-1]

        # get the contact peaks
        contact_peaks = []
        for b0, b1 in zip(flight_batches[:-1], flight_batches[1:]):
            contact_peaks.append(np.argmax(vgrf[b0[-1] : b1[0]]) + b0[-1])
        contact_peaks.append(
            np.argmax(vgrf[flight_batches[-1][-1] :]) + flight_batches[-1][-1]
        )

        # get the contact starts
        contact_starts = []
        contact_batches = continuous_batches(vgrf > float(MINIMUM_CONTACT_FORCE_N))
        for i, batch in enumerate(flight_batches):
            pre = [c for c in contact_batches if c[-1] <= batch[0]]
            if len(pre) == 0:
                raise RuntimeError("no contact phase found")
            pre = pre[-1]
            contact_starts.append(pre[0])

        # separate each jump
        jumps: list[SingleJump] = []
        for i, (pre, post) in enumerate(zip(contact_starts, contact_peaks)):
            start = float(time[pre])
            stop = float(time[post])
            jumps.append(
                SingleJump(
                    bodymass_kg=self.bodymass_kg,
                    straight_legs=self.straight_legs,
                    free_hands=self.free_hands,
                    **{i: v.copy().loc[start:stop, :] for i, v in self.items()},  # type: ignore
                )
            )

        # exclude unnecessary jumps
        sanitized_indices = [
            i + (0 if i >= 0 else len(jumps)) for i in self.excluded_jumps
        ]
        sanitized_indices = sorted(set(sanitized_indices), reverse=True)
        for i in sanitized_indices:
            jumps.pop(i)

        return jumps
