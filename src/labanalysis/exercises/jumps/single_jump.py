"""Single jump exercise module."""

import numpy as np

from ...constants import MINIMUM_CONTACT_FORCE_N, MINIMUM_FLIGHT_TIME_S, G
from ...signalprocessing import continuous_batches
from ...records.body import WholeBody
from ...records import ForcePlatform
from ...timeseries import Point3D


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
    **kwargs
        Additional keyword arguments passed to WholeBody parent class.
    """

    def __init__(
        self,
        bodymass_kg: float,
        free_hands: bool = False,
        straight_legs: bool = False,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        left_hand_ground_reaction_force: ForcePlatform | None = None,
        right_hand_ground_reaction_force: ForcePlatform | None = None,
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
        **kwargs,
    ):
        if (
            right_foot_ground_reaction_force is None
            and left_foot_ground_reaction_force is None
        ):
            raise ValueError(
                "at least one of 'right_foot_ground_reaction_force' or 'left_foot_ground_reaction_force' must be provided."
            )
        super().__init__(
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
            right_trochanter=right_trochanter,
            left_trochanter=left_trochanter,
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
            **kwargs,
        )
        self.set_bodymass_kg(bodymass_kg)
        self.set_free_hands(free_hands)
        self.set_straight_legs(straight_legs)

    def set_bodymass_kg(self, value: float):
        """
        Set the bodymass in kg of the participant.

        Parameters
        ----------
        value : float
            The bodymass in kg.

        Raises
        ------
        ValueError
            IIn case a non positive float is provided.
        """
        try:
            self._bodymass_kg = float(value)
            assert self._bodymass_kg > 0
        except Exception:
            raise ValueError("bodymass must be a positive float")

    def set_free_hands(self, value: bool):
        """
        Set the free hands property of the jump.

        Parameters
        ----------
        value : bool
            The free hands status

        Raises
        ------
        ValueError
            IIn case a non positive float is provided.
        """
        if not isinstance(value, bool):
            raise ValueError("free_hands must be True or False")
        self._free_hands = value

    def set_straight_legs(self, value: bool):
        """
        Set the straight legs property of the jump.

        Parameters
        ----------
        value : bool
            The free hands status

        Raises
        ------
        ValueError
            IIn case a non positive float is provided.
        """
        if not isinstance(value, bool):
            raise ValueError("straight_legs must be True or False")
        self._straight_legs = value

    @property
    def bodymass_kg(self):
        """Return the body mass in kilograms."""
        return self._bodymass_kg

    @property
    def free_hands(self):
        """Return whether hands were free during the jump."""
        return self._free_hands

    @property
    def straight_legs(self):
        """Return whether legs were kept straight during the jump."""
        return self._straight_legs

    @property
    def side(self):
        """
        Determine jump laterality from available force platforms.

        Returns
        -------
        str
            "left", "right", or "bilateral"
        """
        has_left = self.left_foot_ground_reaction_force is not None
        has_right = self.right_foot_ground_reaction_force is not None

        if has_left and has_right:
            return "bilateral"
        elif has_left:
            return "left"
        elif has_right:
            return "right"
        else:
            # Default to bilateral if no force platforms
            return "bilateral"

    @property
    def flight_time(self):
        """return the flight time of the jump in seconds."""
        index = self.flight_phase
        if index is None:
            return None
        index = index.index
        return float(index[-1] - index[0])

    @property
    def contact_time(self):
        index = self.contact_phase
        if index is None:
            return None
        index = index.index
        return float(index[-1] - index[0])

    @property
    def contact_phase(self):
        """
        Returns the ground contact phase as a WholeBody record.

        Returns
        -------
        WholeBody or SingleJump (or subclass)
            Data segment from first ground contact to takeoff.

        Notes
        -----
        For jumps with multiple contact phases (e.g., drop jumps), this returns
        the primary propulsive contact phase (typically the last one).
        """
        vgrf = self.vertical_ground_reaction_force
        if vgrf is None:
            return None
        module = vgrf["force"].to_numpy().flatten()

        # identifico il batch relativi alla fase di contatto
        flight = module < MINIMUM_CONTACT_FORCE_N
        batch = np.arange(np.where(flight)[0][0])

        # ritorno l'oggetto corrispondente al batch più lungo
        return WholeBody(**{i: v.copy().iloc[batch, :] for i, v in self.items()})

    @property
    def flight_phase(self):
        """
        Returns the flight (aerial) phase as a WholeBody record.

        Returns
        -------
        WholeBody or SingleJump (or subclass)
            Data segment during aerial phase after takeoff.

        Notes
        -----
        For jumps with multiple flight phases, this returns the primary
        flight phase (typically the longest one meeting minimum duration).
        """
        vgrf = self.vertical_ground_reaction_force
        if vgrf is None:
            return None
        module = vgrf["force"].to_numpy().flatten()

        # identifico i batch relativi alla fase di volo
        flight = module < MINIMUM_CONTACT_FORCE_N
        batches = continuous_batches(flight)

        # rimuovo i batch di durata non ragionevole
        for i in range(len(batches)):
            batch = batches[i]
            duration = vgrf.index[batch[-1]] - vgrf.index[batch[0]]
            if duration < MINIMUM_FLIGHT_TIME_S:
                batches.pop(i)
        if len(batches) == 0:
            raise ValueError("No valid flight phases have been discovered.")

        # ottengo l'ordinamento dei batch per durata
        index = np.argsort([len(i) for i in batches])

        # ritorno l'oggetto corrispondente al batch più lungo
        batch = batches[index[0]]
        return WholeBody(**{i: v.copy().iloc[batch, :] for i, v in self.items()})

    @property
    def jump_height_from_ft(self):
        """return the jump height in meters calculated from the flight time"""
        flight_time = self.flight_time
        if flight_time is None:
            return None
        return float(G / 8 * flight_time**2)

    @property
    def takeoff_velocity(self):
        cf = self.contact_phase
        if cf is None:
            return None
        rf = cf.resultant_force
        if rf is None:
            return None
        vgrf = rf[self.vertical_axis].copy()
        body_weight = self.bodymass_kg * G
        vgrf = vgrf - body_weight  # type: ignore
        vacc = vgrf / self.bodymass_kg
        time = vgrf.index
        return float(np.trapezoid(vacc, time))  # type: ignore

    @property
    def jump_height_from_tov(self):
        tv = self.takeoff_velocity
        if tv is None:
            return None
        return (tv**2) / (2 * G)

    @property
    def peak_vertical_force(self):
        cp = self.contact_phase
        if cp is None:
            return None
        grf = cp.resultant_force
        if grf is None:
            return None
        return float(grf["force"][self.vertical_axis].to_numpy().max())

    def copy(self):
        """
        Create a deep copy of this SingleJump, preserving custom attributes.

        Returns
        -------
        SingleJump
            A new SingleJump instance with copies of all signals and attributes.

        Notes
        -----
        This method follows the same pattern as EMGSignal and TimeseriesRecord,
        explicitly passing custom non-signal attributes (bodymass_kg, free_hands,
        straight_legs) to the constructor while copying all signal data.
        """
        return SingleJump(
            bodymass_kg=self.bodymass_kg,
            free_hands=self.free_hands,
            straight_legs=self.straight_legs,
            **{k: v.copy() for k, v in self._data.items()},  # type: ignore
        )

    @classmethod
    def from_tdf(
        cls,
        filename: str,
        bodymass_kg: float | int,
        free_hands: bool = False,
        straight_legs: bool = False,
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
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
        """Create a DropJump object from a TDF file."""
        record = WholeBody.from_tdf(
            filename,
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
            free_hands=free_hands,
            straight_legs=straight_legs,
            **{i: v for i, v in record.items()},  # type: ignore
        )


__all__ = ["SingleJump"]
