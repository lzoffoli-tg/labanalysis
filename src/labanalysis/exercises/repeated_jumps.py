"""Repeated jumps exercise module."""

import numpy as np

from ..constants import MINIMUM_CONTACT_FORCE_N, MINIMUM_FLIGHT_TIME_S
from ..signalprocessing import continuous_batches, fillna, butterworth_filt
from ..records.body import WholeBody
from ..records import ForcePlatform
from ..timeseries import Signal1D, Signal3D, EMGSignal, Point3D
from .single_jump import SingleJump


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
            exclude_jumps=exclude_jumps,
            straight_legs=straight_legs,
            free_hands=free_hands,
            **record._data,
        )

    def copy(self):
        return RepeatedJumps(
            bodymass_kg=self.bodymass_kg,
            free_hands=self.free_hands,
            exclude_jumps=self.excluded_jumps,
            straight_legs=self.straight_legs,
            **{i: v.copy() for i, v in self.items()},
        )

    @property
    def jumps(self):
        vgrf = self.resultant_force.copy()
        time = vgrf.index
        vgrf = vgrf.force[self.vertical_axis].to_numpy().flatten()
        vgrf = fillna(arr=vgrf, value=0).flatten()
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
                    **{i: v.copy().loc[start:stop, :] for i, v in self.items()},
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


__all__ = ["RepeatedJumps"]
