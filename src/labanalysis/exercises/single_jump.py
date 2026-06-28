"""Single jump exercise module."""

import numpy as np

from ..constants import MINIMUM_CONTACT_FORCE_N, MINIMUM_FLIGHT_TIME_S
from ..signalprocessing import continuous_batches
from ..records.body import WholeBody
from ..records import ForcePlatform, TimeseriesRecord
from ..timeseries import Signal1D


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
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        **kwargs,
    ):
        super().__init__(bodymass_kg=bodymass_kg, **kwargs)
        self.set_left_foot_ground_reaction_force(left_foot_ground_reaction_force)
        self.set_right_foot_ground_reaction_force(right_foot_ground_reaction_force)

    def set_left_foot_ground_reaction_force(self, fp: ForcePlatform | None):
        if fp is not None and not isinstance(fp, ForcePlatform):
            raise ValueError("left_foot_ground_reaction_force must be a ForcePlatform")
        self._left_foot_ground_reaction_force = fp

    def set_right_foot_ground_reaction_force(self, fp: ForcePlatform | None):
        if fp is not None and not isinstance(fp, ForcePlatform):
            raise ValueError("right_foot_ground_reaction_force must be a ForcePlatform")
        self._right_foot_ground_reaction_force = fp

    @property
    def left_foot_ground_reaction_force(self):
        return self._left_foot_ground_reaction_force

    @property
    def right_foot_ground_reaction_force(self):
        return self._right_foot_ground_reaction_force

    @property
    def ground_reaction_force(self):
        left = self.left_foot_ground_reaction_force
        right = self.right_foot_ground_reaction_force
        if left is None and right is None:
            return None
        elif left is None:
            return right
        elif right is None:
            return left
        else:
            fp = TimeseriesRecord(
                origin=left.origin + right.origin,
                force=left.force + right.force,
                torque=left.torque + right.torque,
            )
            return ForcePlatform(**fp._data)

    @property
    def vertical_ground_reaction_force(self):
        fp = self.ground_reaction_force
        if fp is None:
            return None
        axis = fp.vertical_axis
        return fp.force[axis]

    @property
    def flight_time(self):
        vgrf = self.vertical_ground_reaction_force
        if vgrf is None:
            return None
        module = np.abs(vgrf.to_numpy().flatten())
        contact = module > MINIMUM_CONTACT_FORCE_N
        flight = ~contact
        batches = continuous_batches(flight)
        durations = []
        for batch in batches:
            if len(batch) > 0:
                duration = vgrf.index[batch[-1]] - vgrf.index[batch[0]]
                if duration >= MINIMUM_FLIGHT_TIME_S:
                    durations.append(duration)
        if len(durations) == 0:
            return None
        return Signal1D(
            data=np.array(durations),
            index=np.arange(len(durations)),
            unit="s",
        )

    @property
    def contact_time(self):
        vgrf = self.vertical_ground_reaction_force
        if vgrf is None:
            return None
        module = np.abs(vgrf.to_numpy().flatten())
        contact = module > MINIMUM_CONTACT_FORCE_N
        batches = continuous_batches(contact)
        durations = []
        for batch in batches:
            if len(batch) > 0:
                duration = vgrf.index[batch[-1]] - vgrf.index[batch[0]]
                durations.append(duration)
        if len(durations) == 0:
            return None
        return Signal1D(
            data=np.array(durations),
            index=np.arange(len(durations)),
            unit="s",
        )

    @property
    def jump_height(self):
        flight_time = self.flight_time
        if flight_time is None:
            return None
        g = 9.81
        heights = (g * flight_time.to_numpy().flatten() ** 2) / 8
        return Signal1D(
            data=heights,
            index=flight_time.index,
            unit="m",
        )

    @property
    def takeoff_velocity(self):
        flight_time = self.flight_time
        if flight_time is None:
            return None
        g = 9.81
        velocities = (g * flight_time.to_numpy().flatten()) / 2
        return Signal1D(
            data=velocities,
            index=flight_time.index,
            unit="m/s",
        )

    @property
    def reactive_strength_index(self):
        jump_height = self.jump_height
        contact_time = self.contact_time
        if jump_height is None or contact_time is None:
            return None
        rsi = jump_height.to_numpy().flatten() / contact_time.to_numpy().flatten()
        return Signal1D(
            data=rsi,
            index=jump_height.index,
            unit="m/s",
        )

    @property
    def center_of_mass_displacement(self):
        com = self.center_of_mass
        if com is None:
            return None
        vaxis = com.vertical_axis
        vertical_displacement = com[vaxis].to_numpy().flatten()
        vertical_displacement = vertical_displacement - np.nanmin(vertical_displacement)
        return Signal1D(
            data=vertical_displacement,
            index=com.index,
            unit="m",
        )

    @property
    def peak_vertical_force(self):
        vgrf = self.vertical_ground_reaction_force
        if vgrf is None:
            return None
        module = np.abs(vgrf.to_numpy().flatten())
        contact = module > MINIMUM_CONTACT_FORCE_N
        batches = continuous_batches(contact)
        peaks = []
        for batch in batches:
            if len(batch) > 0:
                peak = np.max(module[batch])
                peaks.append(peak)
        if len(peaks) == 0:
            return None
        return Signal1D(
            data=np.array(peaks),
            index=np.arange(len(peaks)),
            unit="N",
        )

    @property
    def peak_power(self):
        vgrf = self.vertical_ground_reaction_force
        com = self.center_of_mass
        if vgrf is None or com is None:
            return None

        vaxis = com.vertical_axis
        velocity = np.gradient(com[vaxis].to_numpy().flatten(), com.index)
        force = vgrf.to_numpy().flatten()

        module = np.abs(force)
        contact = module > MINIMUM_CONTACT_FORCE_N
        batches = continuous_batches(contact)

        peaks = []
        for batch in batches:
            if len(batch) > 0:
                power = force[batch] * velocity[batch]
                peak = np.max(power)
                peaks.append(peak)

        if len(peaks) == 0:
            return None

        return Signal1D(
            data=np.array(peaks),
            index=np.arange(len(peaks)),
            unit="W",
        )

    @property
    def eccentric_phase_duration(self):
        com = self.center_of_mass
        vgrf = self.vertical_ground_reaction_force
        if com is None or vgrf is None:
            return None

        vaxis = com.vertical_axis
        velocity = np.gradient(com[vaxis].to_numpy().flatten(), com.index)

        module = np.abs(vgrf.to_numpy().flatten())
        contact = module > MINIMUM_CONTACT_FORCE_N
        batches = continuous_batches(contact)

        durations = []
        for batch in batches:
            if len(batch) == 0:
                continue

            vel_batch = velocity[batch]
            descending = vel_batch < 0

            if not np.any(descending):
                continue

            ecc_indices = np.where(descending)[0]
            if len(ecc_indices) > 0:
                duration = com.index[batch[ecc_indices[-1]]] - com.index[batch[ecc_indices[0]]]
                durations.append(duration)

        if len(durations) == 0:
            return None

        return Signal1D(
            data=np.array(durations),
            index=np.arange(len(durations)),
            unit="s",
        )

    @property
    def concentric_phase_duration(self):
        com = self.center_of_mass
        vgrf = self.vertical_ground_reaction_force
        if com is None or vgrf is None:
            return None

        vaxis = com.vertical_axis
        velocity = np.gradient(com[vaxis].to_numpy().flatten(), com.index)

        module = np.abs(vgrf.to_numpy().flatten())
        contact = module > MINIMUM_CONTACT_FORCE_N
        batches = continuous_batches(contact)

        durations = []
        for batch in batches:
            if len(batch) == 0:
                continue

            vel_batch = velocity[batch]
            ascending = vel_batch > 0

            if not np.any(ascending):
                continue

            con_indices = np.where(ascending)[0]
            if len(con_indices) > 0:
                duration = com.index[batch[con_indices[-1]]] - com.index[batch[con_indices[0]]]
                durations.append(duration)

        if len(durations) == 0:
            return None

        return Signal1D(
            data=np.array(durations),
            index=np.arange(len(durations)),
            unit="s",
        )

    def copy(self):
        return SingleJump(**self._get_object_args())


__all__ = ["SingleJump"]
