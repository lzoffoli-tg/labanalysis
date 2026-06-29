"""Single jump exercise module."""

import numpy as np

from ..constants import MINIMUM_CONTACT_FORCE_N, MINIMUM_FLIGHT_TIME_S
from ..signalprocessing import continuous_batches
from ..records.body import WholeBody
from ..records import ForcePlatform, TimeseriesRecord
from ..timeseries import Signal1D, Signal3D, EMGSignal, Point3D


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
        **kwargs,
    ):
        # Store non-signal attributes using object.__setattr__
        # to bypass Record.__setattr__ which would try to store them as signals
        object.__setattr__(self, '_bodymass_kg', bodymass_kg)
        object.__setattr__(self, '_free_hands', free_hands)
        object.__setattr__(self, '_straight_legs', straight_legs)

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
        self.set_left_foot_ground_reaction_force(left_foot_ground_reaction_force)
        self.set_right_foot_ground_reaction_force(right_foot_ground_reaction_force)

    def set_left_foot_ground_reaction_force(self, fp: ForcePlatform | None):
        """
        Set the left foot ground reaction force platform.

        Parameters
        ----------
        fp : ForcePlatform or None
            Force platform object measuring left foot ground reaction forces.

        Raises
        ------
        ValueError
            If fp is not a ForcePlatform instance or None.
        """
        if fp is not None and not isinstance(fp, ForcePlatform):
            raise ValueError("left_foot_ground_reaction_force must be a ForcePlatform")
        self._left_foot_ground_reaction_force = fp

    def set_right_foot_ground_reaction_force(self, fp: ForcePlatform | None):
        """
        Set the right foot ground reaction force platform.

        Parameters
        ----------
        fp : ForcePlatform or None
            Force platform object measuring right foot ground reaction forces.

        Raises
        ------
        ValueError
            If fp is not a ForcePlatform instance or None.
        """
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
        timestamps = []
        for batch in batches:
            if len(batch) > 0:
                duration = vgrf.index[batch[-1]] - vgrf.index[batch[0]]
                if duration >= MINIMUM_FLIGHT_TIME_S:
                    durations.append(duration)
                    timestamps.append(vgrf.index[batch[0]])  # Start time of flight phase
        if len(durations) == 0:
            return None
        return Signal1D(
            data=np.array(durations),
            index=np.array(timestamps),
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
        timestamps = []

        # Minimum contact duration to filter out noise (10ms)
        MINIMUM_CONTACT_DURATION_S = 0.010

        for batch in batches:
            if len(batch) > 0:
                duration = vgrf.index[batch[-1]] - vgrf.index[batch[0]]
                # Filter out very short contacts (likely noise)
                if duration >= MINIMUM_CONTACT_DURATION_S:
                    durations.append(duration)
                    timestamps.append(vgrf.index[batch[0]])  # Start time of contact phase
        if len(durations) == 0:
            return None
        return Signal1D(
            data=np.array(durations),
            index=np.array(timestamps),
            unit="s",
        )

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

        module = np.abs(vgrf.to_numpy().flatten())
        contact = module > MINIMUM_CONTACT_FORCE_N
        batches = continuous_batches(contact)

        if len(batches) == 0:
            return None

        # Minimum contact duration to filter out noise (10ms)
        MINIMUM_CONTACT_DURATION_S = 0.010

        # Filter batches by minimum duration
        valid_batches = []
        for batch in batches:
            if len(batch) > 0:
                duration = vgrf.index[batch[-1]] - vgrf.index[batch[0]]
                if duration >= MINIMUM_CONTACT_DURATION_S:
                    valid_batches.append(batch)

        if len(valid_batches) == 0:
            return None

        # Use the last valid contact batch (primary propulsive phase)
        batch = valid_batches[-1]

        # Use iloc to slice by integer positions (preserves all samples in batch)
        return self.iloc[batch[0]:batch[-1]+1, :]

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

        module = np.abs(vgrf.to_numpy().flatten())
        contact = module > MINIMUM_CONTACT_FORCE_N
        flight = ~contact
        batches = continuous_batches(flight)

        # Find valid flight phases (meeting minimum duration)
        valid_batches = []
        for batch in batches:
            if len(batch) > 0:
                duration = vgrf.index[batch[-1]] - vgrf.index[batch[0]]
                if duration >= MINIMUM_FLIGHT_TIME_S:
                    valid_batches.append(batch)

        if len(valid_batches) == 0:
            return None

        # Use the first valid flight batch
        batch = valid_batches[0]

        # Use iloc to slice by integer positions (preserves all samples in batch)
        return self.iloc[batch[0]:batch[-1]+1, :]

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

        # Get data and timestamps
        jh_data = jump_height.to_numpy().flatten()
        jh_times = jump_height.index
        ct_data = contact_time.to_numpy().flatten()
        ct_times = contact_time.index

        if len(jh_data) == 0 or len(ct_data) == 0:
            return None

        # Pair each flight phase with the nearest preceding contact phase
        rsi_values = []
        rsi_times = []

        for i, jh_time in enumerate(jh_times):
            # Find contact phases that ended before or at this flight phase start
            preceding_contacts = ct_times <= jh_time
            if np.any(preceding_contacts):
                # Get the closest preceding contact
                ct_idx = np.where(preceding_contacts)[0][-1]
                # Avoid division by zero
                if ct_data[ct_idx] > 0:
                    rsi_values.append(jh_data[i] / ct_data[ct_idx])
                    rsi_times.append(jh_time)

        if len(rsi_values) == 0:
            return None

        return Signal1D(
            data=np.array(rsi_values),
            index=np.array(rsi_times),
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
            **{k: v.copy() if hasattr(v, 'copy') else v for k, v in self._data.items()}
        )


__all__ = ["SingleJump"]
