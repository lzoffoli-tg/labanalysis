"""DefaultFreeWeightObject module."""

import numpy as np

from ....constants import G
from ....timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from ....records.forceplatform import ForcePlatform
from ....records.body import WholeBody


class DefaultFreeWeightObject(WholeBody):
    """
    Represents a free weight exercise object with barbell or dumbbell tracking.

    This class extends WholeBody to include specific markers for tracking barbells
    and dumbbells during free weight exercises. It inherits all 42 anatomical markers
    and force platform parameters from WholeBody, plus adds equipment-specific markers.

    Parameters
    ----------
    total_load_kg : int or float
        Total load in kilograms (barbell + plates or dumbbell weight).
    left_dumbbell_medial : Point3D, optional
        Medial marker on left dumbbell.
    left_dumbbell_lateral : Point3D, optional
        Lateral marker on left dumbbell.
    right_dumbbell_medial : Point3D, optional
        Medial marker on right dumbbell.
    right_dumbbell_lateral : Point3D, optional
        Lateral marker on right dumbbell.
    left_barbell : Point3D, optional
        Left side marker on barbell.
    right_barbell : Point3D, optional
        Right side marker on barbell.
    left_hand_ground_reaction_force : ForcePlatform, optional
        Left hand force platform data.
    right_hand_ground_reaction_force : ForcePlatform, optional
        Right hand force platform data.
    left_foot_ground_reaction_force : ForcePlatform, optional
        Left foot force platform data.
    right_foot_ground_reaction_force : ForcePlatform, optional
        Right foot force platform data.
    left_shoulder_anterior : Point3D, optional
        Anterior left shoulder marker.
    left_shoulder_posterior : Point3D, optional
        Posterior left shoulder marker.
    left_acromion : Point3D, optional
        Left acromion marker (shoulder tip).
    right_shoulder_anterior : Point3D, optional
        Anterior right shoulder marker.
    right_shoulder_posterior : Point3D, optional
        Posterior right shoulder marker.
    right_acromion : Point3D, optional
        Right acromion marker (shoulder tip).
    **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
        Additional signals to include (e.g., anatomical markers, EMG).

    Notes
    -----
    This class inherits all 42 WholeBody parameters (38 anatomical markers + 4 force platforms).
    Only the most relevant parameters are listed above. See WholeBody documentation for the
    complete list of available anatomical markers.

    Attributes
    ----------
    total_load_kg : float
        The total load in kilograms.
    barbell_midpoint : Point3D or None
        Calculated midpoint between left and right barbell markers.
    tool_position : Point3D or None
        Position of the exercise tool (barbell or dumbbell center).
    """

    _total_load_kg: float

    def __init__(
        self,
        total_load_kg: int | float,
        left_dumbbell_medial: Point3D | None = None,
        left_dumbbell_lateral: Point3D | None = None,
        right_dumbbell_medial: Point3D | None = None,
        right_dumbbell_lateral: Point3D | None = None,
        left_barbell: Point3D | None = None,
        right_barbell: Point3D | None = None,
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
        signals = {
            **extra_signals,
            **dict(
                left_dumbbell_medial=left_dumbbell_medial,
                left_dumbbell_lateral=left_dumbbell_lateral,
                right_dumbbell_medial=right_dumbbell_medial,
                right_dumbbell_lateral=right_dumbbell_lateral,
                left_barbell=left_barbell,
                right_barbell=right_barbell,
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
        super().__init__(**{i: v for i, v in signals.items() if v is not None})

        self.set_total_load_kg(total_load_kg)

    def set_total_load_kg(self, total_load_kg: int | float):
        """Set the total load in kg"""
        if not isinstance(total_load_kg, (int, float)):
            raise TypeError("total_load_kg must be an integer or a float")
        if total_load_kg < 0:
            raise ValueError("total_load_kg must be a positive value")
        self._total_load_kg = float(total_load_kg)

    @property
    def total_load_kg(self):
        """return the load of the repetition in kg"""
        return self._total_load_kg

    @property
    def barbell_midpoint(self):
        """return the barbell midpoint position as a Point3D"""
        left = self.get("left_barbell")
        right = self.get("right_barbell")
        if left is None or right is None:
            return None
        if not isinstance(left, Point3D) or not isinstance(right, Point3D):
            return None
        return (left + right) / 2

    @property
    def left_dumbbell_midpoint(self):
        """return the left dumbbell midpoint position as a Point3D"""
        medial = self.get("left_dumbell_medial")
        lateral = self.get("left_dumbell_lateral")
        if medial is None or lateral is None:
            return None
        if not isinstance(medial, Point3D) or not isinstance(lateral, Point3D):
            return None
        return (medial + lateral) / 2

    @property
    def right_dumbbell_midpoint(self):
        """return the left dumbbell midpoint position as a Point3D"""
        medial = self.get("right_dumbell_medial")
        lateral = self.get("right_dumbell_lateral")
        if medial is None or lateral is None:
            return None
        if not isinstance(medial, Point3D) or not isinstance(lateral, Point3D):
            return None
        return (medial + lateral) / 2

    @property
    def tool_position(self):
        """return the position of the tool (barbell or dumbbells)"""
        position = self.barbell_midpoint
        if position is None:
            left_dumbbell = self.left_dumbbell_midpoint
            right_dumbbell = self.right_dumbbell_midpoint
            if left_dumbbell is None and right_dumbbell is None:
                return None
            elif left_dumbbell is None:
                position = right_dumbbell
            elif right_dumbbell is None:
                position = left_dumbbell
            else:
                position = (left_dumbbell + right_dumbbell) / 2
        return position

    @property
    def total_displacement(self):
        """return the phase's displacement"""
        velocity = self.velocity
        if velocity is None:
            return np.nan
        norm = np.sum(velocity.copy().to_numpy() ** 2, axis=1) ** 0.5
        return float(np.trapezoid(norm, self.index))

    @property
    def velocity(self):
        """return the repetition's velocity"""
        position = self.tool_position
        if position is None:
            return None
        return Signal3D(
            data=np.gradient(position.to_numpy(), self.index, axis=0),
            index=self.index,
            unit=position.unit + "/s",
            columns=position.columns,
            vertical_axis=position.vertical_axis,
            anteroposterior_axis=position.anteroposterior_axis,
        )

    @property
    def power(self):
        """return the repetition's power in Watts"""
        velocity = self.velocity
        if velocity is None:
            return None
        power = velocity.copy() * self.total_load_kg * G
        power.set_unit("W")
        return power

    @property
    def acceleration(self):
        """return the repetition's acceleration in m/s^2"""
        position = self.tool_position
        if position is None:
            return None
        acc = np.gradient(
            position.copy().to_numpy(),
            self.index,
            axis=0,
            edge_order=2,
        )
        return Signal3D(
            data=acc,
            index=self.index,
            unit=position.unit + "/s**2",
            columns=position.columns,
            vertical_axis=position.vertical_axis,
            anteroposterior_axis=position.anteroposterior_axis,
        )

    @property
    def duration(self):
        """return the phase's duration in seconds"""
        return float(self.index[-1] - self.index[0])

    @property
    def muscle_activations(self):
        """
        Returns coordination and balance metrics from EMG signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with coordination and balance metrics, or empty if not available.
        """

        emgs = self.emgsignals
        out: dict[str, float] = {}
        if emgs.shape[1] == 0:
            return out

        for emg in emgs.values():
            if isinstance(emg, EMGSignal):
                side = emg.side
                name = emg.muscle_name
                unit = emg.unit
                out[f"{side} {name} {unit}"] = float(emg.to_numpy().mean())

        return out

    @property
    def output_metrics(self):
        """return a dictionary with the main output metrics of the phase"""
        metrics: dict[str, float] = {}
        metrics["total_load_kg"] = self.total_load_kg
        metrics["total_displacement_m"] = self.total_displacement
        velocity = self.velocity
        if velocity is None:
            metrics["mean_velocity_m_s"] = np.nan
            metrics["peak_velocity_m_s"] = np.nan
        else:
            velocity = np.linalg.norm(velocity.copy().to_numpy(), axis=1)
            metrics["mean_velocity_m_s"] = np.mean(velocity)
            metrics["peak_velocity_m_s"] = np.max(velocity)
        power = self.power
        if power is None:
            metrics["mean_power_W"] = np.nan
            metrics["peak_power_W"] = np.nan
        else:
            power = np.linalg.norm(power.copy().to_numpy(), axis=1)
            metrics["mean_power_W"] = np.mean(power)
            metrics["peak_power_W"] = np.max(power)
        acc = self.acceleration
        if acc is None:
            metrics["mean_acceleration_m_s2"] = np.nan
            metrics["max_acceleration_m_s2"] = np.nan
        else:
            acc = np.linalg.norm(acc.copy().to_numpy(), axis=1)
            metrics["mean_acceleration_m_s2"] = np.mean(acc)
            metrics["peak_acceleration_m_s2"] = np.max(acc)
        metrics["duration_s"] = self.duration
        metrics.update(self.muscle_activations)
        return metrics
