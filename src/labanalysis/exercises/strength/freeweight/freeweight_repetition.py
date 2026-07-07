"""FreeWeightRepetition module."""

import numpy as np
import pandas as pd

from ....records.forceplatform import ForcePlatform
from ....signalprocessing import find_peaks
from ....timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from .default_freeweight_object import DefaultFreeWeightObject
from .repetition_phase import RepetitionPhase

__all__ = ["FreeWeightRepetition"]


class FreeWeightRepetition(DefaultFreeWeightObject):
    """
    Free Weight Exercise repetition instance

    Parameters
    ----------
    time: Iterable[int | float]
        the array containing the time instant of each repetition in seconds

    barbell_position: Iterable[int | float]
        the array containing the displacement (in meters) of the barbell for each repetition (2 markers)

    load: Iterable[int | float]
        the array containing the load measured at each repetition in kgf

    repetition: int
        the repetition number

    Attributes
    ----------
    raw: DataFrame
        a DataFrame containing the input data

    repetitions: list[DataFrame]
        a list of dataframes each defining one single repetition

    rom_m: float
        the range of movement amplitude in meters

    results_table: DataFrame
        a table containing the data obtained during the test

    summary_table: DataFrame
        a table containing summary statistics about the test

    summary_plot: FigureWidget
        a figure representing the results of the test.
    """

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
        super().__init__(
            total_load_kg=total_load_kg,
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
            right_shoulder_anterior=right_shoulder_anterior,
            right_shoulder_posterior=right_shoulder_posterior,
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
            sc=sc,
            l2=l2,
            **extra_signals,
        )

    @property
    def phases(self):
        """return the phases of the repetition"""
        out: list[RepetitionPhase] = []
        y = self.acceleration
        if y is None:
            return out
        t = y.index.astype(float).flatten()
        v = y.copy()[self.vertical_axis].to_numpy().flatten()
        mins = find_peaks(-v)
        maxs = find_peaks(v)

        def get_starts_and_ends(peaks: np.ndarray, signal: np.ndarray):
            phases: list[float] = []
            for peak in peaks:
                start = np.where(signal[:peak] <= 0)[0]
                start = 0 if len(start) == 0 else (start[-1] + 1)
                end = np.where(signal[peak:] >= 0)[0]
                end = (len(signal) - 1) if len(end) == 0 else (peak + end[0] - 1)
                if start < peak < end:
                    phases += [float(t[start]), float(t[end])]
            return phases

        events = get_starts_and_ends(mins, -v)
        events += get_starts_and_ends(maxs, v)
        events.append(t[-1])
        events = sorted(events)
        for i0, i1 in zip(events[:-1], events[1:]):
            out.append(
                RepetitionPhase(
                    total_load_kg=self.total_load_kg,
                    **{i: v.loc[i0:i1, :] for i, v in self.items()},
                )
            )
        return out

    @property
    def concentric_phase(self):
        """return the concentric phase of the repetition"""
        phase = [p for p in self.phases if p.phase == "concentric"]
        if len(phase) == 0:
            return None
        return phase[0]

    @property
    def eccentric_phase(self):
        """return the eccentric phase of the repetition"""
        phase = [p for p in self.phases if p.phase == "eccentric"]
        if len(phase) == 0:
            return None
        return phase[0]

    @property
    def isometric_phases(self):
        """return the isometric phases of the repetition"""
        out = [p for p in self.phases if p.phase == "isometric"]
        if len(out) == 0:
            return None
        return out[0]

    @property
    def rom(self):
        """return the repetition's ROM in meters"""
        return float(np.max([i.total_displacement for i in self.phases]))

    @property
    def output_metrics(self):
        """return a dictionary with the main output metrics of the phase"""
        return pd.concat(
            [pd.DataFrame(pd.Series(p.output_metrics)).T for p in self.phases],
            ignore_index=True,
        )
