"""Walking-specific gait classes."""

from typing import Literal

import numpy as np
import pandas as pd

from ...constants import *
from ...records.forceplatform import ForcePlatform
from ...signalprocessing import *
from ...timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from .gait_exercise import GaitExercise
from .walking_stride import WalkingStride

__all__ = ["WalkingExercise"]


class WalkingExercise(GaitExercise):
    """
    Represents a walking test.

    Parameters
    ----------
    frame : StateFrame
        A stateframe object containing all the available kinematic, kinetic
        and EMG data related to the test.
    algorithm : Literal['kinematics', 'kinetics'], optional
        Algorithm used for gait cycle detection. 'kinematics' uses marker data,
        'kinetics' uses force platform data.
    left_heel : Point3D or None, optional
        The left heel marker data.
    right_heel : Point3D or None, optional
        The right heel marker data.
    left_toe : Point3D or None, optional
        The left toe marker data.
    right_toe : Point3D or None, optional
        The right toe marker data.
    left_first_metatarsal_head : Point3D or None
        Left first metatarsal head marker.
    left_fifth_metatarsal_head : Point3D or None
        Left fifth metatarsal head marker.
    right_first_metatarsal_head : Point3D or None
        Right first metatarsal head marker.
    right_fifth_metatarsal_head : Point3D or None, optional
        The left metatarsal head marker data.
    right_metatarsal_head : Point3D or None, optional
        The right metatarsal head marker data.
    ground_reaction_force : ForcePlatform or None, optional
        Ground reaction force data.
    ground_reaction_force_threshold : float or int, optional
        Minimum ground reaction force for contact detection.
    height_threshold : float or int, optional
        Maximum vertical height for contact detection.
    vertical_axis : Literal['X', 'Y', 'Z'], optional
        The vertical axis.
    antpos_axis : Literal['X', 'Y', 'Z'], optional
        The anterior-posterior axis.
    """

    def _find_cycles_kinematics(self):
        """
        Find the gait cycles using the kinematics algorithm.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any required marker is missing or no toe-offs have been found.
        """

        # get toe-off times
        time = self.index
        fsamp = float(1 / np.mean(np.diff(time)))
        cycles: list[WalkingStride] = []
        for lbl in ["left_toe", "right_toe"]:

            # get the vertical coordinates of the toe markers
            arr = self[lbl].copy()[self.vertical_axis].to_numpy().flatten()  # type: ignore

            # filter and rescale
            ftoe = arr / np.max(arr)

            # get the minimum reasonable contact time for each step
            frq, pwr = psd(ftoe, fsamp)
            ffrq = frq[np.argmax(pwr)]
            dsamples = int(round(fsamp / ffrq / 2))

            # get the peaks at each cycle
            pks = find_peaks(ftoe, 0.5, dsamples)

            # for each peak obtain the location of the last sample at the
            # required height threshold
            tos = []
            side = lbl.split("_")[0]
            for pk in pks:
                idx = np.where(ftoe[:pk] <= self.height_threshold)[0]
                if len(idx) > 0:
                    line = pd.Series({"time": time[idx[-1]], "side": side})
                    tos += [pd.DataFrame(line).T]

            # wrap the events
            if len(tos) == 0:
                raise ValueError("no toe-offs have been found.")
            tos = pd.concat(tos, ignore_index=True)
            tos = tos.drop_duplicates()
            tos = tos.sort_values("time")
            tos = tos.reset_index(drop=True)

            # check the alternation of the steps
            for i0, i1 in zip(tos.index[:-1], tos.index[1:]):  # type: ignore
                t0 = float(tos.time.values[i0])
                t1 = float(tos.time.values[i1])
                cycles += [
                    self._get_cycle(
                        t0,
                        t1,
                        side,  # type: ignore
                    )
                ]

        # sort the cycles
        cycle_index = np.argsort([i.init_s for i in cycles])
        return [cycles[i] for i in cycle_index]

    def _find_cycles_kinetics(self):
        """
        Find the gait cycles using the kinetics algorithm.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ground_reaction_force not found.
        """

        # get the relevant data
        res = self.resultant_force
        if res is None:
            raise ValueError("ground_reaction_force not found")
        time = res.index
        res_ap = res["force"].copy()[self.anteroposterior_axis].to_numpy().flatten()  # type: ignore
        res_ap -= np.nanmean(res_ap)

        # get the dominant frequency
        fsamp = float(1 / np.mean(np.diff(time)))
        frq, pwr = psd(res_ap, fsamp)
        ffrq = frq[np.argmax(pwr)]

        # find peaks
        min_samp = int(fsamp / ffrq / 2)
        pks = find_peaks(res_ap, 0, min_samp)

        # for each peak pair get the range and obtain the toe-off
        # as the last value occurring before the peaks within the
        # 1 - height_threshold of that range
        toi = []
        for pk in pks:
            thresh = (1 - self.height_threshold) * res_ap[pk]
            loc = np.where(res_ap[:pk] < thresh)[0]
            if len(loc) > 0:
                toi += [loc[-1]]

        # get the latero-lateral centre of pressure
        cop_ml = res["origin"].copy()[self.lateral_axis].to_numpy().flatten()  # type: ignore
        cop_ml -= np.nanmean(cop_ml)

        # get the sin function best fitting the cop_ml
        def _sin_fitted(arr: np.ndarray):
            """fit a sine over arr"""
            rfft = np.fft.rfft(arr - np.mean(arr))
            pwr = psd(arr)[1]
            rfft[pwr < np.max(pwr)] = 0
            return np.fft.irfft(rfft, len(arr))

        sin_ml = _sin_fitted(cop_ml)

        # get the mean latero-lateral position of each toe-off interval
        cnt = [np.arange(i, j + 1) for i, j in zip(toi[:-1], toi[1:])]
        pos = [np.nanmean(sin_ml[i]) for i in cnt]

        # get the sides
        sides = ["left" if i > 0 else "right" for i in pos]

        # generate the steps
        toi_evens = toi[0:-1:2]
        sides_evens = sides[0:-1:2]
        toi_odds = toi[1:-1:2]
        sides_odds = sides[1:-1:2]
        cycles: list[WalkingStride] = []
        for ti, si in zip([toi_evens, toi_odds], [sides_evens, sides_odds]):
            for to, ed, side in zip(ti[:-1], ti[1:], si):
                cycles += [
                    self._get_cycle(
                        float(self.index[to]),
                        float(self.index[ed]),
                        side,  # type: ignore
                    )
                ]

        # sort the cycles
        idx = np.argsort([i.init_s for i in cycles])
        return [cycles[i] for i in idx]

    def _get_cycle(
        self,
        start: float,
        stop: float,
        side: Literal["left", "right"],
    ):
        step = self.copy()[start:stop]
        args = {
            "side": side,
            "ground_reaction_force_threshold": self.ground_reaction_force_threshold,
            "height_threshold": self.height_threshold,
            "algorithm": self.algorithm,
        }
        if step is not None:
            args.update(**{i: v for i, v in step.items()})  # type: ignore
        return WalkingStride(**args)  # type: ignore

    def __init__(
        self,
        algorithm: Literal["kinematics", "kinetics"] = "kinematics",
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
            Initialize a WalkingTest instance.

            Parameters
            ----------
            algorithm : {'kinematics', 'kinetics'}, optional
                Algorithm used for gait cycle detection. 'kinematics' uses marker
                data, 'kinetics' uses force platform data.
            left_heel : Point3D or None, optional
                Marker data for the left heel.
            right_heel : Point3D or None, optional
                Marker data for the right heel.
            left_toe : Point3D or None, optional
                Marker data for the left toe.
            right_toe : Point3D or None, optional
                Marker data for the right toe.
            left_first_metatarsal_head : Point3D or None
            Left first metatarsal head marker.
        left_fifth_metatarsal_head : Point3D or None
            Left fifth metatarsal head marker.
        right_first_metatarsal_head : Point3D or None
            Right first metatarsal head marker.
        right_fifth_metatarsal_head : Point3D or None, optional
                Marker data for the left metatarsal head.
            right_metatarsal_head : Point3D or None, optional
                Marker data for the right metatarsal head.
            grf : ForcePlatform or None, optional
                Ground reaction force data.
            grf_threshold : float or int, optional
                Minimum ground reaction force for contact detection.
            height_threshold : float or int, optional
                Maximum vertical height for contact detection.
            vertical_axis : {'X', 'Y', 'Z'}, optional
                The vertical axis.
            antpos_axis : {'X', 'Y', 'Z'}, optional
                The anterior-posterior axis.
            process_inputs : bool, optional
                If True, process the input data.
            **extra_signals : Signal1D, Signal3D, EMGSignal, Point3D, ForcePlatform
                Additional signals to include.
        """
        super().__init__(
            algorithm=algorithm,
            ground_reaction_force_threshold=ground_reaction_force_threshold,
            height_threshold=height_threshold,
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
            **extra_signals,
        )
