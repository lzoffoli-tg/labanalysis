"""Jump test implementation."""

from os.path import exists
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from ...constants import G, MINIMUM_CONTACT_FORCE_N
from ...timeseries import EMGSignal, Point3D
from ...records import ForcePlatform, TimeseriesRecord
from ...exercises import DropJump, RepeatedJumps, SingleJump
from ...pipelines import get_default_processing_pipeline
from ...signalprocessing import butterworth_filt, fillna
from ...referenceframes import ReferenceFrame
from ..participant import Participant
from ..test_protocol import TestProtocol
from ..test_results import TestResults
from ..normativedata import jumps_normative_values
from .jump_test_results import JumpTestResults


class JumpTest(TestProtocol):
    """
    Test protocol for comprehensive jump performance assessment.

    JumpTest manages and analyzes multiple types of vertical jumps including
    squat jumps (SJ), counter-movement jumps (CMJ), drop jumps (DJ), and
    repeated jumps. The class handles data acquisition from force platforms,
    processes biomechanical signals, normalizes EMG data, and generates
    performance reports with normative comparisons.

    The protocol supports:
    - Multiple jump types with distinct biomechanical characteristics
    - Bilateral and unilateral jump execution
    - EMG normalization and muscle activation analysis
    - Automated data processing pipelines
    - Normative data comparison for performance ranking

    Parameters
    ----------
    participant : Participant
        Participant information including demographics and anthropometrics.
        Must have weight specified.
    normative_data : pd.DataFrame, optional
        Reference data for performance ranking and comparison.
        Default is jumps_normative_values.
    emg_normalization_references : TimeseriesRecord or str or 'self', optional
        Reference signals for EMG amplitude normalization. If 'self', uses
        test data for normalization. Default is empty TimeseriesRecord.
    emg_normalization_function : callable, optional
        Function to compute normalization value from reference (e.g., np.mean,
        np.max). Default is np.mean.
    emg_activation_references : TimeseriesRecord or str or 'self', optional
        Reference signals for determining muscle activation thresholds.
        Default is empty TimeseriesRecord.
    emg_activation_threshold : float, optional
        Threshold multiplier for detecting muscle activation onset.
        Default is 3 (3x reference level).
    relevant_muscle_map : list of str or None, optional
        List of muscle names to include in analysis. If None, includes all
        detected muscles. Default is None.
    squat_jumps : list of SingleJump, optional
        Squat jump trials. Default is empty list.
    counter_movement_jumps : list of SingleJump, optional
        Counter-movement jump trials. Default is empty list.
    drop_jumps : list of DropJump, optional
        Drop jump trials. Default is empty list.
    repeated_jumps : list of SingleJump, optional
        Individual jumps from repeated jump sequences. Default is empty list.

    Attributes
    ----------
    squat_jumps : list of SingleJump
        Squat jump trials (concentric-only jumps from static position).
    counter_movement_jumps : list of SingleJump
        Counter-movement jump trials (jumps with pre-stretch).
    drop_jumps : list of DropJump
        Drop jump trials (plyometric jumps from elevated surface).
    repeated_jumps : list of SingleJump
        Individual jumps from continuous jumping sequences.
    jumps : list
        All jumps combined (all four types concatenated).
    processed_data : JumpTest
        Copy of test with all signals processed through the pipeline.
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline with jump-specific configurations.

    Notes
    -----
    Jump Types:
    - Squat Jump (SJ): Concentric-only jump from static semi-squat position
      (no counter-movement allowed). Measures pure concentric power.
    - Counter-Movement Jump (CMJ): Jump with preliminary downward movement
      to utilize stretch-shortening cycle. Measures reactive strength.
    - Drop Jump (DJ): Jump immediately after landing from a box. Measures
      fast stretch-shortening cycle and reactive strength index (RSI).
    - Repeated Jumps: Continuous jumping for fatigue or endurance assessment.

    Processing Pipeline:
    - Force platforms: 30 Hz lowpass filter, moment calculation
    - EMG signals: 20-450 Hz bandpass, 50ms RMS envelope (vs. 200ms default)
    - Kinematic markers: Standard WholeBody processing pipeline
    - Reference frame: Auto-aligned to bilateral force center for bilateral jumps

    Performance Metrics:
    - Elevation (cm): Jump height calculated from flight time and impulse
    - Flight time (ms): Aerial phase duration
    - Contact time (ms): Ground contact duration
    - Takeoff velocity (m/s): Velocity at takeoff instant
    - RSI (cm/s): Reactive strength index (elevation/contact_time)
    - Force symmetry (%): Left-right force balance for bilateral jumps
    - EMG activation timing (ms): Muscle onset relative to landing
    - EMG pre-activation ratio (%): Pre-landing vs. post-landing EMG

    See Also
    --------
    SingleJump : Single vertical jump record.
    DropJump : Drop jump record with box height.
    RepeatedJumps : Continuous jumping sequence.
    JumpTestResults : Results container with figures and summaries.

    Examples
    --------
    >>> participant = Participant(weight=75, gender='male')
    >>> test = JumpTest.from_files(
    ...     participant=participant,
    ...     squat_jump_files=['sj_trial1.tdf'],
    ...     counter_movement_jump_files=['cmj_trial1.tdf', 'cmj_trial2.tdf'],
    ...     drop_jump_files=['dj_40cm.tdf'],
    ...     drop_jump_heights_cm=[40]
    ... )
    >>> results = test.get_results(include_emg=True)
    >>> print(results.summary)
    """

    @property
    def repeated_jumps(self):
        return self._repeated_jumps

    def add_repeated_jumps(self, *jumps: SingleJump):
        for jump in jumps:
            if not isinstance(jump, SingleJump):
                raise ValueError("jump must be a SingleJump instance.")
            self._repeated_jumps.append(jump)

    def pop_repeated_jumps(self, index: int):
        if not isinstance(index, int):
            raise ValueError("index must be an int.")
        if index < 0 or index > len(self._repeated_jumps) - 1:
            raise ValueError("index out of range.")
        jump = self._repeated_jumps.pop(index)
        return jump

    @property
    def squat_jumps(self):
        return self._squat_jumps

    def add_squat_jumps(self, *jumps: SingleJump):
        for jump in jumps:
            if not isinstance(jump, SingleJump):
                raise ValueError("jump must be a SingleJump instance.")
            self._squat_jumps.append(jump)

    def pop_squat_jumps(self, index: int):
        if not isinstance(index, int):
            raise ValueError("index must be an int.")
        if index < 0 or index > len(self._squat_jumps) - 1:
            raise ValueError("index out of range.")
        squat = self._squat_jumps.pop(index)
        return squat

    @property
    def counter_movement_jumps(self):
        return self._counter_movement_jumps

    def add_counter_movement_jumps(self, *jumps: SingleJump):
        for jump in jumps:
            if not isinstance(jump, SingleJump):
                raise ValueError("jump must be a SingleJump instance.")
            self._counter_movement_jumps.append(jump)

    def pop_counter_movement_jumps(self, index: int):
        if not isinstance(index, int):
            raise ValueError("index must be an int.")
        if index < 0 or index > len(self._counter_movement_jumps) - 1:
            raise ValueError("index out of range.")
        jump = self._counter_movement_jumps.pop(index)
        return jump

    @property
    def drop_jumps(self):
        return self._drop_jumps

    def add_drop_jumps(self, *jumps: DropJump):
        for jump in jumps:
            if not isinstance(jump, DropJump):
                raise ValueError("jump must be a DropJump instance.")
            self._drop_jumps.append(jump)

    def pop_drop_jumps(self, index: int):
        if not isinstance(index, int):
            raise ValueError("index must be an int.")
        if index < 0 or index > len(self._drop_jumps) - 1:
            raise ValueError("index out of range.")
        jump = self._drop_jumps.pop(index)
        return jump

    @property
    def jumps(self):
        return (
            self.squat_jumps
            + self.counter_movement_jumps
            + self.drop_jumps
            + self.repeated_jumps
        )

    def __init__(
        self,
        participant: Participant,
        normative_data: pd.DataFrame = jumps_normative_values,
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
        squat_jumps: list[SingleJump] = [],
        counter_movement_jumps: list[SingleJump] = [],
        drop_jumps: list[DropJump] = [],
        repeated_jumps: list[SingleJump] = [],
    ):
        if not isinstance(participant, Participant):
            raise ValueError("participant must be a Participant class instance.")
        if participant.weight is None:
            raise ValueError("participant's weight must be assigned.")
        super().__init__(
            participant=participant,
            normative_data=normative_data,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            emg_normalization_references=emg_normalization_references,
            relevant_muscle_map=relevant_muscle_map,
        )
        self._squat_jumps: list[SingleJump] = []
        self._counter_movement_jumps: list[SingleJump] = []
        self._drop_jumps: list[DropJump] = []
        self._repeated_jumps: list[SingleJump] = []
        self.add_squat_jumps(*squat_jumps)
        self.add_counter_movement_jumps(*counter_movement_jumps)
        self.add_drop_jumps(*drop_jumps)
        self.add_repeated_jumps(*repeated_jumps)

    @classmethod
    def from_files(
        cls,
        participant: Participant,
        normative_data: pd.DataFrame = jumps_normative_values,
        left_foot_ground_reaction_force: str = "left_foot",
        right_foot_ground_reaction_force: str = "right_foot",
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
        squat_jump_files: list[str] = [],
        squat_jump_free_hands: list[bool] | None = None,
        counter_movement_jump_files: list[str] = [],
        counter_movement_jump_free_hands: list[bool] | None = None,
        drop_jump_files: list[str] = [],
        drop_jump_heights_cm: list[int] | None = None,
        drop_jump_free_hands: list[bool] | None = None,
        repeated_jumps_files: list[str] = [],
        exclude_repeated_jumps: list[list[int]] | None = None,
        repeated_jumps_straight_leg: list[bool] | None = None,
        repeated_jumps_free_hands: list[bool] | None = None,
    ):

        # check the inputs
        if not isinstance(participant, Participant):
            raise ValueError("participant must be a Participant instance.")

        bodymass = participant.weight
        if bodymass is None:
            raise ValueError("participant's bodymass must be provided.")

        if not isinstance(squat_jump_files, list) or not all(
            [isinstance(i, str) and exists(i) for i in squat_jump_files]
        ):
            msg = "squat_jump_files must be a list of valid tdf file paths "
            msg += "corresponding to SingleJump instances."
            raise ValueError(msg)

        if not isinstance(counter_movement_jump_files, list) or not all(
            [isinstance(i, str) and exists(i) for i in counter_movement_jump_files]
        ):
            msg = "counter_movement_jump_files must be a list of valid tdf file"
            msg += " paths corresponding to SingleJump instances."
            raise ValueError(msg)

        if not isinstance(drop_jump_files, list) or not all(
            [isinstance(i, str) and exists(i) for i in drop_jump_files]
        ):
            msg = "drop_jump_files must be a list of valid tdf file"
            msg += " paths corresponding to DropJump instances."
            raise ValueError(msg)

        if drop_jump_heights_cm is None:
            drop_jump_heights_cm = [40 for _ in drop_jump_files]
        if (
            not isinstance(drop_jump_heights_cm, list)
            or not all([isinstance(i, int) for i in drop_jump_heights_cm])
            or len(drop_jump_heights_cm) != len(drop_jump_files)
        ):
            msg = "drop_jump_heights_cm must be a list of int, each representing"
            msg += " the height of each single drop jump."
            raise ValueError(msg)

        if not isinstance(repeated_jumps_files, list) or not all(
            [isinstance(i, str) and exists(i) for i in repeated_jumps_files]
        ):
            msg = "repeated_jumps_files must be a list of valid tdf file"
            msg += " paths corresponding to RepeatedJumps instances."
            raise ValueError(msg)

        if exclude_repeated_jumps is None:
            exclude_repeated_jumps = [[] for _ in repeated_jumps_files]
        if (
            not isinstance(exclude_repeated_jumps, list)
            or not all([isinstance(i, list) for i in exclude_repeated_jumps])
            or not all([isinstance(j, int) for i in exclude_repeated_jumps for j in i])
            or len(exclude_repeated_jumps) != len(repeated_jumps_files)
        ):
            msg = "exclude_repeated_jumps must be a list of lists, each "
            msg += "containing int repesenting the index of the jumps to "
            msg += "exclude from each repeated jump file."
            raise ValueError(msg)

        if repeated_jumps_straight_leg is None:
            repeated_jumps_straight_leg = [False for _ in repeated_jumps_files]
        if (
            not isinstance(repeated_jumps_straight_leg, list)
            or not all([isinstance(i, bool) for i in repeated_jumps_straight_leg])
            or len(repeated_jumps_straight_leg) != len(repeated_jumps_files)
        ):
            msg = "repeated_jumps_straight_leg must be a list of bool, each "
            msg += "representing the height of each single drop jump."
            raise ValueError(msg)

        if squat_jump_free_hands is None:
            squat_jump_free_hands = [False for _ in squat_jump_files]
        if (
            not isinstance(squat_jump_free_hands, list)
            or not all([isinstance(i, bool) for i in squat_jump_free_hands])
            or len(squat_jump_free_hands) != len(squat_jump_files)
        ):
            msg = "squat_jump_free_hands must be a list of bool, each "
            msg += "representing the height of each single drop jump."
            raise ValueError(msg)

        if counter_movement_jump_free_hands is None:
            counter_movement_jump_free_hands = [
                False for _ in counter_movement_jump_files
            ]
        if (
            not isinstance(counter_movement_jump_free_hands, list)
            or not all([isinstance(i, bool) for i in counter_movement_jump_free_hands])
            or len(counter_movement_jump_free_hands) != len(counter_movement_jump_files)
        ):
            msg = "counter_movement_jump_free_hands must be a list of bool, each "
            msg += "representing the height of each single drop jump."
            raise ValueError(msg)

        if drop_jump_free_hands is None:
            drop_jump_free_hands = [False for _ in drop_jump_files]
        if (
            not isinstance(drop_jump_free_hands, list)
            or not all([isinstance(i, bool) for i in drop_jump_free_hands])
            or len(drop_jump_free_hands) != len(drop_jump_files)
        ):
            msg = "drop_jump_free_hands must be a list of bool, each "
            msg += "representing the height of each single drop jump."
            raise ValueError(msg)

        if repeated_jumps_free_hands is None:
            repeated_jumps_free_hands = [False for _ in repeated_jumps_files]
        if (
            not isinstance(repeated_jumps_free_hands, list)
            or not all([isinstance(i, bool) for i in repeated_jumps_free_hands])
            or len(repeated_jumps_free_hands) != len(repeated_jumps_files)
        ):
            msg = "repeated_jumps_free_hands must be a list of bool, each "
            msg += "representing the height of each single drop jump."
            raise ValueError(msg)

        # read the files
        sjs = []
        for file, fh in zip(squat_jump_files, squat_jump_free_hands):
            sjs.append(
                SingleJump.from_tdf(
                    file=file,
                    bodymass_kg=bodymass,
                    free_hands=fh,
                    left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                    right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                )
            )

        cmjs = []
        for file, fh in zip(
            counter_movement_jump_files, counter_movement_jump_free_hands
        ):
            cmjs.append(
                SingleJump.from_tdf(
                    file=file,
                    bodymass_kg=bodymass,
                    free_hands=fh,
                    left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                    right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                )
            )

        djs = []
        for file, height, fh in zip(
            drop_jump_files, drop_jump_heights_cm, drop_jump_free_hands
        ):
            djs.append(
                DropJump.from_tdf(
                    file=file,
                    bodymass_kg=bodymass,
                    free_hands=fh,
                    box_height_cm=height,
                    left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                    right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                )
            )

        rjs = []
        for file, exclude, straight, fh in zip(
            repeated_jumps_files,
            exclude_repeated_jumps,
            repeated_jumps_straight_leg,
            repeated_jumps_free_hands,
        ):
            rjs += RepeatedJumps.from_tdf(
                file=file,
                bodymass_kg=bodymass,
                free_hands=fh,
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                exclude_jumps=exclude,
                straight_legs=straight,
            ).jumps

        return cls(
            participant=participant,
            normative_data=normative_data,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            relevant_muscle_map=relevant_muscle_map,
            squat_jumps=sjs,
            counter_movement_jumps=cmjs,
            drop_jumps=djs,
            repeated_jumps=rjs,
        )

    def get_results(self, include_emg: bool = True):
        return JumpTestResults(self.processed_data, include_emg)

    def copy(self):
        return JumpTest(
            participant=self.participant,
            normative_data=self.normative_data,
            emg_normalization_references=self.emg_activation_references,
            emg_activation_references=self.emg_activation_references,
            emg_activation_threshold=self.emg_activation_threshold,
            relevant_muscle_map=self.relevant_muscle_map,
            squat_jumps=self.squat_jumps,
            counter_movement_jumps=self.counter_movement_jumps,
            drop_jumps=self.drop_jumps,
            repeated_jumps=self.repeated_jumps,
        )

    def _process_record(self, record: TimeseriesRecord):
        # apply the pipeline to the test data
        pipeline = self.processing_pipeline
        exe = pipeline(record, inplace=False)  # type: ignore
        if not isinstance(exe, type(record)):
            raise ValueError("Something went wrong during data processing.")

        # normalize emg data and remove non-relevant muscles
        norms = self.emg_normalization_values
        to_remove: list[str] = []
        for k, m in exe.emgsignals.items():

            # remove if non relevant
            if self.relevant_muscle_map is not None:
                if not any([i.lower() in k.lower() for i in self.relevant_muscle_map]):
                    to_remove.append(k)
                    continue

            # normalize
            if isinstance(m, EMGSignal):
                for (name, side), val in norms.items():
                    if m.muscle_name == name and m.side == side:
                        exe[k] = m / val * 100
                        exe[k].set_unit("%")  # type: ignore
                        break
        if len(to_remove) > 0:
            exe.drop(to_remove, True)

        return exe

    def _process_jump(self, jump: SingleJump | DropJump):
        exe = jump.copy()

        # trim the data to the jump duration
        if not isinstance(jump, DropJump):
            grf = TimeseriesRecord()
            if jump.side in ["right", "bilateral"]:
                grf["right"] = jump.right_foot_ground_reaction_force
            if jump.side in ["left", "bilateral"]:
                grf["left"] = jump.left_foot_ground_reaction_force
            index = grf.strip()
            if index is None:
                raise RuntimeError("strip failed")
            index = index.index
            exe = exe.loc[index[0]:index[-1], :]
            if not isinstance(exe, TimeseriesRecord):
                raise RuntimeError("jump resizing failed.")

        exe = self._process_record(exe)

        # align the reference frame
        if exe.side not in ["right", "left"]:

            def extract_cop(force: Any):
                if not isinstance(force, ForcePlatform):
                    raise ValueError("force must be a ForcePlatform instance.")
                cop = force.origin
                if not isinstance(cop, Point3D):
                    raise ValueError("force must be a ForcePlatform instance.")
                cop = cop.copy()
                return cop.to_numpy().astype(float).mean(axis=0)

            # on bilateral test, we rotate the system of forces to a
            rt = extract_cop(exe.right_foot_ground_reaction_force)
            lt = extract_cop(exe.left_foot_ground_reaction_force)

            def norm(arr):
                return arr / np.sum(arr**2) ** 0.5

            ml = norm(lt - rt)
            vt = np.array([0, 1, 0])
            ap = np.cross(ml, vt)
            origin = (rt + lt) / 2
            ref_frame = ReferenceFrame(origin, ml, vt, ap)
            exe = ref_frame.apply(exe, inplace=False)
            if exe is None:
                raise ValueError("reference frame alignment returned None")

        return exe

    @property
    def processed_data(self):
        out = self.copy()
        for i, jump in enumerate(out.squat_jumps):
            out.squat_jumps[i] = self._process_jump(jump)  # type: ignore
        for i, jump in enumerate(out.counter_movement_jumps):
            out.counter_movement_jumps[i] = self._process_jump(jump)  # type: ignore
        for i, jump in enumerate(out.drop_jumps):
            out.drop_jumps[i] = self._process_jump(jump)  # type: ignore
        for i, jump in enumerate(out.repeated_jumps):
            out.repeated_jumps[i] = self._process_jump(jump)  # type: ignore
        if len(self.emg_normalization_references) > 0:
            out.set_emg_normalization_references(
                self._process_record(self.emg_normalization_references)  # type: ignore
            )
        if len(self.emg_activation_references) > 0:
            out.set_emg_activation_references(
                self._process_record(self.emg_activation_references)  # type: ignore
            )
        return out

    @property
    def processing_pipeline(self):

        # we need a custom force platform processing pipeline due to the
        # drop jump starting condition which might be outside the plates
        def forceplatform_processing_func(fp: ForcePlatform):

            # fill force nans with zeros
            fp.force[:, :] = fillna(fp.force.to_numpy(), value=0, inplace=False)

            # fill position nans via cubic spline
            fp.origin[:, :] = fillna(fp.origin.to_numpy(), inplace=False)

            # lowpass filter both origin and force
            fsamp = float(1 / np.mean(np.diff(fp.index)))
            filt_fun = lambda x: butterworth_filt(
                x,
                fcut=30,
                fsamp=fsamp,  # type: ignore
                order=4,
                ftype="lowpass",
                phase_corrected=True,
            )
            fp.origin.apply(filt_fun, axis=0, inplace=True)
            fp.force.apply(filt_fun, axis=0, inplace=True)

            # update moments
            fp.update_moments(inplace=True)

            # set moments corresponding to the very low vertical force to zero
            module = fp.force.copy().module.to_numpy().flatten()  # type: ignore
            idxs = module < MINIMUM_CONTACT_FORCE_N
            vals = fp.torque.copy().to_numpy()
            vals[idxs, :] = 0
            fp.torque[:, :] = vals

        # we need also a custom EMGSignal processing pipeline to create a
        # short RMS envelope with 50ms rolling window.
        def emgsignal_processing_func(channel: EMGSignal):
            channel[:, :] -= channel.to_numpy().mean()
            fsamp = 1 / np.mean(np.diff(channel.index))
            channel.apply(
                butterworth_filt,
                fcut=[20, 450],
                fsamp=fsamp,
                order=4,
                ftype="bandpass",
                phase_corrected=True,
                inplace=True,
                axis=0,
            )
            channel.apply(
                rms_filt,
                order=int(0.05 * fsamp),
                pad_style="reflect",
                offset=0.5,
                inplace=True,
                axis=0,
            )

        pipeline = get_default_processing_pipeline()
        pipeline.add(
            ForcePlatform=[forceplatform_processing_func],
            EMGSignal=[emgsignal_processing_func],
        )
        return pipeline


__all__ = ["JumpTest"]
