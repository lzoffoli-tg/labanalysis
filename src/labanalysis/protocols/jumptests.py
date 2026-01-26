"""singlejumps test module"""

#! IMPORTS


__all__ = ["JumpTest", "JumpTestResults"]


#! CLASSES

from typing import Any, Callable, Literal
from os.path import exists

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..constants import (
    MINIMUM_CONTACT_FORCE_N,
    RANK_3COLORS,
    RANK_5COLORS,
    SIDE_COLORS,
    G,
)
from ..records.jumping import DropJump, SingleJump, RepeatedJumps
from ..records.pipelines import get_default_processing_pipeline
from ..records.records import ForcePlatform, TimeseriesRecord
from ..records.timeseries import EMGSignal, Point3D
from ..signalprocessing import butterworth_filt, continuous_batches, fillna, rms_filt
from ..utils import hex_to_rgba
from .normativedata import jumps_normative_values
from .protocols import Participant, TestProtocol, TestResults


class JumpTest(TestProtocol):

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
        exe = self.processing_pipeline(record, inplace=False)  # type: ignore
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
            exe = exe.loc(index[0], index[-1])
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
            exe.change_reference_frame(
                ml,
                vt,
                ap,
                origin,
                inplace=True,
            )
            if exe is None:
                raise ValueError("reference frame alignment returned None")

        return exe

    @property
    def processed_data(self):
        out = self.copy()
        for i, jump in enumerate(out.squat_jumps):
            out._squat_jumps[i] = self._process_jump(jump)  # type: ignore
        for i, jump in enumerate(out.counter_movement_jumps):
            out._counter_movement_jumps[i] = self._process_jump(jump)  # type: ignore
        for i, jump in enumerate(out.drop_jumps):
            out._drop_jumps[i] = self._process_jump(jump)  # type: ignore
        for i, jump in enumerate(out.repeated_jumps):
            out._repeated_jumps[i] = self._process_jump(jump)  # type: ignore
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


class JumpTestResults(TestResults):

    def __init__(self, test: JumpTest, include_emg: bool):
        if not isinstance(test, JumpTest):
            raise ValueError("'test' must be an JumpTest instance.")
        super().__init__(test, include_emg)

    def _get_jump_contact_time_ms(self, jump: SingleJump | DropJump):
        time = jump.contact_phase.index
        return int(round((time[-1] - time[0]) * 1000))

    def _get_jump_flight_time_ms(self, jump: SingleJump | DropJump):
        time = jump.flight_phase.index
        return int(round((time[-1] - time[0]) * 1000))

    def _get_takeoff_velocity_ms(
        self,
        jump: SingleJump | DropJump,
        bodyweight: float,
    ):

        # get the ground reaction force during the concentric phase
        con = jump.contact_phase.resultant_force
        if con is None:
            return np.nan
        grf = con.copy().force[jump.vertical_axis].to_numpy().flatten()
        grfy = fillna(arr=grf, value=0).flatten()  # type: ignore
        grft = con.index

        # get the output velocity
        net_grf = grfy - bodyweight * G
        return float(np.trapezoid(net_grf / bodyweight, grft))

    def _get_elevation_cm(
        self,
        jump: SingleJump | DropJump,
        bodyweight: float,
    ):

        # from flight time
        flight_time = jump.flight_phase.index
        flight_time = flight_time[-1] - flight_time[0]
        elevation_from_time = (flight_time**2) * G / 8 * 100

        # from force impulse
        elevation_from_velocity = (
            (self._get_takeoff_velocity_ms(jump, bodyweight) ** 2) / (2 * G) * 100
        )

        # return the lower of the two
        return float(min(elevation_from_time, elevation_from_velocity))

    def _get_muscle_activation_ratio(
        self,
        muscle: EMGSignal,
        landing_instant_s: float,
        t_bodyweight_s: float,
        pre_window_s: float = 0.025,
    ):

        # get the data
        time = muscle.index.copy()
        envelope = muscle.to_numpy().flatten()

        # get the pre-condition
        pre_idx = np.where(
            (time >= landing_instant_s - pre_window_s) & (time < landing_instant_s)
        )[0]
        lr_idx = np.where((time >= landing_instant_s) & (time <= t_bodyweight_s))[0]
        if len(pre_idx) == 0 or len(lr_idx) == 0:
            raise RuntimeError("time window not possible.")

        # get the mean amplitude in the pre-window the the highest activation
        # in the lr window
        pre_emg = float(envelope[pre_idx].mean())
        lr_emg = float(envelope[lr_idx].max())

        # return the ratio
        return pre_emg / lr_emg

    def _get_muscle_activation_time_ms(
        self,
        muscle: EMGSignal,
        threshold: float,
        landing_instant_s: float,
    ):

        # get the data
        time = muscle.index.copy()
        envelope = muscle.to_numpy().flatten()

        # offset time to contact_time_instant_s
        time -= landing_instant_s

        # get the steady activation time
        batches = continuous_batches(envelope >= threshold)
        fsamp = float(1 / np.mean(np.diff(time)))
        samples = int(round(fsamp * 0.025))
        batches = [i for i in batches if len(i) >= samples]
        if len(batches) == 0:
            return time[-1] * 1000
        return time[np.array(batches[0])][0] * 1000

    def _get_summary(self, test: JumpTest):
        muscle_map = test.relevant_muscle_map
        if test.relevant_muscle_map is None:
            muscle_map = []
        else:
            muscle_map = test.relevant_muscle_map.copy()

        def _get_jumps_summary_table(
            jumps: list[SingleJump | DropJump],
            jump_name: str,
        ):
            sides_counter = {}
            sides_df = {}
            for jump in jumps:
                if jump.side not in sides_counter:
                    sides_counter[jump.side] = 1
                    sides_df[jump.side] = pd.DataFrame()
                else:
                    sides_counter[jump.side] += 1
                out = pd.DataFrame()
                contact = jump.contact_phase
                jump_side = "bilateral" if jump.side == "bilateral" else "unilateral"

                # remove unnecessary EMG data
                if self.include_emg:
                    to_remove = []
                    for k in contact.emgsignals.keys():
                        if any([i.lower() in k.lower() for i in muscle_map]):
                            to_remove.append(k)
                else:
                    to_remove = contact.emgsignals.keys()
                contact.drop(to_remove, inplace=True)
                jump.drop(to_remove, inplace=True)

                # get muscle emg amplitude
                for emg in contact.emgsignals.values():
                    if emg.side == jump.side or jump.side == "bilateral":
                        name = emg.muscle_name.replace("_", " ").lower()  # type: ignore
                        name += f" amplitude ({emg.unit})"  # type: ignore
                        val = emg.to_numpy().mean()  # type: ignore
                        out.loc[name, f"{emg.side}"] = float(val)  # type: ignore

                # add activation time and ratios
                refs_keys = test.emg_activation_references.keys()
                if isinstance(jump, DropJump):

                    # get the ground contact time
                    t_gc = contact.index[0]

                    # get the time corresponding to the end of the
                    # loading response
                    grf = jump.resultant_force.force[jump.vertical_axis].copy()
                    time = grf.index.copy()
                    grf = grf.to_numpy().flatten()
                    wgt = test.participant.weight
                    if wgt is None:
                        raise ValueError("participant's weight must be provided.")
                    batches = continuous_batches((grf >= wgt * G) & (time > t_gc))
                    fsamp = float(1 / np.mean(np.diff(time)))
                    min_samples = fsamp * 0.025
                    batches = [i for i in batches if len(i) >= min_samples]
                    if len(batches) == 0:
                        raise RuntimeError("No loading response was detected.")
                    batch = batches[0]
                    t_bw = time[batch][0]

                    # extract the EMG-related metrics
                    for emg in jump.emgsignals.values():
                        if not isinstance(emg, EMGSignal):
                            continue
                        if emg.side == jump.side or jump.side == "bilateral":
                            lbl = "_".join([emg.side, emg.muscle_name])
                            muscle_name = emg.muscle_name.replace("_", " ")
                            muscle_name = muscle_name.lower()

                            # get muscle activation time
                            if lbl in refs_keys:
                                key = (emg.muscle_name, emg.side)
                                threshold = test.emg_activation_thresholds.get(key)
                                if threshold is not None:
                                    val = self._get_muscle_activation_time_ms(
                                        emg,
                                        threshold,
                                        t_gc,
                                    )
                                    name = f"{muscle_name} activation time (ms)"
                                    out.loc[name, emg.side] = val

                            # get muscle activation ratio
                            val = self._get_muscle_activation_ratio(
                                emg,
                                t_gc,
                                t_bw,
                            )
                            name = f"{muscle_name} activation ratio"
                            out.loc[name, emg.side] = val * 100

                # get force
                sides = ["left", "right"] if jump_side == "bilateral" else [jump.side]
                for side in sides:
                    frz = contact.get(f"{side}_foot_ground_reaction_force")
                    if frz is None:
                        continue
                    val = frz["force"][frz.vertical_axis].to_numpy().mean()
                    name = "vertical force (N)"
                    out.loc[name, side] = float(val)

                # add jump parameters
                wgt = test.participant.weight
                if wgt is None:
                    raise ValueError("participant's weight cannot be None.")
                ctime = self._get_jump_contact_time_ms(jump)
                ftime = self._get_jump_flight_time_ms(jump)
                tov = self._get_takeoff_velocity_ms(jump, wgt)
                elevation = self._get_elevation_cm(jump, wgt)
                for side in sides:
                    out.loc["takeoff velocity (m/s)", side] = tov
                    out.loc["elevation (cm)", side] = elevation
                    out.loc["flight time (ms)", side] = ftime
                    out.loc["contact time (ms)", side] = ctime
                    out.loc["flight-to-contact ratio", side] = float(
                        round(ftime / ctime, 2)
                    )
                    out.loc["rsi (cm/s)", side] = float(
                        round(elevation / (ctime / 1000), 2)
                    )

                # add general data
                out.insert(0, "n", sides_counter[jump.side])
                out.insert(0, "side", jump_side)
                type_name = [jump_name]
                if isinstance(jump, DropJump):
                    type_name[0] += f" ({jump.box_height_cm:0.0f}cm)"
                if jump.free_hands:
                    type_name.append("free hands")
                if jump.straight_legs:
                    type_name.append("straight legs")
                out.insert(0, "type", "-".join(type_name))
                out.insert(0, "parameter", out.index)
                out.reset_index(inplace=True, drop=True)

                # aggregate
                sides_df[jump.side] = pd.concat(
                    objs=[sides_df[jump.side], out],
                    ignore_index=True,
                )

            # get the best jump of each side
            for side, df in sides_df.items():
                if side == "bilateral":
                    n = df.loc[df.parameter == "elevation (cm)", ["n", "left"]]
                    n = n.copy().sort_values("left")["n"].to_numpy()[-1]
                else:
                    n = df.loc[df.parameter == "elevation (cm)", ["n", side]]
                    n = n.copy().sort_values(side)["n"].to_numpy()[-1]
                line = df.iloc[[0]].copy()
                line.loc[line.index, "parameter"] = "best jump"
                line.drop(["n"], inplace=True, axis=1)
                if side == "bilateral":
                    line.loc[line.index, "left"] = n
                    line.loc[line.index, "right"] = n
                else:
                    line.loc[line.index, side] = n
                df = pd.concat([df, line], ignore_index=True)
                sides_df[side] = df

            # merge sides
            if "left" in sides_df and "right" in sides_df:
                df_unilateral = sides_df["left"].merge(
                    sides_df["right"],
                    on=["type", "parameter", "n", "side"],
                )
            else:
                df_unilateral = pd.DataFrame()
            if "bilateral" in sides_df:
                df_bilateral = sides_df["bilateral"]
            else:
                df_bilateral = pd.DataFrame()
            best = pd.concat([df_bilateral, df_unilateral], ignore_index=True)

            # calculate symmetries
            if not best.empty:
                num = best.right.to_numpy() - best.left.to_numpy()
                den = (best.right.to_numpy() + best.left.to_numpy()) / 2
                best.loc[best.index, "symmetry (%)"] = num / den * 100
                best.loc[best.parameter == "best jump", "symmetry (%)"] = None

            return best

        rebound_params = ["contact time (ms)", "flight-to-contact ratio", "rsi (cm/s)"]
        sjs = _get_jumps_summary_table(
            test.squat_jumps,  # type: ignore
            "Squat Jump",
        )
        if not sjs.empty:
            sjs = sjs.loc[sjs.parameter.map(lambda x: x not in rebound_params)]
        cmjs = _get_jumps_summary_table(
            test.counter_movement_jumps,  # type: ignore
            "Counter Movement Jump",
        )
        if not cmjs.empty:
            cmjs = cmjs.loc[cmjs.parameter.map(lambda x: x not in rebound_params)]
        sljs = _get_jumps_summary_table(
            test.repeated_jumps,  # type: ignore
            "Repeated Jump",
        )
        djs = _get_jumps_summary_table(
            test.drop_jumps,  # type: ignore
            "Drop Jump",
        )
        return pd.concat([djs, cmjs, sjs, sljs], ignore_index=True)

    def _get_analytics(self, test: JumpTest):
        syms = []
        muscle_map = test.relevant_muscle_map
        if test.relevant_muscle_map is None:
            muscle_map = []
        else:
            muscle_map = test.relevant_muscle_map.copy()

        def get_jump(jump: SingleJump | DropJump, n: int, name: str):
            obj = jump.copy()

            # remove unnecessary EMG data
            if self.include_emg:
                to_remove = []
                for k in jump.emgsignals.keys():
                    if any([i.lower() in k.lower() for i in muscle_map]):
                        to_remove.append(k)
            else:
                to_remove = jump.emgsignals.keys()
            jump.drop(to_remove, inplace=True)

            # get analytics
            obj = obj.to_dataframe()
            obj.insert(0, "jump", n)
            obj.insert(0, "side", jump.side)
            type_name = [name]
            if isinstance(jump, DropJump):
                type_name[0] += f" ({jump.box_height_cm:0.0f}cm)"
            if jump.free_hands:
                type_name.append("free hands")
            if jump.straight_legs:
                type_name.append("straight legs")
            obj.insert(0, "type", "-".join(type_name))

            return obj

        for i, jump in enumerate(test.squat_jumps):
            syms.append(get_jump(jump, i + 1, "Squat Jump"))

        for i, jump in enumerate(test.counter_movement_jumps):
            syms.append(get_jump(jump, i + 1, "Counter Movement Jump"))

        for i, jump in enumerate(test.drop_jumps):
            syms.append(get_jump(jump, i + 1, "Drop Jump"))

        for i, jump in enumerate(test.repeated_jumps):
            syms.append(get_jump(jump, i + 1, "Repeated Jump"))

        return pd.concat(syms, ignore_index=True)

    def _get_grf_figure(self, test: JumpTest):

        def get_data(jump: SingleJump | DropJump, n: int, typed: str):
            grf = jump.copy().resultant_force.copy()
            grf = grf.force.to_dataframe()[[jump.vertical_axis]]  # type: ignore
            grf.columns = pd.Index(["grf"])
            start = jump.flight_phase.index[0]
            grf.insert(0, "time", grf.index - start)
            grf.insert(0, "jump", n)
            grf.insert(0, "side", jump.side)
            type_name = [typed]
            if isinstance(jump, DropJump):
                type_name[0] += f" ({jump.box_height_cm:0.0f}cm)"
            if jump.free_hands:
                type_name.append("free hands")
            if jump.straight_legs:
                type_name.append("straight legs")
            grf.insert(0, "type", "-".join(type_name))
            grf = grf.loc[(grf.time > -1) & (grf.time < 2)]
            return grf

        data = []
        for i, jump in enumerate(test.squat_jumps):
            data.append(get_data(jump, i + 1, "Squat Jump"))
        for i, jump in enumerate(test.counter_movement_jumps):
            data.append(get_data(jump, i + 1, "Counter Movement Jump"))
        for i, jump in enumerate(test.drop_jumps):
            data.append(get_data(jump, i + 1, "Drop Jump"))
        for i, jump in enumerate(test.repeated_jumps):
            data.append(get_data(jump, i + 1, "Single Leg Jump"))
        df = pd.concat(data, ignore_index=True)

        fig = px.line(
            data_frame=df,
            x="time",
            y="grf",
            color="jump",
            facet_row="type",
            facet_col="side",
            facet_col_spacing=0.05,
            facet_row_spacing=0.05,
            template="plotly_white",
        )
        fig.update_traces(opacity=0.5)
        fig.update_yaxes(
            matches=None,
            showticklabels=True,
            title="Ground Reaction Force (N)",
        )
        fig.update_xaxes(
            matches=None,
            showticklabels=True,
            title="Time (s)",
        )
        fig.add_hline(
            y=test.participant.weight * G,
            line_color="red",
            line_dash="dash",
            line_width=1,
            name="Weight (N)",
            showlegend=True,
        )
        fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
        return fig

    def _get_data_and_norms(
        self,
        metric: str,
        test: JumpTest,
        bilateral_is_unique: bool = True,
        ranks: dict[str, str] = RANK_5COLORS,
        symmetric_ranks: bool = False,
        reversed_ranks: bool = False,
    ):

        # retrieve the data of the required metric from summary
        metric_df = self.summary.copy()
        params: list[str] = metric_df.parameter.to_list()
        idx = [i for i, v in enumerate(params) if v.endswith(metric)]
        metric_df = metric_df.iloc[idx]
        metric_df.drop(["symmetry (%)"], axis=1, inplace=True)
        metric_df = metric_df.melt(
            id_vars=["type", "side", "parameter", "n"],
            var_name="limb",
            value_name="value",
        )
        if bilateral_is_unique:
            idx = (metric_df.side != "bilateral") | (metric_df.limb == "left")
            metric_df = metric_df.loc[idx]
            new_limbs = metric_df[["side", "limb"]].apply(
                lambda x: x.side if x.side == "bilateral" else x.limb,
                axis=1,
            )
            metric_df.loc[metric_df.index, "limb"] = new_limbs
        metric_df.reset_index(drop=True, inplace=True)

        # get the data sorted according to the subplots to be rendered
        data: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
        for (t, s), dfr in metric_df.groupby(["type", "side"]):
            key = (str(t), str(s))
            val: dict[str, dict[str, list[float]]] = {}
            for param, dfp in dfr.groupby("parameter"):
                dct: dict[str, list[float]] = {}
                for side, dfs in dfp.groupby("limb"):
                    k = str(side)
                    v = dfs.sort_values("n").value.to_numpy().flatten().tolist()
                    dct[k] = v
                val[str(param)] = dct
            data[key] = val

        # get the normative data sorted according to the subplots to be rendered
        norms: dict[
            tuple[str, str], tuple[list[float], list[float], list[str], list[str]]
        ] = {}
        combs = metric_df[["type", "side"]].drop_duplicates().values.tolist()
        if not test.normative_data.empty:
            gender = test.participant.gender
            if gender is None:
                raise ValueError("Normative Data require gender being specified.")
            gender = gender.lower()[0]
            norm = test.normative_data.copy()
            params: list[str] = norm.parameter.to_list()
            idx = [i for i, v in enumerate(params) if v.endswith(metric)]
            norm = norm.iloc[idx]
            types = norm["type"].str.lower().tolist()
            types = [t.lower().rsplit(" (", 1)[0] for t in types]
            sides = norm["side"].str.lower().tolist()
            genders = [i.lower()[0] for i in norm["gender"]]
            for t, s in combs:
                types_idx = [t.lower().rsplit(" (", 1)[0] in v for v in types]
                types_idx = np.array(types_idx)
                sides_idx = np.array([s in v for v in sides])
                gender_idx = np.array([gender == v for v in genders])
                mask = types_idx & sides_idx & gender_idx
                tnorm = norm.loc[mask]
                if tnorm.shape[0] > 1:
                    msg = "Multiple normative values found for jump elevation."
                    raise ValueError(msg)
                if not tnorm.empty:
                    avg = float(tnorm["mean"].to_numpy()[0])
                    std = float(tnorm["std"].to_numpy()[0])
                    rank_clrs = list(ranks.values())
                    rank_lbls = list(ranks.keys())
                    if reversed_ranks:
                        rank_clrs = rank_clrs[::-1]
                        rank_lbls = rank_lbls[::-1]
                    n_vals = len(ranks)
                    if symmetric_ranks:
                        rank_clrs = rank_clrs[::-1] + rank_clrs
                        rank_lbls = rank_lbls[::-1] + rank_lbls
                        rank_vals = np.arange(n_vals + 1)
                    else:
                        if n_vals % 2 == 1:
                            rank_vals = np.arange((n_vals + 1) // 2) + 1
                        else:
                            rank_vals = np.arange((n_vals + 2) // 2)
                    rank_vals = np.concatenate([rank_vals, -rank_vals]) * std + avg
                    rank_vals = np.unique(rank_vals)[::-1]
                    rank_lows = rank_vals[1:].copy().tolist()
                    rank_tops = rank_vals[:-1].copy().tolist()
                    norms[(t, s)] = (rank_lows, rank_tops, rank_lbls, rank_clrs)

        return data, norms

    def _get_performance_figure(
        self,
        performance_data: dict[str, list[float]],
        performance_norms: tuple[list[float], list[float], list[str], list[str]],
        performance_unit: str,
        performance_metric: str,
        balance_data: list[float] | None = None,
        balance_norms: (
            tuple[list[float], list[float], list[str], list[str]] | None
        ) = None,
    ):

        # generate the figure
        subplot_titles = [performance_metric.capitalize()]
        if balance_data is not None:
            subplot_titles.append("Left/Right Imbalance")
        fig = make_subplots(
            rows=1,
            cols=1 if balance_data is None else 2,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.01,
        )
        fig.update_layout(
            template="plotly_white",
            legend=dict(title_text="Legend"),
            width=1000,
            height=400,
            bargroupgap=0.25,
        )
        fig.update_xaxes(
            showgrid=False,
            showline=False,
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False,
        )

        # get the normative data if available
        if performance_norms is not None:
            rank_lows, rank_tops, rank_lbls, rank_clrs = performance_norms
            rank_lows = np.array(rank_lows)
            rank_tops = np.array(rank_tops)
        else:
            rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

        # plot the bars representing the performance value
        yvals = []
        colors_plotted = []
        for k, (side, performances) in enumerate(performance_data.items()):
            for j, y in enumerate(performances):
                value = round(y, 1)

                # if normative data are available get the main bar color as
                # the color of the rank achieved by the actual value.
                # Otherwise, use the color of the side with which the jump
                # has been performed.
                if len(rank_tops) > 0:
                    idx = np.where(rank_tops >= value)[0]
                    idx = idx[-1] if len(idx) > 0 else 0  # (len(rank_clrs) - 1)
                    color = rank_clrs[idx]
                else:
                    color = SIDE_COLORS[side]  # type: ignore

                # update the y-axis range values
                yvals += rank_lows.tolist() + rank_tops.tolist() + [value]

                # plot the bar
                fig.add_trace(
                    row=1,
                    col=1,
                    trace=go.Bar(
                        x=[k + 1],
                        y=[value],
                        text=[f"Jump {j+1}<br>{value} {performance_unit}"],
                        textposition="outside",
                        textangle=0,
                        showlegend=(j == 0)
                        and performance_norms is None
                        and len(performance_data) > 1,
                        marker_color=[color],
                        marker_line_color=["black"],
                        name=side.capitalize(),
                        legendgroup="Limb",
                        legendgrouptitle_text="Limb",
                        offsetgroup=str(j + 1),
                    ),
                )

        # update the yaxes
        yrange = [np.min(yvals) * 0.9, np.max(yvals) * 1.2]
        fig.update_yaxes(row=1, col=1, range=yrange)

        # update the xaxes
        fig.update_xaxes(
            col=1,
            row=1,
            range=[0, len(performance_data) + 1],
            showticklabels=False,
        )
        if len(performance_data) > 1:
            fig.update_xaxes(
                col=1,
                row=1,
                showticklabels=True,
                tickvals=np.arange(len(performance_data)) + 1,
                tickmode="array",
                ticktext=[str(i).capitalize() for i in list(performance_data.keys())],
            )

        # plot the norms as colored boxes behind the bars
        zipped = zip(rank_lows, rank_tops, rank_lbls, rank_clrs)
        for rlow, rtop, rlbl, rclr in zipped:
            if rlow == np.min(rank_lows) and rlow > yrange[0]:
                rlow = yrange[0]
            if rtop == np.max(rank_tops) and rtop < yrange[1]:
                rtop = yrange[1]
            fig.add_shape(
                type="rect",
                x0=0,
                x1=len(performance_data) + 1,
                y0=rlow,
                y1=rtop,
                line_width=0,
                fillcolor=hex_to_rgba(rclr, 0.25),
                layer="below",
                name=rlbl.capitalize(),
                legendgroup="Rank",
                legendgrouptitle_text="Rank",
                showlegend=rclr not in colors_plotted,
                col=1,
                row=1,
            )
            if rtop < np.max(rank_tops):
                fig.add_annotation(
                    x=len(performance_data) + 1,
                    y=rtop,
                    text=f"{rtop:0.1f} {performance_unit}",
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    font=dict(color=rclr),
                    valign="top",
                    yshift=0,
                    name=rlbl,
                    col=1,  # type: ignore
                    row=1,  # type: ignore
                )

            # ensure that the legend is plotted once
            colors_plotted.append(rclr)

        # plot balance
        if balance_data is not None:

            # get the normative data if available
            if balance_norms is not None:
                rank_lows, rank_tops, rank_lbls, rank_clrs = balance_norms
                rank_lows = np.asarray(rank_lows)
                rank_tops = np.asarray(rank_tops)
            else:
                rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

            # plot the balance of each single jump
            for j, val in enumerate(balance_data):

                # get the bar color as the color of the rank achieved by the
                # jump height. Otherwise, use the color of the side with which the
                # jump has been performed.
                idx = np.where(rank_tops >= abs(val))[0]
                idx = idx[0] if len(idx) > 0 else (len(rank_clrs) - 1)
                color = rank_clrs[idx]

                # get the value and label
                value = max(-50, min(50, val))
                lbl = f"{abs(val):0.1f}%" if -50 <= val <= 50 else ">50.0%"
                lbl = f"Jump {j+1} ({lbl})"

                # plot the bar
                fig.add_trace(
                    col=2,
                    row=1,
                    trace=go.Bar(
                        y=[len(balance_data) - 1 - j],
                        x=[value],
                        text=[lbl],
                        textposition="outside",
                        textangle=0,
                        showlegend=False,
                        marker_color=[color],
                        marker_line_color=["black"],
                        name=f"Jump {j+1}",
                        legendgroup="Jump",
                        legendgrouptitle_text="Jump",
                        orientation="h",
                    ),
                )

            # update rank extremes
            rank_tops[-1] = 120

            # plot the norms as colored boxes behind the bars
            zipped = zip(rank_lows, rank_tops, rank_lbls, rank_clrs)
            for rlow, rtop, rlbl, rclr in zipped:
                fig.add_shape(
                    type="rect",
                    y0=-1,
                    y1=len(balance_data),
                    x0=rlow,
                    x1=rtop,
                    line_width=0,
                    fillcolor=hex_to_rgba(rclr, 0.25),
                    layer="below",
                    name=rlbl.capitalize(),
                    legendgroup="Rank",
                    legendgrouptitle_text="Rank",
                    showlegend=color not in colors_plotted,
                    col=2,
                    row=1,
                )
                fig.add_shape(
                    type="rect",
                    y0=-1,
                    y1=len(balance_data),
                    x0=-rlow,
                    x1=-rtop,
                    line_width=0,
                    fillcolor=hex_to_rgba(rclr, 0.25),
                    layer="below",
                    name=rlbl.capitalize(),
                    legendgroup="Rank",
                    legendgrouptitle_text="Rank",
                    showlegend=False,
                    col=2,
                    row=1,
                )

                # ensure that the legend is plotted once
                colors_plotted.append(rclr)

            # plot the zero line
            fig.add_vline(
                col=2,  # type: ignore
                row=1,  # type: ignore
                x=0,
                line_width=2,
                line_dash="solid",
                showlegend=False,
            )

            # update the xaxes
            xrange = [-np.max(rank_tops), np.max(rank_tops)]
            fig.update_xaxes(
                col=2,
                row=1,
                range=xrange,
                tickmode="array",
                tickvals=[xrange[0] * 0.9, 0, xrange[1] * 0.9],
                ticktext=["Left", "Perfect<br>Balance", "Right"],
                ticklen=0,
            )

            # update the yaxes
            fig.update_yaxes(
                col=2,
                row=1,
                range=[-1, len(balance_data)],
            )

        # check
        return fig

    def _get_muscle_activation_figure(
        self,
        data: dict[str, dict[str, list[float]]],
        norms: tuple[list[float], list[float], list[str], list[str]] | None,
        unit: str,
    ):

        # prepare the figure
        muscles = np.unique(list(data.keys())).tolist()
        sides = np.unique(
            [s.capitalize() for m in data.values() for s in m.keys()]
        ).tolist()
        fig = make_subplots(
            cols=len(sides),
            rows=1,
            horizontal_spacing=0.1,
            subplot_titles=sides,
        )
        fig.update_layout(
            template="plotly_white",
            height=200 * len(muscles),
            width=1200,
            legend=dict(title=dict(text="Legend")),
            bargroupgap=0.1,
        )

        # get the normative data if available
        if norms is not None:
            rank_lows, rank_tops, rank_lbls, rank_clrs = norms
            rank_lows = np.array(rank_lows)
            rank_tops = np.array(rank_tops)
        else:
            rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

        # plot the data
        color_plotted = []
        xvals = {}
        for row, muscle in enumerate(muscles):
            side_dct = data[muscle]
            for col, (side, jump_values) in enumerate(side_dct.items()):

                # plot the jumps
                for n, x in enumerate(jump_values):

                    # update the xrange values
                    if side not in xvals:
                        xvals[side] = []
                    xvals[side].append(x)

                    # if normative data are available get the main bar color as
                    # the color of the rank achieved by the actual value.
                    # Otherwise, use the color of the side with which the jump
                    # has been performed.
                    value = round(x, 1)
                    if len(rank_tops) > 0:
                        idx = np.where(rank_tops >= value)[0]
                        idx = idx[-1] if len(idx) > 0 else 0  # (len(rank_clrs) - 1)
                        color = rank_clrs[idx]
                    else:
                        color = SIDE_COLORS[side]  # type: ignore

                    # get the label
                    lbl = f"{x:0.1f}{unit}"
                    lbl = f"Jump {n+1} ({lbl})"

                    # plot the bar
                    fig.add_trace(
                        row=1,
                        col=col + 1,
                        trace=go.Bar(
                            y=[row],
                            x=[x],
                            text=[lbl],
                            textposition="outside",
                            textangle=0,
                            showlegend=color not in color_plotted and norms is None,
                            marker_color=[color],
                            marker_line_color=["black"],
                            orientation="h",
                            offsetgroup=str(n),
                            name=side,
                            legendgroup="Side",
                            legendgrouptitle_text="Side",
                        ),
                    )

                    # prevent the same color to be plotted again
                    if norms is None:
                        color_plotted.append(color)

        # plot the norms (if available)
        for col, (side, xv) in enumerate(xvals.items()):
            if norms is not None:
                r_lows = rank_lows.copy()
                r_tops = rank_tops.copy()
                r_lows[-1] = min(r_lows[-1], np.min(xv))
                r_lows[-1] *= 1.1 if r_lows[-1] < 0 else 0.9
                r_lows[-1] = min(0, r_lows[-1])
                r_tops[0] = max(r_tops[0], np.max(xv) * 2)
                zipped = zip(r_lows, r_tops, rank_lbls, rank_clrs)
                for rlow, rtop, rlbl, rclr in zipped:
                    fig.add_shape(
                        type="rect",
                        y0=-1,
                        y1=len(muscles),
                        x0=rlow,
                        x1=rtop,
                        line_width=0,
                        fillcolor=hex_to_rgba(rclr, 0.25),
                        layer="below",
                        name=rlbl,
                        legendgroup="Rank",
                        legendgrouptitle_text="Rank",
                        showlegend=rlbl not in color_plotted,
                        row=1,
                        col=col + 1,
                    )

                    # ensure that each rank level is plotted once
                    color_plotted.append(rlbl)

                # update the xrange
                xrange = [min(0, np.min(r_lows)), np.max(r_tops)]

            else:
                xrange = [
                    min(0, np.min(xv) * (1.1 if np.min(xv) < 0 else 0.9)),
                    np.max(xv) * 2,
                ]

            # update x-axis
            tickvals = r_lows[:-1]
            ticktext = [f"{i:.0f}{unit}" for i in tickvals]
            fig.update_xaxes(
                col=col + 1,
                range=xrange,
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=True,
                tickmode="array",
                ticktext=ticktext,
                tickvals=tickvals,
                tickangle=0,
            )

            # update the y-axis
            fig.update_yaxes(
                col=col + 1,
                tickvals=np.arange(len(muscles)).tolist(),
                tickangle=0,
                tickmode="array",
                ticktext=[m.replace(" ", "<br>") for m in muscles],
                showticklabels=True,
                range=[-1, len(muscles)],
            )

            # plot the zero lines
            fig.add_vline(
                x=0,
                line_width=2,
                line_dash="solid",
                showlegend=False,
            )

        return fig

    def _get_elevation_figure(self, test: JumpTest):

        # retrieve the jump height data
        performance_data, performance_norms = self._get_data_and_norms(
            "elevation (cm)",
            test,
        )

        # since we have just one parameter (elevation), we remove the layer
        # defining the parameters for each key
        performance_data = {i: list(v.values())[0] for i, v in performance_data.items()}

        # retrieve the force balance data
        balance_df = self.summary.copy()
        balance_df = balance_df.loc[balance_df.parameter == "vertical force (N)"]
        balance_data: dict[tuple[str, str], list[float]] = {}
        for t, s in performance_data.keys():
            if s == "bilateral":
                balance = balance_df.loc[balance_df["type"] == t].copy()
                balance = balance.loc[balance["side"] == s]
                balance.sort_values("n", inplace=True)
                balance = balance["symmetry (%)"].to_numpy().flatten()
                balance_data[(t, s)] = balance.tolist()

        # prepare the balance norms
        vals = np.array([0, 10, 20, 30, 40, 100])
        lows = vals[:-1].copy().tolist()
        tops = vals[1:].copy().tolist()
        clrs = list(RANK_5COLORS.values())
        lbls = list(RANK_5COLORS.keys())
        balance_norms = {(t, s): (lows, tops, lbls, clrs) for (t, s) in balance_data}

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in performance_data.keys():
            p_data = performance_data.get((t, s))
            p_norms = performance_norms.get((t, s))
            b_data = balance_data.get((t, s))
            b_norms = balance_norms.get((t, s))
            if p_data is not None:
                fig = self._get_performance_figure(
                    p_data,
                    p_norms,  # type: ignore
                    "cm",
                    "Elevation",
                    b_data,
                    b_norms,
                )
                title = f"{t}-{s}".capitalize()
                fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_contact_time_figure(self, test: JumpTest):

        # retrieve the contact time data
        performance_data, performance_norms = self._get_data_and_norms(
            "contact time (ms)",
            test,
            reversed_ranks=True,
        )

        # since we have just one parameter (contact time), we remove the layer
        # defining the parameters for each key
        performance_data = {i: list(v.values())[0] for i, v in performance_data.items()}

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in performance_data.keys():
            p_data = performance_data.get((t, s))
            p_norms = performance_norms.get((t, s))
            if p_data is not None:
                fig = self._get_performance_figure(
                    p_data,
                    p_norms,  # type: ignore
                    "ms",
                    "Contact Time",
                )
                title = f"{t}-{s}".capitalize()
                fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_rsi_figure(self, test: JumpTest):

        # retrieve the rsi data
        performance_data, performance_norms = self._get_data_and_norms(
            "rsi (cm/s)",
            test,
        )

        # since we have just one parameter (rsi), we remove the layer
        # defining the parameters for each key
        performance_data = {i: list(v.values())[0] for i, v in performance_data.items()}

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in performance_data.keys():
            p_data = performance_data.get((t, s))
            p_norms = performance_norms.get((t, s))
            if p_data is not None:
                fig = self._get_performance_figure(
                    p_data,
                    p_norms,  # type: ignore
                    "cm/s",
                    "Reactive Strength Index (RSI)",
                )
                title = f"{t}-{s}".capitalize()
                fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_muscle_activation_ratio_figure(self, test: JumpTest):

        # retrieve the activation ratio data
        data_raw, norms = self._get_data_and_norms(
            "activation ratio",
            test,
            False,
            RANK_3COLORS,
            True,
        )

        # we turn the name of the parameters layer into the muscle names
        data: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
        for i, v in data_raw.items():
            vals = {}
            for j, k in v.items():
                muscle = j.replace(" activation ratio", "").split(" ")
                muscle = " ".join([l.capitalize() for l in muscle])
                vals[muscle] = k
            data[i] = vals

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in data.keys():
            p_data = data.get((t, s))
            p_norms = norms.get((t, s))
            if p_data is not None:
                fig = self._get_muscle_activation_figure(
                    p_data,
                    p_norms,
                    unit="%",
                )

                # update the x-axis
                fig.update_xaxes(
                    title="Pre-Activation (%)",
                    row=len(fig._grid_ref),  # type: ignore
                )

                # update the title
                title = f"{t}-{s}".capitalize()
                fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_muscle_activation_time_figure(self, test: JumpTest):

        # retrieve the activation ratio data
        data_raw, norms = self._get_data_and_norms(
            "activation time (ms)",
            test,
            False,
            RANK_3COLORS,
            True,
        )

        # we turn the name of the parameters layer into the muscle names
        data: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
        for i, v in data_raw.items():
            vals = {}
            for j, k in v.items():
                muscle = j.replace(" activation time (ms)", "").split(" ")
                muscle = " ".join([l.capitalize() for l in muscle])
                vals[muscle] = k
            data[i] = vals

        # generate the figure
        figures: dict[str, go.Figure] = {}
        for t, s in data.keys():
            p_data = data.get((t, s))
            p_norms = norms.get((t, s))
            if p_data is not None:
                fig = self._get_muscle_activation_figure(
                    p_data,
                    p_norms,
                    unit="ms",
                )

                # update the x-axis
                fig.update_xaxes(
                    title="Activation time (ms)",
                    row=len(fig._grid_ref),  # type: ignore
                )
                fig.update_xaxes(
                    tickvals=[-200, 200],
                    tickangle=0,
                    tickmode="array",
                    ticktext=["Before<br>contact", "After<br>contact"],
                    showticklabels=True,
                )

                # update the title
                title = f"{t}-{s}".capitalize()
                fig.update_layout(title=title)
                figures[title] = fig

        return figures

    def _get_figures(self, test: JumpTest):
        out: dict[str, go.Figure] = {}
        out["ground_reaction_forces"] = self._get_grf_figure(test)
        out["elevation"] = self._get_elevation_figure(test)
        if len(test.drop_jumps) > 0 or len(test.repeated_jumps) > 0:
            out["contact_time"] = self._get_contact_time_figure(test)
            out["rsi"] = self._get_rsi_figure(test)
        if len(test.drop_jumps) > 0 and self.include_emg:
            macr = self._get_muscle_activation_ratio_figure(test)
            out["muscle_activation_ratio"] = macr
            mact = self._get_muscle_activation_time_figure(test)
            out["muscle_activation_time"] = mact

        return out
