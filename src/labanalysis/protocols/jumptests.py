"""singlejumps test module"""

#! IMPORTS


__all__ = ["JumpTest"]


#! CLASSES

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..records.jumping import JumpExercise

from ..constants import (
    MINIMUM_CONTACT_FORCE_N,
    RANK_3COLORS,
    RANK_5COLORS,
    G,
    SIDE_COLORS,
)
from ..records import DropJump, SingleJump
from ..records.pipelines import get_default_processing_pipeline
from ..records.records import ForcePlatform, TimeseriesRecord
from ..records.timeseries import EMGSignal, Point3D
from ..signalprocessing import (
    butterworth_filt,
    continuous_batches,
    fillna,
    mean_filt,
    rms_filt,
)
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
        drop_jump_height_cm: int = 40,
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
        counter_movement_jump_files: list[str] = [],
        drop_jump_files: list[str] = [],
        repeated_jumps_files: list[str] = [],
    ):
        if not isinstance(participant, Participant):
            raise ValueError("participant must be a Participant instance.")
        bodymass = participant.weight
        if bodymass is None:
            raise ValueError("participant's bodymass must be provided.")
        if not isinstance(squat_jump_files, list):
            msg = "squat_jump_files must be a list of valid tdf file paths "
            msg += "corresponding to squat jump tests."
            raise ValueError(msg)
        sjs = [
            SingleJump.from_tdf(
                file,
                bodymass,
                left_foot_ground_reaction_force,
                right_foot_ground_reaction_force,
            )
            for file in squat_jump_files
        ]
        cmjs = [
            SingleJump.from_tdf(
                file,
                bodymass,
                left_foot_ground_reaction_force,
                right_foot_ground_reaction_force,
            )
            for file in counter_movement_jump_files
        ]
        djs = [
            DropJump.from_tdf(
                file,
                drop_jump_height_cm,
                bodymass,
                left_foot_ground_reaction_force,
                right_foot_ground_reaction_force,
            )
            for file in drop_jump_files
        ]
        rjs = []
        for file in repeated_jumps_files:
            rjs += JumpExercise.from_tdf(
                file,
                bodymass,
                left_foot_ground_reaction_force,
                right_foot_ground_reaction_force,
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
        exe = self.processing_pipeline(record, inplace=False)
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
        exe = self._process_record(jump)

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
        time = con.index

        # get the output velocity
        net_grf = grfy - bodyweight * G
        return float(np.trapezoid(net_grf, time) / bodyweight)

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
                            if lbl in refs_keys:
                                muscle_name = emg.muscle_name.replace("_", " ")
                                muscle_name = muscle_name.lower()

                                # get muscle activation time
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
                    if isinstance(jump, DropJump):
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
                type_name = jump_name
                if isinstance(jump, DropJump):
                    type_name += f" ({jump.box_height_cm:0.0f}cm)"
                out.insert(0, "type", type_name)
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

        sjs = _get_jumps_summary_table(
            test.squat_jumps,  # type: ignore
            "Squat Jump",
        )
        djs = _get_jumps_summary_table(
            test.drop_jumps,  # type: ignore
            "Drop Jump",
        )
        cmjs = _get_jumps_summary_table(
            test.counter_movement_jumps,  # type: ignore
            "Counter Movement Jump",
        )
        sljs = _get_jumps_summary_table(
            test.repeated_jumps,  # type: ignore
            "Single Leg Jump",
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
            if isinstance(jump, DropJump):
                obj.insert(0, "box_height_cm", jump.box_height_cm)
            obj.insert(0, "type", name)

            return obj

        for i, jump in enumerate(test.squat_jumps):
            syms.append(get_jump(jump, i + 1, "Squat Jump"))
        for i, jump in enumerate(test.counter_movement_jumps):
            syms.append(get_jump(jump, i + 1, "Counter Movement Jump"))
        for i, jump in enumerate(test.drop_jumps):
            syms.append(get_jump(jump, i + 1, "Drop Jump"))
        for i, jump in enumerate(test.repeated_jumps):
            syms.append(get_jump(jump, i + 1, "Single Leg Jump"))

        return pd.concat(syms, ignore_index=True)

    def _get_grf_profiles_figure(self, test: JumpTest):

        def get_data(jump: SingleJump | DropJump, n: int, typed: str):
            grf = jump.copy().resultant_force.copy()
            grf = grf.force.to_dataframe()[[jump.vertical_axis]]  # type: ignore
            grf.columns = pd.Index(["grf"])
            start = jump.flight_phase.index[0]
            grf.insert(0, "time", grf.index - start)
            grf.insert(0, "jump", n)
            grf.insert(0, "type", typed)
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

    def _get_single_jumps_figure(self, test: JumpTest):

        # retrieve the jump height data
        elevation_df = self.summary.copy()
        elevation_df = elevation_df.loc[elevation_df.parameter == "elevation (cm)"]
        elevation_df.drop(
            ["parameter", "symmetry (%)"],
            axis=1,
            inplace=True,
        )
        elevation_df = elevation_df.melt(
            id_vars=["type", "side", "n"],
            var_name="s",
            value_name="value",
        )
        elevation_df = elevation_df.loc[
            (elevation_df.side != "bilateral") | (elevation_df.s == "left")
        ]
        elevation_df.reset_index(drop=True, inplace=True)
        elevation_df.loc[elevation_df.index, "side"] = elevation_df[
            ["side", "s"]
        ].apply(
            lambda x: x.side if x.side == "bilateral" else x.s,
            axis=1,
        )
        elevation_df = elevation_df.drop("s", axis=1)
        elevation_df.insert(
            0,
            "set",
            elevation_df.side.map(
                lambda x: "bilateral" if x == "bilateral" else "unilateral"
            ),
        )

        # get the normative data sorted according to the subplots to be rendered
        elevation_norms = {}
        combs = elevation_df[["type", "set"]].drop_duplicates().values.tolist()
        if not test.normative_data.empty:
            gender = test.participant.gender
            if gender is None:
                raise ValueError("Normative Data require gender being specified.")
            gender = gender.lower()
            norm = test.normative_data.copy()
            norm = norm.loc[norm.parameter == "elevation (cm)"]
            types = norm["type"].str.lower().tolist()
            sides = norm["side"].str.lower().tolist()
            genders = norm["gender"].str.lower().tolist()
            for t, s in combs:
                types_idx = np.array(
                    [
                        t.lower().rsplit(" (", 1)[0] in v.rsplit(" (", 1)[0]
                        for v in types
                    ]
                )
                sides_idx = np.array([s in v for v in sides])
                gender_idx = np.array([gender in v for v in genders])
                mask = types_idx & sides_idx & gender_idx
                tnorm = norm.loc[mask]
                if tnorm.shape[0] > 1:
                    msg = "Multiple normative values found for jump elevation."
                    raise ValueError(msg)
                if not tnorm.empty:
                    avg = float(tnorm["mean"].to_numpy()[0])
                    std = float(tnorm["std"].to_numpy()[0])
                    rank_vals = np.array([+3, +2, +1, -1, -2, -3]) * std
                    rank_vals += avg
                    rank_lows = rank_vals[1:].copy()
                    rank_tops = rank_vals[:-1].copy()
                    rank_clrs = list(RANK_5COLORS.values())
                    rank_lbls = list(RANK_5COLORS.keys())
                    elevation_norms[(t, s)] = (
                        rank_lows,
                        rank_tops,
                        rank_lbls,
                        rank_clrs,
                    )

        # get elevation data sorted according to the subplots to be rendered
        elevation_data = {
            (t, s): {side: float(dfs.value.max()) for side, dfs in dfr.groupby("side")}
            for (t, s), dfr in elevation_df.groupby(["type", "set"])
        }

        # retrieve the force balance data
        balance_df = self.summary.copy()
        balance_df = balance_df.loc[balance_df.parameter == "vertical force (N)"]
        balance_data = {}
        for (t, s), dfr in elevation_df.groupby(["type", "set"]):
            if s != "bilateral":
                balance_data[(t, s)] = np.nan
            else:
                best = int(dfr.n.to_numpy()[np.argsort(dfr.value.to_numpy())[-1]])
                balance = balance_df.loc[balance_df["type"] == t].copy()
                balance = balance.loc[balance["side"] == s]
                balance = balance.loc[balance["n"] == best, "symmetry (%)"]
                balance_data[(t, s)] = float(balance.to_numpy()[0])

        # prepare the balance norms
        vals = np.array([0, 5, 10, 15, 20, 25])
        lows = vals[:-1].copy()
        tops = vals[1:].copy()
        clrs = list(RANK_5COLORS.values())
        lbls = list(RANK_5COLORS.keys())
        balance_norms = {(t, s): (lows, tops, lbls, clrs) for (t, s) in balance_data}

        # generate the figure
        subplot_titles = [
            f"{str(t).rsplit("(")[0].strip()}<br>{str(s).capitalize()}"
            for t, s in elevation_data
        ]
        subplot_titles += ["" for t, s in elevation_data]
        specs = [
            [{"rowspan": 2} for _ in combs],
            [None for _ in combs],
            [{} for _ in combs],
        ]
        fig = make_subplots(
            rows=3,
            cols=len(combs),
            subplot_titles=subplot_titles,
            specs=specs,
        )
        fig.update_layout(
            template="plotly_white",
            legend=dict(title_text="Legend"),
            height=600,
            width=max(1800, 400 * len(combs)),
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
        fig.update_yaxes(col=1, row=1, title="Elevation<br>(cm)")
        fig.update_yaxes(col=1, row=3, title="Force imbalance<br>(%)")

        # plot the elevation data
        norm_plotted = False
        for n, ((t, s), dct) in enumerate(elevation_data.items()):

            # get the normative data if available
            snorm = elevation_norms.get((t, s))
            if snorm is not None:
                rank_lows, rank_tops, rank_lbls, rank_clrs = snorm
            else:
                rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

            # plot the bars representing the jump height
            yvals = []
            for k, (side, y) in enumerate(dct.items()):
                value = round(y, 1)

                # if normative data are available get the main bar color as
                # the color of the rank achieved by the jump height.
                # Otherwise, use the color of the side with which the jump
                # has been performed.
                if len(rank_tops) > 0:
                    idx = np.where(rank_tops >= value)[0]
                    idx = min(len(rank_clrs) - 1, idx[-1] if len(idx) > 0 else 0)
                    color = rank_clrs[idx]
                else:
                    color = SIDE_COLORS[side]  # type: ignore

                # update the y-axis range values
                yvals += rank_lows.tolist() + rank_tops.tolist() + [value]

                # plot the bar
                fig.add_trace(
                    row=1,
                    col=n + 1,
                    trace=go.Bar(
                        x=[k + 1],
                        y=[value],
                        text=[f"{value}cm"],
                        textposition="outside",
                        textangle=0,
                        showlegend=False,
                        marker_color=[color],
                        marker_line_color=["black"],
                        name=f"{t} {side}",
                    ),
                )

            # update the yaxes
            yrange = [np.min(yvals) * 0.9, np.max(yvals) * 1.1]
            fig.update_yaxes(row=1, col=n + 1, range=yrange)

            # update the xaxes
            fig.update_xaxes(
                row=1,
                col=n + 1,
                range=[0, len(dct) + 1],
                tickvals=np.arange(len(dct)) + 1,
                tickmode="array",
                ticktext=[str(i).capitalize() for i in list(dct.keys())],
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
                    x1=len(dct) + 1,
                    y0=rlow,
                    y1=rtop,
                    line_width=0,
                    fillcolor=hex_to_rgba(rclr, 0.25),
                    layer="below",
                    name=rlbl,
                    legendgroup=rlbl,
                    showlegend=not norm_plotted,
                    row=1,
                    col=n + 1,
                )
                if rtop < np.max(rank_tops):
                    fig.add_annotation(
                        x=len(dct) + 1,
                        y=rtop,
                        text=f"{rtop:0.1f}cm",
                        showarrow=False,
                        xanchor="right",
                        yanchor="top",
                        font=dict(color=rclr),
                        valign="top",
                        yshift=0,
                        name=rlbl,
                        row=1,  # type: ignore
                        col=n + 1,  # type: ignore
                    )

            # ensure that the legend is plotted once
            norm_plotted = True

        # plot force balance
        for n, ((t, s), symm) in enumerate(balance_data.items()):
            if np.all(np.isnan(symm)):
                continue

            # get the normative data if available
            snorm = balance_norms.get((t, s))
            if snorm is not None:
                rank_lows, rank_tops, rank_lbls, rank_clrs = snorm
            else:
                rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

            # get the bar color as the color of the rank achieved by the
            # jump height. Otherwise, use the color of the side with which the
            # jump has been performed.
            idx = np.where(rank_tops <= symm)[0]
            idx = min(len(rank_clrs) - 1, idx[-1] if len(idx) > 0 else 0)
            color = rank_clrs[idx]

            # get the value and label
            val = max(-50, min(50, symm))
            lbl = f"{abs(symm):0.1f}%" if -50 <= symm <= 50 else ">50.0%"

            # plot the bar
            fig.add_trace(
                row=3,
                col=n + 1,
                trace=go.Bar(
                    y=[1],
                    x=[val],
                    text=[lbl],
                    textposition="outside",
                    textangle=0,
                    showlegend=False,
                    marker_color=[color],
                    marker_line_color=["black"],
                    name=f"{t} {s}",
                    orientation="h",
                ),
            )

            # plot the norms as colored boxes behind the bars
            zipped = zip(rank_lows, rank_tops, rank_lbls, rank_clrs)
            vals = []
            for rlow, rtop, rlbl, rclr in zipped:
                vals += [rlow, rtop]
                fig.add_shape(
                    type="rect",
                    y0=0,
                    y1=2,
                    x0=rlow,
                    x1=rtop,
                    line_width=0,
                    fillcolor=hex_to_rgba(rclr, 0.25),
                    layer="below",
                    name=rlbl,
                    legendgroup=rlbl,
                    showlegend=not norm_plotted,
                    row=3,
                    col=n + 1,
                )
                fig.add_shape(
                    type="rect",
                    y0=0,
                    y1=2,
                    x0=-rlow,
                    x1=-rtop,
                    line_width=0,
                    fillcolor=hex_to_rgba(rclr, 0.25),
                    layer="below",
                    name=rlbl,
                    legendgroup=rlbl,
                    showlegend=not norm_plotted,
                    row=3,
                    col=n + 1,
                )
                fig.add_annotation(
                    x=rtop,
                    y=0,
                    text=f"{rlbl}<br>{rtop:+0.1f}%",
                    showarrow=False,
                    font=dict(color=rclr),
                    valign="top",
                    yshift=-15,
                    name=rlbl,
                    row=3,  # type: ignore
                    col=n + 1,  # type: ignore
                )
                fig.add_annotation(
                    x=-rtop,
                    y=0,
                    text=f"{rlbl}<br>{-rtop:+0.1f}%",
                    showarrow=False,
                    font=dict(color=rclr),
                    valign="top",
                    yshift=-15,
                    name=rlbl,
                    row=3,  # type: ignore
                    col=n + 1,  # type: ignore
                )

            # ensure that the legend is plotted once
            norm_plotted = True

            # plot the zero line
            fig.add_vline(
                row=3,  # type: ignore
                col=n + 1,  # type: ignore
                x=0,
                line_width=2,
                line_dash="solid",
                showlegend=False,
            )

            # update the xaxes
            xrange = [np.min(vals), np.max(vals)]
            if val * 1.5 < xrange[0] or val * 1.5 > xrange[1]:
                xrange = [-abs(val) * 1.5, abs(val) * 1.5]
            fig.update_xaxes(
                row=3,
                col=n + 1,
                range=xrange,
                showticklabels=False,
            )

            # update the yaxes
            fig.update_yaxes(row=3, col=n + 1, range=[0, 2])

        # check
        return fig

    def _get_drop_jumps_muscle_activation_ratio_figure(self, test: JumpTest):

        # get the drop jump summary
        summ = self.summary.copy()
        summ = summ.loc[summ["type"].map(lambda x: "drop" in x.lower())]
        summ = summ.drop(["type", "symmetry (%)"], axis=1)

        # get the best drop jump for each side looking at the one with the
        # highest RSI
        rsi_df = summ.copy().loc[summ.parameter == "rsi (cm/s)"]
        rsi_df = rsi_df.drop("parameter", axis=1).melt(
            id_vars=["side", "n"],
            value_name="value",
            var_name="limb",
        )
        best_idx = rsi_df.groupby(["side", "limb"])["value"].idxmax()
        best_jumps = rsi_df.loc[best_idx]

        # get the activation ratio
        activation_df = summ.loc[
            summ.parameter.map(lambda x: x.endswith("activation ratio"))
        ]
        muscles = activation_df.parameter.map(
            lambda x: x.split(" activation")[0].capitalize()
        )
        activation_df.loc[activation_df.index, ["muscle"]] = muscles.to_numpy()
        activation_df = activation_df.drop("parameter", axis=1)
        activation_df = activation_df.melt(
            id_vars=["muscle", "side", "n"],
            value_name="value",
            var_name="limb",
        )

        """
        # keep the best jumps
        for (side, limb), dfr in activation_df.groupby(["side", "limb"]):
            mask = (best_jumps.side == side) & (best_jumps.limb == limb)
            best_jump = int(best_jumps.loc[mask, "n"].to_numpy()[0])  # type: ignore
            idx = dfr.loc[dfr.n != best_jump].index
            activation_df.loc[idx, "value"] = None
        activation_df.dropna(inplace=True)
        """
        activation_df = activation_df.groupby(["side", "limb"], as_index=False)
        activation_df = activation_df.mean(numeric_only=True)

        # prepare the dataframe to be used for generating the figure
        activation_df.side = activation_df.side.str.capitalize()
        activation_df.limb = activation_df.limb.str.capitalize()
        activation_df.insert(
            activation_df.shape[1],
            "label",
            activation_df.value.map(lambda x: f"{x:0.2f}"),
        )
        activation_df.insert(
            activation_df.shape[1],
            "color",
            activation_df.limb.map(lambda x: SIDE_COLORS[x.lower()]),
        )
        activation_df.value = activation_df.value - 50

        # get the normative data
        norms = test.normative_data.copy()
        norms = norms.loc[norms.parameter == "activation ratio"]
        norms = norms.loc[
            norms.side == "bilateral"
        ]  # norms are not sensitive to the side
        gender = test.participant.gender
        if gender is None:
            raise ValueError("participant's gender must be provided.")
        gender = gender.lower().capitalize()
        norms = norms.loc[norms.gender == gender]
        avg, std = norms[["mean", "std"]].to_numpy().flatten()

        # get ranks
        ranks = RANK_3COLORS
        rank_lbls = list(ranks.keys())
        rank_lbls = rank_lbls[::-1][:-1] + rank_lbls
        rank_clrs = list(ranks.values())
        rank_clrs = rank_clrs[::-1][:-1] + rank_clrs
        rank_vals = np.array([3, 2, 1, -1, -2, -3]) * std

        # prepare the figure
        titles = []
        for side, bst in best_jumps.groupby("side"):
            vals = {}
            for limb, lst in bst.groupby("limb"):
                line = rsi_df.loc[rsi_df.side == side]
                line = line.loc[line.limb == limb]
                line = line.loc[line.n == lst.n.to_numpy()[0]]
                value = float(line.value.to_numpy()[0])
                vals[limb] = value
            if side == "bilateral":
                titles.append(f"{side}<br>(RSI={vals['left']:0.1f}cm/s)")
            else:
                title = [
                    f"RSI<sub>{i.upper()}</sub>={v:0.1f}cm/s" for i, v in vals.items()
                ]
                title = f"{side}<br>(" + ", ".join(title) + ")"
                titles.append(title)
        fig = make_subplots(
            rows=1,
            cols=len(titles),
            subplot_titles=titles,
        )
        fig.update_layout(
            template="plotly_white",
            height=500,
            width=500 * len(titles),
            legend=dict(title=dict(text="Legend")),
        )
        fig.update_yaxes(showgrid=False, zeroline=False, showline=False)

        # plot the data
        for n, (side, dfr) in enumerate(activation_df.groupby("side")):

            # adjust the extremeties if required
            vals = dfr["value"].to_numpy().flatten().tolist()
            rank_ints = [[i, v] for i, v in zip(rank_vals[:-1], rank_vals[1:])]
            rank_ints = np.asarray(np.sort(np.array(rank_ints), axis=1))

            extra = abs(np.array(vals)).max() * 1.4
            if extra > rank_ints.max():
                rank_ints[0, 1] = extra
                rank_ints[-1, 0] = -extra

            # update the xaxis
            fig.update_xaxes(
                showgrid=False,
                zeroline=False,
                showline=False,
                range=[rank_ints.min(), rank_ints.max()],
                tickmode="array",
                tickvals=[rank_ints.max(), 0, rank_ints.min()],
                ticktext=["Too Stiff", "Balanced", "Too Soft"],
                ticklen=0,
                title="Pre-Activation (%)",
                row=1,
                col=n + 1,
            )

            # render the activation bars
            muscles = []
            for m, (muscle, dfm) in enumerate(dfr.groupby("muscle")):
                for limb, dfs in dfm.groupby("limb"):
                    y = float(dfs["value"].to_numpy()[0])
                    fig.add_trace(
                        row=1,
                        col=n + 1,
                        trace=go.Bar(
                            y=[m + 1],
                            x=[y],
                            marker_color=SIDE_COLORS[str(limb).lower()],
                            marker_line_color="black",
                            text=[f"{y:0.2f}"],
                            textposition="outside",
                            textangle=0,
                            name=limb,
                            legendgroup=limb,
                            offsetgroup=limb,
                            showlegend=bool(n == 0) and bool(m == 0),
                            orientation="h",
                        ),
                    )
                muscles.append(str(muscle).replace(" ", "<br>"))

            # handle the yaxis
            fig.update_yaxes(
                row=1,
                col=n + 1,
                range=[0, len(muscles) + 1],
                tickvals=np.arange(len(muscles)) + 1,
                tickmode="array",
                ticktext=muscles,
                tickangle=0,
            )

            # render the ranks
            zipped = zip(rank_lbls, rank_clrs, rank_ints)
            for k, (lbl, clr, (low, top)) in enumerate(zipped):
                fig.add_shape(
                    type="rect",
                    y0=0,
                    y1=len(muscles) + 1,
                    x0=low,
                    x1=top,
                    line_width=0,
                    fillcolor=hex_to_rgba(clr, 0.25),
                    layer="below",
                    name=lbl,
                    legendgroup=lbl,
                    showlegend=bool(n == 0) & bool(k <= 2),
                    row=1,
                    col=n + 1,
                )

            # add the zero line
            fig.add_vline(
                row=1,  # type: ignore
                col=n + 1,  # type:ignore
                x=0,
                line_width=2,
                line_color="black",
                line_dash="solid",
                showlegend=False,
            )

        return fig

    def _get_drop_jumps_muscle_activation_time_figure(self, test: JumpTest):

        # get the drop jump summary
        summ = self.summary.copy()
        summ = summ.loc[summ["type"].map(lambda x: "drop" in x.lower())]
        summ = summ.drop(["type", "symmetry (%)"], axis=1)

        # get the best drop jump for each side looking at the one with the
        # highest RSI
        contact_df = summ.copy().loc[summ.parameter == "rsi (cm/s)"]
        contact_df = contact_df.drop("parameter", axis=1).melt(
            id_vars=["side", "n"],
            value_name="value",
            var_name="limb",
        )
        best_idx = contact_df.groupby(["side", "limb"])["value"].idxmax()
        best_jumps = contact_df.loc[best_idx]

        # get the activation ratio
        activation_mask = summ.parameter.map(
            lambda x: x.endswith("activation time (ms)")
        )
        activation_df = summ.loc[activation_mask].copy()
        muscles = activation_df.parameter.map(
            lambda x: x.split(" activation")[0].capitalize()
        ).copy()
        activation_df.loc[activation_df.index, "muscle"] = muscles
        activation_df = activation_df.drop("parameter", axis=1)
        activation_df = activation_df.melt(
            id_vars=["muscle", "side", "n"],
            value_name="value",
            var_name="limb",
        )

        """
        # keep the best jumps
        for (side, limb), dfr in activation_df.groupby(["side", "limb"]):
            mask = (best_jumps.side == side) & (best_jumps.limb == limb)
            best_jump = int(best_jumps.loc[mask, "n"].to_numpy()[0])  # type: ignore
            idx = dfr.loc[dfr.n != best_jump].index
            activation_df.loc[idx, "value"] = None
        activation_df.dropna(inplace=True)
        """
        activation_df = activation_df.groupby(["side", "limb"], as_index=False)
        activation_df = activation_df.mean(numeric_only=True)

        # prepare the dataframe to be used for generating the figure
        activation_df.side = activation_df.side.str.capitalize()
        activation_df.limb = activation_df.limb.str.capitalize()
        activation_df.insert(
            activation_df.shape[1],
            "label",
            activation_df.value.map(lambda x: f"{x:0.2f}"),
        )
        activation_df.insert(
            activation_df.shape[1],
            "color",
            activation_df.limb.map(lambda x: SIDE_COLORS[x.lower()]),
        )

        # get the normative data
        # norms are not sensitive to the side nor to gender
        norms = test.normative_data.copy()
        norms = norms.loc[norms.parameter == "activation time (ms)"]
        norms = norms.loc[norms.side == "bilateral"]
        norms = norms.loc[norms.gender.str.lower() == "male"]
        avg, std = norms[["mean", "std"]].to_numpy().flatten()

        # get ranks
        ranks = RANK_3COLORS
        rank_lbls = list(ranks.keys())
        rank_clrs = list(ranks.values())
        rank_vals = np.array([-1, 0, 1, 2]) * std + avg
        rank_ints = [[i, v] for i, v in zip(rank_vals[:-1], rank_vals[1:])]
        rank_ints = np.asarray(np.sort(np.array(rank_ints), axis=1))

        # adjust the extremeties if required
        vals = activation_df["value"].to_numpy().flatten().tolist()
        if np.max(vals) * 1.4 > np.max(rank_vals):
            rank_ints[-1, 1] = 600
        if np.min(vals) * 1.4 < np.min(rank_vals):
            rank_ints[0, 0] = -600

        # prepare the figure
        sides = np.unique(activation_df.side.to_numpy()).tolist()
        fig = make_subplots(
            rows=1,
            cols=len(sides),
            subplot_titles=sides,
        )
        fig.update_layout(
            template="plotly_white",
            height=500,
            width=500 * len(sides),
            legend=dict(title=dict(text="Legend")),
        )
        fig.update_yaxes(showgrid=False, zeroline=False, showline=False)
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showline=False,
            range=[min(0, rank_ints.min()), max(0, rank_ints.max())],
            showticklabels=False,
            title="Activation Time (ms)",
        )

        # plot the data
        for n, (side, dfr) in enumerate(activation_df.groupby("side")):

            # render the activation bars
            muscles = []
            for m, (muscle, dfm) in enumerate(dfr.groupby("muscle")):
                for limb, dfs in dfm.groupby("limb"):
                    x = float(dfs["value"].to_numpy()[0])
                    if x > 300:
                        t = f">+300ms"
                        x = 300
                    elif x < -300:
                        t = f"<-300ms"
                        x = -300
                    fig.add_trace(
                        row=1,
                        col=n + 1,
                        trace=go.Bar(
                            y=[m + 1],
                            x=[x],
                            marker_color=SIDE_COLORS[str(limb).lower()],
                            marker_line_color="black",
                            text=[t],
                            textposition="outside",
                            textangle=0,
                            name=limb,
                            legendgroup=limb,
                            offsetgroup=limb,
                            showlegend=bool(n == 0) and bool(m == 0),
                            orientation="h",
                        ),
                    )
                muscles.append(str(muscle).replace(" ", "<br>"))

            # add the zero line
            for limb, dfs in dfr.groupby("limb"):
                fig.add_vline(
                    x=0,
                    line_width=2,
                    line_dash="solid",
                    showlegend=False,
                    row=1,  # type: ignore
                    col=n + 1,  # type: ignore
                )

            # handle the yaxis
            fig.update_yaxes(
                row=1,
                col=n + 1,
                range=[0, len(muscles) + 1],
                tickvals=np.arange(len(muscles)) + 1,
                tickmode="array",
                ticktext=muscles,
                tickangle=0,
            )

            # render the ranks
            zipped = zip(rank_lbls, rank_clrs, rank_ints)
            for k, (lbl, clr, (low, top)) in enumerate(zipped):
                fig.add_shape(
                    type="rect",
                    y0=0,
                    y1=len(muscles) + 1,
                    x0=low,
                    x1=top,
                    line_width=0,
                    fillcolor=hex_to_rgba(clr, 0.25),
                    layer="below",
                    name=lbl,
                    legendgroup=lbl,
                    showlegend=bool(n == 0) & bool(k <= 2),
                    row=1,
                    col=n + 1,
                )
                if k < len(rank_ints) - 1:
                    fig.add_annotation(
                        y=len(muscles) + 1,
                        x=top,
                        text=f"{top:+0.0f}ms",
                        showarrow=False,
                        xanchor="right",
                        yanchor="top",
                        font=dict(color=clr),
                        align="right",
                        valign="top",
                        yshift=0,
                        name=lbl,
                        row=1,  # type: ignore
                        col=n + 1,  # type: ignore
                    )

        return fig

    def _get_repeated_jumps_figure(self, test: JumpTest):
        # TODO create la figura per le repeated jumps
        return go.Figure()

    def _get_figures(self, test: JumpTest):
        out: dict[str, go.Figure] = {}
        out["ground_reaction_forces"] = self._get_grf_profiles_figure(test)
        out["elevation"] = self._get_single_jumps_figure(test)
        if len(test.drop_jumps) > 0 and self.include_emg:
            macr = self._get_drop_jumps_muscle_activation_ratio_figure(test)
            out["drop_jumps_muscle_activation_ratio"] = macr
            mact = self._get_drop_jumps_muscle_activation_time_figure(test)
            out["drop_jumps_muscle_activation_time"] = mact
        out["repeated_jumps"] = self._get_repeated_jumps_figure(test)
        return out
