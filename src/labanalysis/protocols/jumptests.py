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

from ..constants import (
    MINIMUM_CONTACT_FORCE_N,
    RANK_3COLORS,
    RANK_5COLORS,
    SIDE_COLORS,
    G,
)
from ..records.jumping import DropJump, SingleJump, JumpExercise
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

    def _get_grf_figure(self, test: JumpTest):

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
            range=[-1, None],
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
            gender = gender.lower()
            norm = test.normative_data.copy()
            params: list[str] = norm.parameter.to_list()
            idx = [i for i, v in enumerate(params) if v.endswith(metric)]
            norm = norm.iloc[idx]
            types = norm["type"].str.lower().tolist()
            types = [t.lower().rsplit(" (", 1)[0] for t in types]
            sides = norm["side"].str.lower().tolist()
            genders = norm["gender"].str.lower().tolist()
            for t, s in combs:
                types_idx = [t.lower().rsplit(" (", 1)[0] in v for v in types]
                types_idx = np.array(types_idx)
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
                    rank_clrs = list(ranks.values())
                    rank_lbls = list(ranks.keys())
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
        performance_data: dict[
            tuple[str, str],
            dict[str, list[float]],
        ],
        performance_norms: dict[
            tuple[str, str],
            tuple[list[float], list[float], list[str], list[str]],
        ],
        performance_unit: str,
        balance_data: dict[tuple[str, str], list[float]] = {},
        balance_norms: dict[
            tuple[str, str],
            tuple[list[float], list[float], list[str], list[str]],
        ] = {},
    ):

        # generate the figure
        subplot_titles = [
            f"{str(t).rsplit("(")[0].strip()}<br>{str(s).capitalize()}"
            for t, s in performance_data
        ]
        if len(balance_data) == 0:
            fig = make_subplots(
                rows=1,
                cols=len(performance_data),
                subplot_titles=subplot_titles,
            )
        else:
            specs = [
                [{"rowspan": 2} for _ in range(len(performance_data))],
                [None for _ in range(len(performance_data))],
                [{} for _ in range(len(performance_data))],
            ]
            fig = make_subplots(
                rows=3,
                cols=len(performance_data),
                subplot_titles=subplot_titles,
                specs=specs,
            )
        fig.update_layout(
            template="plotly_white",
            legend=dict(title_text="Legend"),
            height=400 if len(balance_data) == 0 else 600,
            width=max(1800, 400 * len(performance_data)),
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

        # plot the performance data
        norm_plotted = False
        for n, ((t, s), dct) in enumerate(performance_data.items()):

            # get the normative data if available
            snorm = performance_norms.get((t, s))
            if snorm is not None:
                rank_lows, rank_tops, rank_lbls, rank_clrs = snorm
                rank_lows = np.array(rank_lows)
                rank_tops = np.array(rank_tops)
            else:
                rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

            # plot the bars representing the performance value
            yvals = []
            for k, (side, performances) in enumerate(dct.items()):
                for j, y in enumerate(performances):
                    value = round(y, 1)

                    # if normative data are available get the main bar color as
                    # the color of the rank achieved by the actual value.
                    # Otherwise, use the color of the side with which the jump
                    # has been performed.
                    if len(rank_tops) > 0:
                        idx = np.where(rank_tops >= value)[0]
                        idx = idx[-1] if len(idx) > 0 else (len(rank_clrs) - 1)
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
                            text=[f"Jump {j+1}<br>{value} {performance_unit}"],
                            textposition="outside",
                            textangle=0,
                            showlegend=False,
                            marker_color=[color],
                            marker_line_color=["black"],
                            name=f"{t} {side}",
                            offsetgroup=str(j + 1),
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
                        text=f"{rtop:0.1f} {performance_unit}",
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

        # plot balance
        for n, ((t, s), symm) in enumerate(balance_data.items()):
            if np.all(np.isnan(symm)):
                continue

            # get the normative data if available
            snorm = balance_norms.get((t, s))
            if snorm is not None:
                rank_lows, rank_tops, rank_lbls, rank_clrs = snorm
                rank_lows = np.asarray(rank_lows)
                rank_tops = np.asarray(rank_tops)
            else:
                rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

            # plot the balance of each single jump
            for j, val in enumerate(symm):

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
                    row=3,
                    col=n + 1,
                    trace=go.Bar(
                        y=[len(symm) - 1 - j],
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

            # update rank extremes
            if np.max(rank_tops) < np.max(symm) * 2:
                rank_tops[-1] = np.max(symm) * 2

            # plot the norms as colored boxes behind the bars
            zipped = zip(rank_lows, rank_tops, rank_lbls, rank_clrs)
            vals = []
            for rlow, rtop, rlbl, rclr in zipped:
                vals += [rlow, rtop]
                fig.add_shape(
                    type="rect",
                    y0=-1,
                    y1=len(symm),
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
                    y0=-1,
                    y1=len(symm),
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
            xrange = [-np.max(vals), np.max(vals)]
            fig.update_xaxes(
                row=3,
                col=n + 1,
                range=xrange,
                tickmode="array",
                tickvals=[xrange[0] * 0.9, 0, xrange[1] * 0.9],
                ticktext=["Left", "Perfect<br>Balance", "Right"],
                ticklen=0,
            )

            # update the yaxes
            fig.update_yaxes(
                row=3,
                col=n + 1,
                range=[-1, len(symm)],
            )

        # check
        return fig

    def _get_muscle_activation_figure(
        self,
        data: dict[
            tuple[str, str],
            dict[str, dict[str, list[float]]],
        ],
        norms: dict[
            tuple[str, str],
            tuple[list[float], list[float], list[str], list[str]],
        ],
        unit: str,
    ):

        # prepare the figure
        subplot_titles = [
            f"{str(t).rsplit("(")[0].strip()}<br>{str(s).capitalize()}" for t, s in data
        ]
        muscles = [list(m.keys()) for m in list(data.values())]
        muscles = np.unique(np.concatenate(muscles)).tolist()
        fig = make_subplots(
            rows=len(muscles),
            cols=len(subplot_titles),
            subplot_titles=subplot_titles,
            row_titles=muscles,
        )
        fig.update_layout(
            template="plotly_white",
            height=250 * len(muscles),
            width=500 * len(subplot_titles),
            legend=dict(title=dict(text="Legend")),
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            range=[-1, 2],
        )

        # plot the data
        legend_plotted = []
        for col, ((t, s), muscle_dct) in enumerate(data.items()):
            xvals = []
            for row, muscle in enumerate(muscles):
                side_dct = muscle_dct[muscle]
                for y, (side, jump_values) in enumerate(side_dct.items()):
                    for n, x in enumerate(jump_values):

                        # update the xrange values
                        xvals.append(x)

                        # get the bar color as the color of the side with which
                        # the jump has been performed.
                        color = SIDE_COLORS[side]

                        # get the label
                        lbl = f"{x:0.1f}{unit}"
                        lbl = f"Jump {n+1} ({lbl})"

                        # plot the bar
                        fig.add_trace(
                            row=row + 1,
                            col=col + 1,
                            trace=go.Bar(
                                y=[y],
                                x=[x],
                                text=[lbl],
                                textposition="outside",
                                textangle=0,
                                showlegend=side not in legend_plotted,
                                marker_color=[color],
                                marker_line_color=["black"],
                                orientation="h",
                                offsetgroup=str(n),
                                name=side,
                                legendgroup="Side",
                                legendgrouptitle_text="Side",
                            ),
                        )

                        # prevent the same side to be plotted again
                        legend_plotted.append(side)

            # get the normative data if available
            snorm = norms.get((t, s))
            if snorm is not None:
                rank_lows, rank_tops, rank_lbls, rank_clrs = snorm
                rank_lows = np.array(rank_lows)
                rank_tops = np.array(rank_tops)
            else:
                rank_lows = rank_tops = rank_lbls = rank_clrs = np.array([])

            # update xrange
            xmax = max(0, np.max(xvals))
            xmin = min(0, np.min(xvals))
            xdelta = (xmax - xmin) * 0.5
            if snorm is None:
                xrange = [np.min(xvals) - xdelta, np.max(xvals) + xdelta]
            else:
                rmax = np.max(rank_tops)
                rmin = np.min(rank_lows)
                pmax = max(xmax + xdelta, rmax)
                pmin = min(xmin - xdelta, rmin)
                xrange = [pmin, pmax]

                # update norms range if required
                if pmax > rmax:
                    rank_tops[rank_tops == rmax] = pmax
                if pmin < rmin:
                    rank_lows[rank_lows == rmin] = pmin

            # if any x value is lower than zero, keep the limit to zero in the
            # x-axis range
            if np.min(xvals) > 0:
                xrange[0] = max(0, xrange[0])

            # update x-axis
            fig.update_xaxes(
                col=col + 1,
                range=xrange,
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
            )

            # plot normative data
            for row, (muscle, side_dct) in enumerate(muscle_dct.items()):
                zipped = zip(rank_lows, rank_tops, rank_lbls, rank_clrs)
                for rlow, rtop, rlbl, rclr in zipped:
                    fig.add_shape(
                        type="rect",
                        y0=-1,
                        y1=len(muscle_dct),
                        x0=rlow,
                        x1=rtop,
                        line_width=0,
                        fillcolor=hex_to_rgba(rclr, 0.25),
                        layer="below",
                        name=rlbl,
                        legendgroup="Performance<br>Levels",
                        legendgrouptitle_text="Performance<br>Levels",
                        showlegend=rlbl not in legend_plotted,
                        row=row + 1,
                        col=col + 1,
                    )

                    # ensure that each rank level is plotted once
                    legend_plotted.append(rlbl)

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
        vals = np.array([0, 5, 10, 15, 20, 25])
        lows = vals[:-1].copy().tolist()
        tops = vals[1:].copy().tolist()
        clrs = list(RANK_5COLORS.values())
        lbls = list(RANK_5COLORS.keys())
        balance_norms = {(t, s): (lows, tops, lbls, clrs) for (t, s) in balance_data}

        # generate the figure
        fig = self._get_performance_figure(
            performance_data,
            performance_norms,
            "cm",
            balance_data,
            balance_norms,
        )
        fig.update_yaxes(row=1, col=1, title="Elevation<br>(cm)")
        fig.update_yaxes(row=3, col=1, title="Force Imbalance<br>(%)")

        return fig

    def _get_contact_time_figure(self, test: JumpTest):

        # retrieve the contact time data
        performance_data, performance_norms = self._get_data_and_norms(
            "contact time (ms)",
            test,
        )

        # since we have just one parameter (contact time), we remove the layer
        # defining the parameters for each key
        performance_data = {i: list(v.values())[0] for i, v in performance_data.items()}

        # generate the figure
        fig = self._get_performance_figure(
            performance_data,
            performance_norms,
            "ms",
        )
        fig.update_yaxes(row=1, col=1, title="Contact Time<br>(ms)")

        return fig

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
        fig = self._get_performance_figure(
            performance_data,
            performance_norms,
            "cm/s",
        )
        fig.update_yaxes(row=1, col=1, title="Reactive Strength Index<br>(cm/s)")

        return fig

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
        fig = self._get_muscle_activation_figure(
            data,
            norms,
            unit="%",
        )

        # update the x-axis
        fig.update_xaxes(
            title="Pre-Activation (%)",
            row=len(fig._grid_ref),  # type: ignore
        )

        return fig

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
        fig = self._get_muscle_activation_figure(
            data,
            norms,
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

        return fig

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
