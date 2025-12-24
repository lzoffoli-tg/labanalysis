"""singlejumps test module"""

#! IMPORTS


__all__ = ["JumpTest"]


#! CLASSES

from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..constants import RANK_COLORS, G
from ..records import DropJump, SingleJump
from ..records.pipelines import get_default_processing_pipeline
from ..records.records import ForcePlatform, TimeseriesRecord
from ..records.timeseries import EMGSignal, Point3D
from ..signalprocessing import continuous_batches, fillna
from .normativedata import jumps_normative_values
from .protocols import Participant, TestProtocol, TestResults


class JumpTest(TestProtocol):

    @property
    def single_leg_jumps(self):
        return self._single_leg_jumps

    def add_single_leg_jumps(self, *jumps: SingleJump):
        for jump in jumps:
            if not isinstance(jump, SingleJump):
                raise ValueError("jump must be a SingleJump instance.")
            self._single_leg_jumps.append(jump)

    def pop_single_leg_jumps(self, index: int):
        if not isinstance(index, int):
            raise ValueError("index must be an int.")
        if index < 0 or index > len(self._single_leg_jumps) - 1:
            raise ValueError("index out of range.")
        jump = self._single_leg_jumps.pop(index)
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
            + self.single_leg_jumps
        )

    def __init__(
        self,
        participant: Participant,
        normative_data: pd.DataFrame = jumps_normative_values,
        emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
        squat_jumps: list[SingleJump] = [],
        counter_movement_jumps: list[SingleJump] = [],
        drop_jumps: list[DropJump] = [],
        single_leg_jumps: list[SingleJump] = [],
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
        self._single_leg_jumps: list[SingleJump] = []
        self.add_squat_jumps(*squat_jumps)
        self.add_counter_movement_jumps(*counter_movement_jumps)
        self.add_drop_jumps(*drop_jumps)
        self.add_single_leg_jumps(*single_leg_jumps)

    @classmethod
    def from_files(
        cls,
        participant: Participant,
        normative_data: pd.DataFrame = jumps_normative_values,
        left_foot_ground_reaction_force: str = "left_foot",
        right_foot_ground_reaction_force: str = "right_foot",
        drop_jump_height_cm: int = 40,
        emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
        squat_jump_files: list[str] = [],
        counter_movement_jump_files: list[str] = [],
        drop_jump_files: list[str] = [],
        single_leg_jump_files: list[str] = [],
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
        sljs = [
            SingleJump.from_tdf(
                file,
                bodymass,
                left_foot_ground_reaction_force,
                right_foot_ground_reaction_force,
            )
            for file in single_leg_jump_files
        ]
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
            single_leg_jumps=sljs,
        )

    def results(self, include_emg: bool = True):
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
            single_leg_jumps=self.single_leg_jumps,
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
        for i, jump in enumerate(out.single_leg_jumps):
            out._single_leg_jumps[i] = self._process_jump(jump)  # type: ignore
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
        return get_default_processing_pipeline()


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

    def _get_takeoff_velocity_ms(self, jump: SingleJump | DropJump, bodyweight: float):

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

    def _get_elevation_cm(self, jump: SingleJump | DropJump, bodyweight: float):

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

                # add activations
                refs_keys = test.emg_activation_references.keys()
                if isinstance(jump, DropJump):
                    t1 = contact.index[0]
                    t2 = contact.index[-1]
                    for emg in jump.emgsignals.values():
                        if emg.side == jump.side or jump.side == "bilateral":
                            lbl = "_".join([emg.side, emg.muscle_name])  # type: ignore
                            if lbl in refs_keys:
                                val = emg.to_numpy().flatten()
                                thr = test.emg_activation_references[lbl]
                                thr = thr.mean() + thr.std() * test.emg_activation_threshold  # type: ignore
                                time = emg.index
                                contact_time = t2 - t1
                                min_samples = int(
                                    contact_time * 0.2 / np.mean(np.diff(time))
                                )
                                batches = continuous_batches((val >= thr) & (time < t2))
                                batch_samples = np.array([len(i) for i in batches])
                                name = emg.muscle_name.replace("_", " ").lower()  # type: ignore
                                name = f"{name} activation (ms)"  # type: ignore
                                idx = np.where(batch_samples >= min_samples)[0]
                                if len(idx) == 0:
                                    val = contact_time * 1000  # type: ignore
                                else:
                                    batch = batches[idx[0]]
                                    val = (time[batch][0] - t1) * 1000
                                out.loc[name, emg.side] = val  # type: ignore

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
                ratio = float(round(ftime / ctime, 1))
                tov = self._get_takeoff_velocity_ms(jump, wgt)
                elevation = self._get_elevation_cm(jump, wgt)
                for side in sides:
                    out.loc["contact time (ms)", side] = ctime
                    out.loc["flight time (ms)", side] = ftime
                    out.loc["flight-to-contact ratio", side] = ratio
                    out.loc["takeoff velocity (m/s)", side] = tov
                    out.loc["elevation (cm)", side] = elevation

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
                n = df.loc[df.parameter == "elevation (cm)", ["n", side]]
                n = n.copy().sort_values(side)["n"].to_numpy()[-1]
                line = df.iloc[0]
                line.parameter = "best jump"
                line.drop("n", inplace=True)
                line[side] = n
                line = pd.DataFrame(pd.Series(line)).T
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
            test.single_leg_jumps,  # type: ignore
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
        for i, jump in enumerate(test.single_leg_jumps):
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
        for i, jump in enumerate(test.single_leg_jumps):
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
        return go.Figure()
        """
        # get the summary metrics
        summary = self.summary

        # generate the figure
        fig = make_subplots(
            rows=1,
            cols=2,
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=["Elevation", "Force Balance"],
        )

        # plot elevation
        elevation_data = summary.loc[summary.parameter == "elevation (cm)"]
        keys = ["type", "side", "parameter"]
        sides = ["left", "right"]
        for key in sides:
            if key in elevation_data.columns:
                keys.append(key)
        elevation_data = elevation_data[keys].copy()
        side_plotted = []
        yvals = []
        for row, line in elevation_data.iterrows():
            # get normative intervals
            if not test.normative_data.empty:
                norm = test.normative_data.copy()
                norm_types = norm["type"].str.lower().tolist()
                types_idx = [line.type.lower() in v for v in norm_types]
                norm_sides = norm["side"].str.lower().tolist()
                sides_idx = [line.side.lower() in v for v in norm_sides]
                norm_parameters = norm["parameter"].str.lower().tolist()
                params_idx = [line.parameter.lower() in v for v in norm_parameters]
                types_idx = np.array(types_idx)
                sides_idx = np.array(sides_idx)
                params_idx = np.array(params_idx)
                mask = types_idx & sides_idx & params_idx
                norm = norm.loc[mask]
                if norm.shape[0] > 1:
                    msg = "Multiple normative values found for jump elevation."
                    raise ValueError(msg)
                if not norm.empty:
                    avg, std = float(norm["mean"].to_numpy()), float(
                        norm["std"].to_numpy()
                    )
                    rank_values = np.array([avg - 2 * std, avg, avg + 2 * std])
                else:
                    rank_values = np.array([])

            # plot the jumps
            jump_sides = (
                ["left", "right"] if line.side == "unilateral" else ["bilateral"]
            )
            for side in jump_sides:
                x = line.type
                y = float(np.squeeze(line[side]))
                if len(rank_values) == 0:
                    c = "gray"
                    n = "No normative data"
                else:
                    loc = np.where(y <= rank_values)[0]
                    n = (
                        "Excellent"
                        if len(loc) == 0
                        else list(RANK_COLORS.keys())[loc[0]]
                    )
                    c = RANK_COLORS[n]

            if line.side == "bilateral":
                fig.add_trace(
                    row=1,
                    col=1,
                    trace=go.Bar(
                        x=[line.type],
                        y=[float(np.squeeze(line.bilateral))],
                        text=f"{float(np.squeeze(line.bilateral)):0.1f}",
                        marker_color=RANK_COLORS["bilateral"],
                        name="bilateral",
                        offsetgroup="bilateral",
                        showlegend=row == 0,
                    ),
                )
        for i, (typ, dfr) in enumerate(elevation.groupby("type")):
            dfr = dfr.copy().dropna(how="all", axis=1)
            for side in sides:
                if side not in dfr.columns:
                    continue
                y = float(np.squeeze(dfr[side].to_numpy()))
                yvals.append(y)
                color = cmap[np.where(sides == side)[0][0]]
                fig.add_trace(
                    row=1,
                    col=1,
                    trace=go.Bar(
                        x=[typ],
                        y=[y],
                        text=f"{y:0.1f}",
                        marker_color=color,
                        name=side,
                        offsetgroup=side,
                        showlegend=side not in side_plotted,
                    ),
                )
                side_plotted.append(side)

        # set the yrange
        yrange = [np.min(yvals) * 0.9, np.max(yvals) * 1.2]
        fig.update_yaxes(range=yrange, row=1, col=1)

        # add ratios
        cmj = elevation.loc[elevation["type"] == "Counter Movement Jump"]
        if not cmj.empty and "bilateral" in cmj.columns:
            cmj = float(np.squeeze(cmj["bilateral"].to_numpy()))
        else:
            cmj = None
        sj = elevation.loc[elevation["type"] == "Squat Jump"]
        if not sj.empty and "bilateral" in sj.columns:
            sj = float(np.squeeze(sj["bilateral"].to_numpy()))
        else:
            sj = None
        dj = elevation.loc[elevation["type"] == "Drop Jump"]
        if not dj.empty and "bilateral" in dj.columns:
            dj = float(np.squeeze(dj["bilateral"].to_numpy()))
        else:
            dj = None
        msg = []
        if cmj is not None and sj is not None:
            msg.append(f"Counter Movement Jump / Squat Jump = {(cmj / sj):0.2f}")
        if dj is not None and sj is not None:
            msg.append(f"Drop Jump / Squat Jump = {(dj / sj):0.2f}")
        msg = "<br>".join(msg)
        fig.add_annotation(
            row=1,
            col=1,
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.99,
            text=msg,
            valign="top",
            align="center",
            showarrow=False,
        )

        # plot force balance
        balance = summary.loc[summary.parameter == "vertical force balance (%)"]
        balance = balance.dropna(how="all", axis=1)
        if (not balance.empty) and ("bilateral" not in balance.columns):
            raise ValueError("bilateral not in vertical force balance.")
        balance = balance[["type", "bilateral"]]
        yvals = []
        for i, (typ, dfr) in enumerate(balance.groupby("type")):
            lt = float(np.squeeze(dfr.bilateral.to_numpy())) + 50
            rt = 100 - lt
            yvals += [lt, rt]
            for side, val in {"left": lt, "right": rt}.items():
                fig.add_trace(
                    row=1,
                    col=2,
                    trace=go.Bar(
                        x=[typ],
                        y=[val],
                        text=[f"{val:0.1f}%"],
                        name=side,
                        offsetgroup=side,
                        marker_color=cmap[np.where(sides == side)[0][0]],
                        showlegend=bool(i == 0),
                    ),
                )

        # set the yrange
        yrange = [np.min(yvals) * 0.9, np.max(yvals) * 1.1]
        fig.update_yaxes(range=yrange, row=1, col=2)

        # check
        return fig
        """

    def _get_drop_jumps_figure(self, test: JumpTest):
        return go.Figure()
        """

        # extract the relevant data
        summ = self.summary.copy()
        summ = summ.loc[summ["type"] == "Drop Jump"]
        summ = summ.iloc[[i for i, v in enumerate(summ.parameter) if "activation" in v]]
        summ.parameter = summ.parameter.str.replace(" activation (%)", "")
        summ = summ.dropna(how="all", axis=1)
        if summ.empty:
            raise ValueError("No suitable Drop Jumps have been found.")

        sides = ["left", "right", "bilateral"]
        sides = [i for i in sides if i in summ.columns]
        check = 1
        """

    def _get_repeated_jumps_figure(self, test: JumpTest):
        # TODO create la figura per le repeated jumps
        return go.Figure()

    def _get_figures(self, test: JumpTest):
        out: dict[str, go.Figure] = {}
        out["jumps_ground_reaction_forces"] = self._get_grf_profiles_figure(test)
        out["elevation_and_balance"] = self._get_single_jumps_figure(test)
        if len(test.drop_jumps) > 0 and self.include_emg:
            out["drop_jump_activations"] = self._get_drop_jumps_figure(test)
        out["repeated_jumps_fatigue"] = self._get_repeated_jumps_figure(test)
        return out
