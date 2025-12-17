"""singlejumps test module"""

#! IMPORTS


__all__ = ["JumpTest"]


#! CLASSES

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as cmaps
from plotly.subplots import make_subplots

from ..constants import G
from ..records import DropJump, SingleJump
from ..records.pipelines import get_default_processing_pipeline
from ..signalprocessing import continuous_batches, fillna
from .normativedata.normative_data import jumps_normative_values
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

    def copy(self):
        return JumpTest(
            participant=self.participant.copy(),
            normative_data=self.normative_data,
            squat_jumps=self.squat_jumps,
            counter_movement_jumps=self.counter_movement_jumps,
            drop_jumps=self.drop_jumps,
            single_leg_jumps=self.single_leg_jumps,
        )

    @property
    def jumps(self):
        return (
            self.squat_jumps
            + self.counter_movement_jumps
            + self.drop_jumps
            + self.single_leg_jumps
        )

    @property
    def results(self):
        return JumpTestResults(self)

    def __init__(
        self,
        participant: Participant,
        normative_data: pd.DataFrame = jumps_normative_values,
        squat_jumps: list[SingleJump] = [],
        counter_movement_jumps: list[SingleJump] = [],
        drop_jumps: list[DropJump] = [],
        single_leg_jumps: list[SingleJump] = [],
    ):
        if not isinstance(participant, Participant):
            raise ValueError("participant must be a Participant class instance.")
        if participant.weight is None:
            raise ValueError("participant's weight must be assigned.")
        self.set_participant(participant)
        self.set_normative_data(normative_data)
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
        drop_jump_muscle_activation_threshold: float = 3.0,
        squat_jump_files: list[str] = [],
        counter_movement_jump_files: list[str] = [],
        drop_jump_files: list[str] = [],
        single_leg_jump_files: list[str] = [],
    ):
        if not isinstance(drop_jump_muscle_activation_threshold, (int, float)):
            raise ValueError("drop_jump_muscle_activation_threshold must be a float.")
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
                drop_jump_muscle_activation_threshold,
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
        return cls(participant, normative_data, sjs, cmjs, djs, sljs)


class JumpTestResults(TestResults):

    def __init__(self, test: JumpTest):
        if not isinstance(test, JumpTest):
            raise ValueError("'test' must be an JumpTest instance.")
        super().__init__(test)

    def _get_processed_data(self, test: JumpTest):
        new = test.copy()
        pipeline = get_default_processing_pipeline()
        for i in range(len(test.squat_jumps)):
            pipeline(test.squat_jumps[i], True)
        for i in range(len(test.counter_movement_jumps)):
            pipeline(test.counter_movement_jumps[i], True)
        for i in range(len(test.drop_jumps)):
            pipeline(test.drop_jumps[i], True)
        return new

    def _get_jump_contact_time_ms(self, jump: SingleJump | DropJump):
        time = jump.contact_phase.index
        return int(round((time[-1] - time[0]) * 1000))

    def _get_jump_flight_time_ms(self, jump: SingleJump | DropJump):
        time = jump.flight_phase.index
        return int(round((time[-1] - time[0]) * 1000))

    def _get_takeoff_velocity_ms(self, jump: SingleJump | DropJump):

        # get the ground reaction force during the concentric phase
        con = jump.contact_phase.resultant_force
        if con is None:
            return np.nan
        grf = con.copy().force[jump.vertical_axis].to_numpy().flatten()
        grfy = fillna(arr=grf, value=0).flatten()  # type: ignore
        time = con.index

        # get the output velocity
        net_grf = grfy - jump.bodymass_kg * G
        return float(np.trapezoid(net_grf, time) / jump.bodymass_kg)

    def _get_elevation_cm(self, jump: SingleJump | DropJump):

        # from flight time
        flight_time = jump.flight_phase.index
        flight_time = flight_time[-1] - flight_time[0]
        elevation_from_time = (flight_time**2) * G / 8 * 100

        # from force impulse
        elevation_from_velocity = (
            (self._get_takeoff_velocity_ms(jump) ** 2) / (2 * G) * 100
        )

        # return the lower of the two
        return float(min(elevation_from_time, elevation_from_velocity))

    def _get_flight_to_contact_ratio(self, jump: SingleJump | DropJump):
        ft = self._get_jump_flight_time_ms(jump)
        ct = self._get_jump_contact_time_ms(jump)
        return round(ft / ct, 1)

    def _get_jump_summary_table(self, jump: SingleJump | DropJump):
        out = pd.DataFrame()
        contact = jump.contact_phase

        # get muscle balance
        if jump.side == "bilateral":
            emgs = pd.DataFrame()
            for emg in contact.emgsignals.values():
                name = emg.muscle_name.replace("_", " ").lower() + " balance (%)"
                emgs.loc[name, emg.side] = emg.to_numpy().mean()  # type: ignore
            if not emgs.empty:
                emgs = emgs.T
                emgs = (emgs / emgs.sum(axis=0) * 100).T
                out.loc[emgs.index.to_numpy(), jump.side] = emgs.left.to_numpy() - 50

        # get force balance
        if jump.side == "bilateral":
            frzs = pd.DataFrame()
            for side in ["left", "right"]:
                frz = contact.get(f"{side}_foot_ground_reaction_force")
                if frz is None:
                    continue
                val = frz["force"][frz.vertical_axis].to_numpy().mean()
                frzs.loc["vertical force balance (%)", side] = val
            frzs = frzs.T
            frzs = (frzs / frzs.sum(axis=0) * 100).T
            out.loc[frzs.index[0], jump.side] = frzs.left.to_numpy()[0] - 50

        # add activations
        if isinstance(jump, DropJump):
            t0 = jump.landing_phase.index[0]
            contact_phase = jump.contact_phase
            t1 = contact_phase.index[0]
            t2 = contact_phase.index[-1]
            sliced = jump.loc(t0, t2, inplace=False)
            if sliced is None:
                raise ValueError("error occurred slicing a jump")
            for emg in sliced.emgsignals.values():
                if jump.side != "bilateral" and emg.side != jump.side:
                    continue
                val = emg.to_numpy().flatten()
                avg = val.mean()
                std = val.std()
                thr = avg + std * jump.muscle_activation_threshold
                time = emg.index
                min_samples = int(0.2 / np.mean(np.diff(time)))
                batches = continuous_batches((val >= thr) & (time < t2))
                batch_samples = np.array([len(i) for i in batches])
                idx = np.where(batch_samples >= min_samples)[0]
                if len(idx) == 0:
                    out.loc[name, str(emg.side)] = (t2 - t1) * 1000
                batch = batches[idx[0]]
                name = emg.muscle_name.replace("_", " ").lower()
                name += " activation (ms)"
                if jump.side == "bilateral":
                    name = str(emg.side) + " " + name
                out.loc[name, jump.side] = (time[batch][0] - t1) * 1000

        # add jump parameters
        out.loc["contact time (ms)", jump.side] = self._get_jump_contact_time_ms(jump)
        out.loc["flight time (ms)", jump.side] = self._get_jump_flight_time_ms(jump)
        out.loc["flight-to-contact ratio", jump.side] = (
            self._get_flight_to_contact_ratio(jump)
        )
        out.loc["takeoff velocity (m/s)", jump.side] = self._get_takeoff_velocity_ms(
            jump
        )
        out.loc["elevation (cm)", jump.side] = self._get_elevation_cm(jump)

        out.insert(0, "parameter", out.index)
        return out.reset_index(drop=True)

    def _get_summary(self, test: JumpTest):
        syms = []
        for i, jump in enumerate(test.squat_jumps):
            sym = self._get_jump_summary_table(jump)
            sym.insert(0, "n", i + 1)
            sym.insert(0, "type", "Squat Jump")
            syms.append(sym)
        for i, jump in enumerate(test.counter_movement_jumps):
            sym = self._get_jump_summary_table(jump)
            sym.insert(0, "n", i + 1)
            sym.insert(0, "type", "Counter Movement Jump")
            syms.append(sym)
        for i, jump in enumerate(test.drop_jumps):
            sym = self._get_jump_summary_table(jump)
            sym.insert(0, "n", i + 1)
            sym.insert(0, "type", "Drop Jump")
            syms.append(sym)
        for i, jump in enumerate(test.single_leg_jumps):
            sym = self._get_jump_summary_table(jump)
            sym.insert(0, "type", "Drop Jump")
            syms.append(sym)

        summary = pd.concat(syms, ignore_index=True)

        # get the best jump
        best = []
        for typ, dfr in summary.groupby("type"):
            dfr = dfr.copy().dropna(how="all", axis=1)
            for side in ["left", "right", "bilateral"]:
                if side not in dfr.columns:
                    continue
                sub = dfr[["n", "parameter", side]].copy().dropna()
                elevation = sub.loc[sub.parameter == "elevation (cm)"].copy()
                best_jump_n = elevation.sort_values(side, ascending=False)
                best_jump_n = best_jump_n.iloc[0]["n"]
                best.append(dfr.loc[dfr.n == best_jump_n])

        return pd.concat(best, ignore_index=True).drop("n", axis=1)

    def _get_analytics(self, test: JumpTest):
        syms = []
        for i, jump in enumerate(test.squat_jumps):
            sym = jump.to_dataframe()
            sym.insert(0, "jump", i + 1)
            sym.insert(0, "side", jump.side)
            sym.insert(0, "type", "Squat Jump")
            syms.append(sym)
        for jump in test.counter_movement_jumps:
            sym = jump.to_dataframe()
            sym.insert(0, "jump", i + 1)
            sym.insert(0, "side", jump.side)
            sym.insert(0, "type", "Counter Movement Jump")
            syms.append(sym)
        for jump in test.drop_jumps:
            sym = jump.to_dataframe()
            sym.insert(0, "jump", i + 1)
            sym.insert(0, "side", jump.side)
            sym.insert(0, "type", "Drop Jump")
            syms.append(sym)
        for jump in test.single_leg_jumps:
            sym = jump.to_dataframe()
            sym.insert(0, "jump", i + 1)
            sym.insert(0, "side", jump.side)
            sym.insert(0, "type", "Single Leg Jump")
            syms.append(sym)

        return pd.concat(syms, ignore_index=True)

    def _get_single_jumps_figure(self, test: JumpTest):

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
        cmap = cmaps.Plotly
        sides = np.array(["left", "right", "bilateral"])
        elevation = summary.loc[summary.parameter == "elevation (cm)"]
        side_plotted = []
        yvals = []
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

    def _get_drop_jumps_figure(self, test: JumpTest):
        # TODO create la figura per le attivitazioni durante il drop jump
        return go.Figure()

    def _get_repeated_jumps_figure(self, test: JumpTest):
        # TODO create la figura per le repeated jumps
        return go.Figure()

    def _get_figures(self, test: JumpTest):

        return {
            "elevation_and_balance": self._get_single_jumps_figure(test),
            "drop_jump_activations": self._get_repeated_jumps_figure(test),
            "repeated_jumps_fatigue": self._get_repeated_jumps_figure(test),
        }
