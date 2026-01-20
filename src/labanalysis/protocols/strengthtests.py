"""isokinetic test module"""

#! IMPORTS


from typing import Callable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..constants import G, RANK_4COLORS, SIDE_COLORS
from ..io.read.biostrength import PRODUCTS
from ..records import IsokineticExercise, IsometricExercise
from ..records.pipelines import get_default_processing_pipeline
from ..records.records import TimeseriesRecord
from ..records.timeseries import EMGSignal, Signal1D
from ..signalprocessing import butterworth_filt, cubicspline_interp, find_peaks
from .normativedata import isok_1rm_normative_values
from .protocols import Participant, TestProtocol, TestResults

#! CONSTANTS


__all__ = ["Isokinetic1RMTest", "IsometricTest"]


def _get_force_figure(
    tracks: pd.DataFrame,
    summary: pd.DataFrame,
    include_emg: bool = True,
):

    # generate the figure
    def get_muscles():
        lbls = [i for i in summary.parameter if i.endswith(" (%)")]
        return np.unique([i.rsplit(" ", 1)[0] for i in lbls]).tolist()

    def balance_string(left: str, right: str, sep: str = " | "):
        width = max(len(left), len(right))
        nbsp = "\u00a0"  # non-breaking
        ljust = left.rjust(width).replace(" ", nbsp)
        rjust = right.ljust(width).replace(" ", nbsp)
        return sep.join([ljust, rjust])

    plot_emg_muscle_balance = len(get_muscles()) > 0 and include_emg
    sides = np.unique(tracks.side)
    titles = [i.capitalize() for i in sides]
    ncols = len(sides)
    if plot_emg_muscle_balance:
        ncols += 1
        titles += ["Muscle Balance"]
    fig = make_subplots(
        rows=1,
        cols=ncols,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=titles,
        horizontal_spacing=0.1,
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
        width=500 * len(titles),
    )
    if plot_emg_muscle_balance:
        labels = list(RANK_4COLORS.keys())
        colors = list(RANK_4COLORS.values())
        values = [10, 20, 30, 40]
        cscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
        fig.update_layout(
            coloraxis=dict(
                colorscale=cscale,
                cmin=0,
                cmax=50,
                colorbar=dict(
                    title=dict(text="Muscle<br>Imbalance<br>Levels"),
                    len=0.95,
                    y=0.5,
                    tickmode="array",
                    tickvals=values,
                    ticktext=labels,
                ),
            ),
        )

    # plot force profiles
    f_data = tracks.loc[tracks.parameter == "force_amplitude"]
    f_data = f_data.groupby(["time_%", "side", "limb"], as_index=False).max()
    for i, side in enumerate(sides):
        y = f_data.loc[f_data.side == side, "value"].to_numpy()  # type: ignore
        y = y.astype(float).flatten()
        x = f_data.loc[f_data.side == side, "time_%"].to_numpy()  # type: ignore
        x = x.astype(float).flatten()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name="force profile",
                showlegend=False,
                line_color=SIDE_COLORS[side],
            ),
            row=1,
            col=i + 1,
        )
        x_peak = x[np.argmax(y)]
        fig.add_trace(
            go.Scatter(
                x=[x_peak, x_peak],
                y=[0, np.max(y)],
                name="peak",
                line_dash="dash",
                line_color="black",
                opacity=0.5,
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )
        note = [f"{"Peak:"}{np.max(y):0.1f}N"]
        extras = [
            "estimated 1RM (kg)",
            "rate of force development (kN/s)",
            "time to peak force (ms)",
        ]
        for ext in extras:
            est = summary.loc[summary.parameter == ext]
            if est.empty:
                continue
            est = est[side]
            est = float(np.squeeze(est.to_numpy()))
            key = (
                ext.replace("estimated", "Est.")
                .replace("rate of force development", "RFD")
                .replace("time to peak force", "Time@F<sub>MAX</sub>")
            )
            note += [f"{key}: {est:0.1f}"]
        note = "<br>".join(note)
        if x_peak / np.max(x) < 0.40:
            dx = 20
            textposition = "top right"
        elif x_peak / np.max(x) > 0.60:
            dx = -20
            textposition = "top left"
        else:
            dx = 0
            textposition = "top center"
        fig.add_trace(
            row=1,
            col=i + 1,
            trace=go.Scatter(
                x=[x_peak],
                y=[np.max(y)],
                dx=dx,
                text=note,
                mode="markers+text",
                textposition=textposition,
                marker=dict(size=12, color="black"),
                textfont=dict(size=12, color="black"),
                showlegend=False,
                name="force profile",
            ),
        )

    # update force profiles figure layout
    yrange = f_data["value"].to_numpy().flatten()  # type: ignore
    yrange = np.array([np.min(yrange), np.max(yrange)])
    yrange *= np.array([0.9, 1.3])
    yrange = yrange.tolist()
    xrange = f_data["time_%"].to_numpy().flatten()  # type: ignore
    xrange = [np.min(xrange), np.max(xrange)]
    xticks = np.linspace(xrange[0], xrange[1], 5)
    xticks = [int(round(i / 5) * 5) for i in xticks]
    for i in range(len(sides)):
        fig.update_xaxes(
            title="Concentric Phase (%)",
            row=1,
            col=i + 1,
            showline=True,
            linewidth=2,
            linecolor="black",
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            tickvals=xticks,
            tickmode="array",
            range=[min(xticks), max(xticks)],
        )
        fig.update_yaxes(
            title="Force (N)",
            range=yrange,
            row=1,
            col=i + 1,
            showline=True,
            linewidth=2,
            linecolor="black",
            showgrid=False,
            zeroline=False,
            showticklabels=True,
        )

    # plot muscle data
    if plot_emg_muscle_balance:
        if "symmetry (%)" not in summary.columns:
            raise ValueError("'symmetry (%)' missing from summary dataframe")
        if "parameter" not in summary.columns:
            raise ValueError("'parameter' missing from summary dataframe")
        parameters = summary.parameter.to_numpy().flatten()
        symmetries = summary["symmetry (%)"].to_numpy().flatten()
        vals = []
        for i, muscle in enumerate(get_muscles()):
            idx = [j for j, v in enumerate(parameters) if muscle in v]
            if len(idx) == 0:
                continue
            idx = idx[0]
            symm = float(symmetries[idx])
            val = max(-50, min(50, symm))
            vals.append(val)
            lbl = f"{abs(symm):+0.1f}%" if -50 <= symm <= 50 else ">50.0%"
            fig.add_trace(
                trace=go.Bar(
                    x=[val],
                    y=[muscle.replace(" ", "<br>")],
                    orientation="h",
                    text=[lbl],
                    textangle=0,
                    textposition="outside",
                    name="Muscle imbalance",
                    legendgroup="Muscle imbalance",
                    showlegend=False,
                    marker=dict(
                        color=[abs(symm)],
                        coloraxis="coloraxis",
                    ),
                ),
                row=1,
                col=ncols,
            )

        # adjust the muscle balance axes
        vrange = np.max(abs(np.array(vals))) * 1.5
        vrange = max(50, np.max(abs(np.array(vals))) * 1.5)
        vrange = [-vrange, vrange]
        fig.update_xaxes(
            row=1,
            col=ncols,
            title=f"{balance_string("Left", "Right", "|")}<br>(%)",
            zeroline=False,
            showline=False,
            showgrid=False,
            showticklabels=False,
            range=vrange,
        )
        fig.update_yaxes(
            row=1,
            col=ncols,
            title="",
            showgrid=False,
            zeroline=True,
            showline=False,
            zerolinecolor="black",
            zerolinewidth=2,
        )
        fig.add_vline(
            x=0,
            row=1,  # type: ignore
            col=ncols,  # type: ignore
            showlegend=False,
            line_color="black",
            line_width=2,
            line_dash="solid",
        )

    return fig


class Isokinetic1RMTest(TestProtocol):

    def __init__(
        self,
        rm1_coefs: dict[str, float],
        left: IsokineticExercise | None,
        right: IsokineticExercise | None,
        bilateral: IsokineticExercise | None,
        participant: Participant,
        normative_data: pd.DataFrame = isok_1rm_normative_values,
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        super().__init__(
            participant,
            normative_data,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            relevant_muscle_map=relevant_muscle_map,
        )
        self.set_1rm_coefs(rm1_coefs)
        self.set_left_test(left)
        self.set_right_test(right)
        self.set_bilateral_test(bilateral)

    def set_left_test(self, test: IsokineticExercise | None):
        if test is not None and not isinstance(test, IsokineticExercise):
            msg = f"left must be None or an {IsokineticExercise.__name__} instance."
            raise ValueError(msg)
        self._left = test

    @property
    def left(self):
        return self._left

    def set_right_test(self, test: IsokineticExercise | None):
        if test is not None and not isinstance(test, IsokineticExercise):
            msg = f"right must be None or an {IsokineticExercise.__name__} instance."
            raise ValueError(msg)
        self._right = test

    @property
    def right(self):
        return self._right

    def set_bilateral_test(self, test: IsokineticExercise | None):
        if test is not None and not isinstance(test, IsokineticExercise):
            msg = (
                f"bilateral must be None or an {IsokineticExercise.__name__} instance."
            )
            raise ValueError(msg)
        self._bilateral = test

    @property
    def bilateral(self):
        return self._bilateral

    def set_1rm_coefs(self, rm1_coefs: dict[str, float]):
        msg = "rm1_coefs must be a dict with keys 'beta0', 'beta1' and floats "
        msg += "as values."
        if not isinstance(rm1_coefs, dict):
            raise ValueError(msg)
        keys = list(rm1_coefs.keys())
        vals = list(rm1_coefs.values())
        if any([i not in ["beta0", "beta1"] for i in keys]):
            raise ValueError(msg)
        if not all([isinstance(i, (int, float)) for i in vals]):
            raise ValueError(msg)
        self._1rm_coefs = rm1_coefs

    @property
    def rm1_coefs(self):
        return self._1rm_coefs

    def copy(self):
        return Isokinetic1RMTest(
            participant=self.participant.copy(),
            normative_data=self.normative_data,
            rm1_coefs=self.rm1_coefs,
            left=self.left,
            right=self.right,
            bilateral=self.bilateral,
            emg_normalization_references=self.emg_normalization_references,
            emg_normalization_function=self.emg_normalization_function,
            emg_activation_references=self.emg_activation_references,
            emg_activation_threshold=self.emg_activation_threshold,
        )

    @classmethod
    def from_files(
        cls,
        participant: Participant,
        product: Literal[
            "LEG PRESS",
            "LEG PRESS REV",
            "LEG EXTENSION",
            "LEG EXTENSION REV",
            "LEG CURL",
            "LOW ROW",
            "ADJUSTABLE PULLEY REV",
            "CHEST PRESS",
            "SHOULDER PRESS",
        ],
        left_biostrength_filename: str | None = None,
        right_biostrength_filename: str | None = None,
        bilateral_biostrength_filename: str | None = None,
        left_emg_filename: str | None = None,
        right_emg_filename: str | None = None,
        bilateral_emg_filename: str | None = None,
        normative_data: pd.DataFrame = isok_1rm_normative_values,
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
    ):

        # get 1RM coefficients
        prod = PRODUCTS[product]
        rm1_coefs = {i: v for i, v in zip(["beta1", "beta0"], prod._rm1_coefs)}  # type: ignore

        # get left data
        left = {}
        if left_biostrength_filename is not None:
            bio = IsokineticExercise.from_txt(
                left_biostrength_filename,
                product,
                side="left",
            )
            left.update({i: bio[i] for i in ["force", "position"]})
        if left_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(left_emg_filename)
            left.update(emg.emgsignals)
        if len(left) > 0:
            left = IsokineticExercise(
                side="left",
                synchronize_signals=True,
                **left,
            )
        else:
            left = None

        # get right data
        right = {}
        if right_biostrength_filename is not None:
            bio = IsokineticExercise.from_txt(
                right_biostrength_filename,
                product,
                side="right",
            )
            right.update({i: bio[i] for i in ["force", "position"]})
        if right_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(right_emg_filename)
            right.update(emg.emgsignals)
        if len(right) > 0:
            right = IsokineticExercise(
                side="right",
                synchronize_signals=True,
                **right,
            )
        else:
            right = None

        # get bilateral data
        bilateral = {}
        if bilateral_biostrength_filename is not None:
            bio = IsokineticExercise.from_txt(
                bilateral_biostrength_filename,
                product,
                side="bilateral",
            )
            bilateral.update({i: bio[i] for i in ["force", "position"]})
        if bilateral_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(bilateral_emg_filename)
            bilateral.update(emg.emgsignals)
        if len(bilateral) > 0:
            bilateral = IsokineticExercise(
                side="bilateral",
                synchronize_signals=True,
                **bilateral,
            )
        else:
            bilateral = None

        return cls(
            participant=participant,
            normative_data=normative_data,
            rm1_coefs=rm1_coefs,
            left=left,
            right=right,
            bilateral=bilateral,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            relevant_muscle_map=relevant_muscle_map,
        )

    def get_results(
        self,
        include_emg: bool = True,
        estimate_1rm: bool = True,
        include_force_balance: bool = True,
    ):
        return Isokinetic1RMTestResults(
            self.processed_data,
            include_emg,
            estimate_1rm,
            include_force_balance,
        )

    def _process_exercise(self, exercise: IsokineticExercise):
        # apply the pipeline to the test data
        exe = self.processing_pipeline(exercise, inplace=False)
        if not isinstance(exe, IsokineticExercise):
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

        # return processed data
        return exe

    @property
    def processed_data(self):
        out = self.copy()
        if out.left is not None:
            out.set_left_test(self._process_exercise(out.left))
        if out.right is not None:
            out.set_right_test(self._process_exercise(out.right))
        if out.bilateral is not None:
            out.set_bilateral_test(self._process_exercise(out.bilateral))
        return out

    @property
    def processing_pipeline(self):
        def custom_processing_func(signal: Signal1D):
            signal.fillna(inplace=True)
            fsamp = 1 / np.mean(np.diff(signal.index))
            signal.apply(
                butterworth_filt,
                fcut=1,
                fsamp=fsamp,
                order=4,
                ftype="lowpass",
                phase_corrected=True,
                inplace=True,
            )

        pipeline = get_default_processing_pipeline()
        pipeline.add(Signal1D=[custom_processing_func])
        return pipeline


class Isokinetic1RMTestResults(TestResults):

    @property
    def estimate_1rm(self):
        return self._estimate_1rm

    def set_estimate_1rm(self, estimate: bool):
        if not isinstance(estimate, bool):
            raise ValueError("estimate must be a bool instance.")
        self._estimate_1rm = estimate

    @property
    def include_emg(self):
        return self._include_emg

    def set_include_emg(self, estimate: bool):
        if not isinstance(estimate, bool):
            raise ValueError("estimate must be a bool instance.")
        self._include_emg = estimate

    @property
    def include_force_balance(self):
        return self._include_force_balance

    def set_include_force_balance(self, estimate: bool):
        if not isinstance(estimate, bool):
            raise ValueError("estimate must be a bool instance.")
        self._include_force_balance = estimate

    def __init__(
        self,
        test: Isokinetic1RMTest,
        include_emg: bool = True,
        estimate_1rm: bool = False,
        include_force_balance: bool = True,
    ):
        self._summary = pd.DataFrame()
        self._analytics = pd.DataFrame()
        self._figures = {}
        self.set_include_emg(include_emg)
        self.set_include_force_balance(include_force_balance)
        self.set_estimate_1rm(estimate_1rm)
        self._generate_results(test)

    def _get_estimated_1rm(
        self,
        test: IsokineticExercise,
        coefs: dict[str, float],
    ):
        return self._get_peak_force(test) / G * coefs["beta1"] + coefs["beta0"]

    def _get_peak_force(self, test: IsokineticExercise):
        return float(test.force.to_numpy().max())

    def _get_summary(self, test: Isokinetic1RMTest):
        processed = test.processed_data
        trials = [processed.left, processed.right, processed.bilateral]
        sides = ["left", "right", "bilateral"]
        metrics = pd.DataFrame()
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            emg_norms = test.emg_normalization_values
            for rep in trial.repetitions:
                new = pd.DataFrame()
                new.loc["rom (m)", side] = rep.rom_m
                if self.include_emg:
                    for m in rep.emgsignals.values():
                        ename = str(m.muscle_name)
                        eside = str(m.side)
                        keys = emg_norms.keys()
                        check = [i[0] == ename and i[1] == eside for i in keys]
                        ename += " (%)" if any(check) else " (uV)"
                        new.loc[ename, eside] = m.to_numpy().mean()
                metrics = pd.concat([metrics, new])
            if self.estimate_1rm:
                metrics.loc["estimated 1RM (kg)", side] = self._get_estimated_1rm(
                    trial,
                    test.rm1_coefs,
                )
            metrics.loc["peak force (N)", side] = self._get_peak_force(trial)
        metrics.insert(0, "parameter", metrics.index)
        metrics.reset_index(inplace=True, drop=True)
        summary = metrics.groupby("parameter", as_index=False).mean()

        # add left/right symmetries
        for row in range(summary.shape[0]):
            line = summary.iloc[row]
            if "left" in line.index and "right" in line.index:
                lt, rt = line[["left", "right"]].to_numpy().flatten()
                lt = float(lt)
                rt = float(rt)
                symm = 100 * (rt - lt) / ((rt + lt))
                summary.loc[summary.index[row], "symmetry (%)"] = symm

        return summary

    def _get_analytics(self, test: Isokinetic1RMTest):
        analytics = []
        trials = [test.left, test.right, test.bilateral]
        sides = ["left", "right", "bilateral"]
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            for i, rep in enumerate(trial.repetitions):
                cycle = rep.copy()
                if not self.include_emg:
                    cycle.drop(cycle.emgsignals.keys(), inplace=True)
                cycle = cycle.to_dataframe()
                time = cycle.index - cycle.index.min()
                cycle.insert(0, "time_s", time)
                cycle.insert(0, "repetition", i + 1)
                cycle.insert(0, "side", side)
                analytics.append(cycle)
        return pd.concat(analytics, ignore_index=True)

    def _get_figures(self, test: Isokinetic1RMTest):

        # force data
        tracks = self.analytics
        if self.include_emg:
            cols = tracks.columns
        else:
            cols = [
                "side",
                "repetition",
                "time_s",
                "force_amplitude",
                "position_amplitude",
            ]
        force_data = tracks[cols].copy()
        dfs = []
        for (side, rep), dfr in force_data.groupby(by=["side", "repetition"]):
            df = dfr.drop(
                ["side", "repetition", "time_s", "position_amplitude"],
                axis=1,
            ).copy()
            arr = df.to_numpy()
            interp = np.apply_along_axis(
                cubicspline_interp,
                arr=arr,
                axis=0,
                nsamp=201,
            )
            df = pd.DataFrame(dict(zip(df.columns, interp.T)))
            df.insert(
                0,
                "time_%",
                np.linspace(0, 100, df.shape[0]),
            )
            df = df.melt(
                id_vars=["time_%"],
                var_name="parameter",
                value_name="value",
            )
            df.insert(
                0,
                "side",
                side,
            )
            df.insert(
                0,
                "limb",
                "bilateral" if side == "bilateral" else "unilateral",
            )
            df.insert(
                0,
                "repetition",
                rep,
            )
            dfs.append(df)
        dfr = pd.concat(dfs, ignore_index=True)

        out: dict[str, go.Figure] = {}
        for i, v in dfr.groupby("limb"):
            out[i] = _get_force_figure(
                v,
                self.summary,
                self.include_emg,
            )

        return out


class IsometricTest(TestProtocol):

    def __init__(
        self,
        left: IsometricExercise | None,
        right: IsometricExercise | None,
        bilateral: IsometricExercise | None,
        participant: Participant,
        normative_data: pd.DataFrame = pd.DataFrame(),
        emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        super().__init__(
            participant,
            normative_data,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            relevant_muscle_map=relevant_muscle_map,
        )
        self.set_left_test(left)
        self.set_right_test(right)
        self.set_bilateral_test(bilateral)

    def copy(self):
        return IsometricTest(
            participant=self.participant.copy(),
            normative_data=self.normative_data,
            left=self.left,
            right=self.right,
            bilateral=self.bilateral,
            emg_normalization_references=self.emg_normalization_references,
            emg_normalization_function=self.emg_normalization_function,
            emg_activation_references=self.emg_activation_references,
            emg_activation_threshold=self.emg_activation_threshold,
            relevant_muscle_map=self.relevant_muscle_map,
        )

    def get_results(self, include_emg: bool = True):
        return IsometricTestResults(
            self.processed_data,
            include_emg,
        )

    def set_left_test(self, test: IsometricExercise | None):
        if test is not None and not isinstance(test, IsometricExercise):
            msg = f"left must be None or an {IsometricExercise.__name__} instance."
            raise ValueError(msg)
        self._left = test

    @property
    def left(self):
        return self._left

    def set_right_test(self, test: IsometricExercise | None):
        if test is not None and not isinstance(test, IsometricExercise):
            msg = f"right must be None or an {IsometricExercise.__name__} instance."
            raise ValueError(msg)
        self._right = test

    @property
    def right(self):
        return self._right

    def set_bilateral_test(self, test: IsometricExercise | None):
        if test is not None and not isinstance(test, IsometricExercise):
            msg = f"bilateral must be None or an {IsometricExercise.__name__} instance."
            raise ValueError(msg)
        self._bilateral = test

    @property
    def bilateral(self):
        return self._bilateral

    def _process_exercise(self, exercise: IsometricExercise):
        # apply the pipeline to the test data
        exe = self.processing_pipeline(exercise, inplace=False)
        if not isinstance(exe, IsometricExercise):
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

        # return processed data
        return exe

    @property
    def processed_data(self):
        out = self.copy()
        if out.left is not None:
            out.set_left_test(self._process_exercise(out.left))
        if out.right is not None:
            out.set_right_test(self._process_exercise(out.right))
        if out.bilateral is not None:
            out.set_bilateral_test(self._process_exercise(out.bilateral))
        return out

    @property
    def processing_pipeline(self):
        def custom_processing_func(signal: Signal1D):
            signal.fillna(inplace=True)
            fsamp = 1 / np.mean(np.diff(signal.index))
            signal.apply(
                butterworth_filt,
                fcut=1,
                fsamp=fsamp,
                order=4,
                ftype="lowpass",
                phase_corrected=True,
                inplace=True,
            )

        pipeline = get_default_processing_pipeline()
        pipeline.add(Signal1D=[custom_processing_func])
        return pipeline

    @classmethod
    def from_files(
        cls,
        participant: Participant,
        product: Literal[
            "LEG PRESS",
            "LEG PRESS REV",
            "LEG EXTENSION",
            "LEG EXTENSION REV",
            "LEG CURL",
            "LOW ROW",
            "ADJUSTABLE PULLEY REV",
            "CHEST PRESS",
            "SHOULDER PRESS",
        ],
        left_biostrength_filename: str | None = None,
        right_biostrength_filename: str | None = None,
        bilateral_biostrength_filename: str | None = None,
        left_emg_filename: str | None = None,
        right_emg_filename: str | None = None,
        bilateral_emg_filename: str | None = None,
        normative_data: pd.DataFrame = isok_1rm_normative_values,
        emg_normalization_references: TimeseriesRecord = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: TimeseriesRecord = TimeseriesRecord(),
        emg_activation_threshold: float = 3,
        relevant_muscle_map: list[str] | None = None,
    ):

        # get left data
        left = {}
        if left_biostrength_filename is not None:
            bio = IsometricExercise.from_txt(
                left_biostrength_filename,
                product,
                side="left",
            )
            left.update({i: bio[i] for i in ["force", "position"]})
        if left_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(left_emg_filename)
            left.update(emg.emgsignals)
        if len(left) > 0:
            left = IsometricExercise(
                side="left",
                synchronize_signals=True,
                **left,
            )
        else:
            left = None

        # get right data
        right = {}
        if right_biostrength_filename is not None:
            bio = IsometricExercise.from_txt(
                right_biostrength_filename,
                product,
                side="right",
            )
            right.update({i: bio[i] for i in ["force", "position"]})
        if right_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(right_emg_filename)
            right.update(emg.emgsignals)
        if len(right) > 0:
            right = IsometricExercise(
                side="right",
                synchronize_signals=True,
                **right,
            )
        else:
            right = None

        # get bilateral data
        bilateral = {}
        if bilateral_biostrength_filename is not None:
            bio = IsometricExercise.from_txt(
                bilateral_biostrength_filename,
                product,
                side="bilateral",
            )
            bilateral.update({i: bio[i] for i in ["force", "position"]})
        if bilateral_emg_filename is not None:
            emg = TimeseriesRecord.from_tdf(bilateral_emg_filename)
            bilateral.update(emg.emgsignals)
        if len(bilateral) > 0:
            bilateral = IsometricExercise(
                side="bilateral",
                synchronize_signals=True,
                **bilateral,
            )
        else:
            bilateral = None

        return cls(
            participant=participant,
            normative_data=normative_data,
            left=left,
            right=right,
            bilateral=bilateral,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            relevant_muscle_map=relevant_muscle_map,
        )


class IsometricTestResults(TestResults):

    def __init__(self, test: IsometricTest, include_emg: bool):
        if not isinstance(test, IsometricTest):
            raise ValueError("'test' must be an IsometricTest instance.")
        super().__init__(test, include_emg)

    def _get_peak_force(self, test: IsometricExercise):
        return float(test.force.to_numpy().max())

    def _get_rate_of_force_development_kns(self, exe: IsometricExercise):
        force = exe.repetitions[0].force.to_numpy().flatten()
        time = np.array(exe.index)
        peaks = find_peaks(force, height=float(np.max(force) * 0.8))
        peak = np.argmax(force) if len(peaks) == 0 else peaks[0]
        return (force[peak] - force[0]) / (time[peak] - time[0]) / 1000

    def _get_time_to_peak_force_ms(self, exe: IsometricExercise):
        force = exe.repetitions[0].force.to_numpy().flatten()
        time = np.array(exe.index)
        peak = np.argmax(force)
        return (time[peak] - time[0]) * 1000

    def _get_summary(self, test: IsometricTest):
        trials = [test.left, test.right, test.bilateral]
        sides = ["left", "right", "bilateral"]
        metrics = pd.DataFrame()
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            emg_norms = test.emg_normalization_values
            if self.include_emg:
                for rep in trial.repetitions:
                    new = pd.DataFrame()
                    for m in rep.emgsignals.values():
                        ename = str(m.muscle_name)
                        eside = str(m.side)
                        keys = emg_norms.keys()
                        check = [i[0] == ename and i[1] == eside for i in keys]
                        ename += " (%)" if any(check) else " (uV)"
                        new.loc[ename, eside] = m.to_numpy().mean()
                    metrics = pd.concat([metrics, new])
            metrics.loc["rate of force development (kN/s)", side] = (
                self._get_rate_of_force_development_kns(trial)
            )
            metrics.loc["time to peak force (ms)", side] = (
                self._get_time_to_peak_force_ms(trial)
            )
            metrics.loc["peak force (N)", side] = self._get_peak_force(trial)
        metrics.insert(0, "parameter", metrics.index)
        metrics.reset_index(inplace=True, drop=True)
        summary = metrics.groupby("parameter", as_index=False).mean()

        # add left/right symmetries
        for row in range(summary.shape[0]):
            line = summary.iloc[row]
            if "left" in line.index and "right" in line.index:
                lt, rt = line[["left", "right"]].to_numpy().flatten()
                lt = float(lt)
                rt = float(rt)
                symm = 100 * (rt - lt) / ((rt + lt))
                summary.loc[summary.index[row], "symmetry (%)"] = symm

        return summary

    def _get_analytics(self, test: IsometricTest):
        processed = test.processed_data
        analytics = []
        trials = [processed.left, processed.right, processed.bilateral]
        sides = ["left", "right", "bilateral"]
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            for i, rep in enumerate(trial.repetitions):
                cycle = rep.copy()
                if not self.include_emg:
                    cycle.drop(cycle.emgsignals.keys(), inplace=True)
                cycle = cycle.to_dataframe()
                time = cycle.index - cycle.index.min()
                cycle.insert(0, "time_s", time)
                cycle.insert(0, "repetition", i + 1)
                cycle.insert(0, "side", side)
                analytics.append(cycle)
        return pd.concat(analytics, ignore_index=True)

    def _get_figures(self, test: IsometricTest):

        # force data
        tracks = self.analytics
        sides = np.unique(tracks.side).tolist()
        force_data = [
            "side",
            "repetition",
            "time_s",
            "force_amplitude",
            "position_amplitude",
        ]
        force_data = tracks[force_data].copy()
        y_arr = {side: [] for side in sides}
        for (side, rep), dfr in force_data.groupby(by=["side", "repetition"]):
            force = dfr.force_amplitude.to_numpy().flatten()
            fint = cubicspline_interp(force, 201)
            y_arr[side].append(np.atleast_2d(fint))
        tracks = []
        for i, side in enumerate(sides):
            y = np.vstack(y_arr[side]).max(axis=0)
            x = np.linspace(0, 5, len(y))
            df = pd.DataFrame({"Time (s)": x, "y": y})
            df.insert(0, "side", side)
            tracks.append(df)
        tracks = pd.concat(tracks, ignore_index=True)

        return {
            "force_profiles_with_muscle_balance": _get_force_figure(
                tracks,
                self.summary,
            )
        }


# TODO CREATE THE POWER-VELOCITY / FORCE-VELOCITY TEST
