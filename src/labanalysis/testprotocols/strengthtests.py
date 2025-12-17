"""isokinetic test module"""

#! IMPORTS


from typing import Literal

from networkx import is_empty
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as colormaps
from plotly.subplots import make_subplots

from ..records.records import TimeseriesRecord

from ..records.pipelines import ProcessingPipeline

from ..constants import G
from ..io.read.biostrength import PRODUCTS
from ..records import EMGSignal, IsokineticExercise, IsometricExercise, Signal1D
from ..records.pipelines import get_default_processing_pipeline
from ..signalprocessing import cubicspline_interp, find_peaks
from .normativedata import isok_1rm_normative_values
from .protocols import Participant, TestProtocol, TestResults


#! CONSTANTS


__all__ = ["Isokinetic1RMTest", "IsometricTest"]


#! CLASSES


class Isokinetic1RMTest(TestProtocol):

    def __init__(
        self,
        rm1_coefs: dict[str, float],
        left: IsokineticExercise | None,
        right: IsokineticExercise | None,
        bilateral: IsokineticExercise | None,
        participant: Participant,
        normative_data: pd.DataFrame = isok_1rm_normative_values,
    ):
        super().__init__(participant, normative_data)
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
        )

    @property
    def results(self):
        return Isokinetic1RMTestResults(self)


class Isokinetic1RMTestResults(TestResults):

    def __init__(self, test: Isokinetic1RMTest):
        if not isinstance(test, Isokinetic1RMTest):
            raise ValueError("'test' must be an Isokinetic1RMTest instance.")
        super().__init__(test)

    def _get_processed_data(self, test: Isokinetic1RMTest):
        new = test.copy()
        pipeline = get_default_processing_pipeline()
        if new.left is not None:
            new.set_left_test(pipeline(new.left, inplace=False))  # type: ignore
        if new.right is not None:
            new.set_right_test(pipeline(new.right, inplace=False))  # type: ignore
        if new.bilateral is not None:
            new.set_bilateral_test(pipeline(new.bilateral, inplace=False))  # type: ignore
        return new

    def _get_estimated_1rm(
        self,
        test: IsokineticExercise,
        coefs: dict[str, float],
    ):
        return self._get_peak_force(test) / G * coefs["beta1"] + coefs["beta0"]

    def _get_peak_force(self, test: IsokineticExercise):
        return float(test.force.to_numpy().max())

    def _get_summary(self, test: Isokinetic1RMTest):

        # get the muscle activations
        # (if there are no emg data return and empty dataframe)
        emgs = pd.DataFrame()
        sides = ["left", "right", "bilateral"]
        trials = [test.left, test.right, test.bilateral]
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            for emg in trial.emgsignals.values():
                if side != "bilateral" and emg.side != side:
                    continue
                name = emg.muscle_name.replace("_", " ").lower() + " (%)"
                emgs.loc[name, emg.side] = emg.to_numpy().mean()  # type: ignore
        emgs = emgs.T
        summary: pd.DataFrame = (emgs / emgs.sum(axis=0).values * 100).T  # type: ignore

        trials = [test.left, test.right, test.bilateral]
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            metrics = []
            for rep in trial.repetitions:
                metrics += [pd.DataFrame([{"rom_m": rep.rom_m}])]
            out = pd.DataFrame(pd.concat(metrics).mean(axis=0)).T
            summary.loc["rom (m)", side] = out.to_numpy()[0]
            summary.loc["estimated 1RM (kg)", side] = self._get_estimated_1rm(
                trial,
                test.rm1_coefs,
            )
            summary.loc["peak force (N)", side] = self._get_peak_force(trial)
        summary.insert(0, "parameter", summary.index)

        return summary.reset_index(drop=True)

    def _get_analytics(self, test: Isokinetic1RMTest):
        analytics = []
        for side, trial in zip(
            ["left", "right", "bilateral"], [test.left, test.right, test.bilateral]
        ):
            if trial is None:
                continue
            for i, rep in enumerate(trial.repetitions):
                cycle = rep.to_dataframe()
                cycle.insert(0, "time_s", cycle.index)
                cycle.insert(0, "repetition", i + 1)
                cycle.insert(0, "side", side)
                analytics.append(cycle)
        return pd.concat(analytics, ignore_index=True)

    def _get_figures(self, test: Isokinetic1RMTest):

        # get the data
        data = self.analytics
        summ = self.summary
        sides = ["left", "right", "bilateral"]
        sides = [i for i in sides if i in data.side.unique()]
        trials = [test.left, test.right, test.bilateral]
        trials = [i for i in trials if i is not None]
        muscles = [str(i.muscle_name) for t in trials for i in t.emgsignals.values()]
        muscles = np.unique(muscles)
        y = data.force_amplitude.to_numpy()
        yrange = [y.min() * 0.9, y.max() * 1.2]
        cmap = colormaps.Plotly[: len(sides)]

        # generate the figure
        nrows = 1
        ncols = len(sides)
        titles = [i for i in sides]
        if len(muscles) > 0:
            ncols += 1
            titles += ["Muscle EMG balance"]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=titles,
        )
        for i in range(len(sides)):
            fig.update_xaxes(title="ROM (m)", row=1, col=i + 1)
            fig.update_yaxes(range=yrange, row=1, col=i + 1)
        fig.update_yaxes(title="Force (N)", col=1, row=1)
        if len(muscles) > 0:
            fig.update_yaxes(col=ncols, row=1, showticklabels=False, title="")
            fig.add_hline(
                y=50,
                line_color="black",
                line_dash="dash",
                line_width=3,
                opacity=0.5,
                showlegend=False,
                row=1,  # type: ignore
                col=ncols,  # type: ignore
            )

        # plot force profiles
        for i, side in enumerate(sides):
            if side not in summ.columns:
                continue
            reps = []
            for rep in trials[i].repetitions:
                force = rep.force.to_numpy().flatten()
                interp = cubicspline_interp(force, 201)
                reps.append(np.atleast_2d(interp))
            y = np.vstack(reps).max(axis=0)
            x = np.linspace(0, 100, len(y))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name="force profile",
                    showlegend=False,
                    line_color=cmap[i],
                ),
                row=1,
                col=i + 1,
            )
            x_peak = x[np.argmax(y)]
            fig.add_vline(
                row=1,  # type: ignore
                col=i + 1,  # type: ignore
                x=x_peak,
                line_dash="dash",
                line_color="black",
                opacity=0.5,
                showlegend=False,
            )
            note = [f"{"Peak:"}{np.max(y):0.1f}N"]
            est_1rm = summ.loc[summ.parameter == "estimated 1RM (kg)", side]
            est_1rm = float(np.squeeze(est_1rm.to_numpy()))
            note += [f"{"Estimated 1RM:"}{est_1rm:0.1f}kg"]
            note = "<br>".join(note)
            midline = 0.5 * (np.max(x) - np.min(x))
            fig.add_trace(
                row=1,
                col=i + 1,
                trace=go.Scatter(
                    x=[x_peak],
                    y=[np.max(y)],
                    dx=20 if x_peak <= 50 else -20,
                    text=note,
                    mode="markers+text",
                    textposition="top right" if x_peak <= midline else "top left",
                    marker=dict(size=12, color=cmap[i]),
                    textfont=dict(size=12, color=cmap[i]),
                    showlegend=False,
                    name="force profile",
                ),
            )

        # plot muscle data
        muscles_maxy = []
        muscles = [i for i in summ.parameter if i.endswith(" (%)")]
        for m, muscle in enumerate(muscles):
            dfs = summ.copy().loc[summ.parameter == muscle].dropna()
            for i, side in enumerate(sides):
                y = float(np.squeeze(dfs[side].to_numpy()))
                fig.add_trace(
                    row=1,
                    col=ncols,
                    trace=go.Bar(
                        x=[muscle.rsplit(" ")[0]],
                        y=[y],
                        showlegend=m == 0,
                        legendgroup=side,
                        marker_color=cmap[i],
                        name=side,
                        text=[f"{y:0.1f}%"],
                        textposition="outside",
                        offsetgroup=side,
                    ),
                )
                muscles_maxy += [np.max(y)]
        if len(muscles):
            fig.update_layout(barmode="group")
            fig.update_yaxes(
                range=[0, np.max(muscles_maxy) * 1.2],
                col=ncols,
            )

        return {"force_profiles_with_muscle_balance": fig}


class IsometricTest(TestProtocol):

    def __init__(
        self,
        left: IsometricExercise | None,
        right: IsometricExercise | None,
        bilateral: IsometricExercise | None,
        participant: Participant,
        normative_data: pd.DataFrame = isok_1rm_normative_values,
    ):
        super().__init__(participant, normative_data)
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
        )

    @property
    def results(self):
        return IsometricTestResults(self)

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
        )


class IsometricTestResults(TestResults):

    def __init__(self, test: IsometricTest):
        if not isinstance(test, IsometricTest):
            raise ValueError("'test' must be an IsometricTest instance.")
        super().__init__(test)

    def _get_processed_data(self, test: IsometricTest):
        new = test.copy()
        pipeline = get_default_processing_pipeline()
        if new.left is not None:
            new.set_left_test(pipeline(new.left, inplace=False))  # type: ignore
        if new.right is not None:
            new.set_right_test(pipeline(new.right, inplace=False))  # type: ignore
        if new.bilateral is not None:
            new.set_bilateral_test(pipeline(new.bilateral, inplace=False))  # type: ignore
        return new

    def _get_peak_force(self, test: IsometricExercise):
        return float(test.force.to_numpy().max())

    def _get_rate_of_force_development_kns(self, test: IsometricExercise):
        force = test.force.to_numpy().flatten()
        time = np.array(test.index)
        peaks = find_peaks(force, height=float(np.max(force) * 0.8))
        peak = np.argmax(force) if len(peaks) == 0 else peaks[0]
        return (force[peak] - force[0]) / (time[peak] - time[0]) / 1000

    def _get_time_to_peak_force_ms(self, test: IsometricExercise):
        force = test.force.to_numpy().flatten()
        time = np.array(test.index)
        peak = np.argmax(force)
        return (time[peak] - time[0]) * 1000

    def _get_summary(self, test: IsometricTest):

        # get the muscle activations
        # (if there are no emg data return and empty dataframe)
        emgs = pd.DataFrame()
        sides = ["left", "right", "bilateral"]
        trials = [test.left, test.right, test.bilateral]
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            for emg in trial.emgsignals.values():
                if side != "bilateral" and emg.side != side:
                    continue
                name = emg.muscle_name.replace("_", " ").lower() + " (%)"
                emgs.loc[name, emg.side] = emg.to_numpy().mean()  # type: ignore
        emgs = emgs.T
        summary: pd.DataFrame = (emgs / emgs.sum(axis=0).values * 100).T  # type: ignore

        trials = [test.left, test.right, test.bilateral]
        for side, trial in zip(sides, trials):
            if trial is None:
                continue
            summary.loc["peak force (N)", side] = self._get_peak_force(trial)
            summary.loc["time to peak force (ms)", side] = (
                self._get_time_to_peak_force_ms(trial)
            )
            summary.loc["rate of force development (kN/s)", side] = (
                self._get_rate_of_force_development_kns(trial)
            )
        summary.insert(0, "parameter", summary.index)

        return summary.reset_index(drop=True)

    def _get_analytics(self, test: IsometricTest):
        analytics = []
        for side, trial in zip(
            ["left", "right", "bilateral"], [test.left, test.right, test.bilateral]
        ):
            if trial is None:
                continue
            for i, rep in enumerate(trial.repetitions):
                cycle = rep.to_dataframe()
                cycle.insert(0, "time_s", cycle.index)
                cycle.insert(0, "repetition", i + 1)
                cycle.insert(0, "side", side)
                analytics.append(cycle)
        return pd.concat(analytics, ignore_index=True)

    def _get_figures(self, test: IsometricTest):

        # get the data
        data = self.analytics
        summ = self.summary
        sides = ["left", "right", "bilateral"]
        sides = [i for i in sides if i in data.side.unique()]
        trials = [test.left, test.right, test.bilateral]
        trials = [i for i in trials if i is not None]
        muscles = [str(i.muscle_name) for t in trials for i in t.emgsignals.values()]
        muscles = np.unique(muscles)
        y = data.force_amplitude.to_numpy()
        yrange = [y.min() * 0.9, y.max() * 1.2]
        cmap = colormaps.Plotly[: len(sides)]

        # generate the figure
        nrows = 1
        ncols = len(sides)
        titles = [i for i in sides]
        if len(muscles) > 0:
            ncols += 1
            titles += ["Muscle EMG balance"]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=titles,
        )
        for i in range(len(sides)):
            fig.update_xaxes(title="Time (s)", row=1, col=i + 1)
            fig.update_yaxes(range=yrange, row=1, col=i + 1)
        fig.update_yaxes(title="Force (N)", col=1, row=1)
        if len(muscles) > 0:
            fig.update_yaxes(col=ncols, row=1, showticklabels=False, title="")
            fig.add_hline(
                y=50,
                line_color="black",
                line_dash="dash",
                line_width=3,
                opacity=0.5,
                showlegend=False,
                row=1,  # type: ignore
                col=ncols,  # type: ignore
            )

        # plot force profiles
        for i, side in enumerate(sides):
            if side not in summ.columns:
                continue
            reps = []
            for rep in trials[i].repetitions:
                force = rep.force.to_numpy().flatten()
                interp = cubicspline_interp(force, 201)
                reps.append(np.atleast_2d(interp))
            y = np.vstack(reps).max(axis=0)
            x = np.linspace(0, 5, len(y))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name="force profile",
                    showlegend=False,
                    line_color=cmap[i],
                ),
                row=1,
                col=i + 1,
            )
            x_peak = x[np.argmax(y)]
            fig.add_vline(
                row=1,  # type: ignore
                col=i + 1,  # type: ignore
                x=x_peak,
                line_dash="dash",
                line_color="black",
                opacity=0.5,
                showlegend=False,
            )
            note = [f"{"Peak:"}{np.max(y):0.0f}N"]
            tpeak = summ.loc[summ.parameter == "time to peak force (ms)", side]
            tpeak = float(np.squeeze(tpeak.to_numpy()))
            note += [f"{"Time:"}{tpeak:0.0f}ms"]
            rfd = summ.loc[summ.parameter == "rate of force development (kN/s)", side]
            rfd = float(np.squeeze(rfd.to_numpy()))
            note += [f"{"RFD:"}{rfd:0.3f}kN/s"]
            note = "<br>".join(note)
            midline = 0.5 * (np.max(x) - np.min(x))
            fig.add_trace(
                row=1,
                col=i + 1,
                trace=go.Scatter(
                    x=[x_peak],
                    y=[np.max(y)],
                    dx=20 if x_peak <= 50 else -20,
                    text=note,
                    mode="markers+text",
                    textposition="top right" if x_peak <= midline else "top left",
                    marker=dict(size=12, color=cmap[i]),
                    textfont=dict(size=12, color=cmap[i]),
                    showlegend=False,
                    name="force profile",
                ),
            )

        # plot muscle data
        muscles_maxy = []
        muscles = [i for i in summ.parameter if i.endswith(" (%)")]
        for m, muscle in enumerate(muscles):
            dfs = summ.loc[summ.parameter == muscle].dropna()
            for i, side in enumerate(sides):
                y = float(np.squeeze(dfs[side].to_numpy()))
                fig.add_trace(
                    row=1,
                    col=ncols,
                    trace=go.Bar(
                        x=[muscle.rsplit(" ")[0]],
                        y=[y],
                        showlegend=m == 0,
                        legendgroup=side,
                        marker_color=cmap[i],
                        name=side,
                        text=[f"{y:0.1f}%"],
                        textposition="outside",
                        offsetgroup=side,
                    ),
                )
                muscles_maxy += [np.max(y)]
        if len(muscles):
            fig.update_layout(barmode="group")
            fig.update_yaxes(
                range=[0, np.max(muscles_maxy) * 1.2],
                col=ncols,
            )

        return {"force_profiles_with_muscle_balance": fig}


# TODO CREATE THE POWER-VELOCITY / FORCE-VELOCITY TEST
