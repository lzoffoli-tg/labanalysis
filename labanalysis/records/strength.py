"""
isokinetic exercise module
"""

#! IMPORTS

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..constants import G
from ..io.read.biostrength import PRODUCTS as BIOSTRENGTH_PRODUCTS_MAP
from ..io.read.biostrength import BiostrengthProduct
from ..signalprocessing import *
from .timeseries import EMGSignal, Signal1D
from .records import TimeseriesRecord

#! CONSTANTS


__all__ = [
    "DefaultRepetition",
    "DefaultExercise",
    "IsokineticRepetition",
    "IsokineticExercise",
    "IsometricRepetition",
    "IsometricExercise",
]

#! CLASSES


class DefaultRepetition(TimeseriesRecord):
    """
    Isokinetic Test 1RM instance

    Parameters
    ----------
    time: Iterable[int | float]
        the array containing the time instant of each sample in seconds

    position: Iterable[int | float]
        the array containing the displacement of the handles for each sample

    load: Iterable[int | float]
        the array containing the load measured at each sample in kgf

    coefs_1rm: tuple[int | float, int | float]
        the b0 and b1 coefficients used to estimated the 1RM.

    Attributes
    ----------
    raw: DataFrame
        a DataFrame containing the input data

    repetitions: list[DataFrame]
        a list of dataframes each defining one single repetition

    product: BiostrengthProduct
        the product on which the test has been performed

    peak_load: float
        the peak load measured during the isokinetic repetitions

    rom0: float
        the start of the user's range of movement in meters

    rom1: float
        the end of the user's range of movement in meters

    rom: float
        the range of movement amplitude in meters

    results_table: DataFrame
        a table containing the data obtained during the test

    summary_table: DataFrame
        a table containing summary statistics about the test

    summary_plot: FigureWidget
        a figure representing the results of the test.
    """

    # * class variables

    _product: BiostrengthProduct
    _side: Literal["bilateral", "left", "right"]

    # * attributes

    @property
    def side(self):
        """get the side of the test"""
        return self._side

    @property
    def product(self):
        """return the product on which the test has been performed"""
        return self._product

    @property
    def peak_force_N(self):
        """return the ending position of the repetitions"""
        return float(self.force.to_numpy().max())

    @property
    def rom_m(self):
        """return the repetition's ROM"""
        position = self.position.to_numpy().flatten()
        return float(np.max(position) - np.min(position))

    @property
    def muscle_activations(self):
        """
        Returns coordination and balance metrics from EMG signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with coordination and balance metrics, or empty if not available.
        """

        # get the muscle activations
        # (if there are no emg data return and empty dataframe)
        emgs = self.emgsignals
        if emgs.shape[1] == 0:
            return pd.DataFrame()

        # check the presence of left and right muscles
        muscles = {}
        for emg in emgs.values():
            if isinstance(emg, EMGSignal):
                side = emg.side
                if side != self.side:
                    continue
                name = emg.muscle_name
                unit = emg.unit
                muscles[f"{side} {name} {unit}"] = emg.to_numpy().mean()

        return pd.DataFrame(pd.Series(muscles)).T

    @property
    def output_metrics(self):
        """
        Returns summary metrics for the jump.

        Returns
        -------
        pd.DataFrame
            DataFrame with summary metrics for the jump.
        """
        new = {
            "type": self.product.name,
            "side": self.side,
            "peak_force_N": self.peak_force_N,
            "rom_mm": self.rom_m * 1000,
        }
        new = pd.DataFrame(pd.Series(new)).T
        return pd.concat([new, self.muscle_activations], axis=1)

    def __init__(
        self,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        **signals: EMGSignal,
    ):

        # check the input
        if not issubclass(product.__class__, BiostrengthProduct):
            raise ValueError("'product' must be a valid Biostrength Product.")
        if not side in ["bilateral", "left", "right"]:
            raise ValueError("'side' must be any of 'bilateral', 'left', 'right'")

        # check the required data
        if not isinstance(force, Signal1D) and force.unit != "N":
            raise ValueError("force must be a Signal1D with 'N' as unit")
        if not isinstance(position, Signal1D) and position.unit != "m":
            raise ValueError("position must be a Signal1D with 'm' as unit")
        for key, val in signals.items():
            if not isinstance(val, EMGSignal):
                raise ValueError(f"{key} must be an EMGSignal")

        super().__init__(force=force, position=position, **signals)

        # get the raw data
        self._product = product  # type: ignore
        self._side = side


class DefaultExercise(TimeseriesRecord):
    """
    Isokinetic Test 1RM instance

    Parameters
    ----------
    time: Iterable[int | float]
        the array containing the time instant of each sample in seconds

    position: Iterable[int | float]
        the array containing the displacement of the handles for each sample

    load: Iterable[int | float]
        the array containing the load measured at each sample in kgf

    coefs_1rm: tuple[int | float, int | float]
        the b0 and b1 coefficients used to estimated the 1RM.

    Attributes
    ----------
    raw: DataFrame
        a DataFrame containing the input data

    repetitions: list[DataFrame]
        a list of dataframes each defining one single repetition

    product: BiostrengthProduct
        the product on which the test has been performed

    peak_load: float
        the peak load measured during the isokinetic repetitions

    rom0: float
        the start of the user's range of movement in meters

    rom1: float
        the end of the user's range of movement in meters

    rom: float
        the range of movement amplitude in meters

    results_table: DataFrame
        a table containing the data obtained during the test

    summary_table: DataFrame
        a table containing summary statistics about the test

    summary_plot: FigureWidget
        a figure representing the results of the test.
    """

    # * class variables

    _side: Literal["bilateral", "left", "right"]
    _product: BiostrengthProduct
    _repetition_type = DefaultRepetition

    def _get_repetitions_index(self, array: np.ndarray, time: np.ndarray):
        return []

    def _get_repetitions_start_from_biostrength_data(
        self, *args: np.ndarray, **kwargs: np.ndarray
    ):
        return []

    def _get_repetitions_start_from_emg_data(
        self,
        emg: np.ndarray,
        time: np.ndarray,
    ):
        fsamp = int(1 / np.mean(np.diff(time)))
        batches = continuous_batches(
            arr=emg > np.max(emg) * 0.1,
            tolerance=fsamp,
        )
        if len(batches) == 0:
            raise RuntimeError("no repetitions where found on EMG data.")
        start = int(batches[0][0])
        while start > 0 and emg[start] > emg[start - 1]:
            start -= 1
        return start

    def _sync(self, force: Signal1D, position: Signal1D, **muscles: EMGSignal):
        if len(muscles) != 0:

            # get emg data
            emgs = TimeseriesRecord(
                **{i: v.copy() for i, v in muscles.items()},
            )
            m_time = np.round(np.array(emgs.index) * 1000).astype(int)
            m_vals = emgs.to_dataframe().sum(axis=1)
            m_vals = m_vals.values.astype(float).flatten()

            # filter the emg data
            fsamp_muscles = float(1000 / np.mean(np.diff(m_time)))
            m_vals -= m_vals.mean()
            m_vals = butterworth_filt(
                arr=m_vals,
                fcut=[20, 450],
                fsamp=fsamp_muscles,
                order=4,
                ftype="bandpass",
                phase_corrected=True,
            )
            # apply teaker-kaiser operator and get linear envelope
            m_vals = abs(tkeo(m_vals))
            m_vals = mean_filt(m_vals, int(2 * fsamp_muscles), offset=1)

            # get the first derivative
            m_der1 = winter_derivative1(m_vals, m_time)
            m_der1 = np.concatenate([[m_der1[0]], m_der1, [m_der1[-1]]])
            m_der1 = mean_filt(m_der1, int(2 * fsamp_muscles), offset=1)

            # get force and position data
            f_time = np.round(np.array(force.index) * 1000).astype(int)
            f_vals = force.copy().to_numpy()
            p_vals = position.copy().to_numpy()

            # adjust the force and muscles sampling rate
            fsamp_biostrength = float(1000 / np.mean(np.diff(f_time)))
            fsamp = max(fsamp_biostrength, fsamp_muscles)
            tdiff = 1000 / fsamp
            m_time_new = np.arange(m_time[0], m_time[-1] + tdiff, tdiff)
            m_vals = cubicspline_interp(
                y_old=m_vals,
                x_old=m_time,
                x_new=m_time_new,
            )
            f_time_new = np.arange(f_time[0], f_time[-1] + tdiff, tdiff)
            f_vals = cubicspline_interp(
                y_old=f_vals,
                x_old=f_time,
                x_new=f_time_new,
            )
            p_vals = cubicspline_interp(
                y_old=p_vals,
                x_old=f_time,
                x_new=f_time_new,
            )

            # find the start of the repetitions from position
            pos_start = self._get_repetitions_start_from_biostrength_data(
                f_vals, p_vals, f_time_new
            )
            emg_start = self._get_repetitions_start_from_emg_data(m_vals, m_time_new)

            # set the start to zero on both signals
            f_time_new = f_time_new - f_time_new[pos_start]
            m_time = m_time - m_time_new[emg_start]
            m_time_new = m_time_new - m_time_new[emg_start]

            """
            # cross-correlate the force and the muscle data corresponding to
            # the repetitions index
            xcor, lag = xcorr(
                sig1=m_vals[m_time_new >= 0],
                sig2=f_vals[f_time_new >= 0],
                biased=True,
                full=True,
            )

            # smooth the cross-correlation and adjust the emg signal for the lag
            smoothed = mean_filt(xcor, max(1, int(2 * fsamp)))
            m_offset = tdiff * lag[np.argmax(smoothed)]
            m_time_new = m_time_new - m_offset
            m_time = m_time - m_offset
            """

            # regenerate the synchronized signals
            force_sync = Signal1D(
                f_vals.astype(float),  # type: ignore
                (f_time_new / 1000).tolist(),  # type: ignore
                force.unit,
            )
            position_sync = Signal1D(
                p_vals.astype(float),  # type: ignore
                (f_time_new / 1000).tolist(),  # type: ignore
                position.unit,
            )
            muscles_sync: dict[str, EMGSignal] = {}
            for muscle, data in muscles.items():
                m_vals_interp = cubicspline_interp(
                    data.copy().to_numpy(),
                    x_old=m_time,
                    x_new=m_time_new,
                )
                new_emgsignal = EMGSignal(
                    m_vals_interp,  # type: ignore
                    (m_time_new / 1000).tolist(),  # type: ignore
                    muscle_name=data.muscle_name,
                    side=data.side,  # type: ignore
                    unit=data.unit,
                )
                muscles_sync[muscle] = new_emgsignal

            return force_sync, position_sync, muscles_sync

        return force, position, {}

    @property
    def side(self):
        """get the side of the test"""
        return self._side

    def _get_repetitions_splitting_signal(self):
        return np.ndarray([]), np.ndarray([])

    @property
    def repetitions(self):
        """return the tracked repetitions data"""
        arr, time = self._get_repetitions_splitting_signal()
        reps_idx = self._get_repetitions_index(arr, time)
        reps: list[DefaultRepetition] = []
        for rep in reps_idx:
            rep_time = time[rep]
            start = rep_time[0]
            stop = rep_time[-1]
            repetition = self._repetition_type(
                product=self.product.copy().slice(start, stop),
                side=self.side,
                force=self.force.copy()[start:stop],  # type: ignore
                position=self.position.copy()[start:stop],  # type: ignore
                **{i: v.copy()[start:stop] for i, v in self.emgsignals.items()},  # type: ignore
            )
            reps += [repetition]

        return reps

    @property
    def product(self):
        """return the product on which the test has been performed"""
        return self._product

    def reset_time(self, inplace: bool = False):
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        if inplace:
            super().reset_time(inplace)
            self.product._time_s -= np.min(self.product._time_s)
        else:
            out = self.copy()
            out.reset_time(True)
            return out

    def __init__(
        self,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        synchronize_signals: bool = True,
        **signals: EMGSignal,
    ):

        # check the input
        if not issubclass(product.__class__, BiostrengthProduct):
            raise ValueError("'product' must be a valid Biostrength Product.")
        if not side in ["bilateral", "left", "right"]:
            raise ValueError("'side' must be any of 'bilateral', 'left', 'right'")

        # get the required data
        if not isinstance(force, Signal1D) and force.unit != "N":
            raise ValueError("force must be a Signal1D with 'N' as unit")
        if not isinstance(position, Signal1D) and position.unit != "m":
            raise ValueError("position must be a Signal1D with 'm' as unit")
        for key, val in signals.items():
            if not isinstance(val, EMGSignal):
                raise ValueError(f"{key} must be an EMGSignal")

        # apply synchronization if required
        if not isinstance(synchronize_signals, bool):
            raise ValueError("synchronize_signals must be True or False.")
        if synchronize_signals:
            force, position, muscles = self._sync(
                force,
                position,
                **signals,
            )
        else:
            muscles = signals
        super().__init__(
            force=force,
            position=position,
            **muscles,
        )

        # set the class-specific attributes
        self.set_product(product)
        self.set_side(side)

    def set_product(self, product: BiostrengthProduct):
        if not isinstance(product, BiostrengthProduct):
            raise ValueError("product must be a BiostrengthProduct instance.")
        self._product = product

    def set_side(self, side: Literal["left", "right", "bilateral"]):
        if not isinstance(side, str) or side not in ["bilateral", "right", "left"]:
            raise ValueError("'side' must be any of 'bilateral', 'left', 'right'.")
        self._side = side

    @classmethod
    def from_txt(
        cls,
        filename: str,
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
        side: Literal["bilateral", "left", "right"],
    ):
        prod = BIOSTRENGTH_PRODUCTS_MAP[product].from_txt_file(filename)
        load_kgf = prod.load_lever_kgf
        time_s = prod.time_s
        pos_m = prod.position_lever_m
        force = Signal1D(
            load_kgf * G,
            time_s,  # type: ignore
            "N",
        )
        position = Signal1D(
            pos_m,
            time_s,  # type: ignore
            "m",
        )
        return cls(
            force=force,
            position=position,
            product=prod,
            side=side,
        )

    def to_plotly_figure(self):
        fig = super().to_plotly_figure()
        for r, repetition in enumerate(self.repetitions):
            df = repetition.to_dataframe()
            for i, (column, values) in enumerate(df.items()):
                fig.add_trace(
                    row=i + 1,
                    col=1,
                    trace=go.Scatter(
                        x=df.index.to_list(),
                        y=values.values.astype(float).flatten().tolist(),
                        name=f"repetition {r + 1}",
                        mode="lines",
                        showlegend=bool(i == 0),
                        legendgroup=f"repetition {r + 1}",
                    ),
                )
        return fig


class IsokineticRepetition(DefaultRepetition):
    """
    Isokinetic Test 1RM instance

    Parameters
    ----------
    time: Iterable[int | float]
        the array containing the time instant of each sample in seconds

    position: Iterable[int | float]
        the array containing the displacement of the handles for each sample

    load: Iterable[int | float]
        the array containing the load measured at each sample in kgf

    coefs_1rm: tuple[int | float, int | float]
        the b0 and b1 coefficients used to estimated the 1RM.

    Attributes
    ----------
    raw: DataFrame
        a DataFrame containing the input data

    repetitions: list[DataFrame]
        a list of dataframes each defining one single repetition

    product: BiostrengthProduct
        the product on which the test has been performed

    peak_load: float
        the peak load measured during the isokinetic repetitions

    rom0: float
        the start of the user's range of movement in meters

    rom1: float
        the end of the user's range of movement in meters

    rom: float
        the range of movement amplitude in meters

    results_table: DataFrame
        a table containing the data obtained during the test

    summary_table: DataFrame
        a table containing summary statistics about the test

    summary_plot: FigureWidget
        a figure representing the results of the test.
    """

    @property
    def estimated_1rm_kg(self):
        """return the predicted 1RM"""
        b1, b0 = self.product.rm1_coefs
        return self.peak_force_N / G * b1 + b0

    @property
    def output_metrics(self):
        """
        Returns summary metrics for the jump.

        Returns
        -------
        pd.DataFrame
            DataFrame with summary metrics for the jump.
        """
        new = super().output_metrics
        new.insert(
            new.shape[1],
            "estimated_1rm_kg",
            self.estimated_1rm_kg,
        )
        return new

    def __init__(
        self,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        **signals: EMGSignal,
    ):

        # check the input
        if not issubclass(product.__class__, BiostrengthProduct):
            raise ValueError("'product' must be a valid Biostrength Product.")
        if not side in ["bilateral", "left", "right"]:
            raise ValueError("'side' must be any of 'bilateral', 'left', 'right'")

        # check the required data
        if not isinstance(force, Signal1D) and force.unit != "N":
            raise ValueError("force must be a Signal1D with 'N' as unit")
        if not isinstance(position, Signal1D) and position.unit != "m":
            raise ValueError("position must be a Signal1D with 'm' as unit")
        for key, val in signals.items():
            if not isinstance(val, EMGSignal):
                raise ValueError(f"{key} must be an EMGSignal")

        super().__init__(
            product=product,
            side=side,
            force=force,
            position=position,
            **signals,
        )


class IsokineticExercise(DefaultExercise):
    """
    Isokinetic Test 1RM instance

    Parameters
    ----------
    time: Iterable[int | float]
        the array containing the time instant of each sample in seconds

    position: Iterable[int | float]
        the array containing the displacement of the handles for each sample

    load: Iterable[int | float]
        the array containing the load measured at each sample in kgf

    coefs_1rm: tuple[int | float, int | float]
        the b0 and b1 coefficients used to estimated the 1RM.

    Attributes
    ----------
    raw: DataFrame
        a DataFrame containing the input data

    repetitions: list[DataFrame]
        a list of dataframes each defining one single repetition

    product: BiostrengthProduct
        the product on which the test has been performed

    peak_load: float
        the peak load measured during the isokinetic repetitions

    rom0: float
        the start of the user's range of movement in meters

    rom1: float
        the end of the user's range of movement in meters

    rom: float
        the range of movement amplitude in meters

    results_table: DataFrame
        a table containing the data obtained during the test

    summary_table: DataFrame
        a table containing summary statistics about the test

    summary_plot: FigureWidget
        a figure representing the results of the test.
    """

    _repetition_type = IsokineticRepetition

    def _get_repetitions_splitting_signal(self):
        arr = self.position.to_numpy().flatten()
        time = self.position.index
        return arr, time

    def _get_repetitions_index(self, position: np.ndarray, time: np.ndarray):

        # check if position has to be inverted
        if abs(np.min(position)) > abs(np.max(position)):
            position *= -1

        # get the velocity
        velocity = winter_derivative1(position)
        velocity = np.concatenate([[velocity[0]], velocity, [velocity[-1]]])
        velocity = mean_filt(arr=velocity, order=301, offset=1)

        # get the batches where the position is within the 20-80% of the ROM and
        # positive velocity
        rom0 = np.min(position)
        rom1 = np.max(position)
        rom = (position - rom0) / (rom1 - rom0)
        concentric_phase_condition = (velocity > 0) & (rom > 0.2) & (rom < 0.8)
        batches = continuous_batches(concentric_phase_condition)
        if len(batches) == 0:
            raise RuntimeError("No repetitions have been found")

        # get the 3 (or less) repetitions with the greater ROM
        samples = np.argsort([np.max(position[i]) for i in batches])[::-1][:3]
        batches = [batches[i] for i in np.sort(samples)]

        # for each batch (i.e. candidate repetition):
        #   - get the start of each repetition looking at the last zero in the
        #     velocity readings occurring before the first sample of the batch
        #   - get the end of each repetition looing at the first zero in the
        #     velocity readings occurring after the last sample of the batch
        reps_idx: list[list[int]] = []
        for batch in batches:
            start = np.where(velocity[: batch[0]] <= 0)[0]
            start = 0 if len(start) == 0 else start[-1]
            stop = np.where(velocity[batch[-1] :] <= 0)[0]
            stop = len(velocity) if len(stop) == 0 else (stop[0] + batch[-1] + 1)
            reps_idx += [list(range(start, stop, 1))]
        return reps_idx

    def _get_repetitions_start_from_biostrength_data(
        self,
        force: np.ndarray,
        position: np.ndarray,
        time: np.ndarray,
    ):
        start = self._get_repetitions_index(position, time)
        if len(start) == 0:
            raise ValueError("No repetitions have been found in data.")
        return int(start[0][0])

    def __init__(
        self,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        synchronize_signals: bool = True,
        **signals: EMGSignal,
    ):
        super().__init__(
            product=product,
            side=side,
            force=force,
            position=position,
            synchronize_signals=synchronize_signals,
            **signals,
        )


class IsometricRepetition(DefaultRepetition):
    """
    Isokinetic Test 1RM instance

    Parameters
    ----------
    time: Iterable[int | float]
        the array containing the time instant of each sample in seconds

    position: Iterable[int | float]
        the array containing the displacement of the handles for each sample

    load: Iterable[int | float]
        the array containing the load measured at each sample in kgf

    coefs_1rm: tuple[int | float, int | float]
        the b0 and b1 coefficients used to estimated the 1RM.

    Attributes
    ----------
    raw: DataFrame
        a DataFrame containing the input data

    repetitions: list[DataFrame]
        a list of dataframes each defining one single repetition

    product: BiostrengthProduct
        the product on which the test has been performed

    peak_load: float
        the peak load measured during the isokinetic repetitions

    rom0: float
        the start of the user's range of movement in meters

    rom1: float
        the end of the user's range of movement in meters

    rom: float
        the range of movement amplitude in meters

    results_table: DataFrame
        a table containing the data obtained during the test

    summary_table: DataFrame
        a table containing summary statistics about the test

    summary_plot: FigureWidget
        a figure representing the results of the test.
    """

    @property
    def rate_of_force_development(self):
        force = self.force.to_numpy().flatten()
        time = np.array(self.index)
        peaks = find_peaks(force, height=float(np.max(force) * 0.8))
        peak = np.argmax(force) if len(peaks) == 0 else peaks[0]
        return (force[peak] - force[0]) / (time[peak] - time[0])

    @property
    def time_to_peak_force(self):
        force = self.force.to_numpy().flatten()
        time = np.array(self.index)
        peak = np.argmax(force)
        return time[peak] - time[0]

    @property
    def output_metrics(self):
        """
        Returns summary metrics for the jump.

        Returns
        -------
        pd.DataFrame
            DataFrame with summary metrics for the jump.
        """
        new = super().output_metrics
        new.insert(
            new.shape[1],
            "rate_of_force_development_N/s",
            self.rate_of_force_development,
        )
        new.insert(
            new.shape[1],
            "time_to_peak_force_ms",
            self.time_to_peak_force * 1000,
        )
        return new

    def __init__(
        self,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        **signals: EMGSignal,
    ):

        # check the input
        if not issubclass(product.__class__, BiostrengthProduct):
            raise ValueError("'product' must be a valid Biostrength Product.")
        if not side in ["bilateral", "left", "right"]:
            raise ValueError("'side' must be any of 'bilateral', 'left', 'right'")

        # check the required data
        if not isinstance(force, Signal1D) and force.unit != "N":
            raise ValueError("force must be a Signal1D with 'N' as unit")
        if not isinstance(position, Signal1D) and position.unit != "m":
            raise ValueError("position must be a Signal1D with 'm' as unit")
        for key, val in signals.items():
            if not isinstance(val, EMGSignal):
                raise ValueError(f"{key} must be an EMGSignal")

        super().__init__(
            product=product,
            side=side,
            force=force,
            position=position,
            **signals,
        )


class IsometricExercise(DefaultExercise):
    """
    Isokinetic Test 1RM instance

    Parameters
    ----------
    time: Iterable[int | float]
        the array containing the time instant of each sample in seconds

    position: Iterable[int | float]
        the array containing the displacement of the handles for each sample

    load: Iterable[int | float]
        the array containing the load measured at each sample in kgf

    coefs_1rm: tuple[int | float, int | float]
        the b0 and b1 coefficients used to estimated the 1RM.

    Attributes
    ----------
    raw: DataFrame
        a DataFrame containing the input data

    repetitions: list[DataFrame]
        a list of dataframes each defining one single repetition

    product: BiostrengthProduct
        the product on which the test has been performed

    peak_load: float
        the peak load measured during the isokinetic repetitions

    rom0: float
        the start of the user's range of movement in meters

    rom1: float
        the end of the user's range of movement in meters

    rom: float
        the range of movement amplitude in meters

    results_table: DataFrame
        a table containing the data obtained during the test

    summary_table: DataFrame
        a table containing summary statistics about the test

    summary_plot: FigureWidget
        a figure representing the results of the test.
    """

    _repetition_type = IsometricRepetition

    def _get_repetitions_splitting_signal(self):
        arr = self.force.to_numpy().flatten()
        time = self.force.index
        return arr, time

    def _get_repetitions_index(self, force: np.ndarray, time: np.ndarray):

        # check if position has to be inverted
        arr = force.flatten()
        if abs(np.min(arr)) > abs(np.max(arr)):
            arr *= -1

        # get the continuous batches with length higher than 3 seconds
        fsamp = int(1 / np.mean(np.diff(time)))
        batches = continuous_batches(
            arr > np.max(arr) * 0.5,
            tolerance=int(0.1 * fsamp),
        )
        batches = [i for i in batches if len(i) > 3 * fsamp]

        # get the repetitions
        reps: list[list[int]] = []
        for batch in batches:

            # for each batch get the start of the repetition as the last point
            # before the start of the batch without having increments in force
            start = batch[0]
            while start > 0 and arr[start - 1] < arr[start]:
                start -= 1

            # for each batch get the end of the repetition as the first point
            # before the end of the batch without having decrements in force
            stop = batch[-1]
            while stop < len(time) - 1 and arr[stop + 1] < arr[stop]:
                stop += 1

            # add the repetition index
            reps += [list(range(start, stop + 1, 1))]

        return reps

    def _get_repetitions_start_from_biostrength_data(
        self,
        force: np.ndarray,
        position: np.ndarray,
        time: np.ndarray,
    ):
        return self._get_repetitions_index(force, time)[0][0]

    def __init__(
        self,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        synchronize_signals: bool = True,
        **signals: EMGSignal,
    ):
        super().__init__(
            product=product,
            side=side,
            force=force,
            position=position,
            synchronize_signals=synchronize_signals,
            **signals,
        )
