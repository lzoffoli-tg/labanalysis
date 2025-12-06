"""
isokinetic test module
"""

#! IMPORTS

from typing import Literal

import pandas as pd

from ..constants import G
from ..records import IsokineticExercise, IsometricExercise, EMGSignal, Signal1D
from ..io.read.biostrength import PRODUCTS, BiostrengthProduct
from .normativedata import isok_1rm_normative_values
from .protocols import Participant, TestProtocol

#! CONSTANTS


__all__ = ["Isokinetic1RMTest", "IsometricTest"]

#! CLASSES


class Isokinetic1RMTest(IsokineticExercise, TestProtocol):

    def __init__(
        self,
        participant: Participant,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        normative_data: pd.DataFrame = isok_1rm_normative_values,
        synchronize_signals: bool = True,
        **extra_signals: EMGSignal,
    ):
        super().__init__(
            product=product,
            side=side,
            force=force,
            position=position,
            synchronize_signals=synchronize_signals,
            **extra_signals,
        )
        self.set_participant(participant)
        self.set_normative_data(normative_data)

    def copy(self):
        return Isokinetic1RMTest(
            participant=self.participant.copy(),
            product=self.product.copy(),
            side=self.side,
            normative_data=self.normative_data,
            synchronize_signals=False,
            force=self.force,  # type: ignore
            position=self.position,  # type: ignore
            **{i: v.copy() for i, v in self.items() if isinstance(v, EMGSignal)},
        )

    @property
    def results(self):
        analytics = []
        metrics = []
        for i, rep in enumerate(self.repetitions):

            # summary statistics
            line = rep.output_metrics
            line.insert(0, "repetition", i + 1)
            metrics += [line]

            # analytics
            cycle = rep.to_dataframe()
            cycle.columns = cycle.columns.map(lambda x: x.replace("_", " "))
            cycle.insert(0, "time_s", cycle.index)
            cycle.insert(0, "repetition", i + 1)
            cycle.insert(0, "side", rep.side)
            analytics += [cycle]

        # outcomes
        return {
            "summary": pd.concat(metrics, ignore_index=True),
            "analytics": pd.concat(analytics, ignore_index=True),
        }

    @classmethod
    def from_txt(
        cls,
        filename: str,
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
        side: Literal["bilateral", "left", "right"],
        normative_data: pd.DataFrame = isok_1rm_normative_values,
    ):
        prod = PRODUCTS[product].from_txt_file(filename)
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
            participant=participant,
            normative_data=normative_data,
            force=force,
            position=position,
            product=prod,
            side=side,
        )


class IsometricTest(IsometricExercise, TestProtocol):

    def __init__(
        self,
        participant: Participant,
        product: BiostrengthProduct,
        side: Literal["bilateral", "left", "right"],
        force: Signal1D,
        position: Signal1D,
        normative_data: pd.DataFrame = pd.DataFrame(),
        synchronize_signals: bool = True,
        **extra_signals: EMGSignal,
    ):
        super().__init__(
            product=product,
            side=side,
            force=force,
            position=position,
            synchronize_signals=synchronize_signals,
            **extra_signals,
        )
        self.set_participant(participant)
        self.set_normative_data(normative_data)

    def copy(self):
        return IsometricTest(
            participant=self.participant.copy(),
            product=self.product.copy(),
            side=self.side,
            normative_data=self.normative_data,
            synchronize_signals=False,
            force=self.force,  # type: ignore
            position=self.position,  # type: ignore
            **{i: v.copy() for i, v in self.items() if isinstance(v, EMGSignal)},
        )

    @property
    def results(self):
        analytics = []
        metrics = []
        for i, rep in enumerate(self.repetitions):

            # summary statistics
            line = rep.output_metrics
            line.insert(0, "repetition", i + 1)
            metrics += [line]

            # analytics
            cycle = rep.to_dataframe()
            cycle.columns = cycle.columns.map(lambda x: x.replace("_", " "))
            cycle.insert(0, "time_s", cycle.index)
            cycle.insert(0, "repetition", i + 1)
            cycle.insert(0, "side", rep.side)
            analytics += [cycle]

        # outcomes
        return {
            "summary": pd.concat(metrics, ignore_index=True),
            "analytics": pd.concat(analytics, ignore_index=True),
        }

    @classmethod
    def from_txt(
        cls,
        filename: str,
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
        side: Literal["bilateral", "left", "right"],
        normative_data: pd.DataFrame = pd.DataFrame(),
    ):
        prod = PRODUCTS[product].from_txt_file(filename)
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
            participant=participant,
            normative_data=normative_data,
            force=force,
            position=position,
            product=prod,
            side=side,
        )
