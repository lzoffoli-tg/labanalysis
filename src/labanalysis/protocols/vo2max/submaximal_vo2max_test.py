"""Submaximal VO2max test implementation."""

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from ...equations import Bike, Run
from ...records import MetabolicRecord, TimeseriesRecord
from ...pipelines import get_default_processing_pipeline
from ..participant import Participant
from ..test_protocol import TestProtocol
from .submaximal_vo2max_test_results import SubmaximalVO2MaxTestResults


class SubmaximalVO2MaxTest(TestProtocol):

    def __init__(
        self,
        participant: Participant,
        metabolic_record: MetabolicRecord,
        normative_data: pd.DataFrame = pd.DataFrame(),
    ):
        super().__init__(
            participant,
            normative_data,
        )
        self.set_metabolic_record(metabolic_record)

    def set_metabolic_record(self, record: MetabolicRecord):
        if not isinstance(record, MetabolicRecord):
            raise ValueError("record must be a MetabolicRecord instance.")
        self._metabolic_record = record

    @property
    def metabolic_record(self):
        return self._metabolic_record

    def copy(self):
        return SubmaximalVO2MaxTest(
            participant=self.participant.copy(),
            normative_data=self.normative_data,
            metabolic_record=self.metabolic_record,
        )

    @classmethod
    def from_files(
        cls,
        filename: str,
        participant: Participant,
        normative_data: pd.DataFrame = pd.DataFrame(),
        breath_by_breath: bool = False,
    ):
        return cls(
            participant=participant,
            normative_data=normative_data,
            metabolic_record=MetabolicRecord.from_file(
                filename=filename,
                breath_by_breath=breath_by_breath,
            ),
        )

    def get_results(self):
        return SubmaximalVO2MaxTestResults(self.processed_data)

    @property
    def processed_data(self):
        out = self.copy()
        pipeline = self.processing_pipeline
        pipeline(out.metabolic_record, inplace=True)
        return out

    @property
    def processing_pipeline(self):
        return get_default_processing_pipeline()


__all__ = ["SubmaximalVO2MaxTest"]
