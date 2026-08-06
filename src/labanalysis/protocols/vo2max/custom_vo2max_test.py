"""Custom VO2max test implementation."""

import pandas as pd

from ...pipelines import ProcessingPipeline
from .custom_vo2max_test_results import CustomVO2MaxTestResults
from ..participant import Participant
from ..test_protocol import TestProtocol
from ...normative_data.normative_data import vo2max_normative_values


class CustomVO2MaxTest(TestProtocol):
    """
    Custom VO2Max test that handles results regardless of the specific protocol used.
    This class is designed to process metabolic data from various VO2max testing protocols,
    including both submaximal and maximal tests, and generate comprehensive cardiorespiratory
    fitness reports.

    Parameters
    ----------
    participant : Participant
        Participant information including demographics and anthropometrics.
        Must have age/birthdate (for HRmax calculation) and weight (for
        power/speed predictions).
    vo2max : float
        the measured or predicted VO2max value in ml/kg/min.
    walking_speed: float, optional
        The walking speed in km/h, required if movement is "WALKING".
    normative_data : pd.DataFrame, optional
        Reference data for fitness level classification.
    """

    def __init__(
        self,
        participant: Participant,
        vo2max: float,
        walking_speed: float = 4.0,
        normative_data: pd.DataFrame = vo2max_normative_values,
    ):
        super().__init__(
            participant,
            normative_data,
        )
        self.set_vo2max(vo2max)
        self.set_walking_speed(walking_speed)

    def set_vo2max(self, vo2max_value: float):
        if not isinstance(vo2max_value, (int, float)):
            raise TypeError("VO2max value must be a number.")
        if vo2max_value <= 0:
            raise ValueError("VO2max value must be a positive number.")
        self._vo2max = float(vo2max_value)

    def set_walking_speed(self, walking_speed: float):
        if not isinstance(walking_speed, (int, float)):
            raise TypeError("Walking speed must be a number.")
        if walking_speed <= 0:
            raise ValueError("Walking speed must be a positive number.")
        self._walking_speed = float(walking_speed)

    @property
    def vo2max(self):
        return self._vo2max

    @property
    def walking_speed(self):
        return self._walking_speed

    def update_results(self, *args, **kwargs):
        self._results = CustomVO2MaxTestResults(self.processed_data)

    @property
    def processed_data(self):
        return self.copy()

    @property
    def processing_pipeline(self):
        return ProcessingPipeline()


__all__ = ["CustomVO2MaxTest"]
