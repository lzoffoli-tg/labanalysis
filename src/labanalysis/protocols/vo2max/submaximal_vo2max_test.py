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
    """
    Test protocol for estimating maximal oxygen consumption from submaximal exercise.

    SubmaximalVO2MaxTest analyzes incremental exercise data to predict VO2max
    without requiring maximal effort testing. The class processes metabolic data
    from gas exchange analysis, extracts ventilatory thresholds, estimates aerobic
    capacity, and generates comprehensive cardiorespiratory fitness reports.

    The protocol supports:
    - VO2max prediction from heart rate and respiratory exchange ratio (RER)
    - Ventilatory threshold (VT2) detection from VCO2/VO2 relationship
    - FatMax calculation (maximal fat oxidation rate)
    - Automated metabolic data processing
    - Normative data comparison for fitness classification

    Parameters
    ----------
    participant : Participant
        Participant information including demographics and anthropometrics.
        Must have age/birthdate (for HRmax calculation) and weight (for
        power/speed predictions).
    metabolic_record : MetabolicRecord
        Gas exchange data including VO2, VCO2, HR, and derived metrics
        from metabolic cart or breath-by-breath analysis.
    normative_data : pd.DataFrame, optional
        Reference data for fitness level classification. Default is empty DataFrame.

    Attributes
    ----------
    metabolic_record : MetabolicRecord
        Metabolic data container with VO2, VCO2, HR, RQ, and oxidation rates.
    participant : Participant
        Participant demographics and anthropometrics.
    normative_data : pd.DataFrame
        Reference data for normative comparisons.
    processed_data : SubmaximalVO2MaxTest
        Copy of test with all signals processed through the pipeline.
    processing_pipeline : ProcessingPipeline
        Signal processing pipeline (15-point moving average for metabolic data).

    Methods
    -------
    copy()
        Return a copy of the test protocol.
    get_results()
        Process data and return SubmaximalVO2MaxTestResults.
    set_metabolic_record(record)
        Set metabolic data for analysis.
    from_files(filename, participant, ...)
        Load test from metabolic cart output file.

    Notes
    -----
    VO2max Prediction Methods:
    1. **RER-based**: Uses RER > 0.832 to calculate %VO2max from RER via
       Beck et al. (2018) equation: %VO2max = sqrt(2*RER - 1.664) + 0.301
    2. **HR-based**: Extrapolates VO2-HR relationship to predicted HRmax
       (Gellish: 207 - 0.7*age) using data where RER > 0.95
    3. **Final estimate**: min(RER_method, HR_method) for conservative prediction

    Ventilatory Threshold (VT2):
    Detected as the point where VCO2/VO2 curve crosses the identity line
    (VCO2 = VO2), indicating respiratory compensation point. Solved using
    third-order polynomial fit and symbolic equation solving.

    FatMax:
    Maximum fat oxidation rate (g/min) calculated from RER and VO2 using
    Frayn (1983) stoichiometric equations. Reported as absolute (g/min) and
    relative to body mass (g/kg/min).

    Processing Pipeline:
    - 15-point moving average smoothing for all metabolic signals
    - Preserves breath-by-breath or averaged data structure
    - No filtering of cardiac or respiratory data

    Examples
    --------
    >>> from labanalysis.protocols import SubmaximalVO2MaxTest, Participant
    >>>
    >>> # Create participant (age and weight required)
    >>> participant = Participant(
    ...     surname='Runner',
    ...     age=30,
    ...     weight=70,
    ...     gender='Male'
    ... )
    >>>
    >>> # Load test from metabolic cart file
    >>> test = SubmaximalVO2MaxTest.from_files(
    ...     filename='incremental_test.xlsx',
    ...     participant=participant,
    ...     breath_by_breath=False  # Averaged data
    ... )
    >>>
    >>> # Get results with VO2max prediction
    >>> results = test.get_results()
    >>> print(f"Predicted VO2max: {results.summary['vo2max_ml_kg_min'].iloc[0]:.1f} ml/kg/min")
    >>> print(f"VT2: {results.summary['vt2_vo2_ml_kg_min'].iloc[0]:.1f} ml/kg/min")
    >>> print(f"FatMax: {results.summary['fatmax_g_min'].iloc[0]:.2f} g/min")

    See Also
    --------
    SubmaximalVO2MaxTestResults : Results container for VO2max tests.
    MetabolicRecord : Container for gas exchange data.
    TestProtocol : Parent class for test protocols.

    References
    ----------
    .. [1] Beck ON, Kipp SK, Byrnes WC, Kram R. Use aerobic energy expenditure
       instead of oxygen uptake to quantify exercise intensity and predict
       endurance performance. J Appl Physiol 125: 672-674, 2018.
    .. [2] Gellish RL, Goslin BR, Olson RE, et al. Longitudinal modeling of
       the relationship between age and maximal heart rate. Med Sci Sports
       Exerc 39: 822-829, 2007.
    .. [3] Frayn KN. Calculation of substrate oxidation rates in vivo from
       gaseous exchange. J Appl Physiol 55: 628-634, 1983.
    """

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
