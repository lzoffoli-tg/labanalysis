"""jump analysis testing for SingleJump, JumpExercise, and DropJump"""

import sys
from datetime import datetime
from os.path import abspath, dirname, join

import pandas as pd
import numpy as np
from plotly.graph_objects import Figure

# add project root to path like other tests do
sys.path.append(dirname(dirname(abspath(__file__))))

from src.labanalysis.protocols.balancetests import PlankBalanceTest, UprightBalanceTest
from src.labanalysis.protocols.normativedata import (
    plankbalance_normative_values,
    uprightbalance_normative_values,
)
from src.labanalysis.protocols.protocols import Participant
from src.labanalysis.records.records import TimeseriesRecord

participant = Participant(recordingdate=datetime.now())

balance_path = join(dirname(__file__), "assets", "balance_data")
emg_normalization_path = join(balance_path, "stabilità_bilaterale_occhiaperti.tdf")
emg_norm_data = TimeseriesRecord.from_tdf(emg_normalization_path)
emg_norm_data = emg_norm_data.emgsignals


def check_balancetest(test: UprightBalanceTest | PlankBalanceTest):
    results = test.get_results()
    assert hasattr(results, "summary")
    assert isinstance(results.summary, pd.DataFrame)
    assert hasattr(results, "analytics")
    assert isinstance(results.summary, pd.DataFrame)
    assert hasattr(results, "figures")
    assert isinstance(results.figures, dict)
    figures = list(results.figures.values())[0]
    assert all([isinstance(f, Figure) for f in results.figures.values()])


def test_upright_bilateral_eyesopen():
    test = UprightBalanceTest.from_files(
        join(balance_path, "stabilità_bilaterale_occhiaperti.tdf"),
        participant,
        "open",
        "LFOOT_FP",
        "RFOOT_FP",
        uprightbalance_normative_values,
        emg_normalization_references=emg_norm_data,
        emg_normalization_function=np.max,
    )
    check_balancetest(test)
