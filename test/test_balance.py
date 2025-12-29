"""jump analysis testing for SingleJump, JumpExercise, and DropJump"""

import sys
from datetime import datetime
from os.path import abspath, dirname, join

import pandas as pd
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

plank_path = join(
    dirname(__file__),
    "assets",
    "balance_data",
    "plank_occhi_aperti.tdf",
)

singleleg_path = join(
    dirname(__file__),
    "assets",
    "balance_data",
    "stabilità_monopodalico_dx_occhi_aperti.tdf",
)

bilateral_eyesopen_path = join(
    dirname(__file__),
    "assets",
    "balance_data",
    "stabilità_bipodalico_occhi_aperti.tdf",
)

bilateral_eyesshut_path = join(
    dirname(__file__),
    "assets",
    "balance_data",
    "stabilità_bipodalico_occhi_chiusi.tdf",
)

emg_normalization_path = join(
    dirname(__file__),
    "assets",
    "balance_data",
    "normalization_data.tdf",
)
emg_norm_data = TimeseriesRecord.from_tdf(emg_normalization_path)
emg_norm_data = emg_norm_data.emgsignals


def test_upright_bilateral_eyesopen():

    # generate the tests
    test = UprightBalanceTest.from_files(
        bilateral_eyesopen_path,
        participant,
        "open",
        "left_frz",
        "right_frz",
        uprightbalance_normative_values,
        emg_normalization_references=emg_norm_data,
    )

    # check results
    results = test.get_results()
    assert hasattr(results, "summary")
    assert isinstance(results.summary, pd.DataFrame)
    assert hasattr(results, "analytics")
    assert isinstance(results.summary, pd.DataFrame)
    assert hasattr(results, "figures")
    assert isinstance(results.figures, dict)
    figures = list(results.figures.values())[0]
    assert all([isinstance(f, Figure) for f in results.figures.values()])


def test_upright_bilateral_eyesshut():

    # generate the tests
    test = UprightBalanceTest.from_files(
        bilateral_eyesshut_path,
        participant,
        "closed",
        "left_frz",
        "right_frz",
        uprightbalance_normative_values,
        emg_normalization_references=emg_norm_data,
    )

    # check results
    results = test.get_results()
    assert hasattr(results, "summary")
    assert isinstance(results.summary, pd.DataFrame)
    assert hasattr(results, "analytics")
    assert isinstance(results.summary, pd.DataFrame)
    assert hasattr(results, "figures")
    assert isinstance(results.figures, dict)
    figures = list(results.figures.values())[0]
    assert all([isinstance(f, Figure) for f in results.figures.values()])


def test_upright_monolateral():

    # generate the tests
    test = UprightBalanceTest.from_files(
        singleleg_path,
        participant,
        "open",
        None,
        "right_frz",
        uprightbalance_normative_values,
        emg_normalization_references=emg_norm_data,
    )

    # check results
    results = test.get_results()
    assert hasattr(results, "summary")
    assert isinstance(results.summary, pd.DataFrame)
    assert hasattr(results, "analytics")
    assert isinstance(results.summary, pd.DataFrame)
    assert hasattr(results, "figures")
    assert isinstance(results.figures, dict)
    figures = list(results.figures.values())[0]
    assert all([isinstance(f, Figure) for f in results.figures.values()])


def test_plank():

    # generate the tests
    test = PlankBalanceTest.from_files(
        plank_path,
        participant,
        "open",
        "l_pst_frz",
        "r_pst_frz",
        "l_ant_frz",
        "r_ant_frz",
        plankbalance_normative_values,
        emg_normalization_references=emg_norm_data,
    )

    # check results
    results = test.get_results()
    assert hasattr(results, "summary")
    assert isinstance(results.summary, pd.DataFrame)
    assert hasattr(results, "analytics")
    assert isinstance(results.summary, pd.DataFrame)
    assert hasattr(results, "figures")
    assert isinstance(results.figures, dict)
    figures = list(results.figures.values())[0]
    assert all([isinstance(f, Figure) for f in results.figures.values()])
