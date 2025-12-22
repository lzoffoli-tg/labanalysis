"""Tests for Isokinetic1RMTest and Isokinetic1RMTestResults"""

import sys
from datetime import datetime
from os.path import abspath, dirname, join

import pandas as pd
from plotly.graph_objects import Figure

# add project root to path like other tests do
sys.path.append(dirname(dirname(abspath(__file__))))

from src.labanalysis.records.records import TimeseriesRecord
from src.labanalysis.io.read.biostrength import PRODUCTS
from src.labanalysis import Isokinetic1RMTest, Participant, IsometricTest
from src.labanalysis.protocols.normativedata.normative_data import (
    isok_1rm_normative_values,
)


def isok_path(filename: str):
    return join(
        dirname(__file__),
        "assets",
        "isokinetic1rmtest_data",
        "legextensionrev",
        filename,
    )


def isom_path(filename: str):
    return join(
        dirname(__file__),
        "assets",
        "isometrictest_data",
        "legpressrev",
        filename,
    )


def participant():
    # minimal participant for constructing the test
    return Participant(
        surname="Test",
        name="Subject",
        height=180,
        weight=75,
        recordingdate=datetime.now(),
    )


def test_isokinetic1rmtest():
    """Test loading right leg extension isokinetic data from txt file"""
    re_path = isok_path("leg_extension_rev_isok_dx")
    le_path = isok_path("leg_extension_rev_isok_sx")
    product = "LEG EXTENSION REV"
    rm1_coefs = PRODUCTS[product]._rm1_coefs
    rm1_coefs = {i: v for i, v in zip(["beta1", "beta0"], rm1_coefs)}
    test = Isokinetic1RMTest.from_files(
        participant(),
        product,
        le_path + ".txt",
        re_path + ".txt",
        None,
        le_path + ".tdf",
        re_path + ".tdf",
        None,
        isok_1rm_normative_values,
        emg_normalization_references=TimeseriesRecord(),
        emg_activation_references=TimeseriesRecord(),
        emg_activation_threshold=3,
        relevant_muscle_map=["biceps femoris", "vastus medialis"],
    )

    # Verify test instance and basic properties
    assert isinstance(test, Isokinetic1RMTest)

    # Verify repetitions were extracted
    assert hasattr(test.left, "repetitions")
    assert isinstance(test.left.repetitions, list)  # type: ignore
    assert isinstance(test.right.repetitions, list)  # type: ignore
    assert len(test.left.repetitions) > 0, "At least one repetition should be detected"  # type: ignore
    assert len(test.right.repetitions) > 0, "At least one repetition should be detected"  # type: ignore

    # test the results
    assert hasattr(test, "results")
    res = test.results

    # check the summary
    assert hasattr(res, "summary")
    summary = res.summary
    assert isinstance(summary, pd.DataFrame)

    # check the analytics
    assert hasattr(res, "analytics")
    analytics = res.analytics
    assert isinstance(analytics, pd.DataFrame)

    # check the figures
    assert hasattr(res, "figures")
    figures = res.figures
    assert isinstance(figures, dict)
    assert "force_profiles_with_muscle_balance" in list(figures.keys())
    figures = figures["force_profiles_with_muscle_balance"]
    assert isinstance(figures, Figure)


def test_isometrictest():
    """Test loading right leg extension isometric data from txt file"""
    re_path = isom_path("leg_press_rev_isom_dx")
    le_path = isom_path("leg_press_rev_isom_sx")
    product = "LEG PRESS REV"
    test = IsometricTest.from_files(
        participant(),
        product,
        le_path + ".txt",
        re_path + ".txt",
        None,
        le_path + ".tdf",
        re_path + ".tdf",
        None,
        relevant_muscle_map=["vastus medialis", "biceps femoris"],
    )

    # Verify test instance and basic properties
    assert isinstance(test, IsometricTest)

    # Verify repetitions were extracted
    assert hasattr(test.left, "repetitions")
    assert isinstance(test.left.repetitions, list)  # type: ignore
    assert isinstance(test.right.repetitions, list)  # type: ignore
    assert len(test.left.repetitions) > 0, "At least one repetition should be detected"  # type: ignore
    assert len(test.right.repetitions) > 0, "At least one repetition should be detected"  # type: ignore

    # test the results
    assert hasattr(test, "results")
    res = test.results

    # check the summary
    assert hasattr(res, "summary")
    summary = res.summary
    assert isinstance(summary, pd.DataFrame)

    # check the analytics
    assert hasattr(res, "analytics")
    analytics = res.analytics
    assert isinstance(analytics, pd.DataFrame)

    # check the figures
    assert hasattr(res, "figures")
    figures = res.figures
    assert isinstance(figures, dict)
    assert "force_profiles_with_muscle_balance" in list(figures.keys())
    figures = figures["force_profiles_with_muscle_balance"]
    assert isinstance(figures, Figure)
