"""
Test suite for continuous_batches function.

Tests verify identification of contiguous True sequences in boolean arrays
with optional tolerance-based merging.
"""

import numpy as np
import pytest

from labanalysis.signalprocessing import continuous_batches


def test_continuous_batches_simple():
    """
    Test batch detection on simple boolean signal.

    Expected:
        Signal [False, True, True, False, False, True, True, True]
        should produce two batches: [1, 2] and [5, 6, 7]
    """
    signal = np.array([False, True, True, False, False, True, True, True])
    batches = continuous_batches(signal)

    assert len(batches) == 2
    assert batches[0] == [1, 2]
    assert batches[1] == [5, 6, 7]


def test_continuous_batches_with_tolerance():
    """
    Test batch merging with tolerance parameter.

    Expected:
        With tolerance=3, batches separated by gap of 3 (distance 5-2=3)
        should be merged into single batch
    """
    signal = np.array([False, True, True, False, False, True, True, True])
    batches = continuous_batches(signal, tolerance=3)

    # Two batches separated by gap=3 should merge with tolerance=3
    assert len(batches) == 1
    assert set(batches[0]) == {1, 2, 5, 6, 7}


def test_continuous_batches_no_gaps():
    """
    Test batch detection on continuous True signal.

    Expected:
        All True signal should produce single batch with all indices
    """
    signal = np.ones(10, dtype=bool)
    batches = continuous_batches(signal)

    assert len(batches) == 1
    assert batches[0] == list(range(10))


def test_continuous_batches_all_false():
    """
    Test batch detection on all False signal.

    Expected:
        Should return empty list (no batches)
    """
    signal = np.zeros(10, dtype=bool)
    batches = continuous_batches(signal)

    assert len(batches) == 0


def test_continuous_batches_single_sample():
    """
    Test batch detection with single True sample.

    Expected:
        Should detect single-element batch
    """
    signal = np.array([False, False, True, False, False])
    batches = continuous_batches(signal)

    assert len(batches) == 1
    assert batches[0] == [2]


def test_continuous_batches_alternating():
    """
    Test batch detection on alternating True/False signal.

    Expected:
        Should detect individual single-element batches
    """
    signal = np.array([True, False, True, False, True])
    batches = continuous_batches(signal)

    assert len(batches) == 3
    assert batches[0] == [0]
    assert batches[1] == [2]
    assert batches[2] == [4]


def test_continuous_batches_tolerance_exact():
    """
    Test that tolerance merges batches at exact threshold.

    Expected:
        Batches separated by exactly tolerance distance should merge
    """
    # Two batches: [0,1] ends at 1, [3,4] starts at 3, distance = 3-1 = 2
    signal = np.array([True, True, False, True, True])
    batches = continuous_batches(signal, tolerance=2)

    # Distance is exactly 2, should merge with tolerance=2
    assert len(batches) == 1
    assert batches[0] == [0, 1, 3, 4]


def test_continuous_batches_tolerance_no_merge():
    """
    Test that tolerance does NOT merge batches beyond threshold.

    Expected:
        Batches separated by more than tolerance should remain separate
    """
    # Two batches separated by 2 False (gap = 2)
    signal = np.array([True, True, False, False, True, True])
    batches = continuous_batches(signal, tolerance=1)

    # Gap is 2, tolerance is 1, should NOT merge
    assert len(batches) == 2
    assert batches[0] == [0, 1]
    assert batches[1] == [4, 5]


def test_continuous_batches_start_with_true():
    """
    Test batch detection when signal starts with True.

    Expected:
        Should correctly detect batch starting at index 0
    """
    signal = np.array([True, True, False, False])
    batches = continuous_batches(signal)

    assert len(batches) == 1
    assert batches[0] == [0, 1]


def test_continuous_batches_end_with_true():
    """
    Test batch detection when signal ends with True.

    Expected:
        Should correctly detect batch ending at last index
    """
    signal = np.array([False, False, True, True])
    batches = continuous_batches(signal)

    assert len(batches) == 1
    assert batches[0] == [2, 3]


def test_continuous_batches_gait_application():
    """
    Test batch detection on simulated gait contact signal.

    Expected:
        Should detect stance phases (continuous contact periods)
    """
    # Simulated force platform signal: 2 stance phases
    contact = np.array([
        False, False,  # flight
        True, True, True, True, True,  # stance 1
        False, False,  # flight
        True, True, True, True,  # stance 2
        False, False  # flight
    ])
    batches = continuous_batches(contact)

    assert len(batches) == 2
    assert batches[0] == [2, 3, 4, 5, 6]
    assert batches[1] == [9, 10, 11, 12]


def test_continuous_batches_integer_input():
    """
    Test that function handles integer arrays (0/1) correctly.

    Expected:
        Integer array [0, 1, 1, 0] should work like boolean
    """
    signal = np.array([0, 1, 1, 0, 0, 1], dtype=int)
    batches = continuous_batches(signal)

    assert len(batches) == 2
    assert batches[0] == [1, 2]
    assert batches[1] == [5]
