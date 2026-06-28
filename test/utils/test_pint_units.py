"""
Test suite for pint unit registry setup.

Tests verify custom units (bpm, au) and unit registry configuration.
"""

import pint
import pytest

from labanalysis import utils


def test_pint_ureg_defined():
    """
    Test that pint unit registry is initialized.

    Expected:
        ureg should be a pint UnitRegistry with custom units defined
    """
    assert hasattr(utils, 'ureg')
    assert isinstance(utils.ureg, pint.UnitRegistry)


def test_bpm_quantity_defined():
    """
    Test that bpm (beats per minute) quantity is defined.

    Expected:
        bpm_quantity should be 1 bpm with correct units
    """
    assert hasattr(utils, 'bpm_quantity')
    assert utils.bpm_quantity.magnitude == 1
    assert 'bpm' in str(utils.bpm_quantity.units)


def test_au_quantity_defined():
    """
    Test that au (arbitrary units) quantity is defined.

    Expected:
        au_quantity should be 1 au with correct units
    """
    assert hasattr(utils, 'au_quantity')
    assert utils.au_quantity.magnitude == 1
    assert 'au' in str(utils.au_quantity.units)


def test_Q_shortcut_defined():
    """
    Test that Q_ shortcut for Quantity is defined.

    Expected:
        Q_ should be callable and create Quantity objects
    """
    assert hasattr(utils, 'Q_')
    assert callable(utils.Q_)


def test_bpm_unit_arithmetic():
    """
    Test that bpm units work in arithmetic operations.

    Expected:
        Should be able to perform calculations with bpm units
    """
    hr_150 = 150 * utils.ureg.bpm
    hr_180 = 180 * utils.ureg.bpm

    # Addition should work
    hr_sum = hr_150 + 30 * utils.ureg.bpm
    assert hr_sum.magnitude == 180

    # Comparison should work
    assert hr_180 > hr_150


def test_au_unit_dimensionless():
    """
    Test that au is dimensionless.

    Expected:
        au should be dimensionless (like a count)
    """
    value = 100 * utils.ureg.au
    assert value.dimensionless


def test_beat_unit_defined():
    """
    Test that 'beat' base unit is defined.

    Expected:
        Should be able to create beat quantities
    """
    beats = 60 * utils.ureg.beat
    assert beats.magnitude == 60


def test_bpm_to_hz_conversion():
    """
    Test conversion from bpm to Hz (beats per second).

    Expected:
        60 bpm should equal 1 Hz
    """
    hr_60 = 60 * utils.ureg.bpm
    hr_hz = hr_60.to(utils.ureg.Hz)

    assert abs(hr_hz.magnitude - 1.0) < 0.01


def test_Q_creates_quantities():
    """
    Test that Q_ creates Quantity objects correctly.

    Expected:
        Q_ should work like ureg.Quantity
    """
    distance = utils.Q_(100, 'meter')
    assert distance.magnitude == 100
    assert 'meter' in str(distance.units)


def test_type_aliases_defined():
    """
    Test that type aliases are defined.

    Expected:
        FloatArray1D, FloatArray2D, IntArray1D, TextArray1D should exist
    """
    assert hasattr(utils, 'FloatArray1D')
    assert hasattr(utils, 'FloatArray2D')
    assert hasattr(utils, 'IntArray1D')
    assert hasattr(utils, 'TextArray1D')


def test_all_exports():
    """
    Test that __all__ contains expected exports.

    Expected:
        __all__ should list all public functions and types
    """
    assert hasattr(utils, '__all__')
    expected_exports = {
        'magnitude',
        'get_files',
        'split_data',
        'check_entry',
        'check_writing_file',
        'assert_file_extension',
        'FloatArray2D',
        'FloatArray1D',
        'IntArray1D',
        'TextArray1D',
        'bpm_quantity',
        'ureg',
        'au_quantity',
        'Q_',
    }
    assert set(utils.__all__) == expected_exports
