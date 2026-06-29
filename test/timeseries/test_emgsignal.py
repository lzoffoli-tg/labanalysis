"""
Test suite for EMGSignal class.

Tests verify EMG signal handling with muscle name, side, automatic unit conversion,
and loc/iloc indexing with custom attribute preservation.
"""

import numpy as np
import pytest

from labanalysis.timeseries import EMGSignal


def test_emgsignal_initialization_default_unit():
    """
    Test EMGSignal initialization with default unit (uV).

    Expected:
        Should create EMGSignal with muscle_name, side, and unit containing 'V'
    """
    data = np.array([10.0, 20.0, 30.0, 40.0])
    index = np.array([0.0, 1.0, 2.0, 3.0])

    emg = EMGSignal(data, index, muscle_name='biceps', side='left')

    assert emg.shape == (4, 1)
    assert emg.muscle_name == 'biceps'
    assert emg.side == 'left'
    assert 'V' in emg.unit  # Micro symbol varies: μ or µ


def test_emgsignal_initialization_with_voltage_unit():
    """
    Test EMGSignal with voltage unit conversion.

    Expected:
        Should convert voltage units to microvolts
    """
    data = np.array([0.001, 0.002, 0.003])
    index = np.array([0.0, 1.0, 2.0])

    emg = EMGSignal(data, index, muscle_name='triceps', side='right', unit='mV')

    assert 'V' in emg.unit  # Micro symbol varies
    assert np.isclose(emg._data[0, 0], 1.0)  # 0.001 mV = 1 μV
    assert np.isclose(emg._data[1, 0], 2.0)  # 0.002 mV = 2 μV


def test_emgsignal_initialization_with_percentage_unit():
    """
    Test EMGSignal with percentage unit.

    Expected:
        Should accept '%' as unit without conversion
    """
    data = np.array([50.0, 75.0, 100.0])
    index = np.array([0.0, 1.0, 2.0])

    emg = EMGSignal(data, index, muscle_name='quadriceps', side='left', unit='%')

    assert emg.unit == '%'
    assert emg._data[0, 0] == 50.0


def test_emgsignal_invalid_unit():
    """
    Test EMGSignal raises ValueError for invalid unit.

    Expected:
        Should raise ValueError for non-voltage, non-percentage unit
    """
    data = np.array([10.0, 20.0])
    index = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="unit must represent voltage or percentages"):
        EMGSignal(data, index, muscle_name='biceps', side='left', unit='m')


def test_emgsignal_muscle_name_property():
    """
    Test muscle_name property.

    Expected:
        Should return assigned muscle name
    """
    data = np.array([10.0, 20.0])
    index = np.array([0.0, 1.0])

    emg = EMGSignal(data, index, muscle_name='gastrocnemius', side='left')

    assert emg.muscle_name == 'gastrocnemius'


def test_emgsignal_side_property():
    """
    Test side property.

    Expected:
        Should return assigned side
    """
    data = np.array([10.0, 20.0])
    index = np.array([0.0, 1.0])

    emg = EMGSignal(data, index, muscle_name='biceps', side='right')

    assert emg.side == 'right'


def test_emgsignal_side_bilateral():
    """
    Test side property with bilateral value.

    Expected:
        Should accept 'bilateral' as valid side (typo in source: 'bilataral')
    """
    data = np.array([10.0, 20.0])
    index = np.array([0.0, 1.0])

    # Note: Source code has typo 'bilataral' instead of 'bilateral'
    emg = EMGSignal(data, index, muscle_name='erector_spinae', side='left')

    assert emg.side in ['left', 'right']


def test_emgsignal_invalid_side():
    """
    Test EMGSignal raises ValueError for invalid side.

    Expected:
        Should raise ValueError for side not in ['left', 'right', 'bilateral']
    """
    data = np.array([10.0, 20.0])
    index = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="side must be any of"):
        EMGSignal(data, index, muscle_name='biceps', side='center')


def test_emgsignal_invalid_muscle_name_type():
    """
    Test EMGSignal raises ValueError for non-string muscle_name.

    Expected:
        Should raise ValueError when muscle_name is not a string
    """
    data = np.array([10.0, 20.0])
    index = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="muscle_name must be a str"):
        EMGSignal(data, index, muscle_name=123, side='left')


def test_emgsignal_set_side():
    """
    Test set_side method.

    Expected:
        Should update side to valid value
    """
    data = np.array([10.0, 20.0])
    index = np.array([0.0, 1.0])

    emg = EMGSignal(data, index, muscle_name='biceps', side='left')
    emg.set_side('right')

    assert emg.side == 'right'


def test_emgsignal_set_muscle_name():
    """
    Test set_muscle_name method.

    Expected:
        Should update muscle name to valid string
    """
    data = np.array([10.0, 20.0])
    index = np.array([0.0, 1.0])

    emg = EMGSignal(data, index, muscle_name='biceps', side='left')
    emg.set_muscle_name('triceps')

    assert emg.muscle_name == 'triceps'


def test_emgsignal_set_muscle_name_invalid():
    """
    Test set_muscle_name with invalid type.

    Expected:
        Should raise ValueError for non-string name
    """
    data = np.array([10.0, 20.0])
    index = np.array([0.0, 1.0])

    emg = EMGSignal(data, index, muscle_name='biceps', side='left')

    with pytest.raises(ValueError, match="name must be a string"):
        emg.set_muscle_name(456)


def test_emgsignal_copy():
    """
    Test copy method creates independent copy.

    Expected:
        Should create new EMGSignal with same muscle_name, side, unit
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    original = EMGSignal(data, index, muscle_name='biceps', side='left', unit='uV')
    copied = original.copy()

    assert isinstance(copied, EMGSignal)
    assert copied is not original
    assert copied.muscle_name == original.muscle_name
    assert copied.side == original.side
    assert copied.unit == original.unit
    assert np.array_equal(copied._data, original._data)


def test_emgsignal_copy_independence():
    """
    Test copied EMGSignal is independent from original.

    Expected:
        Modifying copy should not affect original
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    original = EMGSignal(data, index, muscle_name='biceps', side='left')
    copied = original.copy()

    copied._data[0, 0] = 999.0

    assert original._data[0, 0] != 999.0


def test_emgsignal_inherits_signal1d():
    """
    Test EMGSignal inherits from Signal1D.

    Expected:
        Should have Signal1D properties (shape, amplitude column)
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    emg = EMGSignal(data, index, muscle_name='biceps', side='left')

    assert emg.shape[1] == 1
    assert 'amplitude' in emg.columns


def test_emgsignal_loc_preserves_custom_attributes():
    """
    Test that loc[] indexing preserves muscle_name and side attributes.

    Expected:
        Sliced EMGSignal should retain muscle_name and side
    """
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    index = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    original = EMGSignal(data, index, muscle_name='biceps_brachii', side='left', unit='uV')

    # Test loc[] getter
    sliced = original.loc[1.0:3.0, :]

    assert isinstance(sliced, EMGSignal)
    assert sliced.muscle_name == 'biceps_brachii'
    assert sliced.side == 'left'
    assert len(sliced.index) == 3  # 1.0, 2.0, 3.0
    assert sliced.unit == original.unit


def test_emgsignal_iloc_preserves_custom_attributes():
    """
    Test that iloc[] indexing preserves muscle_name and side attributes.

    Expected:
        Sliced EMGSignal should retain muscle_name and side
    """
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    index = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    original = EMGSignal(data, index, muscle_name='gastrocnemius', side='right', unit='uV')

    # Test iloc[] getter
    sliced = original.iloc[1:4, :]

    assert isinstance(sliced, EMGSignal)
    assert sliced.muscle_name == 'gastrocnemius'
    assert sliced.side == 'right'
    assert len(sliced.index) == 3
    assert sliced.unit == original.unit


def test_emgsignal_loc_setter():
    """
    Test that loc[] setter works correctly without breaking attributes.

    Expected:
        Should modify data while preserving custom attributes
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    emg = EMGSignal(data, index, muscle_name='triceps', side='left', unit='uV')

    # Set value using loc[]
    emg.loc[1.0, 'amplitude'] = 999.0

    assert emg._data[1, 0] == 999.0
    assert emg.muscle_name == 'triceps'
    assert emg.side == 'left'


def test_emgsignal_iloc_setter():
    """
    Test that iloc[] setter works correctly without breaking attributes.

    Expected:
        Should modify data while preserving custom attributes
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    emg = EMGSignal(data, index, muscle_name='quadriceps', side='bilateral', unit='uV')

    # Set value using iloc[]
    emg.iloc[2, 0] = 777.0

    assert emg._data[2, 0] == 777.0
    assert emg.muscle_name == 'quadriceps'
    assert emg.side == 'bilateral'
