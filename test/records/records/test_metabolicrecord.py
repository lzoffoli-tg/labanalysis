"""
Test suite for MetabolicRecord class.

Tests verify metabolic measurement system with VO2, VCO2, HR, VE, RF.
"""

import numpy as np
import pytest

from labanalysis.records import MetabolicRecord
from labanalysis.timeseries import Signal1D


def test_metabolicrecord_initialization():
    """
    Test MetabolicRecord initialization with all required signals.

    Expected:
        Should create MetabolicRecord with vo2, vco2, hr, ve, rf
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data * 2, index, unit='bpm')
    ve = Signal1D(data * 3, index, unit='L/min')
    rf = Signal1D(data / 2, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)

    assert len(mr) == 5
    assert 'vo2' in mr.keys()
    assert 'vco2' in mr.keys()
    assert 'hr' in mr.keys()
    assert 've' in mr.keys()
    assert 'rf' in mr.keys()


def test_metabolicrecord_breath_by_breath_default():
    """
    Test MetabolicRecord breath_by_breath defaults to False.

    Expected:
        Should have breath_by_breath=False by default
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)

    assert mr.breath_by_breath is False


def test_metabolicrecord_breath_by_breath_custom():
    """
    Test MetabolicRecord with custom breath_by_breath value.

    Expected:
        Should set breath_by_breath to specified value
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf, breath_by_breath=True)

    assert mr.breath_by_breath is True


def test_metabolicrecord_set_vo2_invalid_type():
    """
    Test set_vo2 raises ValueError for non-Signal1D.

    Expected:
        Should raise ValueError
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)

    with pytest.raises(ValueError, match="signal must be a Signal1D instance"):
        mr.set_vo2("invalid")


def test_metabolicrecord_set_vco2_invalid_type():
    """
    Test set_vco2 raises ValueError for non-Signal1D.

    Expected:
        Should raise ValueError
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)

    with pytest.raises(ValueError, match="signal must be a Signal1D instance"):
        mr.set_vco2("invalid")


def test_metabolicrecord_set_hr_invalid_type():
    """
    Test set_hr raises ValueError for non-Signal1D.

    Expected:
        Should raise ValueError
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)

    with pytest.raises(ValueError, match="signal must be a Signal1D instance"):
        mr.set_hr("invalid")


def test_metabolicrecord_set_ve_invalid_type():
    """
    Test set_ve raises ValueError for non-Signal1D.

    Expected:
        Should raise ValueError
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)

    with pytest.raises(ValueError, match="signal must be a Signal1D instance"):
        mr.set_ve("invalid")


def test_metabolicrecord_set_rf_invalid_type():
    """
    Test set_rf raises ValueError for non-Signal1D.

    Expected:
        Should raise ValueError
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)

    with pytest.raises(ValueError, match="signal must be a Signal1D instance"):
        mr.set_rf("invalid")


def test_metabolicrecord_set_breath_by_breath_invalid_type():
    """
    Test set_breath_by_breath raises ValueError for non-bool.

    Expected:
        Should raise ValueError
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)

    with pytest.raises(ValueError, match="value must be True or False"):
        mr.set_breath_by_breath("not_a_bool")


def test_metabolicrecord_rq_property():
    """
    Test RQ (respiratory quotient) property calculation.

    Expected:
        Should return Signal1D with VCO2/VO2 ratio
    """
    vo2_data = np.array([100.0, 200.0, 300.0])
    vco2_data = np.array([80.0, 160.0, 240.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(vo2_data, index, unit='ml/min')
    vco2 = Signal1D(vco2_data, index, unit='ml/min')
    hr = Signal1D(vo2_data, index, unit='bpm')
    ve = Signal1D(vo2_data, index, unit='L/min')
    rf = Signal1D(vo2_data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)
    rq = mr.rq

    assert isinstance(rq, Signal1D)
    assert np.allclose(rq._data.flatten(), [0.8, 0.8, 0.8])


def test_metabolicrecord_fat_oxidation_property():
    """
    Test fat_oxidation property calculation.

    Expected:
        Should return Signal1D with fat oxidation rate
    """
    vo2_data = np.array([1000.0, 2000.0, 3000.0])
    vco2_data = np.array([800.0, 1600.0, 2400.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(vo2_data, index, unit='ml/min')
    vco2 = Signal1D(vco2_data, index, unit='ml/min')
    hr = Signal1D(vo2_data, index, unit='bpm')
    ve = Signal1D(vo2_data, index, unit='L/min')
    rf = Signal1D(vo2_data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)
    fox = mr.fat_oxidation

    assert isinstance(fox, Signal1D)
    assert fox.shape == (3, 1)


def test_metabolicrecord_copy():
    """
    Test copy method.

    Expected:
        Should create deep copy with same signals
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf, breath_by_breath=True)
    copied = mr.copy()

    assert isinstance(copied, MetabolicRecord)
    assert copied is not mr
    assert copied.breath_by_breath == mr.breath_by_breath
    assert len(copied) == len(mr)


def test_metabolicrecord_to_dataframe():
    """
    Test to_dataframe includes RQ and fat oxidation.

    Expected:
        Should include calculated RQ and Fat Oxidation columns
    """
    data = np.array([100.0, 200.0, 300.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)
    df = mr.to_dataframe()

    assert 'RQ' in df.columns
    assert 'Fat Oxidation g/kg/min' in df.columns
    assert df.shape[0] == 3


def test_metabolicrecord_setitem_invalid_key():
    """
    Test setitem with invalid key raises ValueError.

    Expected:
        Should only accept vo2, vco2, hr, ve, rf keys
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)

    with pytest.raises(ValueError, match="only 'vo2', 'vco2', 've', 'rf' and 'hr' attributes"):
        mr['invalid_key'] = Signal1D(data, index, unit='ml/min')


def test_metabolicrecord_setitem_invalid_value():
    """
    Test setitem with non-Signal1D value raises ValueError.

    Expected:
        Should raise ValueError for non-Signal1D value
    """
    data = np.array([10.0, 20.0, 30.0])
    index = np.array([0.0, 1.0, 2.0])

    vo2 = Signal1D(data, index, unit='ml/min')
    vco2 = Signal1D(data, index, unit='ml/min')
    hr = Signal1D(data, index, unit='bpm')
    ve = Signal1D(data, index, unit='L/min')
    rf = Signal1D(data, index, unit='bpm')

    mr = MetabolicRecord(vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf)

    with pytest.raises(ValueError, match="value must be a Signal1D"):
        mr['vo2'] = "not_a_signal"
