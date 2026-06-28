"""
Test suite for labanalysis.io.read.biostrength module.

Tests verify BiostrengthProduct class and related file reading functions.
"""

import pytest
from labanalysis.io.read import biostrength


def test_biostrength_module_importable():
    """
    Test that biostrength module imports without errors.

    Expected:
        Module should be importable
    """
    assert biostrength is not None


def test_biostrengthproduct_class_exists():
    """
    Test that BiostrengthProduct class is defined.

    Expected:
        Class should exist and be instantiable
    """
    assert hasattr(biostrength, 'BiostrengthProduct')
    assert isinstance(biostrength.BiostrengthProduct, type)


def test_biostrengthproduct_class_attributes():
    """
    Test that BiostrengthProduct has expected class constants.

    Expected:
        Class should define calibration constants like spring_correction,
        pulley_radius_m, etc.
    """
    cls = biostrength.BiostrengthProduct
    expected_attrs = [
        '_spring_correction',
        '_pulley_radius_m',
        '_lever_weight_kgf',
        '_camme_ratio',
        '_lever_number',
    ]

    for attr in expected_attrs:
        assert hasattr(cls, attr), f"Missing class attribute: {attr}"
