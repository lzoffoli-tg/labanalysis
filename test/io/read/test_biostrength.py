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
    Test that BiostrengthProduct wrapper has expected structure.

    Expected:
        Wrapper class should have _external_class attribute pointing to
        biostrengthdataconverter package classes.
    """
    cls = biostrength.BiostrengthProduct
    # Wrapper must have _external_class attribute (set in subclasses)
    assert hasattr(cls, '_external_class'), "Missing _external_class attribute"

    # Verify subclasses have _external_class set
    from biostrengthdataconverter import Biostrength as _Biostrength
    assert biostrength.ChestPress._external_class == _Biostrength.ChestPress
    assert biostrength.LegPress._external_class == _Biostrength.LegPress
