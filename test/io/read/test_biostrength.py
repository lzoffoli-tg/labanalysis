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


def test_biostrengthproduct_roller_position_parameter():
    """
    Test that wrapper correctly maps roll_position to roller_position parameter.

    This test ensures that the parameter name mismatch bug doesn't reoccur.
    The external biostrengthdataconverter package uses 'roller_position' for
    LegExtensionREV, while our wrapper API uses 'roll_position' for backward compatibility.

    Expected:
        - LegExtensionREV constructor should accept roll_position and pass it as roller_position
        - LegExtensionREV.from_txt_file should accept roll_position and pass it as roller_position
    """
    import inspect
    from unittest.mock import Mock, patch

    # LegExtensionREV is the only product that uses roller_position parameter
    product_class = biostrength.LegExtensionREV
    external_class = product_class._external_class

    # Verify external class has 'roller_position' parameter (not 'roll_position')
    init_sig = inspect.signature(external_class.__init__)
    assert 'roller_position' in init_sig.parameters, (
        f"{external_class.__name__}.__init__ missing 'roller_position' parameter"
    )
    assert 'roll_position' not in init_sig.parameters, (
        f"{external_class.__name__}.__init__ should not have 'roll_position' parameter"
    )

    from_txt_sig = inspect.signature(external_class.from_txt)
    assert 'roller_position' in from_txt_sig.parameters, (
        f"{external_class.__name__}.from_txt missing 'roller_position' parameter"
    )
    assert 'roll_position' not in from_txt_sig.parameters, (
        f"{external_class.__name__}.from_txt should not have 'roll_position' parameter"
    )

    # Test that wrapper correctly passes the parameter
    with patch.object(biostrength.LegExtensionREV, '_external_class') as mock_external:
        mock_instance = Mock()
        mock_external.return_value = mock_instance
        mock_external.from_txt.return_value = mock_instance

        # Test constructor call
        try:
            product = biostrength.LegExtensionREV(
                time_s=[0.0, 1.0],
                motor_position_rad=[0.0, 1.0],
                motor_load_nm=[0.0, 10.0],
                roll_position=18
            )
            # Verify it was called with roller_position, not roll_position
            mock_external.assert_called_once()
            call_kwargs = mock_external.call_args.kwargs
            assert 'roller_position' in call_kwargs, "Constructor should use 'roller_position'"
            assert call_kwargs['roller_position'] == 18
            assert 'roll_position' not in call_kwargs, "Constructor should not use 'roll_position'"
        except TypeError as e:
            if "roll_position" in str(e):
                pytest.fail(f"Constructor incorrectly uses 'roll_position' instead of 'roller_position': {e}")
            raise

        # Reset mock for next test
        mock_external.reset_mock()

        # Test from_txt_file call
        try:
            product = biostrength.LegExtensionREV.from_txt_file(
                "dummy_file.txt",
                roll_position=18
            )
            # Verify it was called with roller_position, not roll_position
            mock_external.from_txt.assert_called_once()
            call_kwargs = mock_external.from_txt.call_args.kwargs
            assert 'roller_position' in call_kwargs, "from_txt should use 'roller_position'"
            assert call_kwargs['roller_position'] == 18
            assert 'roll_position' not in call_kwargs, "from_txt should not use 'roll_position'"
        except TypeError as e:
            if "roll_position" in str(e):
                pytest.fail(f"from_txt_file incorrectly uses 'roll_position' instead of 'roller_position': {e}")
            raise
