"""Tests for ols.regression module - wrapper for granular regression tests.

This module contains multiple regression classes tested individually in:
- regression/test_base_regression.py
- regression/test_polynomial_regression.py
- regression/test_power_regression.py
- regression/test_exponential_regression.py
- regression/test_multisegment_regression.py

This wrapper ensures the main regression.py module is covered in the 1:1 mapping.
"""

import pytest


@pytest.mark.unit
def test_regression_module_imports():
    """Test that regression module can be imported."""
    from labanalysis.modelling.ols import regression

    assert regression is not None
    # Module contains regression classes
    # Individual classes tested in regression/ subdirectory
