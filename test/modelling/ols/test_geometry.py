"""Tests for ols.geometry module - wrapper for granular geometry tests.

This module contains multiple geometry classes tested individually in:
- geometry/test_circle.py
- geometry/test_ellipse.py
- geometry/test_line2d.py
- geometry/test_line3d.py

This wrapper ensures the main geometry.py module is covered in the 1:1 mapping.
"""

import pytest


@pytest.mark.unit
def test_geometry_module_imports():
    """Test that all geometry classes can be imported."""
    from labanalysis.modelling.ols.geometry import (
        GeometricObject,
        Line2D,
        Line3D,
        Circle,
        Ellipse,
    )

    assert GeometricObject is not None
    assert Line2D is not None
    assert Line3D is not None
    assert Circle is not None
    assert Ellipse is not None
