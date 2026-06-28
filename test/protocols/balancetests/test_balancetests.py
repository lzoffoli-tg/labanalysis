"""Test suite for labanalysis.protocols.balancetests module."""

import pytest
from labanalysis.protocols import balancetests


def test_balancetests_module_importable():
    """Test that balancetests module is importable."""
    assert balancetests is not None


def test_balancetests_has_content():
    """Test that balancetests module has public content."""
    public = [a for a in dir(balancetests) if not a.startswith('_')]
    assert len(public) > 0
