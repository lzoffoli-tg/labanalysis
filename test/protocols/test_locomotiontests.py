"""Test suite for labanalysis.protocols.locomotiontests module."""

import pytest
from labanalysis.protocols import locomotiontests


def test_locomotiontests_module_importable():
    """Test that locomotiontests module is importable."""
    assert locomotiontests is not None


def test_locomotiontests_has_content():
    """Test that locomotiontests module has public content."""
    public = [a for a in dir(locomotiontests) if not a.startswith('_')]
    assert len(public) > 0
