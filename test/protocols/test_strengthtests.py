"""Test suite for labanalysis.protocols.strengthtests module."""

import pytest
from labanalysis.protocols import strengthtests


def test_strengthtests_module_importable():
    """Test that strengthtests module is importable."""
    assert strengthtests is not None


def test_strengthtests_has_content():
    """Test that strengthtests module has public content."""
    public = [a for a in dir(strengthtests) if not a.startswith('_')]
    assert len(public) > 0
