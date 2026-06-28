"""Test suite for labanalysis.protocols.jumptests module."""

import pytest
from labanalysis.protocols import jumptests


def test_jumptests_module_importable():
    """Test that jumptests module is importable."""
    assert jumptests is not None


def test_jumptests_has_content():
    """Test that jumptests module has public content."""
    public = [a for a in dir(jumptests) if not a.startswith('_')]
    assert len(public) > 0
