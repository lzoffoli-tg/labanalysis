"""Test suite for labanalysis.protocols.agilitytests module."""

import pytest
from labanalysis.protocols import agilitytests


def test_agilitytests_module_importable():
    """Test that agilitytests module is importable."""
    assert agilitytests is not None


def test_agilitytests_has_content():
    """Test that agilitytests module has public content."""
    public = [a for a in dir(agilitytests) if not a.startswith('_')]
    assert len(public) > 0
