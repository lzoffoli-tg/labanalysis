"""Tests for agilitytests.shuttle_test module."""

import pytest


@pytest.mark.integration
class TestShuttleTest:
    """Test ShuttleTest class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.protocols.agilitytests.shuttle_test import ShuttleTest
        assert ShuttleTest is not None

    def test_shuttle_test_has_exercises_property(self):
        """Test ShuttleTest has change_of_direction_exercises property."""
        from labanalysis.protocols.agilitytests.shuttle_test import ShuttleTest

        assert hasattr(ShuttleTest, 'change_of_direction_exercises')
        assert isinstance(getattr(ShuttleTest, 'change_of_direction_exercises'), property)

    def test_shuttle_test_docstring_exists(self):
        """Test ShuttleTest has comprehensive docstring."""
        from labanalysis.protocols.agilitytests.shuttle_test import ShuttleTest

        assert ShuttleTest.__doc__ is not None
        assert len(ShuttleTest.__doc__) > 100
        assert 'shuttle' in ShuttleTest.__doc__.lower() or 'agility' in ShuttleTest.__doc__.lower()
