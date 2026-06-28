"""Tests for protocols.participant module."""

import pytest


@pytest.mark.unit
class TestParticipant:
    """Test Participant class."""

    def test_module_imports(self):
        """Test that Participant can be imported."""
        from labanalysis.protocols.participant import Participant
        assert Participant is not None

    def test_participant_basic_creation(self):
        """Test basic Participant creation with minimal params."""
        from labanalysis.protocols.participant import Participant
        from datetime import date

        p = Participant(
            surname="Test",
            name="User",
            recordingdate=date.today()
        )
        assert p.surname == "Test"
        assert p.name == "User"

    def test_participant_has_bmi_property(self):
        """Test Participant has BMI property."""
        from labanalysis.protocols.participant import Participant

        assert hasattr(Participant, 'bmi')
        assert isinstance(getattr(Participant, 'bmi'), property)

    def test_participant_has_hrmax_property(self):
        """Test Participant has hrmax property."""
        from labanalysis.protocols.participant import Participant

        assert hasattr(Participant, 'hrmax')
        assert isinstance(getattr(Participant, 'hrmax'), property)

    def test_participant_docstring_exists(self):
        """Test Participant has comprehensive docstring."""
        from labanalysis.protocols.participant import Participant

        assert Participant.__doc__ is not None
        assert len(Participant.__doc__) > 100
        assert 'participant' in Participant.__doc__.lower()
