"""
Test suite for labanalysis.protocols.protocols module.

Tests protocol base classes and utilities.
"""

import pytest

from labanalysis.protocols import Participant, TestProtocol, TestResults


def test_participant_class_exists():
    """Test that Participant class exists and is importable."""
    assert Participant is not None


def test_test_protocol_class_exists():
    """Test that TestProtocol protocol exists and is importable."""
    assert TestProtocol is not None


def test_test_results_class_exists():
    """Test that TestResults protocol exists and is importable."""
    assert TestResults is not None
