"""
Test suite for labanalysis.io.read.opensim module.

Tests reading OpenSim file formats (.trc, .mot).
Note: These are basic smoke tests. Full format validation would require
matching the exact expected file format.
"""

import pytest

from labanalysis.io.read.opensim import read_trc, read_mot


def test_read_trc_function_exists():
    """Test that read_trc function is importable."""
    assert callable(read_trc)


def test_read_mot_function_exists():
    """Test that read_mot function is importable."""
    assert callable(read_mot)


def test_read_trc_nonexistent_file_raises_error():
    """Test that reading non-existent .trc file raises appropriate error."""
    with pytest.raises((FileNotFoundError, OSError)):
        read_trc("nonexistent_file_12345.trc")


def test_read_mot_nonexistent_file_raises_error():
    """Test that reading non-existent .mot file raises appropriate error."""
    with pytest.raises((FileNotFoundError, OSError)):
        read_mot("nonexistent_file_12345.mot")
