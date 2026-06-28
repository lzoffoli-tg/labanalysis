"""
Test suite for hex_to_rgba function.

Tests verify hex color to RGBA string conversion.
"""

import pytest

from labanalysis.utils import hex_to_rgba


def test_hex_to_rgba_full_hex():
    """
    Test hex_to_rgba with full 6-character hex color.

    Expected:
        #1f77b4 with alpha=1.0 should convert to "rgba(31,119,180,1.0)"
    """
    result = hex_to_rgba("#1f77b4", alpha=1.0)
    assert result == "rgba(31,119,180,1.0)"


def test_hex_to_rgba_short_hex():
    """
    Test hex_to_rgba with 3-character hex color.

    Expected:
        #abc should expand to #aabbcc then convert to rgba
    """
    result = hex_to_rgba("#abc", alpha=0.5)
    assert result == "rgba(170,187,204,0.5)"


def test_hex_to_rgba_no_hash():
    """
    Test hex_to_rgba with hex color without # prefix.

    Expected:
        Should work without # prefix
    """
    result = hex_to_rgba("ff0000", alpha=1.0)
    assert result == "rgba(255,0,0,1.0)"


def test_hex_to_rgba_invalid_hex():
    """
    Test hex_to_rgba with invalid hex color.

    Expected:
        Should raise ValueError for invalid hex format
    """
    with pytest.raises(ValueError, match="Invalid HEX color"):
        hex_to_rgba("#12345", alpha=1.0)


def test_hex_to_rgba_black():
    """
    Test hex_to_rgba with black color.

    Expected:
        #000000 should convert to "rgba(0,0,0,alpha)"
    """
    result = hex_to_rgba("#000000", alpha=1.0)
    assert result == "rgba(0,0,0,1.0)"


def test_hex_to_rgba_white():
    """
    Test hex_to_rgba with white color.

    Expected:
        #ffffff should convert to "rgba(255,255,255,alpha)"
    """
    result = hex_to_rgba("#ffffff", alpha=1.0)
    assert result == "rgba(255,255,255,1.0)"


def test_hex_to_rgba_alpha_variations():
    """
    Test hex_to_rgba with different alpha values.

    Expected:
        Should correctly set alpha channel
    """
    result_opaque = hex_to_rgba("#123456", alpha=1.0)
    assert result_opaque == "rgba(18,52,86,1.0)"

    result_half = hex_to_rgba("#123456", alpha=0.5)
    assert result_half == "rgba(18,52,86,0.5)"

    result_transparent = hex_to_rgba("#123456", alpha=0.0)
    assert result_transparent == "rgba(18,52,86,0.0)"


def test_hex_to_rgba_short_hex_expansions():
    """
    Test that 3-character hex correctly expands.

    Expected:
        #123 should expand to #112233
    """
    result = hex_to_rgba("#123", alpha=1.0)
    # #123 → #112233 → rgb(17, 34, 51)
    assert result == "rgba(17,34,51,1.0)"


def test_hex_to_rgba_case_insensitive():
    """
    Test hex_to_rgba with uppercase hex characters.

    Expected:
        Should handle both uppercase and lowercase hex
    """
    result_lower = hex_to_rgba("#aabbcc", alpha=1.0)
    result_upper = hex_to_rgba("#AABBCC", alpha=1.0)
    assert result_lower == result_upper


def test_hex_to_rgba_typical_plotly_colors():
    """
    Test hex_to_rgba with common Plotly default colors.

    Expected:
        Should correctly convert typical chart colors
    """
    # Plotly default blue
    result = hex_to_rgba("#636EFA", alpha=0.8)
    assert result == "rgba(99,110,250,0.8)"


def test_hex_to_rgba_with_whitespace():
    """
    Test hex_to_rgba strips whitespace from input.

    Expected:
        Should handle hex with leading/trailing whitespace
    """
    result = hex_to_rgba("  #123456  ", alpha=1.0)
    assert result == "rgba(18,52,86,1.0)"


def test_hex_to_rgba_invalid_length():
    """
    Test hex_to_rgba rejects invalid hex lengths.

    Expected:
        Should raise ValueError for lengths other than 3 or 6
    """
    with pytest.raises(ValueError, match="Invalid HEX color"):
        hex_to_rgba("#12", alpha=1.0)

    with pytest.raises(ValueError, match="Invalid HEX color"):
        hex_to_rgba("#1234", alpha=1.0)
