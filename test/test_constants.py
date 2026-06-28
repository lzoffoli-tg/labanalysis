"""
Test suite for labanalysis.constants module.

Tests verify that physical and configuration constants are defined
with correct values and types.
"""

import pytest
from labanalysis import constants


def test_gravity_constant():
    """
    Test that gravity constant is defined and has the standard value.

    Expected:
        G should be 9.80665 m/s^2 (standard gravity acceleration)
    """
    assert hasattr(constants, 'G')
    assert constants.G == 9.80665
    assert isinstance(constants.G, float)


def test_gait_detection_constants():
    """
    Test that gait detection algorithm constants are defined.

    Expected:
        - DEFAULT_MINIMUM_CONTACT_GRF_N should be positive integer (100 N)
        - DEFAULT_MINIMUM_HEIGHT_PERCENTAGE should be between 0 and 1
    """
    assert hasattr(constants, 'DEFAULT_MINIMUM_CONTACT_GRF_N')
    assert hasattr(constants, 'DEFAULT_MINIMUM_HEIGHT_PERCENTAGE')

    assert constants.DEFAULT_MINIMUM_CONTACT_GRF_N == 100
    assert isinstance(constants.DEFAULT_MINIMUM_CONTACT_GRF_N, int)

    assert constants.DEFAULT_MINIMUM_HEIGHT_PERCENTAGE == 0.05
    assert 0 < constants.DEFAULT_MINIMUM_HEIGHT_PERCENTAGE < 1


def test_jump_detection_constants():
    """
    Test that jump detection constants are defined.

    Expected:
        - MINIMUM_CONTACT_FORCE_N should be positive (50 N)
        - MINIMUM_FLIGHT_TIME_S should be positive (0.1 s)
    """
    assert hasattr(constants, 'MINIMUM_CONTACT_FORCE_N')
    assert hasattr(constants, 'MINIMUM_FLIGHT_TIME_S')

    assert constants.MINIMUM_CONTACT_FORCE_N == 50
    assert constants.MINIMUM_FLIGHT_TIME_S == 0.1
    assert constants.MINIMUM_CONTACT_FORCE_N > 0
    assert constants.MINIMUM_FLIGHT_TIME_S > 0


def test_strength_test_constants():
    """
    Test that strength test constants are defined.

    Expected:
        MINIMUM_ISOMETRIC_DISPLACEMENT_M should be positive (0.05 m)
    """
    assert hasattr(constants, 'MINIMUM_ISOMETRIC_DISPLACEMENT_M')
    assert constants.MINIMUM_ISOMETRIC_DISPLACEMENT_M == 0.05
    assert constants.MINIMUM_ISOMETRIC_DISPLACEMENT_M > 0


def test_rank_3colors_structure():
    """
    Test that RANK_3COLORS dictionary has correct structure.

    Expected:
        - Should contain 3 ranks: Normal, Fair, Poor
        - Each should map to a valid hex color string
    """
    assert hasattr(constants, 'RANK_3COLORS')
    assert isinstance(constants.RANK_3COLORS, dict)
    assert len(constants.RANK_3COLORS) == 3

    expected_keys = {'Normal', 'Fair', 'Poor'}
    assert set(constants.RANK_3COLORS.keys()) == expected_keys

    # Verify all values are hex color strings
    for rank, color in constants.RANK_3COLORS.items():
        assert isinstance(color, str)
        assert color.startswith('#')
        assert len(color) == 7  # #RRGGBB format


def test_rank_4colors_structure():
    """
    Test that RANK_4COLORS dictionary has correct structure.

    Expected:
        - Should contain 4 ranks: Good, Normal, Fair, Poor
        - Each should map to a valid hex color string
    """
    assert hasattr(constants, 'RANK_4COLORS')
    assert isinstance(constants.RANK_4COLORS, dict)
    assert len(constants.RANK_4COLORS) == 4

    expected_keys = {'Good', 'Normal', 'Fair', 'Poor'}
    assert set(constants.RANK_4COLORS.keys()) == expected_keys

    for rank, color in constants.RANK_4COLORS.items():
        assert isinstance(color, str)
        assert color.startswith('#')
        assert len(color) == 7


def test_rank_5colors_structure():
    """
    Test that RANK_5COLORS dictionary has correct structure.

    Expected:
        - Should contain 5 ranks: Excellent, Good, Normal, Fair, Poor
        - Each should map to a valid hex color string
    """
    assert hasattr(constants, 'RANK_5COLORS')
    assert isinstance(constants.RANK_5COLORS, dict)
    assert len(constants.RANK_5COLORS) == 5

    expected_keys = {'Excellent', 'Good', 'Normal', 'Fair', 'Poor'}
    assert set(constants.RANK_5COLORS.keys()) == expected_keys

    for rank, color in constants.RANK_5COLORS.items():
        assert isinstance(color, str)
        assert color.startswith('#')
        assert len(color) == 7


def test_side_colors_structure():
    """
    Test that SIDE_COLORS dictionary has correct structure.

    Expected:
        - Should contain 3 sides: bilateral, left, right
        - Each should map to a valid hex color string
    """
    assert hasattr(constants, 'SIDE_COLORS')
    assert isinstance(constants.SIDE_COLORS, dict)
    assert len(constants.SIDE_COLORS) == 3

    expected_keys = {'bilateral', 'left', 'right'}
    assert set(constants.SIDE_COLORS.keys()) == expected_keys

    for side, color in constants.SIDE_COLORS.items():
        assert isinstance(color, str)
        assert color.startswith('#')
        assert len(color) == 7


def test_side_patterns_structure():
    """
    Test that SIDE_PATTERNS dictionary has correct structure.

    Expected:
        - Should contain 3 sides: bilateral, left, right
        - Each should map to a pattern string
    """
    assert hasattr(constants, 'SIDE_PATTERNS')
    assert isinstance(constants.SIDE_PATTERNS, dict)
    assert len(constants.SIDE_PATTERNS) == 3

    expected_keys = {'bilateral', 'left', 'right'}
    assert set(constants.SIDE_PATTERNS.keys()) == expected_keys

    for side, pattern in constants.SIDE_PATTERNS.items():
        assert isinstance(pattern, str)
        assert len(pattern) > 0


def test_rank_colors_consistency():
    """
    Test that common ranks have consistent colors across different rank scales.

    Expected:
        'Normal', 'Fair', 'Poor' should have same colors in all RANK dictionaries
    """
    # Normal, Fair, Poor appear in all three rank scales
    assert constants.RANK_3COLORS['Normal'] == constants.RANK_4COLORS['Normal']
    assert constants.RANK_3COLORS['Normal'] == constants.RANK_5COLORS['Normal']

    assert constants.RANK_3COLORS['Fair'] == constants.RANK_4COLORS['Fair']
    assert constants.RANK_3COLORS['Fair'] == constants.RANK_5COLORS['Fair']

    assert constants.RANK_3COLORS['Poor'] == constants.RANK_4COLORS['Poor']
    assert constants.RANK_3COLORS['Poor'] == constants.RANK_5COLORS['Poor']

    # Good appears in 4 and 5 rank scales
    assert constants.RANK_4COLORS['Good'] == constants.RANK_5COLORS['Good']
