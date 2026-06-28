"""
Test suite for file operation utilities.

Tests verify assert_file_extension and check_writing_file functions.
"""

import pytest

from labanalysis.utils import assert_file_extension


def test_assert_file_extension_valid():
    """
    Test assert_file_extension with valid file extension.

    Expected:
        Should pass without raising for correct extension
    """
    assert_file_extension("test.txt", "txt")
    assert_file_extension("path/to/file.csv", "csv")


def test_assert_file_extension_invalid():
    """
    Test assert_file_extension with invalid extension.

    Expected:
        Should raise AssertionError for wrong extension
    """
    with pytest.raises(AssertionError):
        assert_file_extension("test.txt", "csv")


def test_assert_file_extension_not_string():
    """
    Test assert_file_extension with non-string input.

    Expected:
        Should raise AssertionError for non-string path
    """
    with pytest.raises(AssertionError, match="path must be a str object"):
        assert_file_extension(123, "txt")


def test_assert_file_extension_nested_path():
    """
    Test assert_file_extension with nested path.

    Expected:
        Should correctly extract extension from nested path
    """
    assert_file_extension("path/to/deeply/nested/file.json", "json")


def test_assert_file_extension_multiple_dots():
    """
    Test assert_file_extension with filename containing multiple dots.

    Expected:
        Should use extension after last dot
    """
    assert_file_extension("file.backup.txt", "txt")

    with pytest.raises(AssertionError):
        assert_file_extension("file.backup.txt", "backup")


def test_assert_file_extension_no_extension():
    """
    Test assert_file_extension with file without extension.

    Expected:
        Should raise AssertionError when no extension present
    """
    with pytest.raises(AssertionError):
        assert_file_extension("README", "md")


def test_assert_file_extension_case_sensitive():
    """
    Test assert_file_extension is case-sensitive.

    Expected:
        Should distinguish between 'txt' and 'TXT'
    """
    with pytest.raises(AssertionError):
        assert_file_extension("file.TXT", "txt")


def test_assert_file_extension_windows_path():
    """
    Test assert_file_extension with Windows-style path.

    Expected:
        Should work with backslash separators
    """
    assert_file_extension("C:\\Users\\Documents\\file.xlsx", "xlsx")


def test_assert_file_extension_empty_string():
    """
    Test assert_file_extension with empty extension.

    Expected:
        Should fail for file with actual extension when empty string expected
    """
    with pytest.raises(AssertionError):
        assert_file_extension("file.txt", "")
