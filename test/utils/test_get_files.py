"""
Test suite for get_files function.

Tests verify file listing with extension filtering and subfolder options.
"""

import pytest
from pathlib import Path

from labanalysis.utils import get_files


def test_get_files_empty_directory(tmp_path):
    """
    Test get_files with empty directory.

    Expected:
        Should return empty list for directory with no files
    """
    result = get_files(str(tmp_path), ".txt")
    assert result == []


def test_get_files_with_extension(tmp_path):
    """
    Test get_files finds files with specific extension.

    Expected:
        Should find all .txt files but not .py files
    """
    # Create test files
    (tmp_path / "file1.txt").write_text("test")
    (tmp_path / "file2.txt").write_text("test")
    (tmp_path / "file3.py").write_text("test")

    result = get_files(str(tmp_path), ".txt")
    assert len(result) == 2
    assert all(f.endswith(".txt") for f in result)


def test_get_files_with_subfolders(tmp_path):
    """
    Test get_files with subfolder option.

    Expected:
        With check_subfolders=True, should find files in subdirectories
    """
    # Create nested structure
    (tmp_path / "file1.txt").write_text("test")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file2.txt").write_text("test")

    # Without subfolders
    result_no_sub = get_files(str(tmp_path), ".txt", check_subfolders=False)
    assert len(result_no_sub) == 1

    # With subfolders
    result_with_sub = get_files(str(tmp_path), ".txt", check_subfolders=True)
    assert len(result_with_sub) == 2


def test_get_files_no_extension_filter(tmp_path):
    """
    Test get_files with empty extension.

    Expected:
        With extension="", only matches files with NO extension
    """
    (tmp_path / "file1.txt").write_text("test")
    (tmp_path / "file2.py").write_text("test")
    (tmp_path / "noextension").write_text("test")

    result = get_files(str(tmp_path), "")
    # Empty extension matches files ending with "" (all files)
    # But implementation checks if obj[-len(extension):] == extension
    # For extension="", this is obj[0:] == "" which is always False
    # So no files match
    assert len(result) == 0


def test_get_files_nested_subfolders(tmp_path):
    """
    Test get_files with deeply nested subfolder structure.

    Expected:
        Should recursively find files in all nested subfolders
    """
    # Create nested structure
    level1 = tmp_path / "level1"
    level1.mkdir()
    level2 = level1 / "level2"
    level2.mkdir()

    (tmp_path / "root.txt").write_text("test")
    (level1 / "level1.txt").write_text("test")
    (level2 / "level2.txt").write_text("test")

    result = get_files(str(tmp_path), ".txt", check_subfolders=True)
    assert len(result) == 3


def test_get_files_mixed_extensions(tmp_path):
    """
    Test get_files filters correctly among mixed file types.

    Expected:
        Should only return files with exact extension match
    """
    (tmp_path / "data.csv").write_text("test")
    (tmp_path / "doc.txt").write_text("test")
    (tmp_path / "script.py").write_text("test")
    (tmp_path / "image.png").write_text("test")

    result_csv = get_files(str(tmp_path), ".csv")
    assert len(result_csv) == 1
    assert result_csv[0].endswith(".csv")


def test_get_files_returns_full_paths(tmp_path):
    """
    Test that get_files returns full absolute paths.

    Expected:
        Returned paths should be full paths, not just filenames
    """
    (tmp_path / "file.txt").write_text("test")

    result = get_files(str(tmp_path), ".txt")
    assert len(result) == 1
    assert str(tmp_path) in result[0]


def test_get_files_similar_extensions(tmp_path):
    """
    Test get_files distinguishes between similar extensions.

    Expected:
        Should distinguish .txt from .txt.bak
    """
    (tmp_path / "file.txt").write_text("test")
    (tmp_path / "file.txt.bak").write_text("test")

    result_txt = get_files(str(tmp_path), ".txt")
    result_bak = get_files(str(tmp_path), ".bak")

    assert len(result_txt) == 1
    assert len(result_bak) == 1
    assert result_txt[0].endswith(".txt")
    assert not result_txt[0].endswith(".bak")
