"""
Test suite for labanalysis.messages module.

Tests verify that message dialog functions are importable and have
correct signatures. GUI functionality is not tested (requires display).
"""

import pytest
from labanalysis import messages


def test_module_imports():
    """
    Test that messages module imports without errors.

    Expected:
        Module should be importable and contain dialog functions
    """
    assert messages is not None


def test_showinfo_exists():
    """
    Test that showinfo function exists and is callable.

    Expected:
        showinfo should be a callable function for info dialogs
    """
    assert hasattr(messages, 'showinfo')
    assert callable(messages.showinfo)


def test_showwarning_exists():
    """
    Test that showwarning function exists and is callable.

    Expected:
        showwarning should be a callable function for warning dialogs
    """
    assert hasattr(messages, 'showwarning')
    assert callable(messages.showwarning)


def test_showerror_exists():
    """
    Test that showerror function exists and is callable.

    Expected:
        showerror should be a callable function for error dialogs
    """
    assert hasattr(messages, 'showerror')
    assert callable(messages.showerror)


def test_askquestion_exists():
    """
    Test that askquestion function exists and is callable.

    Expected:
        askquestion should be a callable function for yes/no questions
    """
    assert hasattr(messages, 'askquestion')
    assert callable(messages.askquestion)


def test_askokcancel_exists():
    """
    Test that askokcancel function exists and is callable.

    Expected:
        askokcancel should be a callable function for OK/Cancel dialogs
    """
    assert hasattr(messages, 'askokcancel')
    assert callable(messages.askokcancel)


def test_askyesno_exists():
    """
    Test that askyesno function exists and is callable.

    Expected:
        askyesno should be a callable function for yes/no questions
    """
    assert hasattr(messages, 'askyesno')
    assert callable(messages.askyesno)


def test_askyesnocancel_exists():
    """
    Test that askyesnocancel function exists and is callable.

    Expected:
        askyesnocancel should be a callable function for yes/no/cancel questions
    """
    assert hasattr(messages, 'askyesnocancel')
    assert callable(messages.askyesnocancel)


def test_askretrycancel_exists():
    """
    Test that askretrycancel function exists and is callable.

    Expected:
        askretrycancel should be a callable function for retry/cancel dialogs
    """
    assert hasattr(messages, 'askretrycancel')
    assert callable(messages.askretrycancel)


def test_all_exports():
    """
    Test that __all__ contains expected public functions.

    Expected:
        __all__ should list all 8 public dialog functions
    """
    assert hasattr(messages, '__all__')
    expected_functions = [
        'showinfo',
        'showwarning',
        'showerror',
        'askquestion',
        'askokcancel',
        'askyesno',
        'askyesnocancel',
        'askretrycancel',
    ]
    assert set(messages.__all__) == set(expected_functions)


def test_module_constants():
    """
    Test that internal message constants are defined.

    Expected:
        Module should define icon and type constants (even if private)
    """
    # These are private but used internally
    assert hasattr(messages, '_ERROR')
    assert hasattr(messages, '_INFO')
    assert hasattr(messages, '_QUESTION')
    assert hasattr(messages, '_WARNING')

    assert hasattr(messages, '_OK')
    assert hasattr(messages, '_OKCANCEL')
    assert hasattr(messages, '_YESNO')
    assert hasattr(messages, '_YESNOCANCEL')

    assert messages._ERROR == "error"
    assert messages._INFO == "info"
    assert messages._QUESTION == "question"
    assert messages._WARNING == "warning"
