"""
Shared plotting utilities for protocol visualization.

Contains helpers used across balance and strength test modules.
"""


def balance_string(left: str, right: str, sep: str = " | "):
    """
    Format balanced left-right text for visualization.

    Ensures equal width padding using non-breaking spaces.
    Used in balance tests and strength tests for muscle imbalance displays.

    Parameters
    ----------
    left : str
        Left side text
    right : str
        Right side text
    sep : str, optional
        Separator between left and right, by default " | "

    Returns
    -------
    str
        Formatted string with balanced padding
    """
    width = max(len(left), len(right))
    nbsp = " "  # non-breaking
    ljust = left.rjust(width).replace(" ", nbsp)
    rjust = right.ljust(width).replace(" ", nbsp)
    return sep.join([ljust, rjust])
