"""
Client module for notebook conversion command-line interface.

This module provides a command-line interface for converting Jupyter notebooks
to HTML format using the Converter class.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .converter import Converter

__all__ = ["convert"]


def convert(argv: list[str] | None = None) -> int:
    """
    Command-line interface for converting notebooks to HTML.

    This function provides a CLI for the labanalysis conversion functionality,
    supporting conversion of Jupyter notebook files to HTML format with optional
    execution and template customization.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments to parse. If None, arguments are taken from
        sys.argv. Primarily used for testing.

    Returns
    -------
    int
        Exit code: 0 for success, 2 for conversion errors.

    Examples
    --------
    Convert a notebook without execution:

    >>> convert(["convert", "notebook.ipynb"])
    0

    Convert with execution and custom output:

    >>> convert(["convert", "notebook.ipynb", "--to", "output.html", "--execute"])
    0

    Notes
    -----
    The CLI supports the following command:

    **convert** : Convert notebook to HTML

        * source : Source .ipynb file (required)
        * --to, -t : Output HTML file path (optional)
        * --execute : Execute notebook before converting (flag)
        * --template : Template name (default: "custom_lab")
        * --no-verbose : Disable verbose output (flag)
    """
    parser = argparse.ArgumentParser(prog="labanalysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    conv = subparsers.add_parser("convert", help="Convert notebook to HTML")
    conv.add_argument("source", help="Source .ipynb file")
    conv.add_argument("--to", "-t", dest="to", help="Output HTML file (optional)")
    conv.add_argument(
        "--execute", action="store_true", help="Execute notebook before converting"
    )
    conv.add_argument("--template", default="custom_lab", help="Template name")
    conv.add_argument(
        "--no-verbose",
        action="store_false",
        dest="verbose",
        help="Disable verbose output",
    )

    args = parser.parse_args(argv)

    if args.command == "convert":
        src = Path(args.source)
        out = Path(args.to) if args.to else None

        try:
            converter = Converter(src)
            converter.to_html(
                output_path=out,
                execute=args.execute,
                template=args.template,
                verbose=args.verbose,
            )
            return 0
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 2

    return 0
