"""client module for conversion"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from .converter import Converter

__all__ = ["convert"]


def convert(argv: Optional[List[str]] = None) -> int:
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
