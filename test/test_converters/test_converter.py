"""converter test module"""

import json
from pathlib import Path
import pytest

import sys
from os.path import abspath, dirname

sys.path.append(dirname(dirname(dirname(dirname(abspath(__file__))))))

from src import labanalysis as laban


# Helpers
class DummyExporter:
    def __init__(self, config=None):
        self.config = config

    def from_filename(self, filename):
        return ("<html><body>DUMMY</body></html>", {})


def write_minimal_ipynb(path: Path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# test notebook"]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


def find_html_in_dir(d: Path):
    return list(d.glob("*.html"))


def test_set_source_and_invalid(tmp_path):
    p = tmp_path / "report.ipynb"
    write_minimal_ipynb(p)

    conv = laban.Converter(p)
    assert conv.source_file == p

    # invalid type
    with pytest.raises(ValueError):
        laban.Converter(123)  # type: ignore

    # missing file
    with pytest.raises(ValueError):
        laban.Converter(tmp_path / "no_such.ipynb")


def test_to_html_uses_exporter_and_writes_file(tmp_path, monkeypatch):
    p = tmp_path / "report.ipynb"
    write_minimal_ipynb(p)

    # monkeypatch the exporter used inside converter module
    monkeypatch.setattr(laban.converters.converter, "HTMLExporter", DummyExporter)

    out_hint = tmp_path / "report.html"
    try:
        conv = laban.Converter(p)
        conv.to_html(
            output_path=out_hint, execute=False, template="custom_lab", verbose=False
        )

        html_files = find_html_in_dir(tmp_path)
        assert len(html_files) == 1
        assert html_files[0].suffix == ".html"
        assert "<body>DUMMY</body>" in html_files[0].read_text(encoding="utf-8")
    finally:
        for f in tmp_path.glob("*.html"):
            try:
                f.unlink()
            except Exception:
                pass
        for f in tmp_path.glob("*.ipynb"):
            try:
                f.unlink()
            except Exception:
                pass


def test_cli_convert_calls_converter_and_creates_output(tmp_path, monkeypatch):
    p = tmp_path / "report.ipynb"
    write_minimal_ipynb(p)

    monkeypatch.setattr(laban.converters.converter, "HTMLExporter", DummyExporter)

    out_hint = tmp_path / "report.html"
    try:
        # call the CLI function (convert) as if argv provided from terminal
        ret = laban.convert(["convert", str(p), "--to", str(out_hint)])
        assert ret == 0

        html_files = find_html_in_dir(tmp_path)
        assert len(html_files) == 1
    finally:
        for f in tmp_path.glob("*.html"):
            try:
                f.unlink()
            except Exception:
                pass
        for f in tmp_path.glob("*.ipynb"):
            try:
                f.unlink()
            except Exception:
                pass
