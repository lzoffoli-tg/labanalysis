"""Converter class module"""

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal

from nbconvert import HTMLExporter
from traitlets.config import Config

__all__ = ["Converter"]


class Converter:
    """
    A file conversion class object.

    Examples
    --------
    >>> converter = Converter("example.ipynb")
    >>> converter.to_html()
    """

    def __init__(self, source_file: Path | str):
        self.set_source_file(source_file)

    def set_source_file(self, source_file: Path | str):
        """set the source file path"""
        if isinstance(source_file, str):
            self._source_file = Path(source_file)
        elif isinstance(source_file, Path):
            self._source_file = source_file
        else:
            raise ValueError("Invalid source file type")

        if not self.source_file.exists():
            raise ValueError("Source file not found.")

    @property
    def source_file(self):
        """return the source file path"""
        return self._source_file

    def to_html(
        self,
        output_path: Path | str | None = None,
        execute: bool = False,
        template: Literal["custom_lab"] = "custom_lab",
        verbose: bool = True,
    ):
        """
        Convert source file to html format and save it on the provided output_path.

        Parameters
        ----------
        output_path: Path | str | None
            Path to the output HTML file. If None, the same path of source_file
            is used.
        execute: bool (default: False)
            Whether to execute the notebook cells before conversion.
        template: Literal["custom_lab"] (default: "custom_lab")
            The template to use for conversion.
        verbose: bool (default: True)
            Whether to print verbose output during conversion.

        Returns
        -------
        None
        """

        # check inputs
        if output_path is None:
            output_path = self.source_file.with_suffix(".html")
        elif isinstance(output_path, str):
            output_path = Path(output_path)
        elif isinstance(output_path, Path):
            pass
        else:
            raise ValueError("Invalid output path type")

        if not isinstance(execute, bool):
            raise ValueError("Invalid execute type")

        supported_templates = ["custom_lab"]
        if not isinstance(template, str) and template not in supported_templates:
            raise ValueError(
                f"Invalid template. Supported templates are: {supported_templates}"
            )

        if not isinstance(verbose, bool):
            raise ValueError("verbose must be True or False")

        # Get absolute paths
        script_dir = Path(__file__).parent
        template_basedir = script_dir / "templates"
        if not template_basedir.exists():
            print(f"Error: Template directory not found: {template_basedir}")
            sys.exit(1)

        # Generate output filename with date stamp
        today = datetime.now().strftime("%Y%m%d_%H%M%S_")
        notebook_stem = self.source_file.stem  # Filename without extension
        output_name = f"{today}_{notebook_stem}.html"
        output_path = output_path.parent / output_name

        # check the input type
        starting_extension = self.source_file.suffix
        ending_extension = output_path.suffix
        if starting_extension == ".ipynb" and ending_extension == ".html":
            self._ipynb_to_html(
                output_path,
                execute,
                template_basedir,
                template,
                verbose,
            )
        else:
            raise ValueError(
                f"{starting_extension} to {ending_extension} conversion is not yet supported."
            )

    def _ipynb_to_html(
        self,
        output_path: Path,
        execute: bool,
        template_basedir: Path,
        template: str,
        verbose: bool,
    ):
        """private method to convert .ipynb to HTML"""

        # Print info
        if verbose:
            print("=" * 60)
            print("Report Conversion")
            print("=" * 60)
            print(f"Notebook:  {self.source_file}")
            print(f"Template:  {template}")
            print(f"Execute:   {'Yes' if execute else 'No'}")
            print(f"Output:    {output_path}")
            print("")

        try:
            # Configure exporter
            c = Config()
            c.HTMLExporter.template_name = template
            c.HTMLExporter.extra_template_basedirs = [str(template_basedir.absolute())]

            # Add ExecutePreprocessor if execution is enabled
            if execute:
                c.HTMLExporter.preprocessors = [
                    "nbconvert.preprocessors.ExecutePreprocessor"
                ]
                c.ExecutePreprocessor.timeout = 600  # 10 minutes timeout per cell
                c.ExecutePreprocessor.kernel_name = "python3"

            # Create exporter
            exporter = HTMLExporter(config=c)

            # Convert (will execute first if enabled, then convert)
            if verbose:
                if execute:
                    print("Executing notebook cells...")
                    print("(This may take several minutes depending on computation)")
                    print("")
                else:
                    print("Converting (without execution)...")

            body, resources = exporter.from_filename(str(self.source_file))

            # Write output
            if verbose:
                print(f"Writing to {output_path}...")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(body)

            # Success
            if verbose:
                print("")
                print("[SUCCESS] Conversion successful!")
                print("")
                print(f"Output: {output_path}")
                print(f"Size:   {output_path.stat().st_size / 1024 / 1024:.2f} MB")
                print("")
                print("Open in browser:")
                print(f"  file:///{output_path.absolute().as_posix()}")

        except Exception as e:
            if verbose:
                print("")
                print(f"[ERROR] Conversion failed: {e}")
            traceback.print_exc()
            sys.exit(1)
