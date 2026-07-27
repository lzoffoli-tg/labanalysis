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
    A file conversion utility for transforming notebook files into various formats.

    This class provides methods to convert Jupyter notebook files (.ipynb) to
    HTML format with optional execution of code cells and customizable templates.

    Parameters
    ----------
    source_file : Path | str
        Path to the source file to be converted. Must be an existing file.

    Attributes
    ----------
    source_file : Path
        The path to the source file being converted.

    Examples
    --------
    >>> converter = Converter("example.ipynb")
    >>> converter.to_html()

    >>> converter = Converter("notebook.ipynb")
    >>> converter.to_html(output_path="output.html", execute=True)
    """

    def __init__(self, source_file: Path | str):
        """
        Initialize the Converter with a source file.

        Parameters
        ----------
        source_file : Path | str
            Path to the source file to be converted.

        Raises
        ------
        ValueError
            If the source file does not exist or is of invalid type.
        """
        self.set_source_file(source_file)

    def set_source_file(self, source_file: Path | str):
        """
        Set the source file path for conversion.

        Parameters
        ----------
        source_file : Path | str
            Path to the source file. Can be either a Path object or string.

        Raises
        ------
        ValueError
            If source_file is not a Path or str, or if the file does not exist.
        """
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
        """
        Get the source file path.

        Returns
        -------
        Path
            The path to the source file being converted.
        """
        return self._source_file

    def to_html(
        self,
        output_path: Path | str | None = None,
        execute: bool = False,
        template: Literal["custom_lab"] = "custom_lab",
        verbose: bool = True,
    ):
        """
        Convert source notebook file to HTML format.

        This method converts Jupyter notebook files (.ipynb) to HTML with optional
        code execution and custom template support. The output filename is
        automatically timestamped.

        Parameters
        ----------
        output_path : Path | str | None, default=None
            Path to the output HTML file. If None, uses the same directory as
            source_file with .html extension. The filename will be prepended
            with a timestamp in the format YYYYMMDD_HHMMSS_.
        execute : bool, default=False
            Whether to execute the notebook cells before conversion. If True,
            cells are executed sequentially with a 10-minute timeout per cell.
        template : {"custom_lab"}, default="custom_lab"
            The template to use for HTML conversion. Currently only "custom_lab"
            is supported.
        verbose : bool, default=True
            Whether to print detailed progress information during conversion,
            including execution status, file paths, and output file size.

        Raises
        ------
        ValueError
            If output_path is of invalid type, if execute is not a boolean,
            if template is not supported, if verbose is not a boolean,
            or if the conversion format is not supported.
        SystemExit
            If the template directory is not found.

        Examples
        --------
        >>> converter = Converter("notebook.ipynb")
        >>> converter.to_html()

        >>> converter.to_html(output_path="results/output.html", execute=True)

        >>> converter.to_html(execute=False, verbose=False)
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
        """
        Convert Jupyter notebook to HTML format.

        This is a private method that handles the actual conversion process using
        nbconvert's HTMLExporter. It configures the exporter with the specified
        template and optionally executes notebook cells before conversion.

        Parameters
        ----------
        output_path : Path
            Path where the HTML output will be saved.
        execute : bool
            Whether to execute notebook cells before conversion.
        template_basedir : Path
            Base directory containing the conversion templates.
        template : str
            Name of the template to use for conversion.
        verbose : bool
            Whether to print detailed progress information.

        Raises
        ------
        SystemExit
            If conversion fails due to any error during the process.

        Notes
        -----
        This method uses lazy importing of nbconvert and traitlets to avoid
        loading heavy dependencies when they are not needed.

        When execute=True, cells are executed with a 600-second (10-minute)
        timeout per cell using the 'python3' kernel.
        """

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
