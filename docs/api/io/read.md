# labanalysis.io.read

Data import functions for supported file formats.

**Source**: `src/labanalysis/io/read/`

## Functions

### `read_tdf(path: str | Path)`

Read data from a BTS Bioengineering TDF file.

- `path`: Path to the `.tdf` file.
- Returns a dictionary containing parsed camera, force platform, generic, EMG,
  event, volume, and version data.

### `read_npz(filename: str)`

Read an IRCAM pressure-mat image stream from an NPZ file.

- `filename`: Path to the `.npz` file.
- Returns a dictionary containing the image arrays stored in the file.
