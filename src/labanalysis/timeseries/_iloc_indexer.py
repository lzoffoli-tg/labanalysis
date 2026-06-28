"""
Position-based indexer for Timeseries (.iloc accessor).
"""

import numpy as np


class TimeseriesILocIndexer:
    """Position-based indexer for Timeseries (pandas .iloc analog).

    Provides integer position-based indexing for Timeseries objects.
    Returns views that share the underlying _data buffer with the original object.
    Does NOT support auto-expansion (raises IndexError for out-of-bounds positions).

    Parameters
    ----------
    ts : Timeseries
        The Timeseries object to index.

    Examples
    --------
    >>> ts.iloc[0, 0]  # Get first row, first column
    >>> ts.iloc[:10, :]  # Get first 10 rows, all columns
    >>> ts.iloc[:, [0, 1]] = 0  # Set first two columns to zero
    >>> ts.iloc[-1, :] = 1  # Set last row to 1
    """

    def __init__(self, ts):
        self.ts = ts

    def _parse_key(self, key):
        """Parse key into (row_key, col_key)."""
        if not isinstance(key, tuple):
            return (key, slice(None))
        return (key[0], key[1] if len(key) > 1 else slice(None))

    def _normalize_positions(self, key, axis_size):
        """Convert to integer positions array or None for all positions."""
        if key is None:
            return None
        if isinstance(key, slice) and key == slice(None):
            return None
        if isinstance(key, int):
            if key < -axis_size or key >= axis_size:
                raise IndexError(f"Index {key} out of bounds for axis with size {axis_size}")
            return np.array([key if key >= 0 else axis_size + key])
        if isinstance(key, slice):
            return np.arange(*key.indices(axis_size))
        if isinstance(key, (list, np.ndarray)):
            arr = np.asarray(key)
            if arr.dtype == bool:
                if len(arr) != axis_size:
                    raise IndexError(f"Boolean index has wrong length: {len(arr)} vs {axis_size}")
                return np.where(arr)[0]
            else:
                positions = arr.astype(int)
                if np.any((positions < -axis_size) | (positions >= axis_size)):
                    raise IndexError("Index out of bounds")
                return np.where(positions < 0, positions + axis_size, positions)
        raise TypeError(f"Invalid index type: {type(key)}")

    def __getitem__(self, key):
        """Get data using position-based indexing.

        Returns a VIEW that shares the underlying _data buffer with the original.
        """
        row_key, col_key = self._parse_key(key)
        row_pos = self._normalize_positions(row_key, len(self.ts.index))
        col_pos = self._normalize_positions(col_key, len(self.ts.columns))

        row_labels = None if row_pos is None else self.ts.index[row_pos]
        col_labels = None if col_pos is None else self.ts.columns[col_pos]

        return self.ts._view(rows=row_labels, cols=col_labels)

    def __setitem__(self, key, value):
        """Set data using position-based indexing.

        Does NOT support auto-expansion. Raises IndexError for out-of-bounds positions.
        """
        row_key, col_key = self._parse_key(key)
        row_pos = self._normalize_positions(row_key, len(self.ts.index))
        col_pos = self._normalize_positions(col_key, len(self.ts.columns))

        value_arr = np.asarray(value, dtype=float)
        if value_arr.ndim == 0:
            value_arr = value_arr.reshape(1, 1)
        elif value_arr.ndim == 1:
            if row_pos is not None and col_pos is not None:
                if len(col_pos) == 1:
                    value_arr = value_arr.reshape(-1, 1)
                elif len(row_pos) == 1:
                    value_arr = value_arr.reshape(1, -1)

        if row_pos is None and col_pos is None:
            self.ts._data[:, :] = value_arr
        elif row_pos is None:
            self.ts._data[:, col_pos] = value_arr
        elif col_pos is None:
            self.ts._data[row_pos, :] = value_arr
        else:
            self.ts._data[np.ix_(row_pos, col_pos)] = value_arr


__all__ = ["TimeseriesILocIndexer"]
