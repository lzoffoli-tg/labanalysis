import numpy as np


class TimeseriesILocIndexer:
    """Position-based indexer for Timeseries (.iloc analog)."""

    def __init__(self, ts):
        self.ts = ts

    def _parse_key(self, key):
        if not isinstance(key, tuple):
            return key, slice(None)

        if len(key) == 2:
            return key[0], key[1]

        if len(key) == 1:
            return key[0], slice(None)

        raise IndexError("Too many indices for Timeseries")

    def _normalize_row_key(self, row_key):
        if row_key is None:
            return None

        if isinstance(row_key, slice) and row_key == slice(None):
            return None

        return self._normalize_positions(row_key, len(self.ts.index))

    def _normalize_col_key(self, col_key):
        if col_key is None:
            return None

        if isinstance(col_key, slice) and col_key == slice(None):
            return None

        return self._normalize_positions(col_key, len(self.ts.columns))

    def _normalize_positions(self, key, axis_size):
        if key is None:
            return None

        if isinstance(key, slice):
            start, stop, step = key.indices(axis_size)
            return slice(start, stop, step)

        if isinstance(key, (int, np.integer)):
            pos = int(key)
            if pos < -axis_size or pos >= axis_size:
                raise IndexError("ILoc index out of bounds")
            if pos < 0:
                pos += axis_size
            return pos

        if isinstance(key, (list, tuple, np.ndarray)):
            arr = np.asarray(key)

            if arr.dtype == bool:
                if arr.size != axis_size:
                    raise IndexError("Boolean index has wrong length")
                return np.flatnonzero(arr.astype(bool, copy=False))

            if arr.dtype.kind in "iu":
                pos = arr.astype(int, copy=False)
            else:
                raise TypeError("ILoc indices must be integers")

            if np.any(pos < -axis_size) or np.any(pos >= axis_size):
                raise IndexError("ILoc index out of bounds")

            pos = np.where(pos < 0, pos + axis_size, pos)
            return pos.astype(int, copy=False)

        raise TypeError(f"Invalid index type: {type(key)}")

    def _set_data(self, row_pos, col_pos, value):
        if row_pos is None and col_pos is None:
            self.ts._data[...] = value
        elif row_pos is None:
            self.ts._data[:, col_pos] = value
        elif col_pos is None:
            self.ts._data[row_pos, :] = value
        elif isinstance(row_pos, (int, np.integer)) and isinstance(
            col_pos, (int, np.integer)
        ):
            self.ts._data[row_pos, col_pos] = value
        elif isinstance(row_pos, np.ndarray) and isinstance(col_pos, np.ndarray):
            self.ts._data[np.ix_(row_pos, col_pos)] = value
        else:
            self.ts._data[row_pos, col_pos] = value

    def __getitem__(self, key):
        row_key, col_key = self._parse_key(key)

        row_pos = self._normalize_row_key(row_key)
        col_pos = self._normalize_col_key(col_key)

        if isinstance(row_pos, (int, np.integer)) and isinstance(
            col_pos, (int, np.integer)
        ):
            return self.ts._data[row_pos, col_pos]

        return self.ts._view(rows=row_pos, cols=col_pos)

    def __setitem__(self, key, value):
        row_key, col_key = self._parse_key(key)

        row_pos = self._normalize_row_key(row_key)
        col_pos = self._normalize_col_key(col_key)

        self._set_data(row_pos, col_pos, value)


__all__ = ["TimeseriesILocIndexer"]
