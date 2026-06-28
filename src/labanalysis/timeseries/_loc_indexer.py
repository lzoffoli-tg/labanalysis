"""
Label-based indexer for Timeseries (.loc accessor).
"""

import numpy as np


class TimeseriesLocIndexer:
    """Label-based indexer for Timeseries (pandas .loc analog).

    Provides label-based indexing using time index values (float) and column names (string).
    Returns views that share the underlying _data buffer with the original object.
    Supports auto-expansion of rows and columns when setting values.

    Parameters
    ----------
    ts : Timeseries
        The Timeseries object to index.

    Examples
    --------
    >>> ts.loc[0.5, 'X']  # Get value at time 0.5, column 'X'
    >>> ts.loc[:, ['X', 'Y']]  # Get columns X and Y
    >>> ts.loc[0:1, :] = 0  # Set time range to zero
    >>> ts.loc[[0.1, 0.2], 'X'] = [1, 2]  # Set specific times
    """

    def __init__(self, ts):
        self.ts = ts

    def _parse_key(self, key):
        """Parse key into (row_key, col_key). Single key → (key, slice(None))."""
        if not isinstance(key, tuple):
            return (key, slice(None))
        return (key[0], key[1] if len(key) > 1 else slice(None))

    def _normalize_row_labels(self, row_key):
        """Convert to array of float labels or None for all rows."""
        if row_key is None:
            return None
        if isinstance(row_key, slice) and row_key == slice(None):
            return None
        if isinstance(row_key, (int, float)):
            return np.array([float(row_key)])
        if isinstance(row_key, slice):
            mask = (self.ts.index >= (row_key.start or -np.inf)) & \
                   (self.ts.index <= (row_key.stop or np.inf))
            return self.ts.index[mask]
        if isinstance(row_key, (list, np.ndarray)):
            arr = np.asarray(row_key)
            if arr.dtype == bool:
                return self.ts.index[arr]
            return np.asarray(arr, dtype=float)
        raise TypeError(f"Invalid row key type: {type(row_key)}")

    def _normalize_col_labels(self, col_key):
        """Convert to array of string labels or None for all columns."""
        if col_key is None:
            return None
        if isinstance(col_key, slice) and col_key == slice(None):
            return None
        if isinstance(col_key, str):
            return np.array([col_key])
        if isinstance(col_key, slice):
            if col_key.start is not None:
                start_matches = np.where(self.ts.columns == col_key.start)[0]
                start_idx = start_matches[0] if len(start_matches) > 0 else 0
            else:
                start_idx = 0

            if col_key.stop is not None:
                stop_matches = np.where(self.ts.columns == col_key.stop)[0]
                stop_idx = stop_matches[0] + 1 if len(stop_matches) > 0 else len(self.ts.columns)
            else:
                stop_idx = len(self.ts.columns)

            return self.ts.columns[start_idx:stop_idx]
        if isinstance(col_key, (list, np.ndarray)):
            arr = np.asarray(col_key)
            if arr.dtype == bool:
                return self.ts.columns[arr]
            return np.asarray(arr, dtype='<U100')
        raise TypeError(f"Invalid column key type: {type(col_key)}")

    def __getitem__(self, key):
        """Get data using label-based indexing.

        Returns a VIEW that shares the underlying _data buffer with the original.
        Modifications to the returned object will affect the original.
        """
        row_key, col_key = self._parse_key(key)
        row_labels = self._normalize_row_labels(row_key)
        col_labels = self._normalize_col_labels(col_key)

        if row_labels is None:
            row_indices = None
        else:
            row_indices = []
            for label in row_labels:
                matches = np.where(np.isclose(self.ts.index, label, rtol=1e-6, atol=1e-8))[0]
                if len(matches) > 0:
                    row_indices.append(matches[0])
                else:
                    raise KeyError(f"Index {label} not found")
            row_indices = self.ts.index[np.array(row_indices)]

        if col_labels is None:
            col_indices = None
        else:
            col_indices = []
            for label in col_labels:
                if label in self.ts.columns:
                    col_indices.append(label)
                else:
                    raise KeyError(f"Column '{label}' not found")
            col_indices = np.array(col_indices)

        return self.ts._view(rows=row_indices, cols=col_indices)

    def __setitem__(self, key, value):
        """Set data using label-based indexing.

        Supports auto-expansion: if row/column labels don't exist, they are added.
        After adding rows, the data is sorted by index to maintain sorted order.
        """
        row_key, col_key = self._parse_key(key)
        row_labels = self._normalize_row_labels(row_key)
        col_labels = self._normalize_col_labels(col_key)

        if row_labels is None:
            row_positions = np.arange(len(self.ts.index))
        else:
            row_positions = []
            for label in row_labels:
                matches = np.where(np.isclose(self.ts.index, label, rtol=1e-6, atol=1e-8))[0]
                if len(matches) > 0:
                    row_positions.append(matches[0])
                else:
                    new_idx = len(self.ts.index)
                    self.ts.index = np.append(self.ts.index, label)
                    self.ts._data = np.vstack([self.ts._data, np.full(len(self.ts.columns), np.nan)])
                    row_positions.append(new_idx)
            row_positions = np.array(row_positions)

        if col_labels is None:
            col_positions = np.arange(len(self.ts.columns))
        else:
            col_positions = []
            for label in col_labels:
                if label in self.ts.columns:
                    col_positions.append(np.where(self.ts.columns == label)[0][0])
                else:
                    new_idx = len(self.ts.columns)
                    self.ts.columns = np.append(self.ts.columns, label)
                    self.ts._data = np.hstack([self.ts._data, np.full((len(self.ts.index), 1), np.nan)])
                    col_positions.append(new_idx)
            col_positions = np.array(col_positions)

        value_arr = np.asarray(value, dtype=float)
        if value_arr.ndim == 0:
            value_arr = value_arr.reshape(1, 1)
        elif value_arr.ndim == 1:
            if len(col_positions) == 1:
                value_arr = value_arr.reshape(-1, 1)
            elif len(row_positions) == 1:
                value_arr = value_arr.reshape(1, -1)

        self.ts._data[np.ix_(row_positions, col_positions)] = value_arr

        if row_labels is not None:
            sort_idx = np.argsort(self.ts.index)
            if not np.array_equal(sort_idx, np.arange(len(self.ts.index))):
                self.ts.index = self.ts.index[sort_idx]
                self.ts._data = self.ts._data[sort_idx, :]


__all__ = ["TimeseriesLocIndexer"]
