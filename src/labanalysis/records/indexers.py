"""
Indexer classes for label-based (.loc) and position-based (.iloc) indexing.

This module provides pandas-like indexing capabilities for Timeseries and Record objects.
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
            # Convert slice to array using index bounds
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
            # Slice by column positions
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
            return np.asarray(arr, dtype='<U100')  # Unicode string array
        raise TypeError(f"Invalid column key type: {type(col_key)}")

    def __getitem__(self, key):
        """Get data using label-based indexing.

        Returns a VIEW that shares the underlying _data buffer with the original.
        Modifications to the returned object will affect the original.
        """
        row_key, col_key = self._parse_key(key)
        row_labels = self._normalize_row_labels(row_key)
        col_labels = self._normalize_col_labels(col_key)

        # Convert labels to positions for _view method
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

        # Use existing _view method which returns view objects sharing _data buffer
        return self.ts._view(rows=row_indices, cols=col_indices)

    def __setitem__(self, key, value):
        """Set data using label-based indexing.

        Supports auto-expansion: if row/column labels don't exist, they are added.
        After adding rows, the data is sorted by index to maintain sorted order.
        """
        row_key, col_key = self._parse_key(key)
        row_labels = self._normalize_row_labels(row_key)
        col_labels = self._normalize_col_labels(col_key)

        # Find or create row positions
        if row_labels is None:
            row_positions = np.arange(len(self.ts.index))
        else:
            row_positions = []
            for label in row_labels:
                # Find existing row with np.isclose for float matching
                matches = np.where(np.isclose(self.ts.index, label, rtol=1e-6, atol=1e-8))[0]
                if len(matches) > 0:
                    row_positions.append(matches[0])
                else:
                    # Auto-expand: add new row
                    new_idx = len(self.ts.index)
                    self.ts.index = np.append(self.ts.index, label)
                    self.ts._data = np.vstack([self.ts._data, np.full(len(self.ts.columns), np.nan)])
                    row_positions.append(new_idx)
            row_positions = np.array(row_positions)

        # Find or create column positions
        if col_labels is None:
            col_positions = np.arange(len(self.ts.columns))
        else:
            col_positions = []
            for label in col_labels:
                if label in self.ts.columns:
                    col_positions.append(np.where(self.ts.columns == label)[0][0])
                else:
                    # Auto-expand: add new column
                    new_idx = len(self.ts.columns)
                    self.ts.columns = np.append(self.ts.columns, label)
                    self.ts._data = np.hstack([self.ts._data, np.full((len(self.ts.index), 1), np.nan)])
                    col_positions.append(new_idx)
            col_positions = np.array(col_positions)

        # Assign values
        value_arr = np.asarray(value, dtype=float)
        if value_arr.ndim == 0:
            value_arr = value_arr.reshape(1, 1)
        elif value_arr.ndim == 1:
            if len(col_positions) == 1:
                value_arr = value_arr.reshape(-1, 1)
            elif len(row_positions) == 1:
                value_arr = value_arr.reshape(1, -1)

        self.ts._data[np.ix_(row_positions, col_positions)] = value_arr

        # Sort by index to maintain sorted order (only if rows were added)
        if row_labels is not None:
            sort_idx = np.argsort(self.ts.index)
            if not np.array_equal(sort_idx, np.arange(len(self.ts.index))):
                self.ts.index = self.ts.index[sort_idx]
                self.ts._data = self.ts._data[sort_idx, :]


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
                # Integer array
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

        # Convert positions to labels and delegate to _view() for proper type preservation
        row_labels = None if row_pos is None else self.ts.index[row_pos]
        col_labels = None if col_pos is None else self.ts.columns[col_pos]

        # Use _view() which handles type preservation correctly for all subclasses
        return self.ts._view(rows=row_labels, cols=col_labels)

    def __setitem__(self, key, value):
        """Set data using position-based indexing.

        Does NOT support auto-expansion. Raises IndexError for out-of-bounds positions.
        """
        row_key, col_key = self._parse_key(key)
        row_pos = self._normalize_positions(row_key, len(self.ts.index))
        col_pos = self._normalize_positions(col_key, len(self.ts.columns))

        # Assign values (NO auto-expansion for iloc)
        value_arr = np.asarray(value, dtype=float)
        if value_arr.ndim == 0:
            value_arr = value_arr.reshape(1, 1)
        elif value_arr.ndim == 1:
            # Determine if value should be row or column
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


class RecordLocIndexer:
    """Label-based indexer for Record with tuple column specs.

    Provides label-based indexing for Record objects with support for:
    - Item keys as strings
    - Tuple specs: ('item_key', 'column_name') to access specific columns of items
    - Mixed lists: ['item1', ('item2', 'X')]

    Parameters
    ----------
    rec : Record
        The Record object to index.

    Examples
    --------
    >>> rec.loc[0.5, 'force']  # Get 'force' item at time 0.5
    >>> rec.loc[:, ('force', 'X')]  # Get column 'X' of 'force' item
    >>> rec.loc[:, [('force', 'X'), 'torque']]  # Mixed spec
    """

    def __init__(self, rec):
        self.rec = rec

    def _parse_key(self, key):
        """Parse into (row_key, col_spec)."""
        if not isinstance(key, tuple):
            return (key, None)
        return (key[0], key[1] if len(key) > 1 else None)

    def _parse_col_spec(self, col_spec):
        """Parse column spec into list of (item_key, column_spec) tuples."""
        if col_spec is None:
            return [(key, None) for key in self.rec.keys()]
        if isinstance(col_spec, slice) and col_spec == slice(None):
            return [(key, None) for key in self.rec.keys()]
        if isinstance(col_spec, str):
            return [(col_spec, None)]
        if isinstance(col_spec, tuple) and len(col_spec) == 2:
            if isinstance(col_spec[0], str):
                return [col_spec]
            raise ValueError("First element of tuple must be item key (string)")
        if isinstance(col_spec, (list, np.ndarray)):
            result = []
            for spec in col_spec:
                if isinstance(spec, str):
                    result.append((spec, None))
                elif isinstance(spec, tuple) and len(spec) == 2:
                    result.append(spec)
                else:
                    raise ValueError(f"Invalid column spec: {spec}")
            return result
        raise ValueError(f"Unsupported column specification: {col_spec}")

    def __getitem__(self, key):
        """Get data using label-based indexing."""
        row_key, col_spec = self._parse_key(key)
        item_col_pairs = self._parse_col_spec(col_spec)

        # Extract data for each (item, column) pair
        result_dict = {}
        for item_key, column_spec in item_col_pairs:
            if item_key not in self.rec.keys():
                raise KeyError(f"Item '{item_key}' not found in Record")

            item = self.rec._data[item_key]
            if column_spec is None:
                result_dict[item_key] = item.loc[row_key, :]
            else:
                result_dict[item_key] = item.loc[row_key, column_spec]

        return type(self.rec)(**result_dict)

    def __setitem__(self, key, value):
        """Set data using label-based indexing."""
        row_key, col_spec = self._parse_key(key)
        item_col_pairs = self._parse_col_spec(col_spec)

        if isinstance(value, (int, float, np.ndarray)):
            # Scalar/array broadcast
            for item_key, column_spec in item_col_pairs:
                if item_key not in self.rec.keys():
                    raise KeyError(f"Cannot set non-existent item '{item_key}'. "
                                 f"Add it first using rec['{item_key}'] = Timeseries(...)")
                item = self.rec._data[item_key]
                if column_spec is None:
                    item.loc[row_key, :] = value
                else:
                    item.loc[row_key, column_spec] = value

        elif isinstance(value, dict):
            # Dictionary mapping item_key → value
            for item_key, column_spec in item_col_pairs:
                if item_key not in value:
                    raise ValueError(f"Missing value for item '{item_key}'")
                item = self.rec._data[item_key]
                if column_spec is None:
                    item.loc[row_key, :] = value[item_key]
                else:
                    item.loc[row_key, column_spec] = value[item_key]
        else:
            raise TypeError(f"Unsupported value type: {type(value)}")


class RecordILocIndexer:
    """Position-based indexer for Record with tuple column specs.

    Provides integer position-based indexing for Record objects with support for:
    - Item positions as integers
    - Tuple specs: (item_pos, column_pos) to access specific columns
    - Slices and lists

    Parameters
    ----------
    rec : Record
        The Record object to index.

    Examples
    --------
    >>> rec.iloc[0:10, 0]  # Get first 10 rows of first item
    >>> rec.iloc[:, (0, 1)]  # Get column 1 of first item
    >>> rec.iloc[:, [(0, 1), 2:]]  # Mixed spec
    """

    def __init__(self, rec):
        self.rec = rec

    def _parse_key(self, key):
        """Parse into (row_key, col_spec)."""
        if not isinstance(key, tuple):
            return (key, None)
        return (key[0], key[1] if len(key) > 1 else None)

    def _parse_col_spec(self, col_spec):
        """Parse column spec into list of (item_pos, column_spec) tuples."""
        item_keys = self.rec.keys()
        n_items = len(item_keys)

        if col_spec is None:
            return [(i, None) for i in range(n_items)]

        if isinstance(col_spec, int):
            if col_spec < -n_items or col_spec >= n_items:
                raise IndexError(f"Item position {col_spec} out of bounds for {n_items} items")
            pos = col_spec if col_spec >= 0 else n_items + col_spec
            return [(pos, None)]

        if isinstance(col_spec, slice):
            positions = list(range(*col_spec.indices(n_items)))
            return [(pos, None) for pos in positions]

        if isinstance(col_spec, tuple) and len(col_spec) == 2:
            item_pos, col_pos = col_spec
            if not isinstance(item_pos, int):
                raise TypeError("Item position must be integer")
            if item_pos < -n_items or item_pos >= n_items:
                raise IndexError(f"Item position {item_pos} out of bounds for {n_items} items")
            pos = item_pos if item_pos >= 0 else n_items + item_pos
            return [(pos, col_pos)]

        if isinstance(col_spec, (list, np.ndarray)):
            result = []
            for spec in col_spec:
                parsed = self._parse_col_spec(spec)
                result.extend(parsed)
            return result

        raise ValueError(f"Unsupported column specification: {col_spec}")

    def __getitem__(self, key):
        """Get data using position-based indexing."""
        row_key, col_spec = self._parse_key(key)
        item_col_pairs = self._parse_col_spec(col_spec)
        item_keys = self.rec.keys()

        result_dict = {}
        for item_pos, column_spec in item_col_pairs:
            item_key = item_keys[item_pos]
            item = self.rec._data[item_key]
            if column_spec is None:
                result_dict[item_key] = item.iloc[row_key, :]
            else:
                result_dict[item_key] = item.iloc[row_key, column_spec]

        return type(self.rec)(**result_dict)

    def __setitem__(self, key, value):
        """Set data using position-based indexing."""
        row_key, col_spec = self._parse_key(key)
        item_col_pairs = self._parse_col_spec(col_spec)
        item_keys = self.rec.keys()

        if isinstance(value, (int, float, np.ndarray)):
            for item_pos, column_spec in item_col_pairs:
                item_key = item_keys[item_pos]
                item = self.rec._data[item_key]
                if column_spec is None:
                    item.iloc[row_key, :] = value
                else:
                    item.iloc[row_key, column_spec] = value

        elif isinstance(value, dict):
            for item_pos, column_spec in item_col_pairs:
                item_key = item_keys[item_pos]
                if item_key not in value:
                    raise ValueError(f"Missing value for item '{item_key}'")
                item = self.rec._data[item_key]
                if column_spec is None:
                    item.iloc[row_key, :] = value[item_key]
                else:
                    item.iloc[row_key, column_spec] = value[item_key]
        else:
            raise TypeError(f"Unsupported value type: {type(value)}")
