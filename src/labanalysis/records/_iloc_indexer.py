"""Position-based indexer for Record objects."""

import numpy as np


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


__all__ = ["RecordILocIndexer"]
