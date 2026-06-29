"""Label-based indexer for Record objects."""

import numpy as np


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

        # Get constructor args to preserve custom attributes (e.g., bodymass_kg, box_height_cm)
        if hasattr(self.rec, '_get_constructor_args'):
            constructor_args = self.rec._get_constructor_args()
            # Update with sliced signals
            constructor_args.update(result_dict)
            return type(self.rec)(**constructor_args)
        else:
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


__all__ = ["RecordLocIndexer"]
