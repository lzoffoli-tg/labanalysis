import copy

import numpy as np


class RecordLocIndexer:
    """Label-based indexer for Record objects (.loc analog)."""

    def __init__(self, rec):
        self.rec = rec

    def _parse_key(self, key):
        if not isinstance(key, tuple):
            return key, slice(None)

        if len(key) == 2:
            return key[0], key[1]

        if len(key) == 1:
            return key[0], slice(None)

        raise IndexError("Too many indices for Record")

    def _normalize_row_key(self, row_key):
        if row_key is None:
            return None

        if isinstance(row_key, slice) and row_key == slice(None):
            return None

        return row_key

    def _normalize_col_key(self, col_key):
        if col_key is None:
            return None

        if isinstance(col_key, slice) and col_key == slice(None):
            return None

        return col_key

    @staticmethod
    def _set_attr(obj, name, value):
        object.__setattr__(obj, name, value)

    def _clone_record(self):
        return self.rec.__class__.__new__(self.rec.__class__)

    def _is_record_like(self, value):
        return (
            hasattr(value, "_data")
            and hasattr(value, "keys")
            and hasattr(value, "values")
            and hasattr(value, "loc")
            and hasattr(value, "iloc")
            and isinstance(getattr(value, "_data", None), dict)
        )

    def _apply_to_value(self, value, row_key, col_key):
        if value is None:
            return value

        if isinstance(value, (str, bytes, int, float, bool, complex, np.number)):
            return value

        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            if value.ndim == 1:
                return value[row_key] if row_key is not None else value
            return value[row_key, col_key]

        if isinstance(value, (list, tuple)):
            if row_key is None:
                return value
            return value[row_key]

        if isinstance(value, dict):
            return {
                key: self._apply_to_value(item, row_key, col_key)
                for key, item in value.items()
            }

        if self._is_record_like(value):
            try:
                new_obj = value.__class__.__new__(value.__class__)
                for name, attr_value in value.__dict__.items():
                    if name in {"loc", "iloc"}:
                        continue
                    if name == "_data":
                        self._set_attr(
                            new_obj,
                            name,
                            {
                                key: self._apply_to_value(item, row_key, col_key)
                                for key, item in value._data.items()
                            },
                        )
                    else:
                        self._set_attr(
                            new_obj,
                            name,
                            self._apply_to_value(attr_value, row_key, col_key),
                        )
                return new_obj
            except Exception:
                pass

        if hasattr(value, "loc"):
            try:
                return value.loc[row_key, col_key]
            except Exception:
                try:
                    return value.loc[row_key]
                except Exception:
                    return value

        if hasattr(value, "iloc"):
            try:
                return value.iloc[row_key, col_key]
            except Exception:
                try:
                    return value.iloc[row_key]
                except Exception:
                    return value

        if hasattr(value, "_view"):
            try:
                return value._view(rows=row_key, cols=col_key)
            except Exception:
                try:
                    return value._view(row_key, col_key)
                except Exception:
                    return value

        if hasattr(value, "__getitem__") and not isinstance(value, (dict, type)):
            try:
                return value[row_key, col_key]
            except Exception:
                try:
                    return value[row_key]
                except Exception:
                    return value

        return copy.copy(value)

    def __getitem__(self, key):
        row_key, col_key = self._parse_key(key)

        row_key = self._normalize_row_key(row_key)
        col_key = self._normalize_col_key(col_key)

        new_obj = self._clone_record()

        """
        for name, value in self.rec.__dict__.items():
            if name in {"loc", "iloc"}:
                continue
            self._set_attr(new_obj, name, value)
        """
        for name, value in self.rec.__dict__.items():
            if name in {"loc", "iloc"}:
                continue
            self._set_attr(
                new_obj,
                name,
                self._apply_to_value(value, row_key, col_key),
            )

        return new_obj

    def __setitem__(self, key, value):
        row_key, col_key = self._parse_key(key)

        row_key = self._normalize_row_key(row_key)
        col_key = self._normalize_col_key(col_key)

        for name, attr_value in self.rec.__dict__.items():
            if name in {"loc", "iloc"}:
                continue

            if self._is_record_like(attr_value):
                try:
                    attr_value.loc[row_key, col_key] = value
                    continue
                except Exception:
                    pass

            if hasattr(attr_value, "loc"):
                try:
                    attr_value.loc[row_key, col_key] = value
                    continue
                except Exception:
                    pass

            if hasattr(attr_value, "iloc"):
                try:
                    attr_value.iloc[row_key, col_key] = value
                    continue
                except Exception:
                    pass

            if hasattr(attr_value, "__setitem__"):
                try:
                    attr_value[row_key, col_key] = value
                    continue
                except Exception:
                    pass

            if isinstance(attr_value, np.ndarray):
                try:
                    attr_value[row_key, col_key] = value
                    continue
                except Exception:
                    pass

            if isinstance(attr_value, list):
                try:
                    attr_value[row_key] = value
                    continue
                except Exception:
                    pass

            self._set_attr(self.rec, name, value)


__all__ = ["RecordLocIndexer"]
