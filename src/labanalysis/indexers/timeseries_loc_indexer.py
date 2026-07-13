import numpy as np


class TimeseriesLocIndexer:
    """Label-based indexer for Timeseries (.loc analog)."""

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

        return row_key

    def _normalize_col_key(self, col_key):
        if col_key is None:
            return None

        if isinstance(col_key, slice) and col_key == slice(None):
            return None

        return col_key

    def _resolve_labels(self, selector, labels):
        if selector is None:
            return None

        labels_arr = np.asarray(labels)

        if isinstance(selector, slice):
            return self._resolve_slice(selector, labels_arr)

        if isinstance(selector, (list, tuple, np.ndarray)):
            values = np.asarray(selector)

            if values.dtype == bool:
                if values.size != len(labels_arr):
                    raise IndexError("Boolean selector has wrong length.")
                return np.flatnonzero(values.astype(bool, copy=False))

            mask = np.zeros(len(labels_arr), dtype=bool)

            if labels_arr.dtype.kind in "iuf":
                for value in values.astype(float, copy=False):
                    mask |= np.isclose(
                        labels_arr.astype(float, copy=False),
                        float(value),
                        rtol=1e-6,
                        atol=1e-8,
                    )
            else:
                for value in values.tolist():
                    mask |= labels_arr == value

            return np.flatnonzero(mask)

        if labels_arr.dtype.kind in "iuf":
            matches = np.flatnonzero(
                np.isclose(
                    labels_arr.astype(float, copy=False),
                    float(selector),
                    rtol=1e-6,
                    atol=1e-8,
                )
            )
            if matches.size == 0:
                raise KeyError(selector)
            return int(matches[0])

        matches = np.flatnonzero(labels_arr == selector)
        if matches.size == 0:
            raise KeyError(selector)
        return int(matches[0])

    def _resolve_slice(self, selector, labels_arr):
        if labels_arr.size == 0:
            return np.array([], dtype=int)

        if labels_arr.dtype.kind in "iuf":
            values = labels_arr.astype(float, copy=False)
            start = selector.start
            stop = selector.stop
            step = 1 if selector.step is None else selector.step

            if start is None:
                start = values[0]
            if stop is None:
                stop = values[-1]

            if step > 0:
                mask = (values >= start) & (values <= stop)
            else:
                mask = (values <= start) & (values >= stop)

            positions = np.flatnonzero(mask)

            if step != 1:
                positions = positions[::step]

            return positions.astype(int, copy=False)

        values = labels_arr.tolist()
        start = selector.start
        stop = selector.stop
        step = 1 if selector.step is None else selector.step

        selected = [
            i
            for i, value in enumerate(values)
            if (start is None or value >= start) and (stop is None or value <= stop)
        ]

        if step < 0:
            selected = selected[::-1]

        if abs(step) != 1:
            selected = selected[::step]

        return np.array(selected, dtype=int)

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

        row_key = self._normalize_row_key(row_key)
        col_key = self._normalize_col_key(col_key)

        row_pos = self._resolve_labels(row_key, self.ts.index)
        col_pos = self._resolve_labels(col_key, self.ts.columns)

        if isinstance(row_pos, (int, np.integer)) and isinstance(
            col_pos, (int, np.integer)
        ):
            return self.ts._data[row_pos, col_pos]

        return self.ts._view(rows=row_pos, cols=col_pos)

    def __setitem__(self, key, value):
        row_key, col_key = self._parse_key(key)

        row_key = self._normalize_row_key(row_key)
        col_key = self._normalize_col_key(col_key)

        row_pos = self._resolve_labels(row_key, self.ts.index)
        col_pos = self._resolve_labels(col_key, self.ts.columns)

        self._set_data(row_pos, col_pos, value)


__all__ = ["TimeseriesLocIndexer"]
