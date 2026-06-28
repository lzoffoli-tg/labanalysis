"""
Processing pipeline for biomechanical data.
"""

from typing import Callable

from ..timeseries import Timeseries
from ..records import Record


class ProcessingPipeline:
    """
    A pipeline for processing various types of TimeseriesRecord-compatible
    objects.
    This class allows the user to define a sequence of processing functions
    for each supported object type and apply them to a collection of objects.
    """

    def __init__(self, **callables: Callable | list[Callable]):
        """
        Initialize a ProcessingPipeline.
        """
        object.__setattr__(self, "_items", {})
        self.add(**callables)

    def add(self, **callables: Callable | list[Callable]):
        """
        Add processing functions to the pipeline.

        Parameters
        ----------
        **callables : Callable or list of Callable
            Keyword arguments where keys are object type names and values are
            processing functions or lists of functions.
        """
        for k, v in callables.items():
            self[k] = v

    def remove(self, key: str):
        """
        Remove all processing functions for a given object type.

        Parameters
        ----------
        key : str
            Object type name to remove from pipeline.
        """
        self._items.pop(key)

    def pop(self, key: str):
        """
        Remove and return processing functions for a given object type.

        Parameters
        ----------
        key : str
            Object type name to pop from pipeline.

        Returns
        -------
        Callable or list of Callable
            Processing function(s) that were removed.
        """
        return self._items.pop(key)

    def get(self, key: str):
        """
        Get processing functions for a given object type.

        Parameters
        ----------
        key : str
            Object type name to retrieve.

        Returns
        -------
        list of Callable
            Processing functions for the object type, or empty list if not found.
        """
        default: list[Callable] = []
        return self._items.get(key, default)

    def apply(
        self,
        object: Timeseries | Record,
        inplace: bool = False,
    ):
        """
        Apply the processing pipeline to the given objects.

        Parameters
        ----------
        *objects : variable length argument list
            Objects to process. Can be individual Signal1D, Signal3D, Point3D,
            EMGSignal, ForcePlatform, or TimeseriesRecord instances.
        inplace : bool, optional
            If True, modifies the objects in place. If False, returns the
            processed copies.

        Returns
        -------
        list or None
            If inplace is False, returns a list of processed objects.
            Otherwise, returns None.
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be True or False")
        processed = object.copy() if not inplace else object
        self._apply_recursively(processed)
        if not inplace:
            return processed

    def keys(self):
        return list(self._items.keys())

    def values(self):
        return list(self._items.values())

    def items(self):
        return self._items.items()

    def __repr__(self):
        return self._items.__repr__()

    def __str__(self):
        return self._items.__str__()

    def __setitem__(self, item, value):
        calls = [value] if not isinstance(value, list) else value
        if not all([isinstance(i, Callable) for i in calls]):
            msg = "callables must be Callable objects or lists of "
            msg += "Callable objects."
            raise ValueError(msg)
        self._items[item] = value

    def __getitem__(self, item: str):
        return self._items[item]

    def __getattr__(self, attr: str):
        return self._items[attr]

    def __setattr__(self, attr, value):
        self._items.__setitem__(attr, value)

    def __call__(
        self,
        obj: Timeseries | Record,
        inplace: bool = False,
    ):
        return self.apply(obj, inplace)

    def _apply_recursively(self, obj: Timeseries | Record):
        obj_type = type(obj)
        funcs = self.get(obj_type.__name__)
        if len(funcs) > 0:
            for func in funcs:
                func(obj)
        elif isinstance(obj, Record):
            for val in obj.values():
                self._apply_recursively(val)
