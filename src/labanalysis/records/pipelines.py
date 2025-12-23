"""processing pipeline module"""

# -*- coding: utf-8 -*-


#! IMPORTS


from typing import Callable, List

import numpy as np

from ..signalprocessing import *
from .records import ForcePlatform, Record
from .timeseries import EMGSignal, Point3D, Signal1D, Signal3D, Timeseries

__all__ = [
    "ProcessingPipeline",
    "get_default_processing_pipeline",
    "get_default_emgsignal_processing_func",
    "get_default_signal1d_processing_func",
    "get_default_signal3d_processing_func",
    "get_default_point3d_processing_func",
    "get_default_forceplatform_processing_func",
]


class ProcessingPipeline:
    """
    A pipeline for processing various types of TimeseriesRecord-compatible
    objects.
    This class allows the user to define a sequence of processing functions
    for each supported object type and apply them to a collection of objects.
    """

    def __init__(self, **callables: Callable | List[Callable]):
        """
        Initialize a ProcessingPipeline.
        """
        object.__setattr__(self, "_items", {})
        self.add(**callables)

    def add(self, **callables: Callable | List[Callable]):
        for k, v in callables.items():
            self[k] = v

    def remove(self, key: str):
        self._items.pop(key)

    def pop(self, key: str):
        return self._items.pop(key)

    def get(self, key: str):
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
        object: Timeseries | Record,
        inplace: bool = False,
    ):
        return self.apply(object, inplace)

    def _apply_recursively(self, obj: Timeseries | Record):
        obj_type = type(obj)
        funcs = self.get(obj_type.__name__)
        if len(funcs) > 0:
            for func in funcs:
                func(obj)
        elif isinstance(obj, Record):
            for val in obj.values():
                self._apply_recursively(val)


def get_default_emgsignal_processing_func(channel: EMGSignal):
    channel[:, :] -= channel.to_numpy().mean()
    fsamp = 1 / np.mean(np.diff(channel.index))
    channel.apply(
        butterworth_filt,
        fcut=[20, 450],
        fsamp=fsamp,
        order=4,
        ftype="bandpass",
        phase_corrected=True,
        inplace=True,
        axis=0,
    )
    channel.apply(
        rms_filt,
        order=int(0.2 * fsamp),
        pad_style="reflect",
        offset=0.5,
        inplace=True,
        axis=0,
    )


def get_default_point3d_processing_func(point: Point3D):
    point.fillna(inplace=True)
    fsamp = float(1 / np.mean(np.diff(point.index)))
    point.apply(
        butterworth_filt,
        fcut=6,
        fsamp=fsamp,
        order=4,
        ftype="lowpass",
        phase_corrected=True,
        inplace=True,
    )


def get_default_signal3d_processing_func(signal: Signal3D):
    signal.fillna(inplace=True)
    fsamp = 1 / np.mean(np.diff(signal.index))
    signal.apply(
        butterworth_filt,
        fcut=6,
        fsamp=fsamp,
        order=4,
        ftype="lowpass",
        phase_corrected=True,
        inplace=True,
    )


def get_default_signal1d_processing_func(signal: Signal1D):
    signal.fillna(inplace=True)
    fsamp = 1 / np.mean(np.diff(signal.index))
    signal.apply(
        butterworth_filt,
        fcut=6,
        fsamp=fsamp,
        order=4,
        ftype="lowpass",
        phase_corrected=True,
        inplace=True,
    )


def get_default_forceplatform_processing_func(fp: ForcePlatform):

    def default_3d_processing_func(
        signal: Signal3D | Point3D,
    ):
        signal.fillna(inplace=True, value=0)
        fsamp = 1 / np.mean(np.diff(signal.index))
        signal.apply(
            butterworth_filt,
            fcut=30,
            fsamp=fsamp,
            order=4,
            ftype="lowpass",
            phase_corrected=True,
            inplace=True,
        )

    fp_pipeline = ProcessingPipeline(
        Point3D=[default_3d_processing_func],
        Signal3D=[default_3d_processing_func],
    )
    fp_pipeline(fp, inplace=True)


def get_default_processing_pipeline():

    return ProcessingPipeline(
        EMGSignal=[get_default_emgsignal_processing_func],
        Point3D=[get_default_point3d_processing_func],
        Signal1D=[get_default_signal1d_processing_func],
        Signal3D=[get_default_signal3d_processing_func],
        ForcePlatform=[get_default_forceplatform_processing_func],
    )
