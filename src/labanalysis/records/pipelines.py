"""processing pipeline module"""

# -*- coding: utf-8 -*-


#! IMPORTS


from typing import Callable, List
from . import *
from ..signalprocessing import *
import numpy as np

__all__ = ["ProcessingPipeline", "default_processing_pipeline"]


class ProcessingPipeline:
    """
    A pipeline for processing various types of TimeseriesRecord-compatible
    objects.
    This class allows the user to define a sequence of processing functions
    for each supported object type and apply them to a collection of objects.
    """

    def __init__(self, callables: dict[str, Callable | List[Callable]]):
        """
        Initialize a ProcessingPipeline.
        """
        self.pipeline: dict[str, List[Callable]] = {
            i: [v] if not isinstance(v, list) else v for i, v in callables.items()
        }

    def __call__(
        self,
        object: Timeseries | Record,
        inplace: bool = False,
    ):
        return self.apply(object, inplace)

    def _apply_recursively(self, obj: Timeseries | Record):
        obj_type = type(obj)
        funcs = self.pipeline.get(obj_type.__name__, [])
        if len(funcs) > 0:
            for func in funcs:
                func(obj)
        elif isinstance(obj, Record):
            for val in obj.values():
                self._apply_recursively(val)

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


def default_processing_pipeline(inplace: bool = False):

    def default_emgsignal_processing_func(
        channel: EMGSignal,
        inplace: bool = False,
    ):
        out = channel if inplace else channel.copy()
        out[:, :] -= out.to_numpy().mean()
        fsamp = 1 / np.mean(np.diff(out.index))
        out.apply(
            butterworth_filt,
            fcut=[20, 450],
            fsamp=fsamp,
            order=4,
            ftype="bandpass",
            phase_corrected=True,
            inplace=True,
            axis=1,
        )
        out.apply(
            rms_filt,
            order=int(0.2 * fsamp),
            pad_style="reflect",
            offset=0.5,
            inplace=True,
            axis=1,
        )
        if not inplace:
            return out

    def default_point3d_processing_func(
        point: Point3D,
        inplace: bool = inplace,
    ):
        out = point if inplace else point.copy()
        out.fillna(inplace=True)
        fsamp = float(1 / np.mean(np.diff(out.index)))
        out.apply(
            butterworth_filt,
            fcut=6,
            fsamp=fsamp,
            order=4,
            ftype="lowpass",
            phase_corrected=True,
            inplace=True,
        )
        if not inplace:
            return out

    def default_signal3d_processing_func(
        signal: Signal3D,
        inplace: bool = False,
    ):
        out = signal if inplace else signal.copy()
        out.fillna(inplace=True)
        fsamp = 1 / np.mean(np.diff(out.index))
        out.apply(
            butterworth_filt,
            fcut=6,
            fsamp=fsamp,
            order=4,
            ftype="lowpass",
            phase_corrected=True,
            inplace=True,
        )
        if not inplace:
            return out

    def default_signal1d_processing_func(
        signal: Signal1D,
        inplace: bool = False,
    ):
        out = signal if inplace else signal.copy()
        out.fillna(inplace=True)
        fsamp = 1 / np.mean(np.diff(out.index))
        out.apply(
            butterworth_filt,
            fcut=6,
            fsamp=fsamp,
            order=4,
            ftype="lowpass",
            phase_corrected=True,
            inplace=True,
        )
        if not inplace:
            return out

    def default_forceplatforms_processing_func(
        fp: ForcePlatform, inplace: bool = False
    ):
        out = fp if inplace else fp.copy()

        def default_3d_processing_func(
            signal: Signal3D | Point3D,
            inplace: bool = False,
        ):
            out = signal if inplace else signal.copy()
            out.fillna(inplace=True, value=0)
            fsamp = 1 / np.mean(np.diff(out.index))
            out.apply(
                butterworth_filt,
                fcut=[10, 100],
                fsamp=fsamp,
                order=4,
                ftype="bandstop",
                phase_corrected=True,
                inplace=True,
            )
            if not inplace:
                return out

        fp_pipeline = ProcessingPipeline(
            dict(
                Point3D=[lambda x: default_3d_processing_func(x, inplace)],
                Signal3D=[lambda x: default_3d_processing_func(x, inplace)],
            ),
        )
        fp_pipeline(out, inplace=True)
        if not inplace:
            return out

    return ProcessingPipeline(
        {
            "EMGSignal": [lambda x: default_emgsignal_processing_func(x, inplace)],
            "Point3D": [lambda x: default_point3d_processing_func(x, inplace)],
            "Signal1D": [lambda x: default_signal1d_processing_func(x, inplace)],
            "Signal3D": [lambda x: default_signal3d_processing_func(x, inplace)],
            "ForcePlatform": [
                lambda x: default_forceplatforms_processing_func(x, inplace)
            ],
        }
    )
