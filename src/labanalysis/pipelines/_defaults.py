"""
Default processing functions for biomechanical data pipelines.
"""

import numpy as np

from ..constants import MINIMUM_CONTACT_FORCE_N
from ..signalprocessing import (
    butterworth_filt,
    rms_filt,
    fillna,
    mean_filt,
)
from ..timeseries import EMGSignal, Point3D, Signal1D, Signal3D
from ..records import ForcePlatform, MetabolicRecord
from ._base import ProcessingPipeline


def get_default_emgsignal_processing_func(channel: EMGSignal):
    """
    Apply default EMG signal processing pipeline.

    Removes mean, applies bandpass filter (20-450 Hz), and calculates RMS envelope.

    Parameters
    ----------
    channel : EMGSignal
        EMG signal to process (modified in-place).

    Notes
    -----
    Processing steps:
    1. Remove DC offset (subtract mean)
    2. Bandpass filter (20-450 Hz, 4th order Butterworth)
    3. RMS envelope (200ms window)
    """
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
    """
    Apply default 3D point processing pipeline.

    Fills missing values and applies low-pass filter.

    Parameters
    ----------
    point : Point3D
        3D point trajectory to process (modified in-place).

    Notes
    -----
    Processing steps:
    1. Fill missing values (NaN) via cubic spline interpolation
    2. Low-pass filter (6 Hz cutoff, 4th order Butterworth, phase-corrected)
    """
    point.fillna(mice=True, inplace=True)
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
    """
    Apply default 3D signal processing pipeline.

    Fills missing values and applies low-pass filter.

    Parameters
    ----------
    signal : Signal3D
        3D signal to process (modified in-place).

    Notes
    -----
    Processing steps:
    1. Fill missing values (NaN) via cubic spline interpolation
    2. Low-pass filter (6 Hz cutoff, 4th order Butterworth, phase-corrected)
    """
    signal.fillna(mice=True, inplace=True)
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
    """
    Apply default 1D signal processing pipeline.

    Fills missing values and applies low-pass filter.

    Parameters
    ----------
    signal : Signal1D
        1D signal to process (modified in-place).

    Notes
    -----
    Processing steps:
    1. Fill missing values (NaN) via cubic spline interpolation
    2. Low-pass filter (6 Hz cutoff, 4th order Butterworth, phase-corrected)
    """
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
    """
    Apply default force platform processing pipeline.

    Comprehensive processing for force platform data including contact detection,
    gap filling, filtering, and moment updates.

    Parameters
    ----------
    fp : ForcePlatform
        Force platform record to process (modified in-place).

    Notes
    -----
    Processing steps:
    1. Set forces below minimum contact threshold to NaN
    2. Strip NaN values from beginning and end
    3. Fill force NaNs with zeros
    4. Fill position NaNs via cubic spline
    5. Low-pass filter origin and force (30 Hz, 4th order Butterworth)
    6. Update torque/moments from filtered data
    7. Zero out moments during non-contact periods
    """
    vals = fp.force.copy().to_numpy()
    module = fp.force.copy().module.to_numpy().flatten()
    idxs = module < MINIMUM_CONTACT_FORCE_N
    for i in ["origin", "force", "torque"]:
        vals = fp[i].copy().to_numpy()
        vals[idxs, :] = np.nan
        fp[i][:, :] = vals

    fp.strip(inplace=True)

    fp.force[:, :] = fillna(fp.force.to_numpy(), value=0, inplace=False)

    fp.origin[:, :] = fillna(fp.origin.to_numpy(), mice=True, inplace=False)

    fsamp = float(1 / np.mean(np.diff(fp.index)))
    filt_fun = lambda x: butterworth_filt(
        x,
        fcut=30,
        fsamp=fsamp,
        order=4,
        ftype="lowpass",
        phase_corrected=True,
    )
    fp.origin.apply(filt_fun, axis=0, inplace=True)
    fp.force.apply(filt_fun, axis=0, inplace=True)

    fp.update_moments(inplace=True)

    module = fp.force.copy().module.to_numpy().flatten()
    idxs = module < MINIMUM_CONTACT_FORCE_N
    vals = fp.torque.copy().to_numpy()
    vals[idxs, :] = 0
    fp.torque[:, :] = vals


def get_default_metabolicrecord_processing_func(mr: MetabolicRecord):
    """
    Apply default metabolic data processing pipeline.

    Smooths breath-by-breath data using moving average filter.

    Parameters
    ----------
    mr : MetabolicRecord
        Metabolic record to process (modified in-place).

    Notes
    -----
    For breath-by-breath data, applies 15-point moving average to:
    - VO2 (oxygen consumption)
    - VCO2 (carbon dioxide production)
    - HR (heart rate)
    - VE (ventilation)

    Based on recommendations from Robergs et al. (2010) Sports Medicine 40:95-111.
    """
    if mr.breath_by_breath:
        mr.vo2.apply(mean_filt, order=15, inplace=True, axis=0)
        mr.vco2.apply(mean_filt, order=15, inplace=True, axis=0)
        mr.hr.apply(mean_filt, order=15, inplace=True, axis=0)
        mr.ve.apply(mean_filt, order=15, inplace=True, axis=0)


def get_default_processing_pipeline():
    """
    Create a processing pipeline with default functions for all supported types.

    Returns
    -------
    ProcessingPipeline
        Configured pipeline with default processing functions for:
        - EMGSignal: bandpass filter + RMS envelope
        - Point3D: gap filling + 6Hz lowpass
        - Signal1D: gap filling + 6Hz lowpass
        - Signal3D: gap filling + 6Hz lowpass
        - ForcePlatform: contact detection + filtering + moment updates
        - MetabolicRecord: breath-by-breath smoothing

    See Also
    --------
    get_default_emgsignal_processing_func
    get_default_point3d_processing_func
    get_default_signal1d_processing_func
    get_default_signal3d_processing_func
    get_default_forceplatform_processing_func
    get_default_metabolicrecord_processing_func
    """
    return ProcessingPipeline(
        EMGSignal=[get_default_emgsignal_processing_func],
        Point3D=[get_default_point3d_processing_func],
        Signal1D=[get_default_signal1d_processing_func],
        Signal3D=[get_default_signal3d_processing_func],
        ForcePlatform=[get_default_forceplatform_processing_func],
        MetabolicRecord=[get_default_metabolicrecord_processing_func],
    )
