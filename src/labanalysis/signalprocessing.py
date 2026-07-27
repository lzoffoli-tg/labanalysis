"""
Signal processing and analysis functions for biomechanical data.

This module provides a comprehensive set of functions for processing and
analyzing 1D and 3D signals commonly encountered in biomechanics and
physiological data analysis. Functions include peak detection, filtering,
interpolation, derivative estimation, and geometric transformations.

Functions
---------
Peak Detection and Event Identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
find_peaks
    Detect local maxima in 1D signals.
continuous_batches
    Identify contiguous sequences of True values in boolean arrays.
crossings
    Detect zero-crossing or threshold-crossing points in signals.

Signal Differentiation
~~~~~~~~~~~~~~~~~~~~~~
winter_derivative1
    Compute first derivative using Winter's central difference method.
winter_derivative2
    Compute second derivative using Winter's method.

Signal Filtering
~~~~~~~~~~~~~~~~
butterworth_filt
    Apply Butterworth digital filter with specified parameters.
fir_filt
    Apply FIR (Finite Impulse Response) filter to signals.
mean_filt
    Apply moving average (mean) filter.
median_filt
    Apply moving median filter.
rms_filt
    Apply moving root-mean-square filter.
thresholding_filt
    Apply adaptive thresholding filter to remove outliers.

Interpolation and Resampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cubicspline_interp
    Perform cubic spline interpolation.
fillna
    Fill missing values using direct replacement, MICE, or spline interpolation.

Frequency Domain Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~
psd
    Compute power spectral density using FFT periodogram method.
residual_analysis
    Determine optimal filter cutoff frequency using Winter's residual analysis.

Signal Correlation
~~~~~~~~~~~~~~~~~~
xcorr
    Compute auto-correlation or cross-correlation with lag.

Geometric Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~
gram_schmidt
    Compute orthonormal basis from vectors using Gram-Schmidt process.
to_reference_frame
    Rotate 3D data to a specified reference frame.

Utilities
~~~~~~~~~
nextpow
    Calculate next power of a base for a given value.
freedman_diaconis_bins
    Digitize signal using Freedman-Diaconis bin rule.
padwin
    Pad signal and generate window indices for filtering.
crossovers
    Find piecewise linear regression breakpoints.
outlyingness
    Compute adjusted outlyingness factor for outlier detection.
tkeo
    Compute Teager-Kaiser Energy Operator for signal.

See Also
--------
labanalysis.timeseries : Time-series data structures with integrated filtering.
scipy.signal : Additional signal processing functions.
"""

#! IMPORTS

import warnings
from itertools import product
from types import FunctionType, MethodType
from typing import Literal

import numpy as np
from pandas import DataFrame, Series
from scipy import signal  # type: ignore
from scipy.interpolate import CubicSpline  # type: ignore
from scipy.spatial.transform import Rotation
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

__all__ = [
    "find_peaks",
    "continuous_batches",
    "nextpow",
    "winter_derivative1",
    "winter_derivative2",
    "freedman_diaconis_bins",
    "fir_filt",
    "padwin",
    "thresholding_filt",
    "mean_filt",
    "median_filt",
    "rms_filt",
    "butterworth_filt",
    "cubicspline_interp",
    "residual_analysis",
    "crossovers",
    "psd",
    "crossings",
    "xcorr",
    "outlyingness",
    "gram_schmidt",
    "fillna",
    "tkeo",
    "to_reference_frame",
]


#! FUNCTIONS


def find_peaks(
    arr: np.ndarray,
    height: int | float | None = None,
    distance: int | None = None,
) -> np.ndarray:
    """
    Find peaks in the signal.

    Detects local maxima in a 1D signal where the derivative changes from
    positive to negative. Optionally filters peaks by minimum height and
    minimum separation distance.

    Parameters
    ----------
    arr : np.ndarray
        The input signal (1D array).
    height : int or float or None, optional
        Minimum peak height. Peaks below this value are excluded.
        Default is None (no height filtering).
    distance : int or None, optional
        Minimum distance (in samples) between consecutive peaks.
        When peaks are closer, only the highest is kept.
        Default is None (no distance filtering).

    Returns
    -------
    np.ndarray
        Array of indices where peaks are located in the input signal.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([0, 1, 2, 1, 0, 3, 2, 1, 0])
    >>> peaks = find_peaks(signal)
    >>> peaks
    array([2, 5])

    >>> # With height threshold
    >>> peaks = find_peaks(signal, height=2.5)
    >>> peaks
    array([5])

    >>> # With minimum distance
    >>> peaks = find_peaks(signal, distance=4)
    >>> peaks
    array([5])

    Notes
    -----
    Peak detection algorithm:
    1. Computes first derivative using forward differences
    2. Identifies points where derivative changes from ≥0 to <0
    3. Filters by height threshold if specified
    4. Filters by minimum distance if specified (keeps highest peak)
    """
    # get all peaks
    d1y = arr[1:] - arr[:-1]
    all_peaks = np.where((d1y[1:] < 0) & (d1y[:-1] >= 0))[0] + 1

    # select those peaks at minimum height
    if len(all_peaks) > 0 and height is not None:
        all_peaks = all_peaks[arr[all_peaks] >= height]

    # select those peaks separated at minimum by the given distance
    if len(all_peaks) > 1 and distance is not None:
        i = 1
        while i < len(all_peaks):
            if all_peaks[i] - all_peaks[i - 1] < distance:
                if arr[all_peaks[i]] > arr[all_peaks[i - 1]]:
                    all_peaks = np.append(all_peaks[: i - 1], all_peaks[i:])
                else:
                    all_peaks = np.append(all_peaks[:i], all_peaks[i + 1 :])
            else:
                i += 1

    return all_peaks.astype(int)


def continuous_batches(
    arr: np.ndarray,
    tolerance: int = 0,
):
    """
    Return the list of indices defining batches where consecutive arr values are True.

    Identifies contiguous sequences of True values in a boolean array, optionally
    merging batches separated by short False gaps (tolerance).

    Parameters
    ----------
    arr : np.ndarray
        A 1D boolean array.
    tolerance : int, optional
        Maximum number of False values that can separate two batches
        before they are merged into one. Default is 0 (no merging).

    Returns
    -------
    list of list of int
        A list of lists containing the indices defining each batch of consecutive
        True values.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([False, True, True, False, False, True, True, True])
    >>> batches = continuous_batches(signal)
    >>> batches
    [[1, 2], [5, 6, 7]]

    >>> # With tolerance=1, merge batches separated by 1 False
    >>> batches = continuous_batches(signal, tolerance=1)
    >>> batches
    [[1, 2, 5, 6, 7]]

    Notes
    -----
    Useful for identifying contact phases in force platform data, stance phases
    in gait analysis, or any contiguous event detection in boolean signals.
    """
    locs = arr.astype(int)
    idxs = np.diff(locs)
    idxs = np.concatenate([[locs[0]], idxs])
    crs = locs + idxs
    if locs[-1] == 1:
        crs = np.concatenate([crs, [-1]])
    starts = np.where(crs == 2)[0]
    stops = np.where(crs == -1)[0]
    batches = [list(range(i, v, 1)) for i, v in zip(starts, stops)]

    # join those gaps separated by less than the provided tolerance
    i = 0
    while i < len(batches) - 1:
        if batches[i + 1][0] - batches[i][-1] <= tolerance:
            batches[i] = batches[i] + batches[i + 1]
            batches.pop(i + 1)
        else:
            i += 1

    return batches


def nextpow(
    val: int | float,
    base: int = 2,
) -> int:
    """
    Calculate the smallest power of base that is greater than or equal to val.

    Parameters
    ----------
    val : int or float
        The target value.
    base : int, optional
        The base to be raised to a power. Default is 2.

    Returns
    -------
    int
        The smallest integer power result base^n >= val.

    Examples
    --------
    >>> nextpow(100, base=2)
    128
    >>> nextpow(100, base=10)
    100
    >>> nextpow(1001, base=10)
    10000

    Notes
    -----
    Commonly used to determine optimal FFT sizes (powers of 2) for
    efficient computation, or to round up buffer sizes to convenient
    powers of a base.
    """
    return int(round(base ** np.ceil(np.log(val) / np.log(base))))


def winter_derivative1(
    y_signal: np.ndarray,
    x_signal: np.ndarray | None = None,
    time_diff: float | int = 1,
) -> np.ndarray:
    """
    Compute first derivative using Winter's central difference method.

    Implements the three-point central difference formula recommended by
    Winter (2009) for biomechanical signal differentiation. This method
    provides better noise characteristics than forward or backward differences.

    Parameters
    ----------
    y_signal : np.ndarray
        The signal to be differentiated (1D array).
    x_signal : np.ndarray or None, optional
        Independent variable (e.g., time) corresponding to y_signal.
        If None, assumes uniform sampling with interval `time_diff`.
        Default is None.
    time_diff : float or int, optional
        Sampling interval when x_signal is None. Ignored if x_signal
        is provided. Default is 1.

    Returns
    -------
    np.ndarray
        First derivative of y_signal. Output length is len(y_signal) - 2
        due to central differencing requiring points on both sides.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 100)
    >>> y = np.sin(2 * np.pi * t)
    >>> dydt = winter_derivative1(y, t)
    >>> len(dydt)
    98

    Notes
    -----
    The central difference formula used is:
        dy/dx[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])

    This method is preferred in biomechanics for its superior noise
    handling compared to forward/backward differences, though it reduces
    the output array length by 2 samples.

    References
    ----------
    Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement.
    Fourth Edition. Hoboken, NJ: John Wiley & Sons, Inc.

    See Also
    --------
    winter_derivative2 : Second derivative using Winter's method.
    """

    # get x
    if x_signal is None:
        x_sig = np.arange(len(y_signal)) * time_diff
    else:
        x_sig = x_signal

    # get the derivative
    return (y_signal[2:] - y_signal[:-2]) / (x_sig[2:] - x_sig[:-2])


def winter_derivative2(
    y_signal: np.ndarray,
    x_signal: np.ndarray | None = None,
    time_diff: float | int = 1,
) -> np.ndarray:
    """
    Compute second derivative using Winter's three-point method.

    Implements the three-point finite difference formula for second
    derivatives recommended by Winter (2009) for biomechanical acceleration
    estimation from position or velocity data.

    Parameters
    ----------
    y_signal : np.ndarray
        The signal to be differentiated (1D array).
    x_signal : np.ndarray or None, optional
        Independent variable (e.g., time) corresponding to y_signal.
        If None, assumes uniform sampling with interval `time_diff`.
        Default is None.
    time_diff : float or int, optional
        Sampling interval when x_signal is None. Ignored if x_signal
        is provided. Default is 1.

    Returns
    -------
    np.ndarray
        Second derivative of y_signal. Output length is len(y_signal) - 2.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 100)
    >>> y = 0.5 * 9.81 * t**2  # Free fall position
    >>> d2y = winter_derivative2(y, t)
    >>> np.allclose(d2y, 9.81, atol=0.1)
    True

    Notes
    -----
    The finite difference formula used is:
        d²y/dx²[i] = (y[i+1] - 2*y[i] + y[i-1]) / h²

    where h is the mean sampling interval.

    Commonly used in biomechanics to compute acceleration from
    position data or jerk from velocity data.

    References
    ----------
    Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement.
    Fourth Edition. Hoboken, NJ: John Wiley & Sons, Inc.

    See Also
    --------
    winter_derivative1 : First derivative using Winter's method.
    """

    # get x
    if x_signal is None:
        x_sig = np.arange(len(y_signal)) * time_diff
    else:
        x_sig = np.copy(x_signal)

    # get the derivative
    num = y_signal[2:] + y_signal[:-2] - 2 * y_signal[1:-1]
    den = np.mean(np.diff(x_sig)) ** 2
    return num / den


def freedman_diaconis_bins(
    y_signal: np.ndarray,
) -> np.ndarray:
    """
    Digitize signal into bins using the Freedman-Diaconis rule.

    The Freedman-Diaconis rule determines optimal bin width for histograms
    based on the interquartile range (IQR) and sample size, minimizing
    integrated mean squared error for density estimation.

    Parameters
    ----------
    y_signal : np.ndarray
        The 1D signal to be digitized.

    Returns
    -------
    np.ndarray
        Array with the same shape as y_signal, where each element contains
        the bin index (0-based) for the corresponding sample.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = np.random.randn(1000)
    >>> bins = freedman_diaconis_bins(data)
    >>> bins.shape
    (1000,)
    >>> bins.min(), bins.max()
    (0.0, ...)

    Notes
    -----
    The bin width is calculated as:
        h = 2 * IQR / n^(1/3)

    where IQR is the interquartile range and n is the sample size.

    This rule is particularly robust to outliers compared to other
    binning methods like Sturges' rule or Scott's rule.

    References
    ----------
    Freedman, D., & Diaconis, P. (1981). On the histogram as a density
    estimator: L2 theory. Zeitschrift für Wahrscheinlichkeitstheorie und
    Verwandte Gebiete, 57(4), 453-476. doi: 10.1007/BF01025868

    See Also
    --------
    numpy.histogram : Histogram computation with various binning methods.
    """

    # y IQR
    qnt1 = np.quantile(y_signal, 0.25)
    qnt3 = np.quantile(y_signal, 0.75)
    iqr = qnt3 - qnt1

    # get the width
    wdt = 2 * iqr / (len(y_signal) ** (1 / 3))

    # get the number of intervals
    samp = int(np.floor(1 / wdt)) + 1

    # digitize z
    digitized = np.zeros(y_signal.shape)
    for i in np.arange(samp) + 1:
        loc = np.argwhere((y_signal >= (i - 1) * wdt) & (y_signal < i * wdt))
        digitized[loc] = i - 1
    return digitized


def padwin(
    arr: np.ndarray,
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad signal and generate sliding window indices for filtering operations.

    Creates a padded version of the input signal and corresponding window
    indices for each sample, enabling efficient implementation of moving
    window filters (mean, median, RMS, etc.).

    Parameters
    ----------
    arr : np.ndarray
        The 1D signal to be padded.
    order : int, optional
        Window size (number of samples in each window). Default is 1.
    pad_style : str, optional
        Padding method passed to numpy.pad. Options include 'edge',
        'constant', 'reflect', 'symmetric', 'wrap'. Default is 'edge'.
    offset : float, optional
        Window alignment, value in [0, 1]. Controls where the current
        sample sits within its window:
        - 0.0: current sample at window start (causal)
        - 0.5: current sample at window center (symmetric)
        - 1.0: current sample at window end (anti-causal)
        Default is 0.5 (symmetric window).

    Returns
    -------
    pad : np.ndarray
        The padded signal with length len(arr) + padding.
    mask : np.ndarray
        2D array of shape (len(arr), order) where mask[i] contains the
        indices into the padded signal for the window centered on sample i.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> padded, windows = padwin(signal, order=3, offset=0.5)
    >>> padded
    array([1, 1, 2, 3, 4, 5, 5])
    >>> windows[2]  # Window for sample index 2
    array([1, 2, 3])

    Notes
    -----
    This function is used internally by filtering functions (mean_filt,
    median_filt, rms_filt) to efficiently apply sliding window operations.

    See Also
    --------
    mean_filt : Moving average filter using padwin.
    median_filt : Moving median filter using padwin.
    rms_filt : Moving RMS filter using padwin.
    """
    # get the window range
    stop = order - int(np.floor(order * offset)) - 1
    init = order - stop - 1

    # get the indices of the samples
    idx = np.arange(len(arr)) + init

    # padding
    pad = np.pad(arr, [init, stop], mode=pad_style)  # type: ignore

    # get the windows mask
    rng = np.arange(-init, stop + 1)
    mask = np.atleast_2d([rng + i for i in idx])

    return pad, mask


def thresholding_filt(
    arr: np.ndarray,
    factor: float | int = 3,
    robust: bool = False,
    order: int = 3,
    pad_style: str = "edge",
    offset: float = 0.5,
) -> np.ndarray:
    """
    Apply adaptive thresholding filter to remove outliers and extreme values.

    Replaces values that deviate excessively from their local neighborhood
    (defined by a moving window) with the local central tendency. Useful
    for removing spikes and artifacts while preserving underlying signal shape.

    Parameters
    ----------
    arr : np.ndarray
        The 1D signal to be filtered.
    factor : float or int, optional
        Threshold multiplier for outlier detection. Values farther than
        factor × (local spread) from local center are replaced.
        Default is 3.
    robust : bool, optional
        If True, use median and MAD (median absolute deviation) for
        robust statistics less sensitive to outliers.
        If False, use mean and standard deviation. Default is False.
    order : int, optional
        Window size for local statistics computation. Default is 3.
    pad_style : str, optional
        Padding method for signal edges. Default is 'edge'.
    offset : float, optional
        Window alignment in [0, 1]. Default is 0.5 (symmetric).

    Returns
    -------
    np.ndarray
        Filtered signal with extreme values replaced by local estimates.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1.0, 1.1, 10.0, 0.9, 1.0, 1.1])  # Spike at index 2
    >>> filtered = thresholding_filt(signal, factor=2, order=3)
    >>> filtered[2] < 5  # Spike should be reduced
    True

    Notes
    -----
    The outlier detection criterion is:
        |x[i] - center[i]| > factor × spread[i]

    Where center and spread are either:
    - Non-robust: mean and standard deviation
    - Robust: median and MAD (median absolute deviation)

    Robust mode is recommended when the signal may contain multiple
    outliers that could bias mean/std estimates.

    See Also
    --------
    padwin : Window padding function used internally.
    median_filt : Alternative median-based smoothing.
    """

    # pad the array
    pads, mask = padwin(arr, order, pad_style, offset)

    # get the required values
    if robust:
        vals = np.array([np.median(pads[i]) for i in mask])
        thresh = [np.median(abs(pads[v] - vals[i])) for i, v in enumerate(mask)]
        thresh = np.array(thresh)
    else:
        vals = np.array([np.mean(pads[i]) for i in mask])
        thresh = np.array([np.std(pads[i]) for i in mask])

    # replace the extreme values
    out = np.copy(arr)
    extremes = np.abs(arr - vals) > factor * thresh
    out[extremes] = vals[extremes]

    return out


def mean_filt(
    arr: np.ndarray,
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
) -> np.ndarray:
    """
    Apply moving average (mean) filter to smooth signal.

    Computes the arithmetic mean over a sliding window, effectively
    implementing a low-pass filter that attenuates high-frequency noise.
    Uses cumulative sum for O(n) efficiency.

    Parameters
    ----------
    arr : np.ndarray
        The 1D signal to be filtered.
    order : int, optional
        Window size (number of samples to average). Default is 1.
    pad_style : str, optional
        Padding method for signal edges ('edge', 'reflect', etc.).
        Default is 'edge'.
    offset : float, optional
        Window alignment in [0, 1]. 0.5 centers the window on the
        current sample. Default is 0.5.

    Returns
    -------
    np.ndarray
        Filtered signal with the same length as input.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> smoothed = mean_filt(signal, order=3)
    >>> smoothed
    array([1.33..., 2., 3., 4., 4.66...])

    Notes
    -----
    The moving average is a simple FIR filter with impulse response:
        h[n] = 1/M  for n in [0, M-1], else 0

    where M is the window size (order).

    This implementation uses cumulative sum for efficient O(n) computation
    rather than naive O(n*M) sliding window iteration.

    See Also
    --------
    median_filt : Median-based smoothing filter.
    rms_filt : Root-mean-square filter.
    fir_filt : General FIR filter with custom window.
    """

    # get the window range

    init = int(round(order * offset))
    stop = order - init

    # get the indices of the samples
    idx = np.arange(len(arr)) + init

    # padding
    pad = np.pad(arr, [init, stop], mode=pad_style)  # type: ignore

    # get the cumulative sum of the signal
    csum = np.cumsum(pad).astype(float)

    # get the mean
    return (csum[idx + stop] - csum[idx - init]) / order


def median_filt(
    arr: np.ndarray,
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
) -> np.ndarray:
    """
    Apply moving median filter to signal for robust smoothing.

    Replaces each sample with the median of its local neighborhood,
    providing robust smoothing that preserves edges and is less
    sensitive to outliers compared to mean filtering.

    Parameters
    ----------
    arr : np.ndarray
        The 1D signal to be filtered.
    order : int, optional
        Window size (number of samples). Odd values recommended for
        symmetric windows. Default is 1.
    pad_style : str, optional
        Padding method for signal edges. Default is 'edge'.
    offset : float, optional
        Window alignment in [0, 1]. Default is 0.5 (centered).

    Returns
    -------
    np.ndarray
        Filtered signal with the same length as input.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1.0, 1.0, 10.0, 1.0, 1.0])  # Spike
    >>> filtered = median_filt(signal, order=3)
    >>> filtered[2]  # Spike removed
    1.0

    Notes
    -----
    Median filtering is particularly effective for:
    - Removing salt-and-pepper noise
    - Preserving sharp edges and discontinuities
    - Robust smoothing in presence of outliers

    Unlike mean filtering, the median is not a linear operation,
    so it cannot be expressed as convolution with a kernel.

    For large windows or repeated application, consider using
    scipy.signal.medfilt for potentially better performance.

    See Also
    --------
    mean_filt : Moving average filter.
    thresholding_filt : Adaptive outlier removal.
    scipy.signal.medfilt : Optimized median filter implementation.
    """
    pad, mask = padwin(arr, order, pad_style, offset)
    return np.array([np.median(pad[i]) for i in mask])


def rms_filt(
    arr: np.ndarray,
    order: int = 1,
    pad_style: str = "edge",
    offset: float = 0.5,
) -> np.ndarray:
    """
    Apply moving root-mean-square (RMS) filter to signal.

    Computes the RMS value over a sliding window, commonly used for
    EMG signal processing, vibration analysis, and power estimation.
    Uses cumulative sum for efficient O(n) computation.

    Parameters
    ----------
    arr : np.ndarray
        The 1D signal to be filtered.
    order : int, optional
        Window size (number of samples). Default is 1.
    pad_style : str, optional
        Padding method for signal edges. Default is 'edge'.
    offset : float, optional
        Window alignment in [0, 1]. Default is 0.5 (centered).

    Returns
    -------
    np.ndarray
        RMS-filtered signal with the same length as input.

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1, -1, 1, -1, 1])
    >>> rms = rms_filt(signal, order=3)
    >>> np.allclose(rms, 1.0, atol=0.1)
    True

    Notes
    -----
    The RMS is computed as:
        RMS = sqrt(mean(x²)) = sqrt((1/M) × Σx²)

    where M is the window size (order).

    Common applications:
    - EMG envelope detection for muscle activation analysis
    - Vibration signal analysis
    - AC power/voltage measurements
    - Signal energy estimation

    See Also
    --------
    mean_filt : Moving average filter.
    tkeo : Teager-Kaiser energy operator for EMG.
    """

    # get the window range
    stop = order - int(np.floor(order * offset))
    init = order - stop

    # get the indices of the samples
    idx = np.arange(len(arr)) + init

    # padding
    pad = np.pad(arr, [init, stop], mode=pad_style)  # type: ignore

    # get the squares of the signal
    sqe = pad**2

    # get the cumulative sum of the signal
    csum = np.cumsum(sqe).astype(float)

    # get the root mean of the squares
    return ((csum[idx + stop] - csum[idx - init]) / order) ** 0.5


def fir_filt(
    arr: np.ndarray,
    fcut: float | int | list[float | int] | tuple[float | int] = 1,
    fsamp: float | int = 2,
    order: int = 5,
    ftype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    wtype: Literal[
        "boxcar",
        "triang",
        "blackman",
        "hamming",
        "hann",
        "bartlett",
        "flattop",
        "parzen",
        "bohman",
        "blackmanharris",
        "nuttall",
        "barthann",
        "cosine",
        "exponential",
        "tukey",
        "taylor",
    ] = "hamming",
    pstyle: Literal[
        "constant",
        "edge",
        "linear_ramp",
        "maximum",
        "mean",
        "median",
        "minimum",
        "reflect",
        "symmetric",
        "wrap",
    ] = "edge",
) -> np.ndarray:
    """
    Apply a FIR filter with the specified specs to the signal.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be filtered.
    fcut : float, int, list, or tuple, optional
        The cutoff frequency of the filter.
    fsamp : float or int, optional
        The sampling frequency of the signal.
    order : int, optional
        The order of the filter.
    ftype : str, optional
        The type of filter: "bandpass", "lowpass", "highpass", "bandstop".
    wtype : str, optional
        The type of window to be applied.
    pstyle : str, optional
        The type of padding style.

    Returns
    -------
    np.ndarray
        The filtered signal.
    """
    coefs = signal.firwin(
        order,
        fcut,
        window=wtype,
        pass_zero=ftype,  # type: ignore
        fs=fsamp,
    )
    # Prepare padding parameters based on mode
    pad_kwargs = {"mode": pstyle, "pad_width": (2 * order - 1, 0)}
    if pstyle == "constant":
        pad_kwargs["constant_values"] = arr[0]

    padded = np.pad(arr, **pad_kwargs)  # type: ignore
    avg = np.mean(padded)
    out = signal.lfilter(coefs, 1.0, padded - avg)[(2 * order - 1) :]
    return np.array(out).flatten().astype(float) + avg


def butterworth_filt(
    arr: np.ndarray,
    fcut: float | int | list[float | int] | tuple[float | int] = 1,
    fsamp: float | int = 2,
    order: int = 5,
    ftype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    phase_corrected: bool = True,
) -> np.ndarray:
    """
    Apply a Butterworth filter with the specified parameters.

    Implements a Butterworth digital filter using second-order sections (SOS)
    for improved numerical stability. Optionally applies zero-phase filtering
    using forward-backward filtering.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be filtered (1D array).
    fcut : float, int, list, or tuple, optional
        Cutoff frequency in Hz. For bandpass/bandstop filters, provide
        [low_cut, high_cut]. Default is 1 Hz.
    fsamp : float or int, optional
        Sampling frequency of the signal in Hz. Default is 2 Hz.
    order : int, optional
        Filter order. Higher orders provide steeper roll-off but may
        introduce instability. Default is 5.
    ftype : str, optional
        Filter type: "lowpass", "highpass", "bandpass", or "bandstop".
        Default is "lowpass".
    phase_corrected : bool, optional
        If True, applies filtfilt (forward-backward) for zero phase shift.
        If False, applies single-pass filtering with phase distortion.
        Default is True.

    Returns
    -------
    np.ndarray
        The filtered signal (1D array).

    Examples
    --------
    >>> import numpy as np
    >>> # Generate noisy signal
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(1000)
    >>> # Apply 6 Hz lowpass filter at 1000 Hz sampling
    >>> filtered = butterworth_filt(signal, fcut=6, fsamp=1000, order=4)

    >>> # Bandpass filter 10-50 Hz
    >>> bandpass = butterworth_filt(signal, fcut=[10, 50], fsamp=1000, ftype="bandpass")

    Notes
    -----
    - Cutoff frequencies are normalized to Nyquist frequency (fsamp/2) internally
    - Uses SOS (second-order sections) representation for numerical stability
    - Phase-corrected filtering (filtfilt) effectively doubles the filter order
    - Common in biomechanics: 4th order lowpass at 6 Hz for kinematic data

    See Also
    --------
    residual_analysis : Determine optimal cutoff frequency using Winter's method
    """

    # get the filter coefficients
    fcut = np.atleast_1d(fcut).flatten().astype(float)  # type: ignore
    fcut /= fsamp / 2  # type: ignore
    if len(fcut) == 1:  # type: ignore
        fcut = float(fcut[0])  # type: ignore
    sos = signal.butter(
        order,
        fcut,
        ftype,
        analog=False,
        output="sos",
    )

    # get the filtered data
    if phase_corrected:
        arr = signal.sosfiltfilt(sos, arr)
    else:
        arr = signal.sosfilt(sos, arr)  # type: ignore
    return np.array([arr]).astype(float).flatten()


def cubicspline_interp(
    y_old: np.ndarray,
    nsamp: int | None = None,
    x_old: np.ndarray | None = None,
    x_new: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perform cubic spline interpolation on signal data.

    Interpolates signal values using piecewise cubic polynomials that
    are twice continuously differentiable. Supports both uniform
    resampling (via nsamp) and arbitrary point interpolation (via x_new).

    Parameters
    ----------
    y_old : np.ndarray
        The 1D signal values to interpolate.
    nsamp : int or None, optional
        Number of uniformly-spaced output samples. If provided,
        x_old and x_new are ignored and uniform resampling is performed.
        Default is None.
    x_old : np.ndarray or None, optional
        Independent variable (e.g., time) corresponding to y_old.
        Required if nsamp is None. Default is None.
    x_new : np.ndarray or None, optional
        Target independent variable values for interpolation.
        Required if nsamp is None. Default is None.

    Returns
    -------
    np.ndarray
        Interpolated signal values at the requested points.

    Raises
    ------
    ValueError
        If nsamp is None and either x_old or x_new is not provided.

    Examples
    --------
    >>> import numpy as np
    >>> # Uniform resampling: 10 samples to 100 samples
    >>> y = np.sin(np.linspace(0, 2*np.pi, 10))
    >>> y_interp = cubicspline_interp(y, nsamp=100)
    >>> len(y_interp)
    100

    >>> # Arbitrary point interpolation
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([0, 1, 4, 9])
    >>> x_new = np.array([0.5, 1.5, 2.5])
    >>> y_interp = cubicspline_interp(y, x_old=x, x_new=x_new)

    Notes
    -----
    Cubic spline interpolation is smooth (C² continuous) and does not
    exhibit Runge's phenomenon like high-degree polynomial interpolation.

    The spline is constructed to minimize curvature while passing through
    all data points, making it suitable for smooth biomechanical signals.

    See Also
    --------
    fillna : Fill missing values with natural cubic spline interpolation.
    scipy.interpolate.CubicSpline : Underlying implementation.
    """

    # control of the inputs
    if nsamp is None:
        if x_old is None or x_new is None:
            raise ValueError("the pair x_old / x_new or nsamp must be defined")
    else:
        x_old = np.arange(len(y_old))  # type: ignore
        x_new = np.linspace(np.min(x_old), np.max(x_old), nsamp)  # type: ignore

    # get the cubic-spline interpolated y
    cspline = CubicSpline(x_old, y_old)
    return cspline(x_new).flatten().astype(float)


def residual_analysis(
    arr: np.ndarray,
    ffun: FunctionType | MethodType,
    fnum: int = 1000,
    fmax: float | int | None = None,
    nseg: int = 2,
    minsamp: int = 2,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Perform Winter's residual analysis to determine optimal filter cutoff frequency.

    Implements the method described in Winter (2009) for objectively selecting
    cutoff frequencies by analyzing the residual (difference between filtered
    and unfiltered signal) across multiple cutoff frequencies.

    Parameters
    ----------
    arr : np.ndarray
        The signal to be investigated (1D array).
    ffun : FunctionType or MethodType
        Filtering function to apply. Must accept arr and fcut parameters and
        return filtered signal. Typically butterworth_filt.
    fnum : int, optional
        Number of cutoff frequencies to test. Default is 1000.
    fmax : float or int or None, optional
        Maximum cutoff frequency to test in Hz. If None, uses Nyquist
        frequency. Default is None.
    nseg : int, optional
        Number of segments for piecewise linear fitting of residual curve.
        Default is 2 (finds single crossover point).
    minsamp : int, optional
        Minimum number of frequency samples required per segment.
        Default is 2.

    Returns
    -------
    optimal_fcut : float
        Optimal cutoff frequency (Hz) at the crossover point.
    fcuts : np.ndarray
        Array of tested cutoff frequencies.
    residuals : np.ndarray
        Array of residual RMS values corresponding to each cutoff.

    Examples
    --------
    >>> import numpy as np
    >>> from functools import partial
    >>> # Generate test signal
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(1000)
    >>> # Define filter function
    >>> filt_fun = partial(butterworth_filt, fsamp=1000, order=4)
    >>> # Find optimal cutoff
    >>> fcut_opt, fcuts, residuals = residual_analysis(signal, filt_fun)
    >>> print(f"Optimal cutoff: {fcut_opt:.2f} Hz")

    Notes
    -----
    The method works by:
    1. Filtering signal at multiple cutoff frequencies
    2. Computing RMS of residual (filtered - original) for each
    3. Fitting piecewise linear segments to residual curve
    4. Finding crossover point where segments intersect

    The optimal cutoff is where signal content transitions to noise,
    indicated by change in residual growth rate.

    References
    ----------
    Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement.
    Fourth Edition. Hoboken, NJ: John Wiley & Sons, Inc.

    See Also
    --------
    butterworth_filt : Butterworth filter implementation.
    crossovers : Find intersection points of piecewise linear fits.
    """

    # data check
    if fmax is None:
        pwr, frq = psd(arr, 1)
        idx = int(np.where(np.cumsum(pwr) / np.sum(pwr) >= 0.99)[0][0])  # type: ignore
        fmax = max(float(frq[frq < 0.5][-1]), float(frq[idx]))
    assert 0 < fmax < 0.5, "fmax must lie in the (0, 0.5) range."
    assert minsamp >= 2, "'min_samples' must be >= 2."

    # get the optimal crossing over point
    frq = np.linspace(0, fmax, fnum + 1)[1:].astype(float)
    res = np.array([np.sum((arr - ffun(arr, i)) ** 2) for i in frq])
    res = res.astype(float)
    iopt = crossovers(res, segments=nseg, min_samples=minsamp)[0][-1]
    fopt = float(frq[iopt])

    # return the parameters
    return fopt, frq, res.astype(float)


def _sse(
    xval: np.ndarray,
    yval: np.ndarray,
    segm: list[tuple[int]],
):
    """
    Calculate sum of squared errors for piecewise linear regression.

    This helper function fits separate linear regression lines to each
    segment defined by the breakpoints in segm and computes the total
    sum of squared residuals across all segments.

    Parameters
    ----------
    xval : np.ndarray
        The independent variable (x-axis) values.
    yval : np.ndarray
        The dependent variable (y-axis) values to be fitted.
    segm : list of tuple of int
        Breakpoint indices defining segment boundaries. Each consecutive
        pair of indices defines one segment for linear fitting.

    Returns
    -------
    float
        Sum of squared errors across all segments after fitting a linear
        regression line to each segment independently.

    Notes
    -----
    Used internally by crossovers() to evaluate different segmentation
    strategies when fitting piecewise linear models.
    """
    sse = 0.0
    for i in np.arange(len(segm) - 1):
        coords = np.arange(segm[i], segm[i + 1] + 1)  # type: ignore
        coefs = np.polyfit(xval[coords], yval[coords], 1)
        vals = np.polyval(coefs, xval[coords])
        sse += np.sum((yval[coords] - vals) ** 2)
    return float(sse)


def crossovers(
    arr: np.ndarray,
    x: np.ndarray | None = None,
    segments: int = 2,
    min_samples: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find optimal breakpoints for piecewise linear regression.

    Fits K linear segments to data by exhaustively searching breakpoint
    combinations and selecting the configuration that minimizes total
    sum of squared errors. Used in residual_analysis for detecting
    signal-to-noise transition points.

    Parameters
    ----------
    arr : np.ndarray
        The 1D signal to be fitted with piecewise linear segments.
    x : np.ndarray or None, optional
        Independent variable values. If None, uses array indices.
        Default is None.
    segments : int, optional
        Number of linear segments to fit. Default is 2.
    min_samples : int, optional
        Minimum number of samples required per segment. Must be >= 2.
        Default is 5.

    Returns
    -------
    breakpoints : np.ndarray
        Array of indices where segments join (crossover points).
        Length is segments - 1.
    coefficients : np.ndarray
        Array of shape (segments, 2) containing [slope, intercept]
        for each fitted line segment.

    Examples
    --------
    >>> import numpy as np
    >>> # Create signal with two linear regimes
    >>> x = np.arange(100)
    >>> y = np.concatenate([2*x[:50] + 1, 0.5*x[50:] + 76])
    >>> breakpoints, coefs = crossovers(y, segments=2)
    >>> breakpoints[0]  # Should be near index 50
    50

    Notes
    -----
    Algorithm steps:
    1. Generate all valid breakpoint combinations
    2. Fit linear regression to each segment for each combination
    3. Compute total sum of squared errors for each combination
    4. Return breakpoints and coefficients with minimum error

    Computational complexity is O(C × n), where C is the number of
    valid combinations (grows combinatorially with segments).

    For many segments or long signals, consider hierarchical methods
    or dynamic programming approaches.

    References
    ----------
    Lerman, P. M. (1980). Fitting Segmented Regression Models by Grid
    Search. Applied Statistics, 29(1), 77-84.

    See Also
    --------
    residual_analysis : Uses crossovers to find optimal filter cutoff.
    _sse : Helper function computing segmented regression error.
    """

    # control the inputs
    assert min_samples >= 2, "'min_samples' must be >= 2."

    # get the X axis
    if x is None:
        xaxis = np.arange(len(arr))
    else:
        xaxis = x
    xaxis = np.asarray(xaxis)

    # get all the possible combinations of segments
    combs = []
    for i in np.arange(1, segments):
        start = min_samples * i
        stop = len(arr) - min_samples * (segments - i)
        combs += [np.arange(start, stop)]
    combs = list(product(*combs))

    # remove those combinations having segments shorter than "samples"
    combs = [i for i in combs if np.all(np.diff(i) >= min_samples)]

    # generate the crossovers matrix
    combs = (
        np.zeros((len(combs), 1)),
        np.atleast_2d(combs),
        np.ones((len(combs), 1)) * len(arr) - 1,
    )
    combs = np.hstack(combs).astype(int)

    # calculate the residuals for each combination
    sse = np.array([_sse(xaxis, arr, i) for i in combs])

    # sort the residuals
    sortedsse = np.argsort(sse)

    # get the optimal crossovers order
    crs = xaxis[combs[sortedsse[0]]]

    # get the fitting slopes
    masks = [(xaxis >= i0) & (xaxis <= i1) for i0, i1 in zip(crs[:-1], crs[1:])]
    slopes = [np.polyfit(xaxis[i], arr[i], 1) for i in masks]
    slopes = np.array(slopes).astype(float)

    # return the crossovers
    crs = [np.where(xaxis == i)[0][0] for i in crs[1:-1]]
    return np.array(crs, int), slopes


def psd(
    arr: np.ndarray,
    fsamp: float | int = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using FFT periodogram method.

    Estimates the distribution of signal power across frequencies using
    the Fast Fourier Transform. The DC component and Nyquist frequency
    are handled appropriately for accurate power computation.

    Parameters
    ----------
    arr : np.ndarray
        The 1D signal to analyze.
    fsamp : float or int, optional
        Sampling frequency of the signal in Hz. Default is 1.0.

    Returns
    -------
    frequencies : np.ndarray
        Frequency bins from 0 to Nyquist frequency (fsamp/2) in Hz.
    power : np.ndarray
        Power spectral density at each frequency bin.

    Examples
    --------
    >>> import numpy as np
    >>> # Generate 10 Hz sine wave sampled at 100 Hz
    >>> t = np.linspace(0, 1, 100, endpoint=False)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> freq, power = psd(signal, fsamp=100)
    >>> peak_freq = freq[np.argmax(power)]
    >>> np.isclose(peak_freq, 10, atol=1)
    True

    Notes
    -----
    The power is computed as the squared magnitude of the FFT:
        P[k] = |FFT[k]|²

    The DC component (k=0) and Nyquist frequency are not doubled,
    while all other frequency bins are multiplied by 2 to account
    for the negative frequency components not shown in the one-sided
    spectrum.

    This is a simple periodogram estimator and may have high variance.
    For smoother estimates, consider using scipy.signal.welch or
    scipy.signal.periodogram with appropriate windowing.

    See Also
    --------
    residual_analysis : Uses PSD to determine default maximum frequency.
    scipy.signal.periodogram : Periodogram with windowing options.
    scipy.signal.welch : Welch's method for reduced-variance PSD estimation.
    """

    # get the psd
    fft = np.fft.rfft(arr - np.mean(arr)) / len(arr)
    amp = abs(fft)
    pwr = np.concatenate([[amp[0]], 2 * amp[1:-1], [amp[-1]]]).flatten() ** 2
    frq = np.linspace(0, fsamp / 2, len(pwr))

    # return the data
    return frq.astype(float), pwr.astype(float)


def crossings(
    arr: np.ndarray,
    value: int | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect threshold crossing points in a signal.

    Identifies sample indices where the signal crosses a specified value,
    transitioning from below to above (positive crossing) or above to
    below (negative crossing). Useful for event detection, gait analysis,
    and zero-crossing detection.

    Parameters
    ----------
    arr : np.ndarray
        The 1D signal to analyze.
    value : int or float, optional
        Threshold value for crossing detection. Default is 0.0 (zero-crossing).

    Returns
    -------
    indices : np.ndarray
        Sample indices immediately before each crossing (integer array).
    directions : np.ndarray
        Crossing direction at each index:
        +1 for upward crossing (from below to above threshold)
        -1 for downward crossing (from above to below threshold)

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.array([1, 2, 1, -1, -2, 1, 2])
    >>> indices, directions = crossings(signal, value=0)
    >>> indices
    array([1, 4])
    >>> directions  # Downward then upward
    array([-1,  1])

    Notes
    -----
    The crossing is detected when the sign of (arr - value) changes
    between consecutive samples. The returned index is the last sample
    before the crossing occurred.

    For accurate crossing time estimation, consider interpolating
    between arr[i] and arr[i+1] to find the exact crossing point.

    See Also
    --------
    find_peaks : Detect local maxima in signals.
    continuous_batches : Identify contiguous regions above/below threshold.
    """

    # get the sign of the signal without the offset
    sgn = np.sign(arr - value)

    # get the location of the crossings
    crs = np.where(abs(sgn[1:] - sgn[:-1]) == 2)[0].astype(int)

    # return the crossings
    return crs, -sgn[crs]


def xcorr(
    sig1: np.ndarray,
    sig2: np.ndarray | None = None,
    biased: bool = False,
    full: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute auto-correlation or cross-correlation of signals.

    Calculates the correlation between sig1 and sig2 (or sig1 with itself)
    as a function of time lag, using FFT-based convolution for efficiency.
    Useful for detecting periodic patterns, time delays, and signal similarity.

    Parameters
    ----------
    sig1 : np.ndarray
        The first signal (1D array).
    sig2 : np.ndarray or None, optional
        The second signal for cross-correlation. If None, computes
        auto-correlation of sig1 with itself. Default is None.
    biased : bool, optional
        If True, use biased normalization (divide by signal length).
        If False, use unbiased normalization (divide by number of
        overlapping samples at each lag). Default is False.
    full : bool, optional
        If True, return correlation for both positive and negative lags.
        If False, return only non-negative lags. Default is False.

    Returns
    -------
    correlation : np.ndarray
        Correlation values at each lag.
    lags : np.ndarray
        Time lags in sample units corresponding to each correlation value.
        Range is [-N+1, N-1] if full=True, else [0, N-1].

    Examples
    --------
    >>> import numpy as np
    >>> # Auto-correlation of periodic signal
    >>> t = np.arange(100)
    >>> signal = np.sin(2 * np.pi * 0.1 * t)
    >>> corr, lags = xcorr(signal)
    >>> peak_lag = lags[np.argmax(corr[1:])]  # Ignore lag=0
    >>> np.isclose(peak_lag, 10, atol=1)  # Period is 10 samples
    True

    >>> # Cross-correlation for time delay estimation
    >>> sig1 = np.array([0, 0, 1, 2, 1, 0])
    >>> sig2 = np.array([1, 2, 1, 0, 0, 0])  # sig1 shifted by 2
    >>> corr, lags = xcorr(sig1, sig2)
    >>> delay = lags[np.argmax(corr)]
    >>> delay
    2

    Notes
    -----
    Auto-correlation (sig2=None):
        R[k] = Σ x[n] × x[n+k]

    Cross-correlation (sig2 provided):
        R[k] = Σ x[n] × y[n+k]

    If signals have different lengths, the shorter is zero-padded.

    Unbiased normalization (biased=False) divides by (N - |k|),
    reducing variance but potentially introducing bias near endpoints.

    See Also
    --------
    scipy.signal.correlate : Alternative correlation implementation.
    numpy.correlate : Basic correlation without normalization.
    """

    # take the autocorrelation if only y is provided
    if sig2 is None:
        sigx = np.atleast_2d(sig1)
        sigz = np.vstack([sigx, sigx])

    # take the cross-correlation (ensure the shortest signal is zero-padded)
    else:
        sigx = np.zeros((1, max(len(sig1), len(sig2))))
        sigy = np.copy(sigx)
        sigx[:, : len(sig1)] = sig1
        sigy[:, : len(sig2)] = sig2
        sigz = np.vstack([sigx, sigy])

    # get the matrix shape
    rows, cols = sigz.shape

    # remove the mean from each dimension
    sigv = sigz - np.atleast_2d(np.mean(sigz, 1)).T

    # take the cross-correlation
    xcr = []
    for i in np.arange(rows - 1):
        for j in np.arange(i + 1, rows):
            res = signal.fftconvolve(sigv[i], sigv[j][::-1], "full")
            xcr += [np.atleast_2d(res)]

    # average over all the multiples
    xcr = np.mean(np.concatenate(xcr, axis=0), axis=0)  # type: ignore

    # adjust the output
    lags = np.arange(-(cols - 1), cols)
    if not full:
        xcr = xcr[(cols - 1) :]
        lags = lags[(cols - 1) :]

    # normalize
    xcr /= (cols + 1 - abs(lags)) if not biased else (cols + 1)

    # return the cross-correlation data
    return xcr.astype(float), lags.astype(int)


def outlyingness(
    arr: np.ndarray,
) -> np.ndarray:
    """
    Return the adjusted outlyingness factor.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        The outlyingness score of each element.

    References
    ----------
    Hubert, M., & Van der Veeken, S. (2008). Outlier detection for skewed data. Journal of Chemometrics: A Journal of the Chemometrics Society, 22(3‐4), 235-246.
    """
    qr1, med, qr3 = np.percentile(arr, [0.25, 0.50, 0.75])
    iqr = qr3 - qr1
    low = arr[arr < med]
    upp = arr[arr > med]
    mcs = [((j - med) - (med - i)) / (j - i) for i, j in product(low, upp)]
    mcs = np.median(mcs)
    if mcs > 0:
        wt1 = qr1 - 1.5 * np.e ** (-4 * mcs) * iqr
        wt2 = qr3 + 1.5 * np.e ** (3 * mcs) * iqr
    else:
        wt1 = qr1 - 1.5 * np.e ** (-3 * mcs) * iqr
        wt2 = qr3 + 1.5 * np.e ** (4 * mcs) * iqr
    out = []
    for i in arr:
        if i == med:
            out += [0]
        elif i > med:
            out += [(i - med) / (wt2 - med)]
        else:
            out += [(med - i) / (med - wt1)]
    return np.array(out)


def gram_schmidt(i: np.ndarray, j: np.ndarray, k: np.ndarray | None = None):
    """
    Apply Gram-Schmidt orthonormalization to obtain orthonormal bases from input vectors.

    The Gram-Schmidt process transforms a set of linearly independent vectors
    into an orthonormal basis. This implementation handles batched operations
    on multiple sets of 3D vectors simultaneously, commonly used for defining
    coordinate systems in biomechanics and motion analysis.

    Parameters
    ----------
    i : np.ndarray, shape (N, 3)
        First set of 3D vectors defining the primary axis direction.
        Will be normalized to form the first basis vector e1.
    j : np.ndarray, shape (N, 3)
        Second set of 3D vectors defining the plane containing e1 and e2.
        Will be orthogonalized against e1 and normalized to form e2.
    k : np.ndarray or None, shape (N, 3), optional
        Third set of 3D vectors. If provided, will be orthogonalized
        against e1 and e2 to form e3. If None, e3 is computed as the
        cross product of e1 and e2. Default is None.

    Returns
    -------
    R : np.ndarray, shape (N, 3, 3)
        Stack of rotation matrices where R[n, :, :] is the 3x3 rotation
        matrix with orthonormal basis vectors [e1, e2, e3] as columns
        for the n-th input set.

    Notes
    -----
    The Gram-Schmidt orthonormalization process:
    1. Normalize the first vector: e1 = i / ||i||
    2. Orthogonalize j against e1: u2 = j - proj_{e1}(j)
       Then normalize: e2 = u2 / ||u2||
    3. If k is provided: orthogonalize against e1 and e2: u3 = k - proj_{e1}(k) - proj_{e2}(k)
       Then normalize: e3 = u3 / ||u3||
       If k is None: compute e3 = e1 × e2 (cross product)

    The resulting basis vectors satisfy:
    - ||e1|| = ||e2|| = ||e3|| = 1 (unit vectors)
    - e1 · e2 = e1 · e3 = e2 · e3 = 0 (orthogonal)
    - e1 × e2 = e3 (right-handed coordinate system)

    This is commonly used in biomechanics to define anatomical reference
    frames from landmark positions, where i, j, k represent directions
    derived from anatomical landmarks.

    Examples
    --------
    >>> # Define a simple coordinate system from three direction vectors
    >>> i = np.array([[1, 0, 0], [1, 0, 0]])  # X direction
    >>> j = np.array([[0, 1, 0], [0, 1, 0]])  # Y direction
    >>> R = gram_schmidt(i, j)
    >>> R.shape
    (2, 3, 3)
    """
    # Normalize first vector
    e1 = i / np.linalg.norm(i, axis=1, keepdims=True)

    # Project j onto e1 and orthogonalize
    proj_j_on_e1 = np.sum(j * e1, axis=1, keepdims=True) * e1
    u2 = j - proj_j_on_e1
    e2 = u2 / np.linalg.norm(u2, axis=1, keepdims=True)

    if k is not None:
        # Project k onto e1 and e2, then orthogonalize
        proj_k_on_e1 = np.sum(k * e1, axis=1, keepdims=True) * e1
        proj_k_on_e2 = np.sum(k * e2, axis=1, keepdims=True) * e2
        u3 = k - proj_k_on_e1 - proj_k_on_e2
        e3 = u3 / np.linalg.norm(u3, axis=1, keepdims=True)
    else:
        # Calculate third vector as cross product
        e3 = np.cross(e1, e2)

    # Stack orthonormal vectors as columns of rotation matrix
    return np.stack([e1, e2, e3], axis=2)  # shape (N, 3, 3)


def fillna(
    arr: np.ndarray | DataFrame | Series,
    value: float | int | np.ndarray | list | tuple | None = None,
    mice: bool = False,
    max_iter: int = 50,
    random_state: int | None = None,
    inplace: bool = False,
):
    """
    Fill missing values.

    Imputation priority:

    1. value is not None
       Direct replacement.

    2. mice=True
       IterativeImputer (MICE-like) using all columns as predictors.

    3. Otherwise
       Natural cubic spline interpolation.

    Parameters
    ----------
    arr : ndarray | DataFrame | Series
        Data containing missing values.

    value : scalar | array-like | None, default=None
        Direct replacement values.

        Accepted formats:

        - scalar
        - (n_columns,)
        - (1, n_columns)
        - arr.shape

        No NaN values are allowed.

    mice : bool, default=False
        If True and value is None, use IterativeImputer.

    max_iter : int, default=50
        Number of MICE iterations.

    random_state : int | None, default=None
        Random seed for MICE.

    inplace : bool, default=False
        Modify the original object.

    Returns
    -------
    ndarray | DataFrame | Series | None
        Filled object.

        If inplace=True:
            modifies the input and returns None.

        Otherwise:
            returns a filled copy preserving the original type.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not isinstance(arr, (np.ndarray, DataFrame, Series)):
        raise TypeError(
            "'arr' must be a numpy.ndarray, pandas.DataFrame or pandas.Series."
        )

    original_shape = arr.shape

    # ------------------------------------------------------------------
    # Convert to DataFrame
    # ------------------------------------------------------------------

    if isinstance(arr, np.ndarray):
        arr_float = arr.astype(float)
        if arr_float.ndim == 1:
            obj = DataFrame(
                arr_float.reshape(-1, 1),
                columns=["Y"],
            )
        else:
            obj = DataFrame(
                arr_float,
                columns=[f"Y{i}" for i in range(arr_float.shape[1])],
            )

    elif isinstance(arr, Series):
        obj = DataFrame(
            arr.astype(float),
            columns=["Y"],
        )

    else:
        obj = arr.copy().astype(float)

    # ------------------------------------------------------------------
    # No missing values
    # ------------------------------------------------------------------

    if not obj.isna().values.any():
        if inplace:
            return None
        return arr.copy()

    filled = obj.copy()

    # ==================================================================
    # PRIORITY 1
    # DIRECT REPLACEMENT
    # ==================================================================

    if value is not None:
        value_array = np.asarray(value)
        if np.isnan(value_array.astype(float)).any():
            raise ValueError("'value' cannot contain NaN values.")

        # --------------------------------------------------------------
        # Scalar
        # --------------------------------------------------------------

        if value_array.ndim == 0:
            filled.iloc[:, :] = np.nan_to_num(filled.to_numpy(), nan=float(value_array))

        # --------------------------------------------------------------
        # Shape (n_columns,)
        # --------------------------------------------------------------

        elif value_array.ndim == 1 and value_array.size == obj.shape[1]:
            for i, col in enumerate(filled.columns):
                filled.iloc[:, i] = np.nan_to_num(
                    filled[col].to_numpy(), nan=float(value_array[i])
                )

        # --------------------------------------------------------------
        # Shape (1, n_columns)
        # --------------------------------------------------------------

        elif value_array.ndim == 2 and value_array.shape == (1, obj.shape[1]):
            for i, col in enumerate(filled.columns):
                filled.iloc[:, i] = np.nan_to_num(
                    filled[col].to_numpy(), nan=float(value_array[0, i])
                )

        # --------------------------------------------------------------
        # Shape == arr.shape
        # --------------------------------------------------------------

        elif value_array.shape == obj.shape:
            mask = filled.isna()
            filled = filled.where(
                ~mask,
                value_array,  # type: ignore
            )

        else:
            raise ValueError(
                "Invalid shape for 'value'. Accepted shapes are:\n"
                "- scalar\n"
                "- (n_columns,)\n"
                "- (1, n_columns)\n"
                "- arr.shape"
            )

    # ==================================================================
    # PRIORITY 2
    # MICE
    # ==================================================================

    if mice and filled.isna().values.any():

        filled = DataFrame(
            IterativeImputer(
                estimator=Ridge(alpha=1.0),
                max_iter=max_iter,
                random_state=random_state,
                initial_strategy="median",
                imputation_order="ascending",
                skip_complete=True,
                # keep_empty_features=True,
                # sample_posterior=True,
            ).fit_transform(obj),
            index=obj.index,
            columns=obj.columns,
        )

    # ==================================================================
    # PRIORITY 3
    # NATURAL CUBIC SPLINE
    # ==================================================================

    if filled.isna().values.any():

        for col in filled.columns:
            values = filled[col].to_numpy(dtype=float)
            x_new = np.where(np.isnan(values))[0]
            if len(x_new) == 0:
                continue
            x_old = np.where(~np.isnan(values))[0]
            if len(x_old) == 0:
                warnings.warn(
                    f"Column '{col}' contains only NaN values "
                    f"and cannot be interpolated.",
                    RuntimeWarning,
                )
                continue

            if len(x_old) < 2:
                continue

            try:
                spline = CubicSpline(
                    x_old,
                    values[x_old],
                    bc_type="natural",
                    extrapolate=True,
                )

                filled.loc[
                    filled.index[x_new],
                    col,
                ] = spline(x_new)

            except Exception:
                continue

    # ------------------------------------------------------------------
    # Restore original type
    # ------------------------------------------------------------------

    out = filled.to_numpy(dtype=float).reshape(original_shape)
    if isinstance(arr, np.ndarray):
        if inplace:
            arr[:] = out
            return None
        return out

    if isinstance(arr, Series):
        result = filled.iloc[:, 0]
        if inplace:
            arr.loc[:] = result.values
            return None
        return result

    if inplace:
        arr.loc[:, :] = out
        return None

    return filled


def tkeo(
    arr: np.ndarray,
) -> np.ndarray:
    """
    Compute Teager-Kaiser Energy Operator for signal energy estimation.

    The TKEO is a nonlinear operator sensitive to both amplitude and
    frequency changes, making it effective for detecting transient events,
    muscle activation in EMG signals, and instantaneous energy estimation.

    Parameters
    ----------
    arr : np.ndarray
        The 1D input signal.

    Returns
    -------
    np.ndarray
        Teager-Kaiser energy with the same length as input.
        Edge values are repeated from the first/last computed value.

    Examples
    --------
    >>> import numpy as np
    >>> # Constant amplitude sine wave
    >>> t = np.linspace(0, 1, 100)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> energy = tkeo(signal)
    >>> np.all(energy >= 0)  # TKEO is non-negative for most signals
    True

    Notes
    -----
    The discrete TKEO is defined as:
        Ψ[x[n]] = x[n]² - x[n+1] × x[n-1]

    For a sinusoidal signal x[n] = A×cos(ωn + φ), the TKEO approximates:
        Ψ[x[n]] ≈ A² × ω²

    making it proportional to both amplitude² and frequency².

    Common applications:
    - EMG onset detection (muscle activation)
    - Speech signal analysis
    - Bearing fault detection in vibration signals
    - Transient event detection

    References
    ----------
    Kaiser, J. F. (1990). On a simple algorithm to calculate the
    'energy' of a signal. Proceedings of ICASSP-90, 381-384.

    Li, X., Zhou, P., & Aruin, A. S. (2007). Teager-Kaiser energy
    operation of surface EMG improves muscle activity onset detection.
    Annals of Biomedical Engineering, 35(9), 1532-1538.

    See Also
    --------
    rms_filt : Alternative energy estimation using RMS.
    """
    out = arr[1:-1] ** 2 - arr[2:] * arr[:-2]
    return np.concatenate([[out[0]], out, [out[-1]]]).astype(float)


def to_reference_frame(
    obj: DataFrame | np.ndarray,
    origin: np.ndarray | list[float | int] = [0, 0, 0],
    axis1: np.ndarray | list[float | int] = [1, 0, 0],
    axis2: np.ndarray | list[float | int] = [0, 1, 0],
    axis3: np.ndarray | list[float | int] = [0, 0, 1],
) -> DataFrame | np.ndarray:
    """
    Transform 3D data to a specified reference frame coordinate system.

    Applies translation and rotation to express 3D point coordinates in
    a custom reference frame defined by origin and axis orientations.
    Common in biomechanics for transforming marker data from laboratory
    to anatomical coordinate systems.

    Parameters
    ----------
    obj : DataFrame or np.ndarray
        3D data to transform. Must be shape (N, 3) where columns/axes
        represent [X, Y, Z] coordinates.
    origin : np.ndarray or list of float or int, optional
        3D coordinates [x, y, z] of the new reference frame's origin
        in the current coordinate system. Default is [0, 0, 0].
    axis1 : np.ndarray or list of float or int, optional
        Direction vector [x, y, z] defining the first axis of the
        new reference frame. Default is [1, 0, 0] (X-axis).
    axis2 : np.ndarray or list of float or int, optional
        Direction vector [x, y, z] defining the second axis of the
        new reference frame. Default is [0, 1, 0] (Y-axis).
    axis3 : np.ndarray or list of float or int, optional
        Direction vector [x, y, z] defining the third axis of the
        new reference frame. Default is [0, 0, 1] (Z-axis).

    Returns
    -------
    DataFrame or np.ndarray
        Transformed data in the new reference frame. Output type
        matches input type (DataFrame or ndarray).

    Raises
    ------
    ValueError
        If obj is not a valid 3D dataset (shape[1] != 3), or if
        origin/axis vectors cannot be converted to length-3 arrays.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Transform points to a new reference frame
    >>> points = np.array([[1, 2, 3], [4, 5, 6]])
    >>> transformed = to_reference_frame(
    ...     points,
    ...     origin=[1, 1, 1],
    ...     axis1=[1, 0, 0],
    ...     axis2=[0, 1, 0],
    ...     axis3=[0, 0, 1]
    ... )
    >>> transformed
    array([[0., 1., 2.],
           [3., 4., 5.]])

    Notes
    -----
    Transformation steps:
    1. Translate: subtract origin from all points
    2. Rotate: apply rotation matrix derived from axis1, axis2, axis3
       using Gram-Schmidt orthonormalization

    The axes need not be orthonormal initially; gram_schmidt() is
    applied internally to construct a proper rotation matrix.

    Common use case: Transform motion capture markers from global
    lab coordinates to a local anatomical reference frame defined
    by bony landmarks.

    See Also
    --------
    gram_schmidt : Construct orthonormal basis from axis vectors.
    scipy.spatial.transform.Rotation : Rotation matrix operations.
    """

    def _validate_array(arr: object):
        """
        Validate and convert input to 3-element float array.

        Parameters
        ----------
        arr : object
            Input to validate (should be array-like with 3 elements).

        Returns
        -------
        np.ndarray
            Validated 1D array of length 3.

        Raises
        ------
        ValueError
            If arr cannot be converted to a length-3 float array.
        """
        msg = "origin, axis1, axis2 and axis3 have to be"
        msg += " castable to 1D arrays of len = 3."
        try:
            out = np.array([arr]).astype(float).flatten()
        except Exception:
            raise ValueError(msg)
        if len(out) != 3:
            raise ValueError(msg)
        return out

    # check inputs
    msg = "'obj' must be a numeric pandas DataFrame or a 2D numpy array"
    msg += " with 3 elements along the second dimension."
    try:
        dfr = DataFrame(obj)
        if dfr.shape[1] != 3:
            raise ValueError(msg)
    except Exception:
        raise ValueError(msg)
    ori = np.ones(dfr.shape) * _validate_array(origin)
    ax1 = _validate_array(axis1)
    ax2 = _validate_array(axis2)
    ax3 = _validate_array(axis3)

    # create the rotation matrix
    # gram_schmidt expects (N, 3) arrays, so reshape 1D axes to (1, 3)
    rmat_array = gram_schmidt(ax1.reshape(1, 3), ax2.reshape(1, 3), ax3.reshape(1, 3))
    # Extract the single rotation matrix from shape (1, 3, 3)
    rmat = Rotation.from_matrix(rmat_array[0])

    # apply
    rotated = rmat.apply(dfr.values - ori).astype(float)
    if not isinstance(obj, DataFrame):
        return rotated
    return DataFrame(rotated, columns=obj.columns, index=obj.index)
