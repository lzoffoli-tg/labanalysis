"""
Shared fixtures and utilities for signalprocessing tests.

This module provides:
- Signal generation fixtures (sine waves, noise, chirps, impulses, etc.)
- Validation helper functions for assertions
- Common test utilities and constants
"""

import numpy as np
import pytest
from functools import partial


# ============================================================================
# SIGNAL GENERATION FIXTURES
# ============================================================================

@pytest.fixture
def sine_wave():
    """
    Generate clean sine wave with configurable parameters.

    Returns a function that creates sine waves with specified frequency,
    sampling rate, duration, amplitude, and phase.

    Returns
    -------
    function
        Function that generates (time, signal) tuple.

    Examples
    --------
    >>> def test_example(sine_wave):
    ...     t, signal = sine_wave(freq=10, fsamp=1000, duration=1)
    ...     assert len(signal) == 1000
    """
    def _make_sine(freq=5, fsamp=1000, duration=1, amplitude=1, phase=0):
        t = np.linspace(0, duration, int(fsamp * duration))
        signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
        return t, signal
    return _make_sine


@pytest.fixture
def noisy_signal():
    """
    Add noise to a signal with controlled SNR (Signal-to-Noise Ratio).

    Returns a function that adds Gaussian white noise to any signal
    with a specified SNR in decibels.

    Returns
    -------
    function
        Function that adds noise to signal and returns noisy signal.

    Examples
    --------
    >>> def test_example(sine_wave, noisy_signal):
    ...     t, clean = sine_wave(freq=5)
    ...     noisy = noisy_signal(clean, snr_db=20)
    """
    def _add_noise(signal, snr_db=20):
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
        return signal + noise
    return _add_noise


@pytest.fixture
def step_function():
    """
    Generate step function with configurable step position and amplitude.

    Returns
    -------
    function
        Function that generates step function signal.

    Examples
    --------
    >>> def test_example(step_function):
    ...     signal = step_function(n_samples=100, step_position=50, amplitude=1)
    ...     assert signal[49] == 0
    ...     assert signal[50] == 1
    """
    def _make_step(n_samples=100, step_position=50, amplitude=1, offset=0):
        signal = np.ones(n_samples) * offset
        signal[step_position:] = amplitude + offset
        return signal
    return _make_step


@pytest.fixture
def chirp_signal():
    """
    Generate frequency-swept chirp signal (linear or exponential).

    A chirp is a signal whose frequency increases or decreases over time.
    Useful for testing frequency-dependent filter behavior.

    Returns
    -------
    function
        Function that generates (time, chirp_signal) tuple.

    Examples
    --------
    >>> def test_example(chirp_signal):
    ...     t, signal = chirp_signal(f0=1, f1=50, duration=1, fsamp=1000)
    ...     # Signal sweeps from 1 Hz to 50 Hz over 1 second
    """
    def _make_chirp(f0=1, f1=50, duration=1, fsamp=1000, method='linear'):
        t = np.linspace(0, duration, int(fsamp * duration))
        if method == 'linear':
            # Linear frequency sweep
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
        elif method == 'exponential':
            # Exponential frequency sweep
            k = (f1 / f0) ** (1 / duration)
            phase = 2 * np.pi * f0 * (k**t - 1) / np.log(k)
        else:
            raise ValueError(f"Unknown method: {method}")
        return t, np.sin(phase)
    return _make_chirp


@pytest.fixture
def impulse_train():
    """
    Generate impulse train (Dirac comb) with evenly spaced impulses.

    Useful for testing impulse response of filters.

    Returns
    -------
    function
        Function that generates impulse train signal.

    Examples
    --------
    >>> def test_example(impulse_train):
    ...     signal = impulse_train(n_samples=100, n_impulses=5)
    ...     assert np.sum(signal > 0) == 5
    """
    def _make_impulses(n_samples=100, n_impulses=5, amplitude=1):
        signal = np.zeros(n_samples)
        positions = np.linspace(10, n_samples-10, n_impulses, dtype=int)
        signal[positions] = amplitude
        return signal
    return _make_impulses


@pytest.fixture
def polynomial_signal():
    """
    Generate signal from polynomial function.

    Useful for testing derivative functions with known analytical solutions.

    Returns
    -------
    function
        Function that generates (x, y) tuple where y = polynomial(x).

    Examples
    --------
    >>> def test_example(polynomial_signal):
    ...     # y = 2x² + 3x + 1
    ...     x, y = polynomial_signal(coefficients=[2, 3, 1], x_range=(0, 10))
    """
    def _make_polynomial(coefficients, x_range=(0, 10), n_samples=100):
        x = np.linspace(x_range[0], x_range[1], n_samples)
        y = np.polyval(coefficients, x)
        return x, y
    return _make_polynomial


@pytest.fixture
def multi_frequency_signal():
    """
    Generate signal with multiple frequency components.

    Useful for testing spectral analysis and filtering.

    Returns
    -------
    function
        Function that generates (time, signal) tuple with multiple frequencies.

    Examples
    --------
    >>> def test_example(multi_frequency_signal):
    ...     # Signal with 5 Hz and 15 Hz components
    ...     t, signal = multi_frequency_signal(
    ...         frequencies=[5, 15],
    ...         amplitudes=[1.0, 0.5],
    ...         fsamp=1000
    ...     )
    """
    def _make_multi_freq(frequencies, amplitudes=None, fsamp=1000, duration=1):
        if amplitudes is None:
            amplitudes = np.ones(len(frequencies))

        t = np.linspace(0, duration, int(fsamp * duration))
        signal = np.zeros_like(t)

        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t)

        return t, signal
    return _make_multi_freq


# ============================================================================
# VALIDATION HELPER FUNCTIONS
# ============================================================================

def assert_frequency_attenuated(original, filtered, freq, fsamp, min_attenuation_db=-20):
    """
    Assert that a specific frequency is attenuated in filtered signal.

    Uses PSD (Power Spectral Density) to measure power at target frequency
    in both original and filtered signals, then verifies attenuation.

    Parameters
    ----------
    original : np.ndarray
        Original signal before filtering
    filtered : np.ndarray
        Filtered signal
    freq : float
        Frequency to check (Hz)
    fsamp : float
        Sampling frequency (Hz)
    min_attenuation_db : float
        Minimum required attenuation in dB (negative value, e.g., -20)

    Raises
    ------
    AssertionError
        If attenuation is less than required

    Examples
    --------
    >>> # Verify that 50 Hz is attenuated by at least 20 dB
    >>> assert_frequency_attenuated(original, filtered, freq=50, fsamp=1000, min_attenuation_db=-20)
    """
    # Import here to avoid circular dependency
    import sys
    from os.path import join, dirname, abspath
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "src"))
    from labanalysis.signalprocessing import psd

    f_orig, p_orig = psd(original, fsamp)
    f_filt, p_filt = psd(filtered, fsamp)

    # Find power at target frequency
    idx = np.argmin(np.abs(f_orig - freq))

    # Calculate attenuation
    if p_orig[idx] > 1e-10:  # Avoid division by zero
        attenuation_db = 10 * np.log10(p_filt[idx] / p_orig[idx])
        assert attenuation_db < min_attenuation_db, \
            f"Frequency {freq} Hz attenuated by {attenuation_db:.2f} dB, " \
            f"required < {min_attenuation_db} dB"


def assert_orthonormal_basis(vectors, rtol=1e-6, atol=1e-10):
    """
    Assert that a set of vectors form an orthonormal basis.

    Checks that:
    - Each vector has unit length (normalized)
    - All vectors are mutually orthogonal (dot product = 0)

    Parameters
    ----------
    vectors : list of np.ndarray
        List of vectors to check (each should be 1D array)
    rtol : float
        Relative tolerance for norm check
    atol : float
        Absolute tolerance for dot product check

    Raises
    ------
    AssertionError
        If vectors are not orthonormal

    Examples
    --------
    >>> e1 = np.array([1, 0, 0])
    >>> e2 = np.array([0, 1, 0])
    >>> e3 = np.array([0, 0, 1])
    >>> assert_orthonormal_basis([e1, e2, e3])
    """
    n = len(vectors)

    for i in range(n):
        # Check normalization
        norm = np.linalg.norm(vectors[i])
        np.testing.assert_allclose(norm, 1.0, rtol=rtol,
                                   err_msg=f"Vector {i} not normalized: norm={norm}")

        # Check orthogonality with other vectors
        for j in range(i+1, n):
            dot_product = np.dot(vectors[i], vectors[j])
            np.testing.assert_allclose(dot_product, 0.0, atol=atol,
                                       err_msg=f"Vectors {i} and {j} not orthogonal: "
                                               f"dot product={dot_product}")


def assert_smooth_signal(signal, max_second_derivative=None):
    """
    Assert that a signal is smooth (continuous second derivative).

    Checks for discontinuities by examining the second derivative.
    Optionally enforces a maximum bound on the second derivative.

    Parameters
    ----------
    signal : np.ndarray
        Signal to check for smoothness
    max_second_derivative : float, optional
        Maximum allowed absolute value of second derivative

    Raises
    ------
    AssertionError
        If signal has discontinuities or exceeds max second derivative

    Examples
    --------
    >>> # Check that interpolated signal is smooth
    >>> assert_smooth_signal(interpolated_signal, max_second_derivative=10.0)
    """
    d2y = np.diff(signal, n=2)

    if max_second_derivative is not None:
        max_d2y = np.max(np.abs(d2y))
        assert max_d2y < max_second_derivative, \
            f"Signal not smooth: max|d²y| = {max_d2y} > {max_second_derivative}"

    # Check for discontinuities (large jumps in second derivative)
    if len(d2y) > 1:
        d3y = np.diff(d2y)
        if len(d3y) > 0 and np.std(d3y) > 0:
            discontinuity_threshold = 10 * np.std(d3y)
            discontinuities = np.abs(d3y) > discontinuity_threshold
            n_discontinuities = np.sum(discontinuities)
            assert n_discontinuities == 0, \
                f"Signal has {n_discontinuities} discontinuities in second derivative"


def assert_preserves_dc_component(original, filtered, rtol=0.01):
    """
    Assert that filtering preserves DC component (mean value).

    DC component is the zero-frequency component, i.e., the mean.
    Lowpass filters should preserve this.

    Parameters
    ----------
    original : np.ndarray
        Original signal
    filtered : np.ndarray
        Filtered signal
    rtol : float
        Relative tolerance for comparison

    Raises
    ------
    AssertionError
        If DC component is not preserved within tolerance

    Examples
    --------
    >>> # Lowpass filter should preserve mean
    >>> assert_preserves_dc_component(original, lowpass_filtered)
    """
    dc_original = np.mean(original)
    dc_filtered = np.mean(filtered)

    np.testing.assert_allclose(dc_filtered, dc_original, rtol=rtol,
                               err_msg=f"DC component not preserved: "
                                       f"original={dc_original}, filtered={dc_filtered}")


def assert_frequency_detected(signal, fsamp, expected_freq, tolerance_hz=2):
    """
    Assert that PSD detects a specific frequency in the signal.

    Finds the dominant frequency in the signal's power spectrum and
    verifies it matches the expected frequency within tolerance.

    Parameters
    ----------
    signal : np.ndarray
        Signal to analyze
    fsamp : float
        Sampling frequency (Hz)
    expected_freq : float
        Expected dominant frequency (Hz)
    tolerance_hz : float
        Tolerance for frequency detection (Hz)

    Raises
    ------
    AssertionError
        If detected frequency differs from expected by more than tolerance

    Examples
    --------
    >>> # Verify that 10 Hz sine wave is detected
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> assert_frequency_detected(signal, fsamp=1000, expected_freq=10, tolerance_hz=1)
    """
    # Import here to avoid circular dependency
    import sys
    from os.path import join, dirname, abspath
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "src"))
    from labanalysis.signalprocessing import psd

    frq, pwr = psd(signal, fsamp)
    peak_idx = np.argmax(pwr)
    detected_freq = frq[peak_idx]

    error = abs(detected_freq - expected_freq)
    assert error < tolerance_hz, \
        f"Expected frequency {expected_freq} Hz, detected {detected_freq} Hz " \
        f"(error {error:.2f} Hz > tolerance {tolerance_hz} Hz)"


# ============================================================================
# COMMON TEST CONSTANTS
# ============================================================================

# Common filter orders for parametric testing
FILTER_ORDERS = [1, 3, 5, 7, 11, 21]

# Common offset values for window-based filters
FILTER_OFFSETS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Common filter types
FILTER_TYPES = ["lowpass", "highpass", "bandpass", "bandstop"]

# Common window types for FIR filters
WINDOW_TYPES = ["hamming", "hann", "blackman", "bartlett"]

# Common padding styles
PADDING_STYLES = ["edge", "constant", "reflect", "symmetric"]
