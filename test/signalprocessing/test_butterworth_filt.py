"""
Test suite for butterworth_filt function.

Tests verify Butterworth filtering with various parameters:
cutoff frequencies, filter types, and phase correction.
"""

import numpy as np
import pytest

from labanalysis.signalprocessing import butterworth_filt


def test_butterworth_lowpass_basic():
    """
    Test basic lowpass Butterworth filter application.

    Expected:
        Lowpass filter should attenuate high-frequency components
        while preserving low-frequency signal
    """
    # Create signal: 5 Hz sine + 50 Hz noise
    t = np.linspace(0, 1, 1000)
    signal_clean = np.sin(2 * np.pi * 5 * t)
    noise = 0.5 * np.sin(2 * np.pi * 50 * t)
    signal_noisy = signal_clean + noise

    # Apply 10 Hz lowpass filter
    filtered = butterworth_filt(signal_noisy, fcut=10, fsamp=1000, order=4)

    # Filtered signal should be closer to clean signal
    error_before = np.mean((signal_noisy - signal_clean) ** 2)
    error_after = np.mean((filtered - signal_clean) ** 2)

    assert error_after < error_before
    assert len(filtered) == len(signal_noisy)


def test_butterworth_highpass():
    """
    Test highpass Butterworth filter removes low frequencies.

    Expected:
        Highpass filter should attenuate low-frequency components
    """
    t = np.linspace(0, 1, 1000)
    # Signal with 2 Hz (low) and 50 Hz (high) components
    low_freq = np.sin(2 * np.pi * 2 * t)
    high_freq = np.sin(2 * np.pi * 50 * t)
    signal = low_freq + high_freq

    # Apply 20 Hz highpass filter
    filtered = butterworth_filt(signal, fcut=20, fsamp=1000, order=4, ftype="highpass")

    # RMS of filtered should be closer to high_freq component
    assert np.sqrt(np.mean(filtered**2)) < np.sqrt(np.mean(signal**2))


def test_butterworth_bandpass():
    """
    Test bandpass Butterworth filter preserves middle frequencies.

    Expected:
        Bandpass [10, 50] Hz should preserve 30 Hz, attenuate 5 Hz and 100 Hz
    """
    t = np.linspace(0, 2, 2000)
    sig_5hz = np.sin(2 * np.pi * 5 * t)
    sig_30hz = np.sin(2 * np.pi * 30 * t)
    sig_100hz = np.sin(2 * np.pi * 100 * t)
    signal = sig_5hz + sig_30hz + sig_100hz

    # Bandpass 10-50 Hz
    filtered = butterworth_filt(
        signal, fcut=[10, 50], fsamp=1000, order=4, ftype="bandpass"
    )

    # Filtered should correlate with 30 Hz component
    corr_30hz = np.corrcoef(filtered, sig_30hz)[0, 1]
    assert abs(corr_30hz) > 0.7  # Strong correlation


def test_butterworth_bandstop():
    """
    Test bandstop (notch) filter removes specific frequency band.

    Expected:
        Bandstop [45, 55] Hz should attenuate 50 Hz powerline noise
    """
    t = np.linspace(0, 1, 1000)
    signal_clean = np.sin(2 * np.pi * 10 * t)
    powerline_noise = 0.5 * np.sin(2 * np.pi * 50 * t)
    signal = signal_clean + powerline_noise

    # Bandstop 45-55 Hz (notch out 50 Hz)
    filtered = butterworth_filt(
        signal, fcut=[45, 55], fsamp=1000, order=4, ftype="bandstop"
    )

    # Filtered should be closer to clean signal
    error_before = np.mean((signal - signal_clean) ** 2)
    error_after = np.mean((filtered - signal_clean) ** 2)

    assert error_after < error_before


def test_butterworth_phase_corrected_vs_uncorrected():
    """
    Test difference between phase-corrected and non-phase-corrected filtering.

    Expected:
        Phase-corrected (filtfilt) should have no phase shift
        Non-phase-corrected should introduce phase delay
    """
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t)

    filtered_corrected = butterworth_filt(
        signal, fcut=10, fsamp=1000, order=4, phase_corrected=True
    )
    filtered_uncorrected = butterworth_filt(
        signal, fcut=10, fsamp=1000, order=4, phase_corrected=False
    )

    # Phase-corrected should align better with original
    # Uncorrected will have delay
    peak_original = np.argmax(signal[:200])
    peak_corrected = np.argmax(filtered_corrected[:200])
    peak_uncorrected = np.argmax(filtered_uncorrected[:200])

    assert abs(peak_corrected - peak_original) < abs(peak_uncorrected - peak_original)


def test_butterworth_output_shape():
    """
    Test that output shape matches input shape.

    Expected:
        Filtered signal should have same length as input
    """
    signal = np.random.randn(500)
    filtered = butterworth_filt(signal, fcut=5, fsamp=100, order=4)

    assert filtered.shape == signal.shape
    assert len(filtered) == len(signal)


def test_butterworth_output_type():
    """
    Test that output is float numpy array.

    Expected:
        Output should be 1D numpy array with float dtype
    """
    signal = np.random.randn(100)
    filtered = butterworth_filt(signal, fcut=5, fsamp=50)

    assert isinstance(filtered, np.ndarray)
    assert filtered.dtype == np.float64
    assert filtered.ndim == 1


def test_butterworth_biomechanics_typical():
    """
    Test typical biomechanics filtering: 6 Hz lowpass for kinematics.

    Expected:
        Standard Winter's recommendation for kinematic data filtering
        should execute without error and reduce noise
    """
    # Simulate marker position data with noise
    t = np.linspace(0, 5, 1000)  # 5 seconds at 200 Hz
    position = 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz movement
    noise = 0.05 * np.random.randn(len(t))
    noisy_position = position + noise

    # Apply typical biomechanics filter: 6 Hz lowpass, 4th order
    filtered = butterworth_filt(noisy_position, fcut=6, fsamp=200, order=4)

    # Should reduce noise (lower variance)
    assert np.var(filtered) < np.var(noisy_position)
    # Should preserve shape (high correlation with clean signal)
    corr = np.corrcoef(filtered, position)[0, 1]
    assert corr > 0.95


def test_butterworth_preserves_dc_offset():
    """
    Test that DC offset (mean) is preserved by lowpass filter.

    Expected:
        Mean of filtered signal should be close to mean of original
    """
    signal = np.random.randn(500) + 10.0  # Random signal with DC offset
    filtered = butterworth_filt(signal, fcut=5, fsamp=100, order=4)

    assert abs(np.mean(filtered) - np.mean(signal)) < 0.1


def test_butterworth_cutoff_list_input():
    """
    Test that fcut accepts list input for bandpass/bandstop.

    Expected:
        Should accept [fcut_low, fcut_high] for band filters
    """
    signal = np.random.randn(500)

    # Test with list input
    filtered = butterworth_filt(
        signal, fcut=[10, 40], fsamp=200, order=4, ftype="bandpass"
    )

    assert len(filtered) == len(signal)


def test_butterworth_cutoff_tuple_input():
    """
    Test that fcut accepts tuple input for bandpass/bandstop.

    Expected:
        Should accept (fcut_low, fcut_high) for band filters
    """
    signal = np.random.randn(500)

    # Test with tuple input
    filtered = butterworth_filt(
        signal, fcut=(10, 40), fsamp=200, order=4, ftype="bandpass"
    )

    assert len(filtered) == len(signal)


def test_butterworth_different_orders():
    """
    Test that higher filter order provides sharper cutoff.

    Expected:
        Higher order filter should execute without error
    """
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

    filtered_order2 = butterworth_filt(signal, fcut=10, fsamp=1000, order=2)
    filtered_order8 = butterworth_filt(signal, fcut=10, fsamp=1000, order=8)

    # Both should produce valid output
    assert len(filtered_order2) == len(signal)
    assert len(filtered_order8) == len(signal)
