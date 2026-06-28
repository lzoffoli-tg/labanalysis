"""
Test suite for interpolation and analysis functions.

Tests verify: cubicspline_interp, residual_analysis, crossovers, fillna, gram_schmidt, to_reference_frame.
"""

import numpy as np
import pandas as pd
import pytest

from labanalysis.signalprocessing import (
    cubicspline_interp,
    residual_analysis,
    crossovers,
    fillna,
    gram_schmidt,
    to_reference_frame,
)


class TestCubicSplineInterp:
    """Tests for cubic spline interpolation."""

    def test_cubicspline_nsamp(self):
        """Test interpolation with specified number of samples."""
        y = np.array([0, 1, 4, 9, 16])  # x^2 at x=0,1,2,3,4
        y_interp = cubicspline_interp(y, nsamp=9)

        assert len(y_interp) == 9

    def test_cubicspline_with_x_coordinates(self):
        """Test interpolation with custom x coordinates."""
        x_old = np.array([0, 1, 2, 3])
        y_old = np.array([0, 1, 4, 9])
        x_new = np.array([0.5, 1.5, 2.5])
        y_new = cubicspline_interp(y_old, x_old=x_old, x_new=x_new)

        assert len(y_new) == len(x_new)

    def test_cubicspline_linear_signal(self):
        """Test interpolation preserves linear relationship."""
        x_old = np.array([0, 1, 2, 3, 4])
        y_old = 2 * x_old + 3
        y_interp = cubicspline_interp(y_old, nsamp=20)

        # Interpolated values should still follow linear relationship
        x_new = np.linspace(0, 4, 20)
        expected = 2 * x_new + 3
        assert np.allclose(y_interp, expected, atol=0.01)


class TestResidualAnalysis:
    """Tests for residual analysis (Winter's method)."""

    def test_residual_analysis_basic(self):
        """Test residual analysis finds reasonable cutoff."""
        from functools import partial
        from labanalysis.signalprocessing import butterworth_filt

        # Signal: 5 Hz sine + noise
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.randn(1000)

        # Create filter function
        filt_fun = partial(butterworth_filt, fsamp=1000, order=4)

        fopt, frq, res = residual_analysis(signal, filt_fun, fnum=100, fmax=0.4)

        # Optimal cutoff should be in valid range
        assert 0 < fopt < 0.5
        assert len(frq) == len(res)

    def test_residual_analysis_output_types(self):
        """Test residual analysis output types."""
        from functools import partial
        from labanalysis.signalprocessing import butterworth_filt

        signal = np.random.randn(200)
        filt_fun = partial(butterworth_filt, fsamp=200, order=4)

        fopt, frq, res = residual_analysis(signal, filt_fun, fnum=50, fmax=0.4)

        assert isinstance(fopt, float)
        assert isinstance(frq, np.ndarray)
        assert isinstance(res, np.ndarray)


class TestCrossovers:
    """Tests for piecewise linear regression crossovers."""

    def test_crossovers_two_segments(self):
        """Test finding crossover point between two linear segments."""
        # Create signal with clear breakpoint
        x1 = np.linspace(0, 5, 50)
        y1 = 2 * x1 + 1
        x2 = np.linspace(5, 10, 50)
        y2 = -1 * x2 + 20
        y = np.concatenate([y1, y2])

        crs, slopes = crossovers(y, segments=2, min_samples=5)

        # Should find one crossover near index 50
        assert len(crs) == 1
        assert 40 < crs[0] < 60

    def test_crossovers_slopes(self):
        """Test that slopes are returned correctly."""
        y = np.concatenate([np.linspace(0, 10, 50), np.linspace(10, 5, 50)])
        crs, slopes = crossovers(y, segments=2, min_samples=5)

        # Should return slopes for both segments
        assert slopes.shape[0] == 2
        assert slopes.shape[1] == 2  # slope and intercept

    def test_crossovers_with_x_axis(self):
        """Test crossovers with custom x-axis."""
        x = np.linspace(0, 10, 100)
        y = np.where(x < 5, 2 * x, -x + 15)
        crs, slopes = crossovers(y, x=x, segments=2, min_samples=5)

        assert len(crs) >= 1


class TestFillna:
    """Tests for missing data imputation."""

    def test_fillna_constant_value(self):
        """Test filling NaN with constant value."""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        filled = fillna(arr, value=0.0)

        assert not np.any(np.isnan(filled))
        assert filled[1] == 0.0
        assert filled[3] == 0.0

    def test_fillna_cubic_spline(self):
        """Test filling NaN with cubic spline interpolation."""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        filled = fillna(arr)

        assert not np.any(np.isnan(filled))
        # Should interpolate linearly for this case
        assert abs(filled[1] - 2.0) < 0.5
        assert abs(filled[3] - 4.0) < 0.5

    def test_fillna_dataframe(self):
        """Test filling NaN in DataFrame."""
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 2, 3]})
        filled = fillna(df)

        assert not filled.isnull().any().any()

    def test_fillna_with_regressors(self):
        """Test filling with linear regression."""
        arr = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        regressors = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        filled = fillna(arr, regressors=regressors)

        assert not np.any(np.isnan(filled))

    def test_fillna_inplace_array(self):
        """Test in-place filling for numpy array."""
        arr = np.array([1.0, np.nan, 3.0])
        fillna(arr, value=0.0, inplace=True)

        assert not np.any(np.isnan(arr))
        assert arr[1] == 0.0


class TestGramSchmidt:
    """Tests for Gram-Schmidt orthonormalization."""

    def test_gram_schmidt_orthogonal_input(self):
        """Test with already orthogonal vectors."""
        i = np.array([[1, 0, 0]])
        j = np.array([[0, 1, 0]])
        R = gram_schmidt(i, j)

        assert R.shape == (1, 3, 3)
        # Result should be identity matrix
        assert np.allclose(R[0], np.eye(3), atol=0.01)

    def test_gram_schmidt_non_orthogonal(self):
        """Test with non-orthogonal vectors."""
        i = np.array([[1, 0, 0]])
        j = np.array([[1, 1, 0]])  # Not orthogonal to i
        R = gram_schmidt(i, j)

        # Verify orthonormality
        e1 = R[0, :, 0]
        e2 = R[0, :, 1]
        e3 = R[0, :, 2]

        # Should be unit vectors
        assert np.abs(np.linalg.norm(e1) - 1.0) < 0.01
        assert np.abs(np.linalg.norm(e2) - 1.0) < 0.01
        assert np.abs(np.linalg.norm(e3) - 1.0) < 0.01

        # Should be orthogonal
        assert abs(np.dot(e1, e2)) < 0.01
        assert abs(np.dot(e1, e3)) < 0.01
        assert abs(np.dot(e2, e3)) < 0.01

    def test_gram_schmidt_with_third_vector(self):
        """Test with all three vectors provided."""
        i = np.array([[1, 0, 0]])
        j = np.array([[0, 1, 0]])
        k = np.array([[0, 0, 1]])
        R = gram_schmidt(i, j, k)

        assert R.shape == (1, 3, 3)

    def test_gram_schmidt_batch(self):
        """Test with multiple sets of vectors (batch processing)."""
        # Two sets of vectors
        i = np.array([[1, 0, 0], [1, 0, 0]])
        j = np.array([[0, 1, 0], [1, 1, 0]])
        R = gram_schmidt(i, j)

        assert R.shape == (2, 3, 3)


class TestToReferenceFrame:
    """Tests for reference frame transformation."""

    def test_to_reference_frame_identity(self):
        """Test transformation with identity reference frame."""
        points = np.array([[1, 2, 3], [4, 5, 6]])
        transformed = to_reference_frame(
            points, origin=[0, 0, 0], axis1=[1, 0, 0], axis2=[0, 1, 0], axis3=[0, 0, 1]
        )

        # Should remain unchanged for identity frame
        assert np.allclose(transformed, points, atol=0.01)

    def test_to_reference_frame_translation(self):
        """Test pure translation (no rotation)."""
        points = np.array([[1, 2, 3]])
        transformed = to_reference_frame(
            points, origin=[1, 1, 1], axis1=[1, 0, 0], axis2=[0, 1, 0], axis3=[0, 0, 1]
        )

        # Should be translated by -origin
        expected = np.array([[0, 1, 2]])
        assert np.allclose(transformed, expected, atol=0.01)

    def test_to_reference_frame_dataframe(self):
        """Test with DataFrame input."""
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["X", "Y", "Z"])
        transformed = to_reference_frame(df, origin=[0, 0, 0])

        assert isinstance(transformed, pd.DataFrame)
        assert list(transformed.columns) == ["X", "Y", "Z"]

    def test_to_reference_frame_rotation_90deg(self):
        """Test 90-degree rotation about Z axis."""
        points = np.array([[1, 0, 0]])  # Point on X-axis
        # Rotate reference frame 90° about Z: new X = old Y, new Y = -old X
        transformed = to_reference_frame(
            points, origin=[0, 0, 0], axis1=[0, 1, 0], axis2=[-1, 0, 0], axis3=[0, 0, 1]
        )

        # Should produce valid transformed output
        assert transformed.shape == points.shape
        assert not np.allclose(transformed, points)
