"""Test ReferenceFrame class and transformations."""

import numpy as np
import pandas as pd
import pytest

import labanalysis as laban


class TestReferenceFrameConstructor:
    """Test ReferenceFrame constructor and initialization."""

    def test_constructor_1d_inputs(self):
        """Test constructor with 1D inputs (single sample)."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]

        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        assert ref_frame._n_samples == 1
        assert ref_frame._is_single_frame is True
        assert ref_frame.origin.shape == (1, 3)
        assert ref_frame.lateral_axis.shape == (1, 3)
        assert ref_frame.vertical_axis.shape == (1, 3)
        assert ref_frame.rotation_matrix.shape == (1, 3, 3)

    def test_constructor_2d_inputs(self):
        """Test constructor with 2D inputs (multiple samples)."""
        n_samples = 10
        origin = np.random.rand(n_samples, 3)
        lateral_axis = np.random.rand(n_samples, 3)
        vertical_axis = np.random.rand(n_samples, 3)

        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        assert ref_frame._n_samples == n_samples
        assert ref_frame._is_single_frame is False
        assert ref_frame.origin.shape == (n_samples, 3)
        assert ref_frame.rotation_matrix.shape == (n_samples, 3, 3)

    def test_constructor_with_axis_3(self):
        """Test constructor with explicit third axis."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        anteroposterior_axis = [0.0, 0.0, 1.0]

        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis, anteroposterior_axis)

        assert ref_frame.anteroposterior_axis is not None
        assert ref_frame.anteroposterior_axis.shape == (1, 3)

    def test_constructor_without_axis_3(self):
        """Test that axis_3 is computed when not provided."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]

        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        # anteroposterior_axis should be None since it was computed
        assert ref_frame.anteroposterior_axis is None

    def test_constructor_invalid_shape_columns(self):
        """Test that constructor raises error for wrong number of columns."""
        origin = [0.0, 0.0]  # Only 2 columns
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]

        with pytest.raises(ValueError, match="must have 3 columns"):
            laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

    def test_constructor_mismatched_rows(self):
        """Test that constructor raises error for mismatched row counts."""
        origin = np.random.rand(5, 3)
        lateral_axis = np.random.rand(10, 3)  # Different number of rows
        vertical_axis = np.random.rand(5, 3)

        with pytest.raises(ValueError, match="same number of rows"):
            laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

    def test_orthonormalization(self):
        """Test that rotation matrix is orthonormal."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.1, 0.0]  # Not orthogonal
        vertical_axis = [0.0, 1.0, 0.1]

        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)
        R = ref_frame.rotation_matrix[0]

        # Check orthonormality: R^T @ R = I
        identity = R.T @ R
        assert np.allclose(identity, np.eye(3), atol=1e-10)

        # Check determinant = 1 (right-handed)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestReferenceFrameProperties:
    """Test ReferenceFrame property accessors."""

    def test_property_access(self):
        """Test that all properties are accessible."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]

        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        # Test property access
        assert np.allclose(ref_frame.origin, [[1.0, 2.0, 3.0]])
        assert np.allclose(ref_frame.lateral_axis, [[1.0, 0.0, 0.0]])
        assert np.allclose(ref_frame.vertical_axis, [[0.0, 1.0, 0.0]])
        assert ref_frame.rotation_matrix.shape == (1, 3, 3)


class TestReferenceFrameApplyNumpy:
    """Test ReferenceFrame.apply() with numpy arrays."""

    def test_apply_numpy_1d(self):
        """Test transformation of 1D numpy array."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.array([2.0, 3.0, 4.0])
        result = ref_frame.apply(data)

        # Should return 1D array (same dimensionality as input)
        assert result.ndim == 1
        assert result.shape == (3,)

    def test_apply_numpy_2d(self):
        """Test transformation of 2D numpy array."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(100, 3)
        result = ref_frame.apply(data)

        assert result.shape == (100, 3)

    def test_apply_numpy_identity_transformation(self):
        """Test that identity transformation preserves data."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        anteroposterior_axis = [0.0, 0.0, 1.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis, anteroposterior_axis)

        data = np.random.rand(50, 3)
        result = ref_frame.apply(data)

        assert np.allclose(result, data, atol=1e-10)

    def test_apply_numpy_translation_only(self):
        """Test pure translation (no rotation)."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        anteroposterior_axis = [0.0, 0.0, 1.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis, anteroposterior_axis)

        data = np.array([[2.0, 3.0, 4.0]])
        result = ref_frame.apply(data)

        # Result should be data - origin
        expected = data - np.array([[1.0, 2.0, 3.0]])
        assert np.allclose(result, expected, atol=1e-10)

    def test_apply_numpy_broadcasting(self):
        """Test broadcasting of single-frame to multiple samples."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        assert ref_frame._is_single_frame is True

        # Apply to 100 samples
        data = np.random.rand(100, 3)
        result = ref_frame.apply(data)

        assert result.shape == (100, 3)

    def test_apply_numpy_multiframe_matching(self):
        """Test multi-frame transformation with matching sample count."""
        n_samples = 50
        origin = np.random.rand(n_samples, 3)
        lateral_axis = np.random.rand(n_samples, 3)
        vertical_axis = np.random.rand(n_samples, 3)
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(n_samples, 3)
        result = ref_frame.apply(data)

        assert result.shape == (n_samples, 3)

    def test_apply_numpy_multiframe_mismatch_error(self):
        """Test that mismatched sample counts raise error."""
        origin = np.random.rand(50, 3)
        lateral_axis = np.random.rand(50, 3)
        vertical_axis = np.random.rand(50, 3)
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(100, 3)  # Different sample count

        with pytest.raises(ValueError, match="Shape mismatch"):
            ref_frame.apply(data)

    def test_apply_numpy_invalid_columns(self):
        """Test that wrong number of columns raises error."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(100, 2)  # Only 2 columns

        with pytest.raises(ValueError, match="must have 3 columns"):
            ref_frame.apply(data)


class TestReferenceFrameApplyDataFrame:
    """Test ReferenceFrame.apply() with pandas DataFrames."""

    def test_apply_dataframe(self):
        """Test transformation of DataFrame with 3 numeric columns."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        df = pd.DataFrame(
            {"X": [2.0, 3.0, 4.0], "Y": [3.0, 4.0, 5.0], "Z": [4.0, 5.0, 6.0]}
        )

        result = ref_frame.apply(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)

    def test_apply_dataframe_inplace(self):
        """Test in-place transformation of DataFrame."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        df = pd.DataFrame(
            {"X": [2.0, 3.0, 4.0], "Y": [3.0, 4.0, 5.0], "Z": [4.0, 5.0, 6.0]}
        )
        original_id = id(df)

        result = ref_frame.apply(df, inplace=True)

        assert result is None
        assert id(df) == original_id  # Same object

    def test_apply_dataframe_wrong_columns(self):
        """Test that DataFrame with wrong number of columns raises error."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})  # Only 2 columns

        with pytest.raises(ValueError, match="exactly 3 numeric columns"):
            ref_frame.apply(df)


class TestReferenceFrameApplyTimeseries:
    """Test ReferenceFrame.apply() with Timeseries objects."""

    def test_apply_signal3d(self):
        """Test transformation of Signal3D."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(100, 3)
        index = np.arange(100) / 100.0
        signal = laban.Signal3D(data, index, unit="m", columns=["X", "Y", "Z"])

        result = ref_frame.apply(signal)

        assert isinstance(result, laban.Signal3D)
        assert result.shape == signal.shape
        assert result.unit == signal.unit

    def test_apply_point3d(self):
        """Test transformation of Point3D."""
        origin = [0.5, 1.0, 0.2]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(100, 3)
        index = np.arange(100) / 100.0
        point = laban.Point3D(data, index, columns=["X", "Y", "Z"], unit="m")

        result = ref_frame.apply(point)

        assert isinstance(result, laban.Point3D)
        assert result.shape == point.shape
        assert result.unit == point.unit

    def test_apply_timeseries_inplace(self):
        """Test in-place transformation of Timeseries."""
        origin = [1.0, 2.0, 3.0]  # Non-zero origin to actually change data
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(50, 3)
        index = np.arange(50) / 100.0
        signal = laban.Signal3D(data, index, unit="m", columns=["X", "Y", "Z"])
        original_id = id(signal)
        original_data = signal._data.copy()

        result = ref_frame.apply(signal, inplace=True)

        assert result is None
        assert id(signal) == original_id
        assert not np.array_equal(signal._data, original_data)

    def test_apply_signal1d_error(self):
        """Test that applying to 1D signal raises error."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(100, 1)
        index = np.arange(100) / 100.0
        signal = laban.Signal1D(data, index, unit="V")

        with pytest.raises(ValueError, match="must have 3 columns"):
            ref_frame.apply(signal)


class TestReferenceFrameApplyForcePlatform:
    """Test ReferenceFrame.apply() with ForcePlatform objects."""

    def test_apply_forceplatform(self):
        """Test transformation of ForcePlatform."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        # Create ForcePlatform
        n_samples = 100
        index = np.arange(n_samples) / 100.0
        origin_data = np.random.rand(n_samples, 3)
        force_data = np.random.rand(n_samples, 3)
        torque_data = np.random.rand(n_samples, 3)

        fp_origin = laban.Point3D(origin_data, index, columns=["X", "Y", "Z"], unit="m")
        fp_force = laban.Signal3D(force_data, index, columns=["X", "Y", "Z"], unit="N")
        fp_torque = laban.Signal3D(
            torque_data, index, columns=["X", "Y", "Z"], unit="Nm"
        )

        fp = laban.ForcePlatform(origin=fp_origin, force=fp_force, torque=fp_torque)

        result = ref_frame.apply(fp)

        assert isinstance(result, laban.ForcePlatform)
        assert result.origin.shape == fp.origin.shape
        assert result.force.shape == fp.force.shape
        assert result.torque.shape == fp.torque.shape

    def test_apply_forceplatform_inplace(self):
        """Test in-place transformation of ForcePlatform."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        # Create ForcePlatform
        n_samples = 50
        index = np.arange(n_samples) / 100.0
        origin_data = np.random.rand(n_samples, 3)
        force_data = np.random.rand(n_samples, 3)
        torque_data = np.random.rand(n_samples, 3)

        fp_origin = laban.Point3D(origin_data, index, columns=["X", "Y", "Z"], unit="m")
        fp_force = laban.Signal3D(force_data, index, columns=["X", "Y", "Z"], unit="N")
        fp_torque = laban.Signal3D(
            torque_data, index, columns=["X", "Y", "Z"], unit="Nm"
        )

        fp = laban.ForcePlatform(origin=fp_origin, force=fp_force, torque=fp_torque)
        original_id = id(fp)

        result = ref_frame.apply(fp, inplace=True)

        assert result is None
        assert id(fp) == original_id


class TestReferenceFrameCallable:
    """Test ReferenceFrame callable interface."""

    def test_callable_interface(self):
        """Test that ReferenceFrame can be called directly."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(50, 3)

        # Call via __call__
        result_call = ref_frame(data)

        # Call via apply
        result_apply = ref_frame.apply(data)

        assert np.allclose(result_call, result_apply)


class TestReferenceFrameWithTimeseriesApply:
    """Test using ReferenceFrame via Timeseries.apply() method."""

    def test_timeseries_apply_referenceframe(self):
        """Test that ts.apply(rf) works correctly."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(50, 3)
        index = np.arange(50) / 100.0
        ts = laban.Signal3D(data, index, unit="m", columns=["X", "Y", "Z"])

        # Apply via ts.apply(rf) instead of rf.apply(ts)
        result = ts.apply(rf)

        assert isinstance(result, laban.Signal3D)
        assert result.shape == ts.shape
        assert result.unit == ts.unit
        assert not np.array_equal(result._data, ts._data)

    def test_timeseries_apply_referenceframe_inplace(self):
        """Test that ts.apply(rf, inplace=True) works correctly."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(50, 3)
        index = np.arange(50) / 100.0
        ts = laban.Signal3D(data, index, unit="m", columns=["X", "Y", "Z"])
        original_data = ts._data.copy()
        original_id = id(ts)

        # Apply inplace via ts.apply(rf, inplace=True)
        result = ts.apply(rf, inplace=True)

        assert result is None
        assert id(ts) == original_id
        assert not np.array_equal(ts._data, original_data)

    def test_point3d_apply_referenceframe(self):
        """Test that Point3D.apply(rf) works correctly."""
        origin = [0.5, 1.0, 0.2]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(100, 3)
        index = np.arange(100) / 100.0
        point = laban.Point3D(data, index, columns=["X", "Y", "Z"], unit="m")

        # Apply via point.apply(rf)
        result = point.apply(rf)

        assert isinstance(result, laban.Point3D)
        assert result.shape == point.shape
        assert result.unit == point.unit

    def test_timeseries_apply_with_broadcasting(self):
        """Test broadcasting when using ts.apply(rf) with single-frame reference."""
        # Single-frame reference
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        assert rf._is_single_frame is True

        # Multi-sample timeseries
        data = np.random.rand(100, 3)
        index = np.arange(100) / 100.0
        ts = laban.Signal3D(data, index, unit="m", columns=["X", "Y", "Z"])

        # Broadcasting should work
        result = ts.apply(rf)

        assert result.shape == ts.shape


class TestReferenceFrameApplyInverse:
    """Test ReferenceFrame.apply_inverse() method."""

    def test_apply_inverse_numpy_roundtrip(self):
        """Test that apply_inverse reverses apply for numpy arrays."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.1, 0.0]
        vertical_axis = [0.0, 1.0, 0.1]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(50, 3)
        transformed = rf.apply(data)
        recovered = rf.apply_inverse(transformed)

        assert np.allclose(recovered, data, atol=1e-10)

    def test_apply_inverse_numpy_1d(self):
        """Test apply_inverse with 1D numpy array."""
        origin = [0.5, 1.0, 0.2]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.array([2.0, 3.0, 4.0])
        transformed = rf.apply(data)
        recovered = rf.apply_inverse(transformed)

        assert np.allclose(recovered, data, atol=1e-10)

    def test_apply_inverse_signal3d_roundtrip(self):
        """Test that apply_inverse reverses apply for Signal3D."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(100, 3)
        index = np.arange(100) / 100.0
        signal = laban.Signal3D(data, index, unit="m", columns=["X", "Y", "Z"])

        transformed = rf.apply(signal)
        recovered = rf.apply_inverse(transformed)

        assert np.allclose(recovered._data, signal._data, atol=1e-10)
        assert recovered.unit == signal.unit

    def test_apply_inverse_point3d_roundtrip(self):
        """Test that apply_inverse reverses apply for Point3D."""
        origin = [0.5, 1.0, 0.2]
        lateral_axis = [1.0, 0.1, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(100, 3)
        index = np.arange(100) / 100.0
        point = laban.Point3D(data, index, columns=["X", "Y", "Z"], unit="m")

        transformed = rf.apply(point)
        recovered = rf.apply_inverse(transformed)

        assert np.allclose(recovered._data, point._data, atol=1e-10)

    def test_apply_inverse_dataframe_roundtrip(self):
        """Test that apply_inverse reverses apply for DataFrame."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        df = pd.DataFrame({
            "X": [1.0, 2.0, 3.0],
            "Y": [2.0, 3.0, 4.0],
            "Z": [3.0, 4.0, 5.0]
        })

        transformed = rf.apply(df)
        recovered = rf.apply_inverse(transformed)

        assert np.allclose(recovered[["X", "Y", "Z"]].values, df[["X", "Y", "Z"]].values, atol=1e-10)

    def test_apply_inverse_forceplatform_roundtrip(self):
        """Test that apply_inverse reverses apply for ForcePlatform."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        n_samples = 100
        index = np.arange(n_samples) / 100.0
        cop_data = np.random.rand(n_samples, 3)
        force_data = np.random.rand(n_samples, 3) * 100
        torque_data = np.random.rand(n_samples, 3) * 10

        cop = laban.Point3D(cop_data, index, columns=["X", "Y", "Z"], unit="m")
        force = laban.Signal3D(force_data, index, columns=["X", "Y", "Z"], unit="N")
        torque = laban.Signal3D(torque_data, index, columns=["X", "Y", "Z"], unit="Nm")

        fp = laban.ForcePlatform(origin=cop, force=force, torque=torque)

        transformed = rf.apply(fp)
        recovered = rf.apply_inverse(transformed)

        # Check origin, force (torque may differ due to free moment calculation)
        assert np.allclose(recovered.origin._data, fp.origin._data, atol=1e-10)
        assert np.allclose(recovered.force._data, fp.force._data, atol=1e-10)

    def test_apply_inverse_broadcasting(self):
        """Test apply_inverse with broadcasting."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        assert rf._is_single_frame is True

        data = np.random.rand(100, 3)
        transformed = rf.apply(data)
        recovered = rf.apply_inverse(transformed)

        assert np.allclose(recovered, data, atol=1e-10)

    def test_apply_inverse_inplace(self):
        """Test apply_inverse with inplace=True."""
        origin = [1.0, 2.0, 3.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        rf = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.random.rand(50, 3)
        index = np.arange(50) / 100.0
        signal = laban.Signal3D(data.copy(), index, unit="m", columns=["X", "Y", "Z"])
        original_data = signal._data.copy()

        # Apply transformation
        transformed = rf.apply(signal)

        # Apply inverse inplace
        result = rf.apply_inverse(transformed, inplace=True)

        assert result is None
        assert np.allclose(transformed._data, original_data, atol=1e-10)


class TestReferenceFrameEdgeCases:
    """Test edge cases and special scenarios."""

    def test_nan_propagation(self):
        """Test that NaN values are preserved in transformation."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0], [7.0, np.nan, 9.0]])

        result = ref_frame.apply(data)

        # Check that NaN positions are preserved
        assert np.isnan(result[1, 0])
        assert np.isnan(result[2, 1])

    def test_empty_array(self):
        """Test transformation of empty array."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        data = np.empty((0, 3))
        result = ref_frame.apply(data)

        assert result.shape == (0, 3)

    def test_unsupported_type(self):
        """Test that unsupported types raise TypeError."""
        origin = [0.0, 0.0, 0.0]
        lateral_axis = [1.0, 0.0, 0.0]
        vertical_axis = [0.0, 1.0, 0.0]
        ref_frame = laban.ReferenceFrame(origin, lateral_axis, vertical_axis)

        with pytest.raises(TypeError, match="Unsupported type"):
            ref_frame.apply("invalid_type")


class TestTimeseriesInputs:
    """Test ReferenceFrame accepts Point3D, Signal3D, and Timeseries."""

    def test_point3d_origin(self):
        """Test Point3D can be used as origin."""
        # Create Point3D origin
        origin_data = np.array([[1.0, 2.0, 3.0]])
        origin_pt = laban.Point3D(
            data=origin_data,
            index=[0.0],
            columns=["X", "Y", "Z"]
        )

        # Create reference frame with Point3D origin
        rf = laban.ReferenceFrame(
            origin=origin_pt,
            lateral_axis=[1.0, 0.0, 0.0],
            vertical_axis=[0.0, 1.0, 0.0]
        )

        # Verify origin extracted correctly
        np.testing.assert_array_almost_equal(rf.origin[0], [1.0, 2.0, 3.0])

    def test_signal3d_axes(self):
        """Test Signal3D can be used for axes."""
        # Create Signal3D axes (2 samples)
        lateral_data = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        lateral_sig = laban.Signal3D(
            data=lateral_data,
            index=[0.0, 1.0],
            unit="m/s",
            columns=["X", "Y", "Z"]
        )

        vertical_data = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        vertical_sig = laban.Signal3D(
            data=vertical_data,
            index=[0.0, 1.0],
            unit="m/s",
            columns=["X", "Y", "Z"]
        )

        # Origin must also have 2 samples to match
        origin_data = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        # Create reference frame with Signal3D axes
        rf = laban.ReferenceFrame(
            origin=origin_data,
            lateral_axis=lateral_sig,
            vertical_axis=vertical_sig
        )

        # Verify axes extracted correctly
        assert rf._n_samples == 2
        np.testing.assert_array_almost_equal(rf.lateral_axis[0], [1.0, 0.0, 0.0])

    def test_mixed_input_types(self):
        """Test mixing Point3D, arrays, and lists."""
        # Mix input types
        origin_pt = laban.Point3D(
            data=np.array([[0.5, 1.0, 0.2]]),
            index=[0.0],
            columns=["X", "Y", "Z"]
        )

        lateral_array = np.array([[1.0, 0.0, 0.0]])
        vertical_list = [[0.0, 1.0, 0.0]]

        # Should work with mixed types
        rf = laban.ReferenceFrame(
            origin=origin_pt,
            lateral_axis=lateral_array,
            vertical_axis=vertical_list
        )

        assert rf._n_samples == 1

    def test_point3d_arithmetic_result(self):
        """Test Point3D from arithmetic (e.g., hip - knee) works."""
        hip = laban.Point3D(
            data=np.array([[0.0, 1.0, 0.0]]),
            index=[0.0],
            columns=["X", "Y", "Z"]
        )
        knee = laban.Point3D(
            data=np.array([[0.0, 0.5, 0.0]]),
            index=[0.0],
            columns=["X", "Y", "Z"]
        )

        # Arithmetic produces Point3D
        vertical_vec = hip - knee

        # Should accept Point3D result
        rf = laban.ReferenceFrame(
            origin=knee,
            lateral_axis=[1.0, 0.0, 0.0],
            vertical_axis=vertical_vec
        )

        # Verify extraction
        np.testing.assert_array_almost_equal(rf.origin[0], [0.0, 0.5, 0.0])

    def test_incompatible_type_raises_error(self):
        """Test that incompatible types raise TypeError."""
        # Try passing invalid type
        with pytest.raises(TypeError, match="must be np.ndarray, list, tuple, or Timeseries"):
            laban.ReferenceFrame(
                origin="invalid",  # String not supported
                lateral_axis=[1.0, 0.0, 0.0],
                vertical_axis=[0.0, 1.0, 0.0]
            )

    def test_timeseries_wrong_columns_raises_error(self):
        """Test that Timeseries with wrong column count raises error."""
        # Create Signal1D (1 column, not 3)
        signal_1d = laban.Signal1D(
            data=np.array([[1.0], [2.0]]),
            index=[0.0, 1.0],
            unit="m"
        )

        # Should raise ValueError during shape validation
        with pytest.raises(ValueError, match="must have 3 columns"):
            laban.ReferenceFrame(
                origin=signal_1d,  # Wrong shape
                lateral_axis=[1.0, 0.0, 0.0],
                vertical_axis=[0.0, 1.0, 0.0]
            )

    def test_multi_sample_point3d(self):
        """Test multi-sample Point3D works correctly."""
        n_samples = 50
        origin_data = np.random.rand(n_samples, 3)
        origin_pt = laban.Point3D(
            data=origin_data,
            index=list(range(n_samples)),
            columns=["X", "Y", "Z"]
        )

        lateral_data = np.random.rand(n_samples, 3)
        lateral_pt = laban.Point3D(
            data=lateral_data,
            index=list(range(n_samples)),
            columns=["X", "Y", "Z"]
        )

        vertical_data = np.random.rand(n_samples, 3)
        vertical_pt = laban.Point3D(
            data=vertical_data,
            index=list(range(n_samples)),
            columns=["X", "Y", "Z"]
        )

        # Create reference frame with multi-sample Point3D
        rf = laban.ReferenceFrame(
            origin=origin_pt,
            lateral_axis=lateral_pt,
            vertical_axis=vertical_pt
        )

        assert rf._n_samples == n_samples
        np.testing.assert_array_almost_equal(rf.origin, origin_data)
