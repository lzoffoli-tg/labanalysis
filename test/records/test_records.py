"""Test Record and Timeseries attribute/item interchangeable access"""

import numpy as np
import pytest

import labanalysis as laban


def create_mock_point3d(n_samples=100, offset=0.0):
    """Create a mock Point3D with random data."""
    data = np.random.randn(n_samples, 3) + offset
    index = np.arange(n_samples) / 100.0  # 100 Hz
    return laban.Point3D(
        data=data,
        index=index,
        columns=["X", "Y", "Z"],
        unit="mm"
    )


class TestRecordAttributeItemAccess:
    """Test interchangeable access between attributes and items for Record classes."""

    def test_record_property_access_as_item(self):
        """Test that Record properties can be accessed using item notation."""
        # Create a WholeBody with ankle markers
        left_ankle_medial = create_mock_point3d(n_samples=10, offset=0.0)
        left_ankle_lateral = create_mock_point3d(n_samples=10, offset=0.1)

        wb = laban.WholeBody(
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral
        )

        # left_ankle is a property that calculates the midpoint
        # Test that both access methods work
        result_attr = wb.left_ankle
        result_item = wb['left_ankle']

        assert isinstance(result_attr, laban.Point3D)
        assert isinstance(result_item, laban.Point3D)
        assert np.allclose(result_attr.to_numpy(), result_item.to_numpy())

    def test_record_item_access_as_attribute(self):
        """Test that Record items can be accessed using attribute notation."""
        left_ankle_lateral = create_mock_point3d(n_samples=10, offset=0.0)

        wb = laban.WholeBody(left_ankle_lateral=left_ankle_lateral)

        # left_ankle_lateral is an item in _data
        # Test that both access methods work
        result_item = wb['left_ankle_lateral']
        result_attr = wb.left_ankle_lateral

        assert isinstance(result_attr, laban.Point3D)
        assert isinstance(result_item, laban.Point3D)
        assert np.allclose(result_attr.to_numpy(), result_item.to_numpy())

    def test_record_item_priority_over_property(self):
        """Test that items in _data have priority over properties with same name."""
        left_ankle_medial = create_mock_point3d(n_samples=10, offset=0.0)
        left_ankle_lateral = create_mock_point3d(n_samples=10, offset=0.1)

        wb = laban.WholeBody(
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral
        )

        # When accessing 'left_ankle_lateral' which exists in _data,
        # it should return the item, not compute a property
        result = wb['left_ankle_lateral']
        assert isinstance(result, laban.Point3D)
        assert np.allclose(result.to_numpy(), left_ankle_lateral.to_numpy())

    def test_record_nonexistent_key_raises_error(self):
        """Test that accessing non-existent key raises KeyError."""
        wb = laban.WholeBody()

        with pytest.raises(KeyError, match="'nonexistent'"):
            _ = wb['nonexistent']

    def test_timeseriesrecord_with_forceplatform(self):
        """Test attribute/item access with nested Record types like ForcePlatform."""
        n_samples = 10

        # Create ForcePlatform components
        origin = create_mock_point3d(n_samples, offset=0.0)
        force_data = np.random.randn(n_samples, 3)
        force = laban.Signal3D(
            data=force_data,
            index=origin.index,
            unit="N",
            columns=["X", "Y", "Z"]
        )
        torque_data = np.random.randn(n_samples, 3)
        torque = laban.Signal3D(
            data=torque_data,
            index=origin.index,
            unit="Nm",
            columns=["X", "Y", "Z"]
        )

        fp = laban.ForcePlatform(origin=origin, force=force, torque=torque)

        # Create TimeseriesRecord
        rec = laban.TimeseriesRecord(fp1=fp)

        # Test both access methods
        result_attr = rec.fp1
        result_item = rec['fp1']

        assert isinstance(result_attr, laban.ForcePlatform)
        assert isinstance(result_item, laban.ForcePlatform)


class TestTimeseriesAttributeItemAccess:
    """Test interchangeable access between attributes and items for Timeseries classes."""

    def test_timeseries_column_access_as_item(self):
        """Test that Timeseries columns can be accessed using item notation."""
        point = create_mock_point3d(n_samples=10)

        # 'X' is a column name
        # Test that both access methods work
        result_attr = point.X
        result_item = point['X']

        assert isinstance(result_attr, laban.Timeseries)
        assert isinstance(result_item, laban.Timeseries)
        assert np.allclose(result_attr.to_numpy(), result_item.to_numpy())

    def test_timeseries_property_access_as_item(self):
        """Test that Timeseries properties can be accessed using item notation."""
        point = create_mock_point3d(n_samples=10)

        # 'module' is a property that calculates the magnitude
        # Test that both access methods work
        result_attr = point.module
        result_item = point['module']

        assert isinstance(result_attr, laban.Signal1D)
        assert isinstance(result_item, laban.Signal1D)
        assert np.allclose(result_attr.to_numpy(), result_item.to_numpy())

    def test_timeseries_column_priority_over_property(self):
        """Test that columns have priority over properties with same name."""
        data = np.random.randn(10, 3)
        index = np.arange(10) / 100.0

        # Create a Point3D with standard columns
        point = laban.Point3D(
            data=data,
            index=index,
            columns=["X", "Y", "Z"],
            unit="mm"
        )

        # Accessing 'X' should return the column, not any property
        result = point['X']
        assert isinstance(result, laban.Timeseries)
        assert result.shape == (10, 1)

    def test_timeseries_nonexistent_key_raises_error(self):
        """Test that accessing non-existent key raises KeyError."""
        point = create_mock_point3d(n_samples=10)

        with pytest.raises(KeyError, match="'nonexistent'"):
            _ = point['nonexistent']

    def test_signal1d_attribute_item_access(self):
        """Test attribute/item access for Signal1D."""
        data = np.random.randn(10)
        index = np.arange(10) / 100.0

        signal = laban.Signal1D(data=data, index=index, unit="m/s")

        # 'amplitude' is the default column name for Signal1D
        result_attr = signal.amplitude
        result_item = signal['amplitude']

        assert isinstance(result_attr, laban.Timeseries)
        assert isinstance(result_item, laban.Timeseries)
        assert np.allclose(result_attr.to_numpy(), result_item.to_numpy())

    def test_signal3d_axis_properties_as_items(self):
        """Test that Signal3D axis properties can be accessed as items."""
        data = np.random.randn(10, 3)
        index = np.arange(10) / 100.0

        signal = laban.Signal3D(
            data=data,
            index=index,
            unit="N",
            columns=["X", "Y", "Z"],
            vertical_axis="Y",
            anteroposterior_axis="Z"
        )

        # Test property access
        assert signal.vertical_axis == "Y"
        assert signal['vertical_axis'] == "Y"

        assert signal.anteroposterior_axis == "Z"
        assert signal['anteroposterior_axis'] == "Z"

        assert signal.lateral_axis == "X"
        assert signal['lateral_axis'] == "X"

    def test_emgsignal_attribute_item_access(self):
        """Test attribute/item access for EMGSignal."""
        data = np.random.randn(10)
        index = np.arange(10) / 100.0

        emg = laban.EMGSignal(
            data=data,
            index=index,
            muscle_name="vastus_lateralis",
            side="left",
            unit="uV"
        )

        # Test muscle_name and side properties
        assert emg.muscle_name == "vastus_lateralis"
        assert emg['muscle_name'] == "vastus_lateralis"

        assert emg.side == "left"
        assert emg['side'] == "left"


class TestComplexScenarios:
    """Test complex scenarios with nested access patterns."""

    def test_wholebody_nested_property_access(self):
        """Test accessing nested properties through item notation."""
        # Create a full ankle with medial and lateral markers
        left_ankle_medial = create_mock_point3d(n_samples=10, offset=0.0)
        left_ankle_lateral = create_mock_point3d(n_samples=10, offset=0.1)

        # Create knee markers for leg properties
        left_knee_medial = create_mock_point3d(n_samples=10, offset=2.0)
        left_knee_lateral = create_mock_point3d(n_samples=10, offset=2.1)

        wb = laban.WholeBody(
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral,
            left_knee_medial=left_knee_medial,
            left_knee_lateral=left_knee_lateral
        )

        # Test that we can access computed joint centers
        left_ankle_attr = wb.left_ankle
        left_ankle_item = wb['left_ankle']

        assert isinstance(left_ankle_attr, laban.Point3D)
        assert isinstance(left_ankle_item, laban.Point3D)
        assert np.allclose(left_ankle_attr.to_numpy(), left_ankle_item.to_numpy())

        # Test knee access
        left_knee_attr = wb.left_knee
        left_knee_item = wb['left_knee']

        assert isinstance(left_knee_attr, laban.Point3D)
        assert isinstance(left_knee_item, laban.Point3D)
        assert np.allclose(left_knee_attr.to_numpy(), left_knee_item.to_numpy())

    def test_record_mixed_access_patterns(self):
        """Test mixing attribute and item access in the same operation."""
        left_ankle_medial = create_mock_point3d(n_samples=10, offset=0.0)
        left_ankle_lateral = create_mock_point3d(n_samples=10, offset=0.1)

        wb = laban.WholeBody(
            left_ankle_medial=left_ankle_medial,
            left_ankle_lateral=left_ankle_lateral
        )

        # Mix attribute and item access
        ankle_via_attr = wb.left_ankle
        lateral_via_item = wb['left_ankle_lateral']
        medial_via_attr = wb.left_ankle_medial

        # All should be valid Point3D objects
        assert isinstance(ankle_via_attr, laban.Point3D)
        assert isinstance(lateral_via_item, laban.Point3D)
        assert isinstance(medial_via_attr, laban.Point3D)


class TestRecordStrip:
    """Test strip method with independent parameter for Record and subclasses."""

    def test_record_strip_independent_true_preserves_original_behavior(self):
        """Test that independent=True maintains original independent stripping."""
        # Element A: NaN at start and end
        data_a = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
        sig_a = laban.Signal1D(data_a, index=[0, 1, 2, 3, 4], unit="m")

        # Element B: NaN at different positions (note: strip keeps rows from first to last non-NaN)
        data_b = np.array([1.0, np.nan, 2.0, 3.0, np.nan])
        sig_b = laban.Signal1D(data_b, index=[0, 1, 2, 3, 4], unit="m")

        rec = laban.Record(signal_a=sig_a, signal_b=sig_b)
        result = rec.strip(independent=True, inplace=False)

        # signal_a should strip to [1, 2, 3] (from first to last non-NaN)
        assert np.allclose(result['signal_a'].index, [1, 2, 3])
        # signal_b should strip to [0, 1, 2, 3] (from first to last non-NaN, includes intermediate NaN)
        assert np.allclose(result['signal_b'].index, [0, 1, 2, 3])

    def test_record_strip_independent_false_shared_timeframe(self):
        """Test that independent=False creates shared timeframe."""
        # Element A: has data from index 1 to 3
        data_a = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
        sig_a = laban.Signal1D(data_a, index=[0, 1, 2, 3, 4], unit="m")

        # Element B: has data from index 0 to 4
        data_b = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        sig_b = laban.Signal1D(data_b, index=[0, 1, 2, 3, 4], unit="m")

        rec = laban.Record(signal_a=sig_a, signal_b=sig_b)
        result = rec.strip(independent=False, inplace=False)

        # Both should span from index 0 (first non-NaN in B) to 4 (last non-NaN in B)
        assert np.allclose(result['signal_a'].index, [0, 1, 2, 3, 4])
        assert np.allclose(result['signal_b'].index, [0, 1, 2, 3, 4])

    def test_record_strip_default_uses_shared_timeframe(self):
        """Test that default behavior (no independent specified) uses shared timeframe."""
        data_a = np.array([np.nan, 1.0, 2.0, np.nan])
        sig_a = laban.Signal1D(data_a, index=[0, 1, 2, 3], unit="m")

        data_b = np.array([1.0, np.nan, np.nan, 2.0])
        sig_b = laban.Signal1D(data_b, index=[0, 1, 2, 3], unit="m")

        rec = laban.Record(signal_a=sig_a, signal_b=sig_b)
        result = rec.strip(inplace=False)

        # Should use shared timeframe [0, 3]
        assert np.allclose(result['signal_a'].index, [0, 1, 2, 3])
        assert np.allclose(result['signal_b'].index, [0, 1, 2, 3])

    def test_record_strip_inplace_true_modifies_original(self):
        """Test that inplace=True modifies the original Record."""
        data_a = np.array([np.nan, 1.0, 2.0, np.nan])
        sig_a = laban.Signal1D(data_a, index=[0, 1, 2, 3], unit="m")

        rec = laban.Record(signal_a=sig_a)
        original_id = id(rec)

        result = rec.strip(independent=False, inplace=True)

        assert result is None
        assert id(rec) == original_id
        assert np.allclose(rec['signal_a'].index, [1, 2])

    def test_record_strip_inplace_false_preserves_original(self):
        """Test that inplace=False preserves the original Record."""
        data_a = np.array([np.nan, 1.0, 2.0, np.nan])
        sig_a = laban.Signal1D(data_a, index=[0, 1, 2, 3], unit="m")

        rec = laban.Record(signal_a=sig_a)
        original_index = rec['signal_a'].index.copy()

        result = rec.strip(independent=False, inplace=False)

        assert result is not rec
        assert np.allclose(rec['signal_a'].index, original_index)
        assert np.allclose(result['signal_a'].index, [1, 2])

    def test_record_strip_empty_record(self):
        """Test stripping an empty Record."""
        rec = laban.Record()
        result = rec.strip(independent=False, inplace=False)

        assert len(result._data) == 0

    def test_record_strip_all_nan_elements(self):
        """Test stripping when all elements are all NaN."""
        data_a = np.array([np.nan, np.nan, np.nan])
        sig_a = laban.Signal1D(data_a, index=[0, 1, 2], unit="m")

        data_b = np.array([np.nan, np.nan, np.nan])
        sig_b = laban.Signal1D(data_b, index=[0, 1, 2], unit="m")

        rec = laban.Record(signal_a=sig_a, signal_b=sig_b)
        result = rec.strip(independent=False, inplace=False)

        # Should remain unchanged
        assert np.allclose(result['signal_a'].index, [0, 1, 2])
        assert np.allclose(result['signal_b'].index, [0, 1, 2])

    def test_record_strip_single_element(self):
        """Test that single-element Record behaves identically for both modes."""
        data = np.array([np.nan, 1.0, 2.0, np.nan])
        sig = laban.Signal1D(data, index=[0, 1, 2, 3], unit="m")

        rec_ind = laban.Record(signal=sig)
        rec_shared = laban.Record(signal=sig)

        result_ind = rec_ind.strip(independent=True, inplace=False)
        result_shared = rec_shared.strip(independent=False, inplace=False)

        assert np.allclose(result_ind['signal'].index, result_shared['signal'].index)

    def test_record_strip_axis_0_only(self):
        """Test stripping only rows (axis=0) with shared timeframe."""
        # Create 3D signal with some NaN columns
        data = np.array([
            [np.nan, 1.0, np.nan],
            [1.0, 2.0, np.nan],
            [2.0, 3.0, np.nan],
            [np.nan, 4.0, np.nan]
        ])
        sig = laban.Signal3D(data, index=[0, 1, 2, 3], unit="m", columns=["X", "Y", "Z"])
        rec = laban.Record(signal=sig)

        result = rec.strip(axis=0, independent=False, inplace=False)

        # Should strip rows from first to last non-all-NaN row, keeping all columns
        # Rows 0-3 all have at least one non-NaN value
        assert np.allclose(result['signal'].index, [0, 1, 2, 3])
        assert result['signal'].shape[1] == 3  # All columns preserved

    def test_record_strip_axis_1_always_independent(self):
        """Test that axis=1 stripping is always independent regardless of parameter."""
        # Column Z is all NaN at the end
        data = np.array([
            [1.0, 2.0, np.nan],
            [3.0, 4.0, np.nan],
            [5.0, 6.0, np.nan]
        ])
        sig = laban.Signal3D(data, index=[0, 1, 2], unit="m", columns=["X", "Y", "Z"])
        rec = laban.Record(signal=sig)

        result_shared = rec.strip(axis=1, independent=False, inplace=False)
        result_ind = rec.strip(axis=1, independent=True, inplace=False)

        # Both should strip to same columns (X and Y only, Z is all NaN)
        assert result_shared['signal'].shape[1] == 2
        assert result_ind['signal'].shape[1] == 2
        assert list(result_shared['signal'].columns) == ["X", "Y"]
        assert list(result_shared['signal'].columns) == list(result_ind['signal'].columns)

    def test_record_strip_axis_none_shared_time_independent_columns(self):
        """Test axis=None: shared time stripping, independent column stripping."""
        # Signal A: has data in rows [1,2] and columns [X,Y]
        data_a = np.array([
            [np.nan, np.nan, np.nan],
            [1.0, 2.0, np.nan],
            [3.0, 4.0, np.nan],
            [np.nan, np.nan, np.nan]
        ])
        sig_a = laban.Signal3D(data_a, index=[0, 1, 2, 3], unit="m", columns=["X", "Y", "Z"])

        # Signal B: has data in rows [0,3] and columns [Y,Z]
        data_b = np.array([
            [np.nan, 1.0, 2.0],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, 3.0, 4.0]
        ])
        sig_b = laban.Signal3D(data_b, index=[0, 1, 2, 3], unit="m", columns=["X", "Y", "Z"])

        rec = laban.Record(signal_a=sig_a, signal_b=sig_b)
        result = rec.strip(axis=None, independent=False, inplace=False)

        # Shared time: should span [0, 3]
        assert np.allclose(result['signal_a'].index, [0, 1, 2, 3])
        assert np.allclose(result['signal_b'].index, [0, 1, 2, 3])

        # Independent columns: each keeps its own non-empty columns
        assert "X" in result['signal_a'].columns
        assert "Y" in result['signal_a'].columns
        assert "Y" in result['signal_b'].columns
        assert "Z" in result['signal_b'].columns

    def test_record_strip_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        data = np.array([1.0, 2.0, 3.0])
        sig = laban.Signal1D(data, index=[0, 1, 2], unit="m")
        rec = laban.Record(signal=sig)

        with pytest.raises(ValueError, match="independent must be True or False"):
            rec.strip(independent="yes")

        with pytest.raises(ValueError, match="independent must be True or False"):
            rec.strip(independent=1)

        with pytest.raises(ValueError, match="inplace must be True or False"):
            rec.strip(inplace="yes")

        with pytest.raises(ValueError, match="axis must be None or 0 or 1"):
            rec.strip(axis=2)

    def test_timeseriesrecord_strip_shared_timeframe(self):
        """Test that TimeseriesRecord inherits shared timeframe stripping."""
        # Create Point3D with NaN at edges
        data_point = np.array([
            [np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [np.nan, np.nan, np.nan]
        ])
        point = laban.Point3D(data_point, index=[0, 1, 2, 3], unit="mm", columns=["X", "Y", "Z"])

        # Create Signal3D with data at different positions
        data_signal = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [4.0, 5.0, 6.0]
        ])
        signal = laban.Signal3D(data_signal, index=[0, 1, 2, 3], unit="N", columns=["X", "Y", "Z"])

        rec = laban.TimeseriesRecord(marker=point, force=signal)
        result = rec.strip(independent=False, inplace=False)

        # Should span from 0 (first non-NaN in signal) to 3 (last non-NaN in signal)
        assert np.allclose(result['marker'].index, [0, 1, 2, 3])
        assert np.allclose(result['force'].index, [0, 1, 2, 3])

    def test_timeseriesrecord_strip_independent_mode(self):
        """Test that TimeseriesRecord can use independent mode."""
        data_point = np.array([
            [np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan]
        ])
        point = laban.Point3D(data_point, index=[0, 1, 2], unit="mm", columns=["X", "Y", "Z"])

        data_signal = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],
            [4.0, 5.0, 6.0]
        ])
        signal = laban.Signal3D(data_signal, index=[0, 1, 2], unit="N", columns=["X", "Y", "Z"])

        rec = laban.TimeseriesRecord(marker=point, force=signal)
        result = rec.strip(independent=True, inplace=False)

        # Each should have its own stripped timeframe (from first to last non-all-NaN)
        assert np.allclose(result['marker'].index, [1])
        # force has non-NaN at indices 0 and 2, so keeps [0, 1, 2]
        assert np.allclose(result['force'].index, [0, 1, 2])

    def test_forceplatform_strip_shared_timeframe(self):
        """Test that ForcePlatform inherits shared timeframe stripping."""
        n_samples = 10

        # Create elements where shared non-NaN range is [2, 7]
        # Origin: NaN at rows 0-1, data at 2-9
        origin_data = np.ones((n_samples, 3))
        origin_data[0:2, :] = np.nan
        origin = laban.Point3D(
            data=origin_data,
            index=np.arange(n_samples) / 100.0,
            unit="mm",
            columns=["X", "Y", "Z"]
        )

        # Force: data at 0-7, NaN at 8-9
        force_data = np.ones((n_samples, 3))
        force_data[8:10, :] = np.nan
        force = laban.Signal3D(
            data=force_data,
            index=np.arange(n_samples) / 100.0,
            unit="N",
            columns=["X", "Y", "Z"]
        )

        # Torque: NaN at 0-1, data at 2-7, NaN at 8-9
        torque_data = np.ones((n_samples, 3))
        torque_data[0:2, :] = np.nan
        torque_data[8:10, :] = np.nan
        torque = laban.Signal3D(
            data=torque_data,
            index=np.arange(n_samples) / 100.0,
            unit="Nm",
            columns=["X", "Y", "Z"]
        )

        fp = laban.ForcePlatform(origin=origin, force=force, torque=torque)
        result = fp.strip(independent=False, inplace=False)

        # Shared timeframe: from row 0 (force has data) to row 9 (origin has data)
        # Actually: at least one element has non-NaN at each of rows 0-9
        # Row 0: origin NaN, force OK, torque NaN → keep
        # Row 1: origin NaN, force OK, torque NaN → keep
        # Rows 2-7: all have data → keep
        # Row 8: origin OK, force NaN, torque NaN → keep
        # Row 9: origin OK, force NaN, torque NaN → keep
        expected_start = 0 / 100.0
        expected_stop = 9 / 100.0

        assert np.isclose(result['origin'].index[0], expected_start)
        assert np.isclose(result['origin'].index[-1], expected_stop)
        assert np.isclose(result['force'].index[0], expected_start)
        assert np.isclose(result['force'].index[-1], expected_stop)
        assert np.isclose(result['torque'].index[0], expected_start)
        assert np.isclose(result['torque'].index[-1], expected_stop)

    def test_forceplatform_strip_independent_mode(self):
        """Test that ForcePlatform can use independent mode."""
        n_samples = 5

        origin_data = np.array([
            [np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [np.nan, np.nan, np.nan]
        ])
        origin = laban.Point3D(
            data=origin_data,
            index=[0, 1, 2, 3, 4],
            unit="mm",
            columns=["X", "Y", "Z"]
        )

        force_data = np.random.randn(n_samples, 3)
        force = laban.Signal3D(
            data=force_data,
            index=[0, 1, 2, 3, 4],
            unit="N",
            columns=["X", "Y", "Z"]
        )

        torque_data = np.random.randn(n_samples, 3)
        torque = laban.Signal3D(
            data=torque_data,
            index=[0, 1, 2, 3, 4],
            unit="Nm",
            columns=["X", "Y", "Z"]
        )

        fp = laban.ForcePlatform(origin=origin, force=force, torque=torque)
        result = fp.strip(independent=True, inplace=False)

        # Origin should be stripped to [1, 2, 3]
        assert np.allclose(result['origin'].index, [1, 2, 3])
        # Force and torque should keep all indices
        assert len(result['force'].index) == 5
        assert len(result['torque'].index) == 5

    def test_metabolicrecord_strip_shared_timeframe(self):
        """Test that MetabolicRecord inherits shared timeframe stripping."""
        # Create signals with different NaN patterns
        vo2_data = np.array([np.nan, 10.0, 12.0, 14.0, np.nan])
        vo2 = laban.Signal1D(vo2_data, index=[0, 1, 2, 3, 4], unit="mL/kg/min")

        vco2_data = np.array([8.0, np.nan, 10.0, np.nan, 12.0])
        vco2 = laban.Signal1D(vco2_data, index=[0, 1, 2, 3, 4], unit="mL/kg/min")

        hr_data = np.array([120.0, 125.0, 130.0, 135.0, 140.0])
        hr = laban.Signal1D(hr_data, index=[0, 1, 2, 3, 4], unit="1/min")

        ve_data = np.array([30.0, 32.0, 34.0, 36.0, 38.0])
        ve = laban.Signal1D(ve_data, index=[0, 1, 2, 3, 4], unit="L/min")

        rf_data = np.array([20.0, 22.0, 24.0, 26.0, 28.0])
        rf = laban.Signal1D(rf_data, index=[0, 1, 2, 3, 4], unit="1/min")

        rec = laban.MetabolicRecord(
            vo2=vo2, vco2=vco2, hr=hr, ve=ve, rf=rf, breath_by_breath=False
        )
        result = rec.strip(independent=False, inplace=False)

        # All should span from 0 (first non-NaN in vco2) to 4 (last non-NaN in vco2)
        assert np.allclose(result['vo2'].index, [0, 1, 2, 3, 4])
        assert np.allclose(result['vco2'].index, [0, 1, 2, 3, 4])
        assert np.allclose(result['hr'].index, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
