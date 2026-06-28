"""Quick test for attribute/item interchangeable access feature"""

import numpy as np
import pytest

import labanalysis as laban


def test_wholebody_item_access_to_property():
    """
    Test the main use case from the specification:
    ts = WholeBody(left_ankle_medial=Point3D(...), left_ankle_lateral=Point3D(...))

    If user does:
    ts.left_ankle (which is a property of WholeBody)

    Then also ts['left_ankle'] must return the same result.
    """
    # Create test data
    data = np.random.rand(10, 3)
    index = np.arange(10, dtype=float)

    left_ankle_medial = laban.Point3D(
        data=data,
        index=index,
        columns=['X', 'Y', 'Z']
    )

    left_ankle_lateral = laban.Point3D(
        data=data + 0.1,
        index=index,
        columns=['X', 'Y', 'Z']
    )

    ts = laban.WholeBody(
        left_ankle_medial=left_ankle_medial,
        left_ankle_lateral=left_ankle_lateral
    )

    # Test: ts.left_ankle is a property
    result_property = ts.left_ankle

    # Test: ts['left_ankle'] must work too
    result_item = ts['left_ankle']

    # They must be equal
    assert isinstance(result_property, laban.Point3D)
    assert isinstance(result_item, laban.Point3D)
    assert np.allclose(result_property.to_numpy(), result_item.to_numpy())
    print("✓ ts.left_ankle == ts['left_ankle']")


def test_wholebody_attribute_access_to_item():
    """
    Test the reverse case:
    ts['left_ankle_lateral'] works (it's an item)

    Then also ts.left_ankle_lateral must work
    (this already worked before via __getattr__, but we verify it still works)
    """
    data = np.random.rand(10, 3)
    index = np.arange(10, dtype=float)

    left_ankle_lateral = laban.Point3D(
        data=data,
        index=index,
        columns=['X', 'Y', 'Z']
    )

    ts = laban.WholeBody(left_ankle_lateral=left_ankle_lateral)

    # Test: ts['left_ankle_lateral'] is an item
    result_item = ts['left_ankle_lateral']

    # Test: ts.left_ankle_lateral must work too
    result_attr = ts.left_ankle_lateral

    # They must be equal
    assert isinstance(result_item, laban.Point3D)
    assert isinstance(result_attr, laban.Point3D)
    assert np.allclose(result_item.to_numpy(), result_attr.to_numpy())
    print("✓ ts['left_ankle_lateral'] == ts.left_ankle_lateral")


def test_point3d_item_access_to_property():
    """
    Test for Timeseries:
    point.module is a property

    Then also point['module'] must work
    """
    data = np.random.rand(10, 3)
    index = np.arange(10, dtype=float)

    point = laban.Point3D(
        data=data,
        index=index,
        columns=['X', 'Y', 'Z']
    )

    # Test: point.module is a property
    result_property = point.module

    # Test: point['module'] must work too
    result_item = point['module']

    # They must be equal
    assert isinstance(result_property, laban.Signal1D)
    assert isinstance(result_item, laban.Signal1D)
    assert np.allclose(result_property.to_numpy(), result_item.to_numpy())
    print("✓ point.module == point['module']")


def test_point3d_attribute_access_to_column():
    """
    Test for Timeseries:
    point['X'] works (it's a column)

    Then also point.X must work
    (this already worked before via __getattr__, but we verify it still works)
    """
    data = np.random.rand(10, 3)
    index = np.arange(10, dtype=float)

    point = laban.Point3D(
        data=data,
        index=index,
        columns=['X', 'Y', 'Z']
    )

    # Test: point['X'] is a column
    result_item = point['X']

    # Test: point.X must work too
    result_attr = point.X

    # They must be equal
    assert isinstance(result_item, laban.Timeseries)
    assert isinstance(result_attr, laban.Timeseries)
    assert np.allclose(result_item.to_numpy(), result_attr.to_numpy())
    print("✓ point['X'] == point.X")


def test_priority_items_over_properties():
    """
    Test that items in _data have priority over properties when accessed via [].
    """
    data = np.random.rand(10, 3)
    index = np.arange(10, dtype=float)

    left_ankle_medial = laban.Point3D(
        data=data,
        index=index,
        columns=['X', 'Y', 'Z']
    )

    left_ankle_lateral = laban.Point3D(
        data=data + 0.1,
        index=index,
        columns=['X', 'Y', 'Z']
    )

    ts = laban.WholeBody(
        left_ankle_medial=left_ankle_medial,
        left_ankle_lateral=left_ankle_lateral
    )

    # 'left_ankle_lateral' exists in _data
    # ts['left_ankle_lateral'] should return the item from _data, not try to call a property
    result = ts['left_ankle_lateral']

    assert isinstance(result, laban.Point3D)
    assert np.allclose(result.to_numpy(), left_ankle_lateral.to_numpy())
    print("✓ Items in _data have priority over properties")


def test_nonexistent_key_raises_error():
    """Test that accessing non-existent keys raises appropriate errors."""
    ts = laban.WholeBody()

    with pytest.raises(KeyError):
        _ = ts['nonexistent_item']
    print("✓ Non-existent keys raise KeyError")

    data = np.random.rand(10, 3)
    index = np.arange(10, dtype=float)
    point = laban.Point3D(data=data, index=index, columns=['X', 'Y', 'Z'])

    with pytest.raises(KeyError):
        _ = point['nonexistent_column']
    print("✓ Non-existent columns raise KeyError")


if __name__ == "__main__":
    print("Testing attribute/item interchangeable access...")
    print("=" * 60)

    test_wholebody_item_access_to_property()
    test_wholebody_attribute_access_to_item()
    test_point3d_item_access_to_property()
    test_point3d_attribute_access_to_column()
    test_priority_items_over_properties()
    test_nonexistent_key_raises_error()

    print("=" * 60)
    print("All tests passed! ✓")
