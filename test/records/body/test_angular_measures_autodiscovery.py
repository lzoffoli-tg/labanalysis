"""
Test auto-discovery of angular measures.

Verifies that _angular_measures is correctly populated by introspecting
all angular properties of the WholeBody class.
"""

import pytest
from labanalysis.records.body import WholeBody


def test_angular_measures_auto_discovery():
    """Test that _angular_measures is auto-populated correctly."""
    angular_measures = WholeBody._angular_measures

    # Expected properties (as of this implementation)
    expected_properties = {
        # Ankle (4)
        "left_ankle_flexionextension",
        "right_ankle_flexionextension",
        "left_ankle_inversioneversion",
        "right_ankle_inversioneversion",
        # Knee (4)
        "left_knee_flexionextension",
        "right_knee_flexionextension",
        "left_knee_varusvalgus",
        "right_knee_varusvalgus",
        # Hip (6)
        "left_hip_flexionextension",
        "right_hip_flexionextension",
        "left_hip_abductionadduction",
        "right_hip_abductionadduction",
        "left_hip_internalexternalrotation",
        "right_hip_internalexternalrotation",
        # Pelvis (4)
        "pelvis_anteroposterior_tilt_global",
        "pelvis_lateraltilt_global",
        "pelvis_rotation_global",
        "pelvis_rotation_local",
        # Trunk (1)
        "trunk_rotation",
        # Shoulder (8)
        "left_shoulder_abductionadduction",
        "right_shoulder_abductionadduction",
        "left_shoulder_flexionextension",
        "right_shoulder_flexionextension",
        "left_shoulder_internalexternalrotation",
        "right_shoulder_internalexternalrotation",
        "left_shoulder_elevationdepression",
        "right_shoulder_elevationdepression",
        # Scapular (2)
        "left_scapular_protractionretraction",
        "right_scapular_protractionretraction",
        # Elbow (2)
        "left_elbow_flexionextension",
        "right_elbow_flexionextension",
        # Neck (2)
        "neck_lateralflexion",
        "neck_flexionextension",
        # Spine (2)
        "lumbar_lordosis",
        "dorsal_kyphosis",
    }

    # Convert to set for comparison
    discovered = set(angular_measures)

    # Check that all expected properties are discovered
    missing = expected_properties - discovered
    if missing:
        pytest.fail(
            f"Expected angular properties not discovered: {sorted(missing)}\n"
            f"Total expected: {len(expected_properties)}\n"
            f"Total discovered: {len(discovered)}"
        )

    # Check for unexpected properties (properties discovered but not expected)
    unexpected = discovered - expected_properties
    if unexpected:
        # This is a warning, not a failure - new properties might have been added
        print(
            f"\nWarning: Unexpected angular properties discovered: {sorted(unexpected)}\n"
            f"If these are newly implemented properties, update this test."
        )

    # Verify count
    assert len(discovered) >= len(expected_properties), (
        f"Expected at least {len(expected_properties)} angular measures, "
        f"but found only {len(discovered)}"
    )

    print(f"\nTotal angular measures discovered: {len(discovered)}")
    print("\nDiscovered properties:")
    for i, prop in enumerate(sorted(discovered), 1):
        print(f"  {i:2d}. {prop}")


def test_no_deprecated_properties():
    """Test that deprecated properties are not in _angular_measures."""
    angular_measures = WholeBody._angular_measures

    # Properties that should NOT be in the list (deprecated or removed)
    deprecated = {
        "shoulder_lateraltilt",  # replaced by _global and _local variants
        "pelvis_anteroposterior_tilt",  # replaced by _global variant
    }

    discovered = set(angular_measures)
    found_deprecated = discovered & deprecated

    if found_deprecated:
        pytest.fail(
            f"Deprecated properties found in _angular_measures: {sorted(found_deprecated)}\n"
            f"These should have been replaced with their _global/_local variants."
        )


def test_all_properties_are_accessible():
    """Test that all properties in _angular_measures exist on the class."""
    angular_measures = WholeBody._angular_measures

    for prop_name in angular_measures:
        assert hasattr(WholeBody, prop_name), (
            f"Property '{prop_name}' listed in _angular_measures "
            f"but not found on WholeBody class"
        )

        attr = getattr(WholeBody, prop_name)
        assert isinstance(attr, property), (
            f"'{prop_name}' in _angular_measures is not a property "
            f"(it's a {type(attr).__name__})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
