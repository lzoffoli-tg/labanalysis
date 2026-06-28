"""Aggregation methods for combining segment lengths and joint angles."""

import numpy as np
from ...timeseries import Timeseries

__all__ = ["AggregationMixin"]


class AggregationMixin:
    """
    Mixin providing aggregation methods for WholeBody.

    Provides methods to collect and combine:
    - All segment lengths and widths into a single Timeseries
    - All joint angles into a single Timeseries
    - Deep copy functionality
    """

    @property
    def segment_lengths(self):
        """
        All segment lengths and widths combined into a single Timeseries.

        Returns a Timeseries containing all computed segment dimensions including:
        - Foot heights, lengths, and widths (left/right)
        - Ankle widths (left/right)
        - Leg lengths (left/right)
        - Thigh lengths (left/right)
        - Knee widths (left/right)
        - Lower limb total lengths (left/right)
        - Arm lengths (left/right)
        - Forearm lengths (left/right)
        - Elbow widths (left/right)
        - Upper limb total lengths (left/right)
        - Body dimensions: shoulder_width, hip_width, trunk_length, pelvis_height

        Returns
        -------
        Timeseries
            Combined timeseries with all segment length and width measurements.
            Each column represents one segment dimension (e.g., 'left_foot_height',
            'right_thigh_length', 'left_ankle_width', 'shoulder_width').

        Notes
        -----
        Only includes properties that are computable from available markers.
        If markers are missing, the corresponding columns will not be present.

        Examples
        --------
        >>> body = WholeBody(left_heel=..., right_heel=..., ...)
        >>> lengths = body.segment_lengths
        >>> print(lengths.columns)
        ['left_foot_height', 'right_foot_height', 'left_leg_length', 'shoulder_width', ...]
        """
        length_properties = [
            "left_foot_height",
            "right_foot_height",
            "left_foot_length",
            "right_foot_length",
            "left_foot_width",
            "right_foot_width",
            "left_ankle_width",
            "right_ankle_width",
            "left_leg_length",
            "right_leg_length",
            "left_thigh_length",
            "right_thigh_length",
            "left_knee_width",
            "right_knee_width",
            "left_lower_limb_length",
            "right_lower_limb_length",
            "left_arm_length",
            "right_arm_length",
            "left_forearm_length",
            "right_forearm_length",
            "left_elbow_width",
            "right_elbow_width",
            "left_upper_limb_length",
            "right_upper_limb_length",
            "shoulder_width",
            "hip_width",
            "trunk_length",
            "pelvis_height",
        ]

        # Collect all available length measurements
        data_list = []
        column_names = []
        common_index = None
        common_unit = None

        for prop_name in length_properties:
            try:
                value = getattr(self, prop_name)
                if value is not None:
                    # Get the data as 1D array
                    data_1d = value.to_numpy().flatten()
                    data_list.append(data_1d)
                    column_names.append(prop_name)

                    # Use first property's index and unit as reference
                    if common_index is None:
                        common_index = value.index
                        common_unit = value.unit
            except (AttributeError, TypeError, ValueError):
                # Skip if property cannot be computed (missing markers)
                continue

        if not data_list:
            raise ValueError(
                "No segment lengths could be computed. Check that required markers are provided."
            )

        # Stack all columns into 2D array
        data_2d = np.column_stack(data_list)

        return Timeseries(
            data=data_2d,
            index=common_index,
            columns=column_names,
            unit=common_unit,
        )

    @property
    def joint_angles(self):
        """
        All joint angles combined into a single Timeseries.

        Returns a Timeseries containing all computed joint angles including:
        - Ankle angles: flexion/extension, inversion/eversion (left/right)
        - Knee angles: flexion/extension, varus/valgus (left/right)
        - Hip angles: flexion/extension, abduction/adduction, internal/external rotation (left/right)
        - Pelvis angles: lateral tilt, rotation, anteroposterior tilt (global frame)
        - Trunk angles: lateral flexion, rotation, flexion/extension (global and local frames)
        - Shoulder angles: flexion/extension, abduction/adduction, internal/external rotation,
          lateral tilt, elevation/depression (left/right, global and local frames)
        - Scapular angles: protraction/retraction (left/right)
        - Elbow angles: flexion/extension (left/right)
        - Neck angles: flexion/extension (local and global), lateral tilt
        - Spine curvature: lumbar lordosis, dorsal kyphosis

        Returns
        -------
        Timeseries
            Combined timeseries with all joint angle measurements in degrees.
            Each column represents one angular measurement (e.g., 'left_knee_flexionextension',
            'pelvis_rotation_global', 'lumbar_lordosis').

        Notes
        -----
        Only includes properties that are computable from available markers.
        If markers are missing, the corresponding columns will not be present.

        Sign conventions vary by joint - see individual angle property docstrings for details.

        Examples
        --------
        >>> body = WholeBody(left_asis=..., right_asis=..., ...)
        >>> angles = body.joint_angles
        >>> print(angles.columns)
        ['left_ankle_flexionextension', 'right_knee_flexionextension', ...]
        """
        angle_properties = [
            # Ankle angles
            "left_ankle_flexionextension",
            "right_ankle_flexionextension",
            "left_ankle_inversioneversion",
            "right_ankle_inversioneversion",
            # Knee angles
            "left_knee_flexionextension",
            "right_knee_flexionextension",
            "left_knee_varusvalgus",
            "right_knee_varusvalgus",
            # Hip angles
            "left_hip_flexionextension",
            "right_hip_flexionextension",
            "left_hip_abductionadduction",
            "right_hip_abductionadduction",
            "left_hip_internalexternalrotation",
            "right_hip_internalexternalrotation",
            # Pelvis angles
            "pelvis_anteroposterior_tilt_global",
            # Trunk angles
            "trunk_rotation",
            # Shoulder angles
            "left_shoulder_flexionextension",
            "right_shoulder_flexionextension",
            "left_shoulder_abductionadduction",
            "right_shoulder_abductionadduction",
            "left_shoulder_internalexternalrotation",
            "right_shoulder_internalexternalrotation",
            "left_shoulder_elevationdepression",
            "right_shoulder_elevationdepression",
            # Scapular angles
            "left_scapular_protractionretraction",
            "right_scapular_protractionretraction",
            # Elbow angles
            "left_elbow_flexionextension",
            "right_elbow_flexionextension",
            # Neck angles
            "neck_flexionextension",
            "neck_lateralflexion",
            # Spine curvature
            "lumbar_lordosis",
            "dorsal_kyphosis",
        ]

        # Collect all available angle measurements
        data_list = []
        column_names = []
        common_index = None
        common_unit = None

        for prop_name in angle_properties:
            try:
                value = getattr(self, prop_name)
                if value is not None:
                    # Get the data as 1D array
                    data_1d = value.to_numpy().flatten()
                    data_list.append(data_1d)
                    column_names.append(prop_name)

                    # Use first property's index and unit as reference
                    if common_index is None:
                        common_index = value.index
                        common_unit = value.unit
            except (AttributeError, TypeError, ValueError):
                # Skip if property cannot be computed (missing markers)
                continue

        if not data_list:
            raise ValueError(
                "No joint angles could be computed. Check that required markers are provided."
            )

        # Stack all columns into 2D array
        data_2d = np.column_stack(data_list)

        return Timeseries(
            data=data_2d,
            index=common_index,
            columns=column_names,
            unit=common_unit,
        )

    def copy(self):
        """
        Create a deep copy of the WholeBody object.

        Returns
        -------
        WholeBody
            A new WholeBody instance with copies of all markers and signals.

        Notes
        -----
        All internal Point3D, Signal1D, Signal3D, ForcePlatform, and EMGSignal
        objects are deep copied, so modifications to the copy do not affect
        the original instance.
        """
        # Import here to avoid circular dependency
        from .wholebody import WholeBody
        return WholeBody(**{i: v.copy() for i, v in self.items()})
