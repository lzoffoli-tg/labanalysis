"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Point3D
from ....referenceframes import ReferenceFrame

class AnkleJointsMixin:
    """AnkleJoints properties for WholeBody."""

    @property
    def left_ankle(self):
        """
        Left ankle joint center.
        Calculated as the midpoint between lateral and medial ankle markers
        (malleoli). If only one marker is available, returns that marker.
        Returns
        -------
        Point3D
            Ankle joint center position. Returns NaN if both markers are missing.
        See Also
        --------
        right_ankle : Right ankle joint center
        left_ankle_referenceframe : Reference frame for angular measurements
        """
        lat = self._get_point("left_ankle_lateral")
        med = self._get_point("left_ankle_medial")
        # Both missing
        if lat is None and med is None:
            warnings.warn(
                "Cannot calculate left_ankle: missing markers ['left_ankle_lateral', 'left_ankle_medial']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_point3d(ref)
        # Only one missing - use the other
        elif lat is None:
            return med
        elif med is None:
            return lat
        # Both present - calculate average
        else:
            try:
                return Point3D(
                    data=(lat._data + med._data) / 2,
                    index=np.unique(np.concatenate([lat.index, med.index])),
                    columns=lat.columns,
                )
            except Exception as e:
                return lat

    @property
    def left_ankle_referenceframe(self):
        """
        Left ankle reference frame for angular measurements.
        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:
        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)
        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.
        Origin
        ------
        Left ankle center (use `self.left_ankle` property)
        Construction
        ------------
        1. lateral_axis: LEFT (left_ankle_lateral → left_ankle_medial)
        2. vertical_axis: UP (left_ankle → left_knee)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame object with origin at ankle center and orthonormalized axes.
        See Also
        --------
        left_ankle : Ankle center (origin of this frame)
        right_ankle_referenceframe : Right ankle reference frame
        left_ankle_flexionextension : Flexion angle using this frame
        left_ankle_inversioneversion : Inversion angle using this frame
        """
        ankle = self.left_ankle
        knee = self.left_knee
        try:
            ankle_lat = self._get_point("left_ankle_lateral")
            ankle_med = self._get_point("left_ankle_medial")
            # Construct lateral_axis: LEFT (lateral to medial)
            axis_x = (ankle_lat - ankle_med).to_numpy()
        except Exception:
            # Default LEFT: use lateral_axis property to determine which column
            lateral_idx = np.where(ankle.columns == ankle.lateral_axis)[0][0]
            axis_x = np.zeros((ankle.shape[0], 3))
            axis_x[:, lateral_idx] = -1  # LEFT (negative for left side)
        # Construct vertical_axis: UP (ankle to knee)
        axis_y = (knee - ankle).to_numpy()
        # Create ReferenceFrame
        return ReferenceFrame(origin=ankle, lateral_axis=axis_x, vertical_axis=axis_y)

    @property
    def right_ankle(self):
        """
        Right ankle joint center.
        Calculated as the midpoint between lateral and medial ankle markers
        (malleoli). If only one marker is available, returns that marker.
        Returns
        -------
        Point3D
            Ankle joint center position. Returns NaN if both markers are missing.
        See Also
        --------
        left_ankle : Left ankle joint center
        right_ankle_referenceframe : Reference frame for angular measurements
        """
        lat = self._get_point("right_ankle_lateral")
        med = self._get_point("right_ankle_medial")
        # Both missing
        if lat is None and med is None:
            warnings.warn(
                "Cannot calculate right_ankle: missing markers ['right_ankle_lateral', 'right_ankle_medial']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_point3d(ref)
        # Only one missing - use the other
        elif lat is None:
            return med
        elif med is None:
            return lat
        # Both present - calculate average
        else:
            try:
                return Point3D(
                    data=(lat._data + med._data) / 2,
                    index=np.unique(np.concatenate([lat.index, med.index])),
                    columns=lat.columns,
                )
            except Exception as e:
                return lat

    @property
    def right_ankle_referenceframe(self):
        """
        Right ankle reference frame for angular measurements.
        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:
        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)
        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.
        Origin
        ------
        Right ankle center (use `self.right_ankle` property)
        Construction
        ------------
        1. lateral_axis: RIGHT (right_ankle_lateral → right_ankle_medial)
        2. vertical_axis: UP (right_ankle → right_knee)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame object with origin at ankle center and orthonormalized axes.
        See Also
        --------
        right_ankle : Ankle center (origin of this frame)
        left_ankle_referenceframe : Left ankle reference frame
        right_ankle_flexionextension : Flexion angle using this frame
        right_ankle_inversioneversion : Inversion angle using this frame
        """
        ankle = self.right_ankle
        knee = self.right_knee
        try:
            ankle_lat = self._get_point("right_ankle_lateral")
            ankle_med = self._get_point("right_ankle_medial")
            # Construct lateral_axis: RIGHT (lateral to medial)
            axis_x = (ankle_lat - ankle_med).to_numpy()
        except Exception:
            # Default RIGHT: use lateral_axis property to determine which column
            lateral_idx = np.where(ankle.columns == ankle.lateral_axis)[0][0]
            axis_x = np.zeros((ankle.shape[0], 3))
            axis_x[:, lateral_idx] = 1  # RIGHT (positive for right side)
        # Construct vertical_axis: UP (ankle to knee)
        axis_y = (knee - ankle).to_numpy()
        # Construct anteroposterior_axis: compute and negate to keep pointing FORWARD (left-handed system)
        axis_z = -np.cross(axis_x, axis_y)
        # Create ReferenceFrame (left-handed: det(R) = -1)
        return ReferenceFrame(
            origin=ankle,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )
