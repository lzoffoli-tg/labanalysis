"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Point3D
from ....referenceframes import ReferenceFrame

class WristJointsMixin:
    """WristJoints properties for WholeBody."""

    @property
    def left_wrist(self):
        """
        Left wrist joint center.
        Calculated as the midpoint between lateral and medial wrist markers.
        If only one marker is available, returns that marker.
        Returns
        -------
        Point3D
            Wrist joint center position. Returns NaN if both markers are missing.
        See Also
        --------
        right_wrist : Right wrist joint center
        left_wrist_referenceframe : Reference frame for angular measurements
        """
        lat = self._get_point("left_wrist_lateral")
        med = self._get_point("left_wrist_medial")
        if lat is None and med is None:
            warnings.warn(
                "Cannot calculate left_wrist: missing markers ['left_wrist_lateral', 'left_wrist_medial']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_point3d(ref)
        elif lat is None:
            return med
        elif med is None:
            return lat
        else:
            try:
                return Point3D(
                    (lat + med).to_numpy() / 2,
                    index=np.unique(np.concatenate([lat.index, med.index])),
                    columns=lat.columns,
                )
            except Exception as e:
                return lat

    @property
    def left_wrist_referenceframe(self):
        """
        Left wrist reference frame for angular measurements.
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
        Left wrist center (use `self.left_wrist` property)
        Construction
        ------------
        1. lateral_axis: LEFT (left_wrist_lateral → left_wrist_medial)
        2. vertical_axis: UP (left_wrist → left_elbow)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at left wrist center and orthonormal axes.
        See Also
        --------
        left_wrist : Wrist center (origin of this frame)
        right_wrist_referenceframe : Right wrist reference frame
        """
        wrist = self.left_wrist
        elbow = self.left_elbow
        try:
            wrist_lat = self._get_point("left_wrist_lateral")
            wrist_med = self._get_point("left_wrist_medial")
            # Construct lateral_axis: LEFT (lateral to medial)
            axis_x = (wrist_lat - wrist_med).to_numpy()
        except Exception:
            # Default LEFT: use lateral_axis property to determine which column
            lateral_idx = np.where(wrist.columns == wrist.lateral_axis)[0][0]
            axis_x = np.zeros((wrist.shape[0], 3))
            axis_x[:, lateral_idx] = -1  # LEFT (negative for left side)
        # Construct vertical_axis: UP (wrist to elbow)
        axis_y = (elbow - wrist).to_numpy()
        return ReferenceFrame(origin=wrist, lateral_axis=axis_x, vertical_axis=axis_y)

    @property
    def right_wrist(self):
        """
        Right wrist joint center.
        Calculated as the midpoint between lateral and medial wrist markers.
        If only one marker is available, returns that marker.
        Returns
        -------
        Point3D
            Wrist joint center position. Returns NaN if both markers are missing.
        See Also
        --------
        left_wrist : Left wrist joint center
        right_wrist_referenceframe : Reference frame for angular measurements
        """
        lat = self._get_point("right_wrist_lateral")
        med = self._get_point("right_wrist_medial")
        if lat is None and med is None:
            warnings.warn(
                "Cannot calculate right_wrist: missing markers ['right_wrist_lateral', 'right_wrist_medial']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_point3d(ref)
        elif lat is None:
            return med
        elif med is None:
            return lat
        else:
            try:
                return Point3D(
                    (lat + med).to_numpy() / 2,
                    index=np.unique(np.concatenate([lat.index, med.index])),
                    columns=lat.columns,
                )
            except Exception as e:
                return lat

    @property
    def right_wrist_referenceframe(self):
        """
        Right wrist reference frame for angular measurements.
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
        Right wrist center (use `self.right_wrist` property)
        Construction
        ------------
        1. lateral_axis: RIGHT (right_wrist_lateral → right_wrist_medial)
        2. vertical_axis: UP (right_wrist → right_elbow)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at right wrist center and orthonormal axes.
        See Also
        --------
        right_wrist : Wrist center (origin of this frame)
        left_wrist_referenceframe : Left wrist reference frame
        """
        wrist = self.right_wrist
        elbow = self.right_elbow
        try:
            wrist_lat = self._get_point("right_wrist_lateral")
            wrist_med = self._get_point("right_wrist_medial")
            # Construct lateral_axis: RIGHT (lateral to medial)
            axis_x = (wrist_lat - wrist_med).to_numpy()
        except Exception:
            # Default RIGHT: use lateral_axis property to determine which column
            lateral_idx = np.where(wrist.columns == wrist.lateral_axis)[0][0]
            axis_x = np.zeros((wrist.shape[0], 3))
            axis_x[:, lateral_idx] = 1  # RIGHT (positive for right side)
        # Construct vertical_axis: UP (wrist to elbow)
        axis_y = (elbow - wrist).to_numpy()
        # Construct anteroposterior_axis: compute and negate to keep pointing FORWARD (left-handed system)
        axis_z = -np.cross(axis_x, axis_y)
        return ReferenceFrame(
            origin=wrist,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )
