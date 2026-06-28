"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Point3D
from ....referenceframes import ReferenceFrame

class ElbowJointsMixin:
    """ElbowJoints properties for WholeBody."""

    @property
    def left_elbow(self):
        """
        Left elbow joint center.
        Calculated as the midpoint between lateral and medial elbow markers
        (epicondyles). If only one marker is available, returns that marker.
        Returns
        -------
        Point3D
            Elbow joint center position. Returns NaN if both markers are missing.
        See Also
        --------
        right_elbow : Right elbow joint center
        left_elbow_referenceframe : Reference frame for angular measurements
        """
        lat = self._get_point("left_elbow_lateral")
        med = self._get_point("left_elbow_medial")
        if lat is None and med is None:
            warnings.warn(
                "Cannot calculate left_elbow: missing markers ['left_elbow_lateral', 'left_elbow_medial']. Returning NaN.",
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
                    index=np.unique(np.concatenate([lat.index, med.index])).tolist(),
                    columns=lat.columns,
                )
            except Exception as e:
                return lat

    @property
    def left_elbow_referenceframe(self):
        """
        Left elbow reference frame for angular measurements.
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
        Left elbow center (use `self.left_elbow` property)
        Construction
        ------------
        1. lateral_axis: LEFT (left_elbow_lateral → left_elbow_medial)
        2. vertical_axis: UP (left_elbow → left_shoulder)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at left elbow center and orthonormal axes.
        See Also
        --------
        left_elbow : Elbow center (origin of this frame)
        right_elbow_referenceframe : Right elbow reference frame
        left_elbow_flexionextension : Elbow flexion angle using this frame
        """
        elbow = self.left_elbow
        shoulder = self.left_shoulder
        elbow_lat = self._get_point("left_elbow_lateral")
        elbow_med = self._get_point("left_elbow_medial")
        # Construct lateral_axis: LEFT (lateral to medial)
        axis_x = (elbow_lat - elbow_med).to_numpy()
        # Construct vertical_axis: UP (elbow to shoulder)
        axis_y = (shoulder - elbow).to_numpy()
        return ReferenceFrame(origin=elbow, lateral_axis=axis_x, vertical_axis=axis_y)

    @property
    def right_elbow(self):
        """
        Right elbow joint center.
        Calculated as the midpoint between lateral and medial elbow markers
        (epicondyles). If only one marker is available, returns that marker.
        Returns
        -------
        Point3D
            Elbow joint center position. Returns NaN if both markers are missing.
        See Also
        --------
        left_elbow : Left elbow joint center
        right_elbow_referenceframe : Reference frame for angular measurements
        """
        lat = self._get_point("right_elbow_lateral")
        med = self._get_point("right_elbow_medial")
        if lat is None and med is None:
            warnings.warn(
                "Cannot calculate right_elbow: missing markers ['right_elbow_lateral', 'right_elbow_medial']. Returning NaN.",
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
    def right_elbow_referenceframe(self):
        """
        Right elbow reference frame for angular measurements.
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
        Right elbow center (use `self.right_elbow` property)
        Construction
        ------------
        1. lateral_axis: RIGHT (right_elbow_lateral → right_elbow_medial)
        2. vertical_axis: UP (right_elbow → right_shoulder)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at right elbow center and orthonormal axes.
        See Also
        --------
        right_elbow : Elbow center (origin of this frame)
        left_elbow_referenceframe : Left elbow reference frame
        right_elbow_flexionextension : Elbow flexion angle using this frame
        """
        elbow = self.right_elbow
        shoulder = self.right_shoulder
        elbow_lat = self._get_point("right_elbow_lateral")
        elbow_med = self._get_point("right_elbow_medial")
        # Construct lateral_axis: RIGHT (lateral to medial)
        axis_x = (elbow_lat - elbow_med).to_numpy()
        # Construct vertical_axis: UP (elbow to shoulder)
        axis_y = (shoulder - elbow).to_numpy()
        # Construct anteroposterior_axis: compute and negate to keep pointing FORWARD (left-handed system)
        axis_z = -np.cross(axis_x, axis_y)
        return ReferenceFrame(
            origin=elbow,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )
