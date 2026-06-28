"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Point3D
from ....referenceframes import ReferenceFrame

class KneeJointsMixin:
    """KneeJoints properties for WholeBody."""

    @property
    def left_knee(self):
        """
        Left knee joint center.
        Calculated as the midpoint between lateral and medial knee markers
        (femoral epicondyles). If only one marker is available, returns that marker.
        Returns
        -------
        Point3D
            Knee joint center position. Returns NaN if both markers are missing.
        See Also
        --------
        right_knee : Right knee joint center
        left_knee_referenceframe : Reference frame for angular measurements
        """
        lat = self._get_point("left_knee_lateral")
        med = self._get_point("left_knee_medial")
        if lat is None and med is None:
            warnings.warn(
                "Cannot calculate left_knee: missing markers ['left_knee_lateral', 'left_knee_medial']. Returning NaN.",
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
                    data=(lat._data + med._data) / 2,
                    index=np.unique(np.concatenate([lat.index, med.index])),
                    columns=lat.columns,
                )
            except Exception as e:
                return lat

    @property
    def left_knee_referenceframe(self):
        """
        Left knee reference frame for angular measurements.
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
        Left knee center (use `self.left_knee` property)
        Construction
        ------------
        1. lateral_axis: LEFT (left_knee_lateral → left_knee_medial)
        2. vertical_axis: UP (left_knee → left_hip)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at left knee center and orthonormal axes.
        See Also
        --------
        left_knee : Knee center (origin of this frame)
        right_knee_referenceframe : Right knee reference frame
        left_knee_flexionextension : Knee flexion angle using this frame
        left_knee_varusvalgus : Knee varus/valgus angle using this frame
        """
        knee = self.left_knee
        hip = self.left_hip
        try:
            knee_lat = self._get_point("left_knee_lateral")
            knee_med = self._get_point("left_knee_medial")
            # Construct lateral_axis: LEFT (lateral to medial)
            axis_x = (knee_lat - knee_med).to_numpy()
        except Exception:
            # Default LEFT: use lateral_axis property to determine which column
            lateral_idx = np.where(knee.columns == knee.lateral_axis)[0][0]
            axis_x = np.zeros((knee.shape[0], 3))
            axis_x[:, lateral_idx] = -1  # LEFT (negative for left side)
        # Construct vertical_axis: UP (knee to hip)
        axis_y = (hip - knee).to_numpy()
        return ReferenceFrame(origin=knee, lateral_axis=axis_x, vertical_axis=axis_y)

    @property
    def right_knee(self):
        """
        Right knee joint center.
        Calculated as the midpoint between lateral and medial knee markers
        (femoral epicondyles). If only one marker is available, returns that marker.
        Returns
        -------
        Point3D
            Knee joint center position. Returns NaN if both markers are missing.
        See Also
        --------
        left_knee : Left knee joint center
        right_knee_referenceframe : Reference frame for angular measurements
        """
        lat = self._get_point("right_knee_lateral")
        med = self._get_point("right_knee_medial")
        if lat is None and med is None:
            warnings.warn(
                "Cannot calculate right_knee: missing markers ['right_knee_lateral', 'right_knee_medial']. Returning NaN.",
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
                    data=(lat._data + med._data) / 2,
                    index=np.unique(np.concatenate([lat.index, med.index])),
                    columns=lat.columns,
                )
            except Exception as e:
                return lat

    @property
    def right_knee_referenceframe(self):
        """
        Right knee reference frame for angular measurements.
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
        Right knee center (use `self.right_knee` property)
        Construction
        ------------
        1. lateral_axis: RIGHT (right_knee_lateral → right_knee_medial)
        2. vertical_axis: UP (right_knee → right_hip)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at right knee center and orthonormal axes.
        See Also
        --------
        right_knee : Knee center (origin of this frame)
        left_knee_referenceframe : Left knee reference frame
        right_knee_flexionextension : Knee flexion angle using this frame
        right_knee_varusvalgus : Knee varus/valgus angle using this frame
        """
        knee = self.right_knee
        hip = self.right_hip
        try:
            knee_lat = self._get_point("right_knee_lateral")
            knee_med = self._get_point("right_knee_medial")
            # Construct lateral_axis: RIGHT (lateral to medial)
            axis_x = (knee_lat - knee_med).to_numpy()
        except Exception:
            # Default RIGHT: use lateral_axis property to determine which column
            lateral_idx = np.where(knee.columns == knee.lateral_axis)[0][0]
            axis_x = np.zeros((knee.shape[0], 3))
            axis_x[:, lateral_idx] = 1  # RIGHT (positive for right side)
        # Construct vertical_axis: UP (knee to hip)
        axis_y = (hip - knee).to_numpy()
        # Construct anteroposterior_axis: compute and negate to keep pointing FORWARD (left-handed system)
        axis_z = -np.cross(axis_x, axis_y)
        return ReferenceFrame(
            origin=knee,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )
