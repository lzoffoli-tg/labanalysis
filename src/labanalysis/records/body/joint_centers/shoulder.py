"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Point3D
from ....referenceframes import ReferenceFrame

class ShoulderJointsMixin:
    """ShoulderJoints properties for WholeBody."""

    @property
    def left_shoulder(self):
        """
        Left shoulder joint center.
        Calculated using two methods (priority order):
        1. Midpoint of anterior and posterior shoulder markers (if available)
        2. De Leva (1996) regression from acromion marker (fallback)
        Returns
        -------
        Point3D
            Shoulder joint center position. Returns NaN if all required markers are missing.
        Notes
        -----
        De Leva (1996) fallback uses -12.25% of upper arm length (acromion to elbow)
        as offset from acromion toward C7 (proximal direction).
        See Also
        --------
        right_shoulder : Right shoulder joint center
        left_shoulder_referenceframe : Reference frame for angular measurements
        _calculate_shoulder_from_acromion : De Leva regression method
        """
        # Try primary method: midpoint of anterior and posterior markers
        ant = self._get_point("left_shoulder_anterior")
        pos = self._get_point("left_shoulder_posterior")
        if ant is not None and pos is not None:
            return Point3D(
                (ant + pos).to_numpy() / 2,
                index=np.unique(np.concatenate([ant.index, pos.index])),
                columns=ant.columns,
            )
        # Fallback: De Leva regression from acromion
        acr = self._get_point("left_acromion")
        if acr is not None:
            try:
                return self._calculate_shoulder_from_acromion(acr, side="left")
            except Exception:
                pass
        # All methods failed
        warnings.warn(
            "Cannot calculate left_shoulder: missing markers ['left_shoulder_anterior', 'left_shoulder_posterior'] and 'left_acromion'. Returning NaN.",
            UserWarning
        )
        ref = self._find_any_valid_marker()
        return self._create_nan_point3d(ref)

    @property
    def left_shoulder_referenceframe(self):
        """
        Left shoulder reference frame for angular measurements.
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
        Left shoulder joint center (use `self.left_shoulder` property, De Leva 1996)
        Construction
        ------------
        1. lateral_axis: LEFT (neck_base → left_shoulder)
        2. vertical_axis: UP (pelvis_center → neck_base)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at left shoulder center and orthonormal axes.
        See Also
        --------
        left_shoulder : Shoulder joint center (origin of this frame)
        neck_base : Neck base (used for X-axis)
        pelvis_center : Pelvis center (used for Y-axis)
        right_shoulder_referenceframe : Right shoulder reference frame
        left_shoulder_flexionextension : Shoulder flexion angle using this frame
        left_shoulder_abductionadduction : Shoulder abduction angle using this frame
        left_shoulder_internalexternalrotation : Shoulder rotation angle using this frame
        """
        shoulder = self.left_shoulder
        neck_base = self.neck_base
        pelvis_center = self.pelvis_center
        # Construct lateral_axis: LEFT (neck_base to shoulder, points outward/LEFT for left shoulder)
        axis_x = (shoulder - neck_base).to_numpy()
        axis_x = axis_x / np.linalg.norm(axis_x, axis=1, keepdims=True)
        # Construct vertical_axis: UP (pelvis_center to neck_base)
        axis_y = (neck_base - pelvis_center).to_numpy()
        axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)
        # Construct anteroposterior_axis: FORWARD (cross product)
        axis_z = np.cross(axis_x, axis_y)
        return ReferenceFrame(
            origin=shoulder,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def right_shoulder(self):
        """
        Right shoulder joint center.
        Calculated using two methods (priority order):
        1. Midpoint of anterior and posterior shoulder markers (if available)
        2. De Leva (1996) regression from acromion marker (fallback)
        Returns
        -------
        Point3D
            Shoulder joint center position. Returns NaN if all required markers are missing.
        Notes
        -----
        De Leva (1996) fallback uses -12.25% of upper arm length (acromion to elbow)
        as offset from acromion toward C7 (proximal direction).
        See Also
        --------
        left_shoulder : Left shoulder joint center
        right_shoulder_referenceframe : Reference frame for angular measurements
        _calculate_shoulder_from_acromion : De Leva regression method
        """
        # Try primary method: midpoint of anterior and posterior markers
        ant = self._get_point("right_shoulder_anterior")
        pos = self._get_point("right_shoulder_posterior")
        if ant is not None and pos is not None:
            return Point3D(
                (ant + pos).to_numpy() / 2,
                index=np.unique(np.concatenate([ant.index, pos.index])),
                columns=ant.columns,
            )
        # Fallback: De Leva regression from acromion
        acr = self._get_point("right_acromion")
        if acr is not None:
            try:
                return self._calculate_shoulder_from_acromion(acr, side="right")
            except Exception:
                pass
        # All methods failed
        warnings.warn(
            "Cannot calculate right_shoulder: missing markers ['right_shoulder_anterior', 'right_shoulder_posterior'] and 'right_acromion'. Returning NaN.",
            UserWarning
        )
        ref = self._find_any_valid_marker()
        return self._create_nan_point3d(ref)

    @property
    def right_shoulder_referenceframe(self):
        """
        Right shoulder reference frame for angular measurements.
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
        Right shoulder joint center (use `self.right_shoulder` property, De Leva 1996)
        Construction
        ------------
        1. lateral_axis: RIGHT (neck_base → right_shoulder)
        2. vertical_axis: UP (pelvis_center → neck_base)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at right shoulder center and orthonormal axes.
        See Also
        --------
        right_shoulder : Shoulder joint center (origin of this frame)
        neck_base : Neck base (used for X-axis)
        pelvis_center : Pelvis center (used for Y-axis)
        left_shoulder_referenceframe : Left shoulder reference frame
        right_shoulder_flexionextension : Shoulder flexion angle using this frame
        right_shoulder_abductionadduction : Shoulder abduction angle using this frame
        right_shoulder_internalexternalrotation : Shoulder rotation angle using this frame
        """
        shoulder = self.right_shoulder
        neck_base = self.neck_base
        pelvis_center = self.pelvis_center
        # Construct lateral_axis: RIGHT (neck_base to shoulder, points outward/RIGHT for right shoulder)
        axis_x = (shoulder - neck_base).to_numpy()
        axis_x = axis_x / np.linalg.norm(axis_x, axis=1, keepdims=True)
        # Construct vertical_axis: UP (pelvis_center to neck_base)
        axis_y = (neck_base - pelvis_center).to_numpy()
        axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)
        # Construct anteroposterior_axis: compute and negate to keep pointing FORWARD (left-handed system)
        axis_z = -np.cross(axis_x, axis_y)
        return ReferenceFrame(
            origin=shoulder,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )
