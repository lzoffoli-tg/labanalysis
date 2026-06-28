"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal3D
from ....referenceframes import ReferenceFrame

class HipJointsMixin:
    """HipJoints properties for WholeBody."""

    @property
    def left_hip(self):
        """
        Left hip joint center.
        Estimated using De Leva (1996) regression equations based on pelvis dimensions
        and left trochanter marker position. The hip joint center is offset from the
        trochanter according to pelvis width and height.
        Returns
        -------
        Point3D
            Hip joint center position. Returns NaN if trochanter marker is missing.
        Notes
        -----
        Uses De Leva (1996) relative offsets:
        - Mediolateral: -19% of pelvis width
        - Vertical: -14% of pelvis height
        - Anteroposterior: 36% of pelvis width
        See Also
        --------
        right_hip : Right hip joint center
        pelvis_center : Pelvis center (ASIS/PSIS centroid)
        left_hip_referenceframe : Reference frame for angular measurements
        """
        troch = self._get_point("left_trochanter")
        if troch is None:
            warnings.warn(
                "Cannot calculate left_hip: missing marker 'left_trochanter'. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_point3d(ref)
        # get the De Leva (1996) approximations
        p0 = self.pelvis_center
        width = self.pelvis_width.to_numpy()
        height = self.pelvis_height.to_numpy()
        offset_ml = -0.19 * width
        offset_vt = -0.14 * height
        offset_ap = 0.36 * width
        offsets = Signal3D(
            data=np.hstack([offset_ml, offset_vt, offset_ap]),
            index=p0.index,
            columns=p0.columns,
            unit=p0.unit,
        )
        return self._get_translated_point_along_plane(
            troch,
            offsets,
            self._pelvis_plane,
        )

    @property
    def left_hip_referenceframe(self):
        """
        Left hip reference frame (based on pelvis frame with X pointing LEFT).
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
        Left hip joint center (use `self.left_hip` property, De Leva 1996)
        Construction
        ------------
        Uses pelvis reference frame directly (X already points LEFT for left side).
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at left hip center and axes from pelvis frame.
        See Also
        --------
        left_hip : Hip joint center (origin of this frame)
        right_hip_referenceframe : Right hip reference frame
        pelvis_referenceframe : Base pelvis reference frame
        left_hip_flexionextension : Hip flexion angle using this frame
        left_hip_abductionadduction : Hip abduction angle using this frame
        left_hip_internalexternalrotation : Hip rotation angle using this frame
        """
        # For left hip, use pelvis frame axes with hip origin
        pelvis_rf = self.pelvis_referenceframe
        return ReferenceFrame(
            origin=self.left_hip,
            lateral_axis=pelvis_rf.lateral_axis,
            vertical_axis=pelvis_rf.vertical_axis,
            anteroposterior_axis=pelvis_rf.anteroposterior_axis,
        )

    @property
    def pelvis_center(self):
        """
        Pelvis center (centroid of 4 ASIS/PSIS markers).
        Returns
        -------
        Point3D
            Pelvis center point (average of ASIS and PSIS markers).
        """
        l_asis = self._get_point("left_asis")
        r_asis = self._get_point("right_asis")
        l_psis = self._get_point("left_psis")
        r_psis = self._get_point("right_psis")
        markers = [l_asis, r_asis, l_psis, r_psis]
        marker_names = ["left_asis", "right_asis", "left_psis", "right_psis"]
        valid_markers = [m for m in markers if m is not None]
        if len(valid_markers) == 0:
            warnings.warn(
                f"Cannot calculate pelvis_center: missing markers {marker_names}. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_point3d(ref)
        elif len(valid_markers) < 4:
            missing = [marker_names[i] for i, m in enumerate(markers) if m is None]
            warnings.warn(
                f"Calculating pelvis_center with partial markers (missing {missing}).",
                UserWarning
            )
        result = valid_markers[0]
        for m in valid_markers[1:]:
            result = result + m
        return result / len(valid_markers)

    @property
    def right_hip(self):
        """
        Right hip joint center.
        Estimated using De Leva (1996) regression equations based on pelvis dimensions
        and right trochanter marker position. The hip joint center is offset from the
        trochanter according to pelvis width and height.
        Returns
        -------
        Point3D
            Hip joint center position. Returns NaN if trochanter marker is missing.
        Notes
        -----
        Uses De Leva (1996) relative offsets:
        - Mediolateral: +19% of pelvis width
        - Vertical: -14% of pelvis height
        - Anteroposterior: +36% of pelvis width
        See Also
        --------
        left_hip : Left hip joint center
        pelvis_center : Pelvis center (ASIS/PSIS centroid)
        right_hip_referenceframe : Reference frame for angular measurements
        """
        troch = self._get_point("right_trochanter")
        if troch is None:
            warnings.warn(
                "Cannot calculate right_hip: missing marker 'right_trochanter'. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_point3d(ref)
        # get the De Leva (1996) approximations
        p0 = self.pelvis_center
        width = self.pelvis_width.to_numpy()
        height = self.pelvis_height.to_numpy()
        offset_ml = +0.19 * width
        offset_vt = -0.14 * height
        offset_ap = +0.36 * width
        offsets = Signal3D(
            data=np.hstack([offset_ml, offset_vt, offset_ap]),
            index=p0.index,
            columns=p0.columns,
            unit=p0.unit,
        )
        return self._get_translated_point_along_plane(
            troch,
            offsets,
            self._pelvis_plane,
        )

    @property
    def right_hip_referenceframe(self):
        """
        Right hip reference frame (mirrored pelvis frame with X pointing RIGHT).
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
        Right hip joint center (use `self.right_hip` property, De Leva 1996)
        Construction
        ------------
        Mirrors the pelvis reference frame:
        - X: negated pelvis X (points RIGHT instead of LEFT)
        - Y: same pelvis Y (points UP)
        - Z: same pelvis Z (points FORWARD, det(R) = -1, left-handed)
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at right hip center, mirrored axes (left-handed).
        See Also
        --------
        right_hip : Hip joint center (origin of this frame)
        left_hip_referenceframe : Left hip reference frame
        pelvis_referenceframe : Base pelvis reference frame
        right_hip_flexionextension : Hip flexion angle using this frame
        right_hip_abductionadduction : Hip abduction angle using this frame
        right_hip_internalexternalrotation : Hip rotation angle using this frame
        """
        pelvis_rf = self.pelvis_referenceframe
        # For right hip: lateral axis points RIGHT, vertical UP, anteroposterior FORWARD
        # This creates a left-handed system (det(R) = -1) to maintain Z pointing forward
        axis_x = -pelvis_rf.lateral_axis  # Lateral: RIGHT
        axis_y = pelvis_rf.vertical_axis  # Vertical: UP
        axis_z = (
            pelvis_rf.anteroposterior_axis
        )  # Anteroposterior: FORWARD (same as pelvis)
        return ReferenceFrame(
            origin=self.right_hip,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )
