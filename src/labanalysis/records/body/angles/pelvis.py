"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D
from ....referenceframes import ReferenceFrame

class PelvisAnglesMixin:
    """PelvisAngles properties for WholeBody."""

    @property
    def pelvis_anteroposterior_tilt_global(self):
        """
        Calculate pelvis anteroposterior tilt (pitch) in global sagittal plane.
        The angle represents the forward or backward tilting of the pelvis
        relative to the global vertical, measured in the global sagittal plane.
        Interpretation
        --------------
        - **Positive (+)**: Anterior tilt (antiversione del bacino)
          ASIS markers are lower/more forward than PSIS markers relative to gravity.
          Lumbar lordosis typically increases.
          Common in hyperlordosis, tight hip flexors.
        - **Negative (-)**: Posterior tilt (retroversione del bacino)
          PSIS markers are lower/more forward than ASIS markers relative to gravity.
          Lumbar lordosis typically decreases.
          Common in hypolordosis, tight hamstrings.
        - **0°**: Neutral position (ASIS and PSIS at same height in global frame)
        Calculation Method
        ------------------
        Measures the angle of the pelvis plane (defined by ASIS-PSIS vector)
        relative to the global sagittal plane (vertical and anteroposterior axes).
        The tilt vector from PSIS midpoint to ASIS midpoint is projected onto
        the global sagittal plane:
        - self.anteroposterior_axis = global forward direction
        - self.vertical_axis = global vertical direction (gravity)
        The angle is arctan2(anteroposterior_component, vertical_component), where
        positive values indicate anterior tilt (ASIS forward/lower than PSIS).
        Returns
        -------
        Signal1D
            Pelvis anteroposterior tilt angle in degrees.
            Positive = anterior tilt (ASIS forward/lower)
            Negative = posterior tilt (PSIS forward/lower)
        See Also
        --------
        pelvis_lateral_tilt_global : Pelvis frontal plane tilt
        pelvis_rotation_global : Pelvis transverse plane rotation
        lumbar_lordosis : Lumbar spine curvature
        """
        try:
            # Get pelvis markers
            l_asis = self._get_point("left_asis")
            r_asis = self._get_point("right_asis")
            l_psis = self._get_point("left_psis")
            r_psis = self._get_point("right_psis")
            # Calculate tilt vector (PSIS midpoint to ASIS midpoint)
            psis_mid = (l_psis + r_psis) / 2
            asis_mid = (l_asis + r_asis) / 2
            tilt_vec = (asis_mid - psis_mid).to_numpy()
            # Project onto global sagittal plane (anteroposterior_axis, vertical_axis)
            cols = l_asis.columns
            ap_idx = np.where(cols == self.anteroposterior_axis)[0][0]
            vertical_idx = np.where(cols == self.vertical_axis)[0][0]
            ap_comp = tilt_vec[:, ap_idx]
            vertical_comp = tilt_vec[:, vertical_idx]
            # arctan2(-vertical, anteroposterior): positive when ASIS is lower (anterior tilt)
            # Negative vertical component means ASIS lower than PSIS (downward vector)
            angle = np.degrees(np.arctan2(-vertical_comp, ap_comp))
            return Signal1D(data=angle, index=l_asis.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate pelvis_anteroposterior_tilt_global: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def pelvis_anteroposterior_tilt_local(self):
        """
        Calculate pelvis anteroposterior tilt in trunk reference frame's sagittal plane.

        The angle represents the forward or backward tilting of the pelvis
        relative to the trunk orientation, measured in the trunk's sagittal plane.

        Interpretation
        --------------
        - **Positive (+)**: Anterior tilt (relative to trunk orientation)
          ASIS markers are forward/lower than PSIS markers, accounting for trunk flexion.
        - **Negative (-)**: Posterior tilt (relative to trunk orientation)
          PSIS markers are forward/lower than ASIS markers, accounting for trunk flexion.
        - **0°**: Neutral position (ASIS-PSIS aligned in trunk sagittal plane)

        Calculation Method
        ------------------
        Uses a trunk-aligned reference frame defined by:
        - **Vertical axis**: trunk axis from pelvis_center to neck_base
        - **Sagittal plane**: defined by the trunk vertical axis and its 90° projection
          onto the pelvis sagittal plane (pointing FORWARD)

        The ASIS-PSIS tilt vector is transformed into this trunk reference frame
        and projected onto the trunk's sagittal plane.

        The angle is calculated as arctan2(anteroposterior_component, vertical_component)
        in the trunk reference frame, giving the tilt angle relative to the trunk's
        vertical.

        This measurement differs from pelvis_anteroposterior_tilt_global by accounting for
        trunk flexion/extension - it measures pelvic tilt relative to the trunk
        orientation rather than absolute vertical.

        Returns
        -------
        Signal1D
            Pelvis anteroposterior tilt angle in degrees relative to trunk orientation.
            Positive = anterior tilt (ASIS forward/lower)
            Negative = posterior tilt (PSIS forward/lower)

        See Also
        --------
        pelvis_anteroposterior_tilt_global : Pelvis anteroposterior tilt in global frame
        pelvis_lateral_tilt_local : Pelvis lateral tilt in local frame
        pelvis_referenceframe : Pelvis reference frame
        lumbar_lordosis : Lumbar spine curvature
        """
        try:
            # Get pelvis markers
            l_asis = self._get_point("left_asis")
            r_asis = self._get_point("right_asis")
            l_psis = self._get_point("left_psis")
            r_psis = self._get_point("right_psis")

            # Calculate tilt vector (PSIS midpoint to ASIS midpoint)
            psis_mid = (l_psis + r_psis) / 2
            asis_mid = (l_asis + r_asis) / 2
            tilt_vec = (asis_mid - psis_mid).to_numpy()

            # Get trunk vertical axis (pelvis_center → neck_base)
            pelvis_center = self.pelvis_center
            neck_base = self.neck_base
            trunk_vertical = (neck_base - pelvis_center).to_numpy()
            trunk_vertical = trunk_vertical / np.linalg.norm(trunk_vertical, axis=1, keepdims=True)

            # Get pelvis reference frame anteroposterior axis (FORWARD direction in pelvis frame)
            pelvis_rf = self.pelvis_referenceframe
            pelvis_ap = pelvis_rf.rotation_matrix[:, :, 2]  # Column 2 = anteroposterior_axis (FORWARD)

            # Project pelvis anteroposterior axis onto plane perpendicular to trunk vertical
            # This gives the FORWARD direction in the trunk's sagittal plane
            # projection = ap - (ap · vertical) * vertical
            dot_product = np.sum(pelvis_ap * trunk_vertical, axis=1, keepdims=True)
            trunk_ap = pelvis_ap - dot_product * trunk_vertical
            trunk_ap = trunk_ap / np.linalg.norm(trunk_ap, axis=1, keepdims=True)

            # Project tilt vector onto trunk sagittal plane (trunk_ap, trunk_vertical)
            ap_comp = np.sum(tilt_vec * trunk_ap, axis=1)
            vertical_comp = np.sum(tilt_vec * trunk_vertical, axis=1)

            # arctan2(-vertical, ap): positive when ASIS is lower (anterior tilt)
            angle = np.degrees(np.arctan2(-vertical_comp, ap_comp))

            return Signal1D(data=angle, index=l_asis.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate pelvis_anteroposterior_tilt_local: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def pelvis_lateral_tilt_global(self):
        """
        Calculate pelvis lateral tilt (roll) in frontal plane.
        The angle represents the left or right side tilting of the pelvis
        relative to the global horizontal, measured in the frontal plane.
        Interpretation
        --------------
        - **Positive (+)**: Left hip higher than right hip (inclinazione sinistra del bacino)
          The left hip joint center is elevated relative to the right.
          Common in left hip drop compensation or scoliosis.
        - **Negative (-)**: Right hip higher than left hip (inclinazione destra del bacino)
          The right hip joint center is elevated relative to the left.
          Common in right hip drop, Trendelenburg gait.
        - **0°**: Neutral position (hip joints level in frontal plane)
        Calculation Method
        ------------------
        Uses hip joint centers (computed using De Leva 1996 regression) to define
        the hip-to-hip vector.
        The vector from right hip to left hip is projected onto the pelvis reference
        frame's frontal plane (defined by lateral_axis and vertical_axis).
        The lateral_axis points LEFT (from right midpoint to left midpoint), and
        the vertical_axis points UP (from hip_center to pelvis_center).
        The result is arctan2(vertical_component, lateral_component), giving the
        tilt angle relative to horizontal. For a neutral pelvis (hips level),
        the hip-to-hip vector projects positively onto the lateral_axis (same direction),
        giving arctan2(0, positive) = 0°.
        Returns
        -------
        Signal1D
            Pelvis lateral tilt angle in degrees.
            Positive = left hip higher than right hip
            Negative = right hip higher than left hip
        See Also
        --------
        pelvis_anteroposterior_tilt_global : Pelvis sagittal plane tilt
        pelvis_rotation_global : Pelvis transverse plane rotation
        trunk_lateralflexion : Trunk frontal plane flexion
        """
        try:
            # Get hip joint centers
            left_hip = self.left_hip
            right_hip = self.right_hip
            # Calculate hip-to-hip vector (right to left)
            hip_vector = (left_hip - right_hip).to_numpy()

            # Get pelvis reference frame lateral axis (anatomical LEFT direction)
            pelvis_rf = self.pelvis_referenceframe
            pelvis_lateral = pelvis_rf.rotation_matrix[:, :, 0]  # Column 0 = lateral_axis (LEFT)

            # For global measurement, use GLOBAL vertical axis (absolute up/down)
            # Find which column in the data corresponds to the global vertical axis
            cols = left_hip.columns
            vertical_idx = np.where(cols == self.vertical_axis)[0][0]

            # Build global vertical unit vector (e.g., [0, 1, 0] for Y-up coordinate system)
            global_vertical = np.zeros_like(hip_vector)
            global_vertical[:, vertical_idx] = 1.0

            # Project hip vector onto frontal plane:
            # - lateral_comp: projection onto anatomical LEFT (pelvis lateral_axis)
            # - vertical_comp: projection onto global UP (absolute vertical)
            # For neutral pelvis (hips level), lateral_comp is positive (right→left aligns with LEFT)
            lateral_comp = np.sum(hip_vector * pelvis_lateral, axis=1)
            vertical_comp = hip_vector[:, vertical_idx]  # Direct component along global vertical

            # arctan2(vertical, lateral): positive when left hip is higher
            angle = np.degrees(np.arctan2(vertical_comp, lateral_comp))
            return Signal1D(data=angle, index=left_hip.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate pelvis_lateral_tilt_global: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def pelvis_lateral_tilt_local(self):
        """
        Calculate pelvis lateral tilt in trunk reference frame's frontal plane.

        The angle represents the left or right side tilting of the pelvis
        relative to the trunk orientation, measured in the trunk's frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Left hip higher than right hip (relative to trunk orientation)
          The left hip is elevated relative to the right, accounting for trunk flexion.
        - **Negative (-)**: Right hip higher than left hip (relative to trunk orientation)
          The right hip is elevated relative to the left, accounting for trunk flexion.
        - **0°**: Neutral position (hips level in trunk frontal plane)

        Calculation Method
        ------------------
        Uses a trunk-aligned reference frame defined by:
        - **Vertical axis**: trunk axis from pelvis_center to neck_base
        - **Frontal plane**: defined by the trunk vertical axis and its 90° projection
          onto the pelvis frontal plane (pointing LEFT)

        The hip-to-hip vector is transformed into this trunk reference frame
        and projected onto the trunk's frontal plane.

        The angle is calculated as arctan2(vertical_component, lateral_component)
        in the trunk reference frame, giving the tilt angle relative to the trunk's
        horizontal.

        This measurement differs from pelvis_lateral_tilt_global by accounting for
        trunk flexion/extension - it measures pelvic tilt relative to the trunk
        orientation rather than absolute vertical.

        Returns
        -------
        Signal1D
            Pelvis lateral tilt angle in degrees relative to trunk orientation.
            Positive = left hip higher than right hip
            Negative = right hip higher than left hip

        See Also
        --------
        pelvis_lateral_tilt_global : Pelvis lateral tilt in global frame
        shoulder_lateral_tilt_local : Shoulder lateral tilt in local frame
        pelvis_referenceframe : Pelvis reference frame
        trunk_lateralflexion : Trunk lateral flexion
        """
        try:
            # Get hip joint centers
            left_hip = self.left_hip
            right_hip = self.right_hip

            # Get trunk vertical axis (pelvis_center → neck_base)
            pelvis_center = self.pelvis_center
            neck_base = self.neck_base
            trunk_vertical = (neck_base - pelvis_center).to_numpy()
            trunk_vertical = trunk_vertical / np.linalg.norm(trunk_vertical, axis=1, keepdims=True)

            # Get pelvis reference frame lateral axis (LEFT direction in pelvis frame)
            pelvis_rf = self.pelvis_referenceframe
            pelvis_lateral = pelvis_rf.rotation_matrix[:, :, 0]  # Column 0 = lateral_axis (LEFT)

            # Project pelvis lateral axis onto plane perpendicular to trunk vertical
            # This gives the LEFT direction in the trunk's frontal plane
            # projection = lateral - (lateral · vertical) * vertical
            dot_product = np.sum(pelvis_lateral * trunk_vertical, axis=1, keepdims=True)
            trunk_lateral = pelvis_lateral - dot_product * trunk_vertical
            trunk_lateral = trunk_lateral / np.linalg.norm(trunk_lateral, axis=1, keepdims=True)

            # Calculate hip-to-hip vector (right to left)
            hip_vector = (left_hip - right_hip).to_numpy()

            # Project hip vector onto trunk frontal plane (trunk_lateral, trunk_vertical)
            lateral_comp = np.sum(hip_vector * trunk_lateral, axis=1)
            vertical_comp = np.sum(hip_vector * trunk_vertical, axis=1)

            # arctan2(vertical, lateral): positive when left hip is higher
            angle = np.degrees(np.arctan2(vertical_comp, lateral_comp))

            return Signal1D(data=angle, index=left_hip.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate pelvis_lateral_tilt_local: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def pelvis_referenceframe(self):
        """
        Pelvis reference frame for angular measurements.
        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:
        - **lateral_axis**: Mediolateral direction (LEFT, from right to left ASIS-PSIS midpoints)
        - **vertical_axis**: Superior-inferior direction (UP, from midpoint of hip joints to pelvis_center)
        - **anteroposterior_axis**: Anterior-posterior direction (FORWARD, Gram-Schmidt cross product)
        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.
        Origin
        ------
        Pelvis center (centroid of ASIS and PSIS markers)
        Construction
        ------------
        1. lateral_axis: LEFT (from right midpoint to left midpoint)
        2. vertical_axis: UP (mean of left_hip and right_hip → pelvis_center)
        3. anteroposterior_axis: FORWARD (Gram-Schmidt cross product)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at pelvis center and orthonormal axes.
        See Also
        --------
        pelvis_center : Pelvis center (origin of this frame)
        left_hip : Left hip joint center (used for vertical axis)
        right_hip : Right hip joint center (used for vertical axis)
        trunk_lateralflexion : Trunk lateral flexion using this frame
        pelvis_anteroposterior_tilt_global : Pelvis anteroposterior tilt using this frame
        trunk_rotation : Trunk rotation using this frame
        """
        # Get pelvis points
        l_asis = self._get_point("left_asis")
        r_asis = self._get_point("right_asis")
        l_psis = self._get_point("left_psis")
        r_psis = self._get_point("right_psis")
        # Calculate midpoints
        left_mid = (l_asis + l_psis) / 2
        right_mid = (r_asis + r_psis) / 2
        centroid = (l_asis + r_asis + l_psis + r_psis) / 4
        # Get hip_center for vertical_axis construction (mean of left and right hip joints)
        left_hip = self.left_hip
        right_hip = self.right_hip
        hip_center = (left_hip + right_hip) / 2
        # Construct lateral_axis: LEFT (right midpoint to left midpoint)
        axis_x = (left_mid - right_mid).to_numpy()
        axis_x = axis_x / np.linalg.norm(axis_x, axis=1, keepdims=True)
        # Construct vertical_axis: UP (hip_center to pelvis_center)
        axis_y = (centroid - hip_center).to_numpy()
        axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)
        # Construct anteroposterior_axis: FORWARD (cross product)
        axis_z = np.cross(axis_x, axis_y)
        return ReferenceFrame(
            origin=centroid,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )

    @property
    def pelvis_rotation_global(self):
        """
        Calculate pelvis axial rotation (yaw) in transverse plane.
        The angle represents the left or right rotational orientation of
        the pelvis relative to the global forward direction, measured in
        the transverse (horizontal) plane.
        Interpretation
        --------------
        - **Positive (+)**: Left hip forward relative to right hip (rotazione sinistra del bacino)
          The left hip joint center is more anterior than the right.
          The pelvis rotates counterclockwise (viewed from above).
        - **Negative (-)**: Right hip forward relative to left hip (rotazione destra del bacino)
          The right hip joint center is more anterior than the left.
          The pelvis rotates clockwise (viewed from above).
        - **0°**: Neutral position (hip joints aligned in transverse plane)
        Calculation Method
        ------------------
        Uses hip joint centers (computed using De Leva 1996 regression) to define
        the hip-to-hip vector.
        The vector from right hip to left hip is projected onto the global transverse
        plane (defined by lateral_axis and anteroposterior_axis in the global coordinate system).
        The angle is calculated using the global coordinate components:
        - self.lateral_axis = global lateral direction
        - self.anteroposterior_axis = global anteroposterior direction
        The result is arctan2(anteroposterior_component, lateral_component), giving
        the rotation angle in the transverse plane.
        Returns
        -------
        Signal1D
            Pelvis rotation angle in degrees.
            Positive = left hip forward relative to right hip
            Negative = right hip forward relative to left hip
        See Also
        --------
        pelvis_anteroposterior_tilt_global : Pelvis sagittal plane tilt
        pelvis_lateral_tilt_global : Pelvis frontal plane tilt
        trunk_rotation : Trunk transverse plane rotation
        """
        try:
            # Get hip joint centers
            left_hip = self.left_hip
            right_hip = self.right_hip
            # Calculate hip-to-hip vector (right to left)
            hip_vector = (left_hip - right_hip).to_numpy()
            # Project onto global transverse plane (lateral_axis, anteroposterior_axis)
            cols = left_hip.columns
            lateral_idx = np.where(cols == self.lateral_axis)[0][0]
            ap_idx = np.where(cols == self.anteroposterior_axis)[0][0]
            lateral_comp = hip_vector[:, lateral_idx]
            ap_comp = hip_vector[:, ap_idx]
            # arctan2(anteroposterior, lateral): positive when left hip is forward
            angle = np.degrees(np.arctan2(ap_comp, lateral_comp))
            return Signal1D(data=angle, index=left_hip.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate pelvis_rotation_global: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def pelvis_rotation_local(self):
        """
        Calculate pelvis rotation in neck's transverse plane.
        The angle represents the left or right rotational orientation of
        the pelvis measured in the transverse plane of the neck reference
        frame.
        Interpretation
        --------------
        - **Positive (+)**: Left hip forward relative to right hip
          The left hip joint center is more anterior than the right
          when measured in the neck's transverse plane.
        - **Negative (-)**: Right hip forward relative to left hip
          The right hip joint center is more anterior than the left
          when measured in the neck's transverse plane.
        - **0°**: Neutral position (hip joints aligned in neck's transverse plane)
        Calculation Method
        ------------------
        Uses hip joint centers (computed using De Leva 1996 regression) to define
        the hip-to-hip vector.
        The vector from right hip to left hip is transformed into the neck
        reference frame coordinate system. The transverse plane components
        (lateral and anteroposterior) are extracted, and the angle is
        calculated as arctan2(anteroposterior_component, lateral_component).
        The neck reference frame is defined with:
        - Origin at neck_base (midpoint between SC and C7)
        - Vertical axis: pelvis_center → neck_base (UP)
        - Anteroposterior axis: C7 → SC (FORWARD)
        - Lateral axis: cross product (LEFT)
        Returns
        -------
        Signal1D
            Pelvis rotation angle in degrees.
            Positive = left hip forward relative to right hip
            Negative = right hip forward relative to left hip
        See Also
        --------
        pelvis_rotation_global : Pelvis rotation in global transverse plane
        pelvis_lateral_tilt_local : Pelvis lateral tilt in local reference plane
        neck_referenceframe : Neck local reference frame
        """
        try:
            # Get hip joint centers
            left_hip = self.left_hip
            right_hip = self.right_hip
            # Calculate hip-to-hip vector (right to left)
            hip_vector = (left_hip - right_hip).to_numpy()
            # Transform to neck reference frame
            neck_rframe = self.neck_referenceframe
            neck_rmat = neck_rframe.rotation_matrix
            hip_vector_rf = np.einsum("nij,nj->ni", neck_rmat, hip_vector)
            # Extract transverse plane components
            # Index [0] = lateral_axis, [2] = anteroposterior_axis
            lateral_comp = hip_vector_rf[:, 0]
            ap_comp = hip_vector_rf[:, 2]
            # Calculate angle: positive when left hip is forward
            angle = np.degrees(np.arctan2(ap_comp, lateral_comp))
            return Signal1D(data=angle, index=left_hip.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate pelvis_rotation_local: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
