"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D


class TrunkAnglesMixin:
    """TrunkAngles properties for WholeBody."""

    @property
    def trunk_rotation(self):
        """
        Calculate trunk axial rotation (yaw) in transverse plane.
        The angle represents the rotational orientation of the shoulders
        relative to the pelvis, measured in the transverse (horizontal) plane.
        Interpretation
        --------------
        - **Positive (+)**: Left rotation (rotazione sinistra del tronco)
          The shoulders rotate counterclockwise (viewed from above).
          The right shoulder moves forward relative to the left shoulder.
          Common in left trunk rotation movements.
        - **Negative (-)**: Right rotation (rotazione destra del tronco)
          The shoulders rotate clockwise (viewed from above).
          The left shoulder moves forward relative to the right shoulder.
          Common in right trunk rotation movements.
        - **0°**: Neutral position (shoulders aligned with pelvis in transverse plane)
        Calculation Method
        ------------------
        Uses pelvis reference frame with:
        - Origin: Pelvis center (centroid of 4 ASIS/PSIS markers)
        - lateral_axis: LEFT (right midpoint → left midpoint)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The shoulder axis vector (C7 → sternoclavicular junction) is transformed
        to the pelvis reference frame and projected onto the transverse plane
        (defined by lateral_axis and anteroposterior_axis).
        The calculation uses rotation matrix indices:
        - rotation_matrix[:, :, 0] = lateral_axis component
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        The angle is arctan2(lateral_component, anteroposterior_component), where
        positive values indicate left rotation (right shoulder forward).
        Returns
        -------
        Signal1D
            Trunk rotation angle in degrees.
            Positive = left rotation (right shoulder forward)
            Negative = right rotation (left shoulder forward)
        See Also
        --------
        trunk_lateralflexion : Trunk frontal plane flexion
        pelvis_rotation : Pelvis transverse plane rotation
        shoulder_lateral_tilt : Shoulder frontal plane tilt
        """
        try:
            # Get shoulder axis markers
            c7 = self._get_point("c7")
            sc = self._get_point("sc")
            # Get pelvis reference frame
            rmat = self.pelvis_referenceframe.rotation_matrix
            # Calculate shoulder axis vector (C7 to sternoclavicular junction)
            shoulder_axis = (sc - c7).to_numpy()
            # Transform to pelvis reference frame
            shoulder_axis_rf = np.einsum("nij,nj->ni", rmat, shoulder_axis)  # type: ignore
            # Extract components in reference frame coordinates
            # Index [0] = lateral_axis component (by ReferenceFrame construction)
            # Index [2] = anteroposterior_axis component (by ReferenceFrame construction)
            # arctan2 of lateral and anteroposterior components gives rotation angle
            # Positive lateral component (LEFT) = left rotation = positive angle
            angle = (
                np.arctan2(
                    shoulder_axis_rf[:, 0],
                    shoulder_axis_rf[:, 2],
                )
                * 180
                / np.pi
            )
            # Return angle
            return Signal1D(data=angle, index=c7.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate trunk_rotation: missing required markers. Returning NaN.",
                UserWarning,
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def trunk_lateralflexion_local(self):
        """
        Calculate trunk lateral flexion in pelvis frontal plane.

        The angle represents the lateral bending of the trunk relative to
        the pelvis orientation, measured in the pelvis frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Trunk bending to the LEFT
          The neck_base is displaced to the left of pelvis_center when viewed
          in the pelvis frontal plane.
        - **Negative (-)**: Trunk bending to the RIGHT
          The neck_base is displaced to the right of pelvis_center when viewed
          in the pelvis frontal plane.
        - **0°**: Neutral position (trunk vertical in pelvis frontal plane)

        Calculation Method
        ------------------
        Calculates the angle of the trunk segment (pelvis_center → neck_base)
        projected onto the pelvis frontal plane (defined by pelvis lateral_axis
        and vertical_axis).

        The trunk vector is projected onto the pelvis frontal plane and the
        angle is calculated as arctan2(lateral_component, vertical_component),
        giving the lateral flexion angle relative to the pelvis vertical axis.

        Returns
        -------
        Signal1D
            Trunk lateral flexion angle in degrees.
            Positive = trunk bending left
            Negative = trunk bending right

        See Also
        --------
        trunk_lateralflexion_global : Trunk lateral flexion in global frame
        pelvis_lateral_tilt_local : Pelvis lateral tilt in local frame
        pelvis_referenceframe : Pelvis reference frame
        """
        try:
            # Get trunk segment points
            pelvis_center = self.pelvis_center
            neck_base = self.neck_base

            # Calculate trunk vector (pelvis_center → neck_base)
            trunk_vec = (neck_base - pelvis_center).to_numpy()

            # Get pelvis reference frame axes
            pelvis_rf = self.pelvis_referenceframe
            pelvis_lateral = pelvis_rf.rotation_matrix[
                :, :, 0
            ]  # Column 0 = lateral_axis (LEFT)
            pelvis_vertical = pelvis_rf.rotation_matrix[
                :, :, 1
            ]  # Column 1 = vertical_axis (UP)

            # Project trunk vector onto pelvis frontal plane (lateral, vertical)
            lateral_comp = np.sum(trunk_vec * pelvis_lateral, axis=1)
            vertical_comp = np.sum(trunk_vec * pelvis_vertical, axis=1)

            # arctan2(lateral, vertical): positive when trunk bends left
            angle = np.degrees(np.arctan2(lateral_comp, vertical_comp))

            return Signal1D(data=angle, index=pelvis_center.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate trunk_lateralflexion_local: missing required markers. Returning NaN.",
                UserWarning,
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def trunk_lateralflexion_global(self):
        """
        Calculate trunk lateral flexion in global frontal plane.

        The angle represents the lateral bending of the trunk relative to
        the global vertical, measured in the global frontal plane.

        Interpretation
        --------------
        - **Positive (+)**: Trunk bending to the LEFT
          The neck_base is displaced to the left of pelvis_center when viewed
          in the global frontal plane (absolute reference).
        - **Negative (-)**: Trunk bending to the RIGHT
          The neck_base is displaced to the right of pelvis_center when viewed
          in the global frontal plane (absolute reference).
        - **0°**: Neutral position (trunk aligned with global vertical)

        Calculation Method
        ------------------
        Calculates the angle of the trunk segment (pelvis_center → neck_base)
        projected onto the global frontal plane (defined by global lateral_axis
        and vertical_axis).

        The trunk vector lateral and vertical components are extracted using
        the global coordinate system axes, and the angle is calculated as
        arctan2(lateral_component, vertical_component).

        Returns
        -------
        Signal1D
            Trunk lateral flexion angle in degrees.
            Positive = trunk bending left
            Negative = trunk bending right

        See Also
        --------
        trunk_lateralflexion_local : Trunk lateral flexion in pelvis frame
        pelvis_lateral_tilt_global : Pelvis lateral tilt in global frame
        """
        try:
            # Get trunk segment points
            pelvis_center = self.pelvis_center
            neck_base = self.neck_base

            # Calculate trunk vector (pelvis_center → neck_base)
            trunk_vec = (neck_base - pelvis_center).to_numpy()

            # Project onto global frontal plane (lateral_axis, vertical_axis)
            cols = pelvis_center.columns
            lateral_idx = np.where(cols == self.lateral_axis)[0][0]
            vertical_idx = np.where(cols == self.vertical_axis)[0][0]

            lateral_comp = trunk_vec[:, lateral_idx]
            vertical_comp = trunk_vec[:, vertical_idx]

            # arctan2(lateral, vertical): positive when trunk bends left
            angle = np.degrees(np.arctan2(lateral_comp, vertical_comp))

            return Signal1D(data=angle, index=pelvis_center.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate trunk_lateralflexion_global: missing required markers. Returning NaN.",
                UserWarning,
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
