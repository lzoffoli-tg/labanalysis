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
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
