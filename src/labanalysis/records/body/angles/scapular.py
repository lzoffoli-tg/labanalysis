"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D

class ScapularAnglesMixin:
    """ScapularAngles properties for WholeBody."""

    @property
    def left_scapular_protractionretraction(self):
        """
        Calculate left scapular protraction/retraction angle in transverse plane.
        The angle represents the horizontal position of the left shoulder relative
        to neck_base (base of the neck), indicating scapular protraction or
        retraction.
        Interpretation
        --------------
        - **Positive (+)**: Scapular protraction (protrazione scapolare)
          The shoulder is positioned anteriorly (forward) relative to neck base.
          Common in rounded shoulder posture.
        - **Negative (-)**: Scapular retraction (retrazione scapolare)
          The shoulder is positioned posteriorly (backward) relative to neck base.
          Common in military/upright posture.
        - **0°**: Neutral position (shoulder aligned with neck base in transverse plane)
        Calculation Method
        ------------------
        Uses neck reference frame with:
        - Origin: neck_base = midpoint(C7, sternoclavicular_junction)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (C7 → sternoclavicular_junction)
        - lateral_axis: LEFT (cross product vertical × anteroposterior, Gram-Schmidt)
        The left shoulder position is transformed to the neck reference frame and
        projected onto the transverse plane (defined by lateral_axis and
        anteroposterior_axis).
        The calculation uses self.lateral_axis and self.anteroposterior_axis to
        identify rotation matrix components. Positive values indicate protraction
        (shoulder forward), negative values indicate retraction (shoulder backward).
        Returns
        -------
        Signal1D
            Scapular protraction/retraction angle in degrees.
            Positive = protraction (forward shoulder position)
            Negative = retraction (backward shoulder position)
        See Also
        --------
        right_scapular_protractionretraction : Right scapular protraction/retraction
        neck_referenceframe : Neck reference frame used for this calculation
        shoulder_lateraltilt_global : Shoulder elevation in frontal plane
        trunk_rotation_global : Trunk rotation that may affect shoulder position
        """
        try:
            shoulder = self.left_shoulder
            neck_base = self.neck_base
            rmat = self.neck_referenceframe.rotation_matrix
            # Calculate angle of left shoulder in neck reference frame's transverse plane
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                shoulder,
                neck_base,
                rmat,
                self.lateral_axis,
                self.anteroposterior_axis,
            )
            # In neck frame, left shoulder has:
            # - lateral_axis > 0 (shoulder is left, +lateral_axis points LEFT)
            # - anteroposterior_axis > 0 when protracted (shoulder is forward, +anteroposterior_axis points FORWARD)
            # arctan2 of anteroposterior and lateral components gives protraction/retraction angle
            # For symmetric sign convention: positive = protraction
            angle_result = angle.to_numpy()
            return Signal1D(data=angle_result, index=shoulder.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_scapular_protractionretraction: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_scapular_protractionretraction(self):
        """
        Calculate right scapular protraction/retraction angle in transverse plane.
        The angle represents the horizontal position of the right shoulder relative
        to neck_base (base of the neck), indicating scapular protraction or
        retraction.
        Interpretation
        --------------
        - **Positive (+)**: Scapular protraction (protrazione scapolare)
          The shoulder is positioned anteriorly (forward) relative to neck base.
          Common in rounded shoulder posture.
        - **Negative (-)**: Scapular retraction (retrazione scapolare)
          The shoulder is positioned posteriorly (backward) relative to neck base.
          Common in military/upright posture.
        - **0°**: Neutral position (shoulder aligned with neck base in transverse plane)
        Calculation Method
        ------------------
        Uses neck reference frame with:
        - Origin: neck_base = midpoint(C7, sternoclavicular_junction)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (C7 → sternoclavicular_junction)
        - lateral_axis: LEFT (cross product vertical × anteroposterior, Gram-Schmidt)
        The right shoulder position is transformed to the neck reference frame and
        projected onto the transverse plane (defined by lateral_axis and
        anteroposterior_axis).
        The calculation uses self.lateral_axis and self.anteroposterior_axis to
        identify rotation matrix components. The lateral component is negated for
        the right side to maintain consistent sign convention: positive = protraction
        (shoulder forward), negative = retraction (shoulder backward).
        Returns
        -------
        Signal1D
            Scapular protraction/retraction angle in degrees.
            Positive = protraction (forward shoulder position)
            Negative = retraction (backward shoulder position)
        See Also
        --------
        left_scapular_protractionretraction : Left scapular protraction/retraction
        neck_referenceframe : Neck reference frame used for this calculation
        shoulder_lateraltilt_global : Shoulder elevation in frontal plane
        trunk_rotation_global : Trunk rotation that may affect shoulder position
        """
        try:
            shoulder = self.right_shoulder
            neck_base = self.neck_base
            rmat = self.neck_referenceframe.rotation_matrix
            # Calculate angle of right shoulder in neck reference frame's transverse plane
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                shoulder,
                neck_base,
                rmat,
                self.lateral_axis,
                self.anteroposterior_axis,
            )
            # In neck frame, right shoulder has:
            # - lateral_axis < 0 (shoulder is right, opposite to +lateral_axis which points LEFT)
            # - anteroposterior_axis > 0 when protracted (shoulder is forward, +anteroposterior_axis points FORWARD)
            # arctan2 with lateral < 0 gives angle in wrong quadrant
            # Recalculate angle using negated lateral component to match left side convention
            angle_result = np.degrees(np.arctan2(y, -x))
            return Signal1D(data=angle_result, index=shoulder.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_scapular_protractionretraction: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
