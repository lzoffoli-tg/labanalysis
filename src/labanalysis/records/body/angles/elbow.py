"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D

class ElbowAnglesMixin:
    """ElbowAngles properties for WholeBody."""

    @property
    def left_elbow_flexionextension(self):
        """
        Calculate left elbow flexion/extension angle in sagittal plane.
        The angle represents the bending or straightening of the elbow joint,
        indicating flexion (bent elbow) or extension (straight elbow).
        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The elbow is bent, bringing the wrist toward the shoulder.
          Common in lifting, reaching, biceps curl.
        - **Negative (-)**: Extension (estensione)
          The elbow is straightened beyond neutral.
          Rare; indicates hyperextension.
        - **0°**: Neutral position (fully straight elbow)
        Calculation Method
        ------------------
        Uses left elbow reference frame with:
        - Origin: Left elbow center (midpoint of lateral and medial elbow markers)
        - lateral_axis: LEFT (elbow_lateral → elbow_medial)
        - vertical_axis: UP (elbow → shoulder)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The wrist position is transformed to the elbow reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The calculation uses rotation matrix indices:
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component
        Zero degrees corresponds to full elbow extension (wrist directly below elbow).
        Positive values indicate flexion (bent elbow).
        Returns
        -------
        Signal1D
            Elbow flexion/extension angle in degrees.
            Positive = flexion (bent elbow)
            Range: [0°, 180°]
        See Also
        --------
        right_elbow_flexionextension : Right elbow flexion angle
        left_shoulder_flexionextension : Left shoulder flexion angle
        """
        try:
            elbow = self.left_elbow
            wrist = self.left_wrist
            rmat = self.left_elbow_referenceframe.rotation_matrix
            # Wrist vector from elbow origin
            wrist_vec = (wrist - elbow).to_numpy()
            # Transform to elbow reference frame
            wrist_rf = np.einsum("nij,nj->ni", rmat, wrist_vec)
            # Extract components in reference frame coordinates
            # Index [2] = anteroposterior_axis component (by ReferenceFrame construction)
            # Index [1] = vertical_axis component (by ReferenceFrame construction)
            # Elbow flexion is measured from the extended position (wrist below elbow, vertical negative)
            # arctan2 of anteroposterior and vertical components gives flexion/extension angle
            # At extended (wrist straight down): anteroposterior≈0, vertical<0 → angle ≈ 0°
            # At flexed (wrist forward): anteroposterior>0, vertical<0 → angle < 0°, so negate to get positive flexion
            flexion = -np.arctan2(wrist_rf[:, 2], -wrist_rf[:, 1]) * 180 / np.pi
            return Signal1D(data=flexion, index=elbow.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_elbow_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_elbow_flexionextension(self):
        """
        Calculate right elbow flexion/extension angle in sagittal plane.
        The angle represents the bending or straightening of the elbow joint,
        indicating flexion (bent elbow) or extension (straight elbow).
        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The elbow is bent, bringing the wrist toward the shoulder.
          Common in lifting, reaching, biceps curl.
        - **Negative (-)**: Extension (estensione)
          The elbow is straightened beyond neutral.
          Rare; indicates hyperextension.
        - **0°**: Neutral position (fully straight elbow)
        Calculation Method
        ------------------
        Uses right elbow reference frame with:
        - Origin: Right elbow center (midpoint of lateral and medial elbow markers)
        - lateral_axis: RIGHT (elbow_lateral → elbow_medial)
        - vertical_axis: UP (elbow → shoulder)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)
        The wrist position is transformed to the elbow reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The calculation uses rotation matrix indices:
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component
        Zero degrees corresponds to full elbow extension (wrist directly below elbow).
        Positive values indicate flexion (bent elbow).
        Returns
        -------
        Signal1D
            Elbow flexion/extension angle in degrees.
            Positive = flexion (bent elbow)
            Negative = extension (hyperextension)
        See Also
        --------
        left_elbow_flexionextension : Left elbow flexion angle
        right_shoulder_flexionextension : Right shoulder flexion angle
        """
        try:
            elbow = self.right_elbow
            wrist = self.right_wrist
            rmat = self.right_elbow_referenceframe.rotation_matrix
            # Wrist vector from elbow origin
            wrist_vec = (wrist - elbow).to_numpy()
            # Transform to elbow reference frame
            wrist_rf = np.einsum("nij,nj->ni", rmat, wrist_vec)
            # Extract components in reference frame coordinates
            # Index [2] = anteroposterior_axis component (by ReferenceFrame construction)
            # Index [1] = vertical_axis component (by ReferenceFrame construction)
            # Elbow flexion is measured from the extended position (wrist below elbow, vertical negative)
            # For left-handed frame: anteroposterior_axis is negated to point forward
            # arctan2 of anteroposterior and vertical components gives flexion/extension angle
            # At extended (wrist straight down): anteroposterior≈0, vertical<0 → angle ≈ 0°
            # At flexed (wrist forward): anteroposterior<0 (negated), vertical<0 → angle > 0°, but we need to negate result
            flexion = -np.arctan2(wrist_rf[:, 2], -wrist_rf[:, 1]) * 180 / np.pi
            return Signal1D(data=flexion, index=elbow.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_elbow_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
