"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D

class KneeAnglesMixin:
    """KneeAngles properties for WholeBody."""

    @property
    def left_knee_flexionextension(self):
        """
        Calculate left knee flexion/extension angle in sagittal plane.
        The angle represents the bending or straightening of the knee joint,
        indicating flexion (bent knee) or extension (straight knee).
        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The knee is bent, bringing the heel toward the buttock.
          Common in squatting, running, jumping preparation.
        - **Negative (-)**: Extension (estensione)
          The knee is straightened beyond neutral.
          Rare; indicates hyperextension.
        - **0°**: Neutral position (fully straight knee)
        Calculation Method
        ------------------
        Uses left knee reference frame with:
        - Origin: Left knee center (midpoint of lateral and medial knee markers)
        - lateral_axis: LEFT (knee_lateral → knee_medial)
        - vertical_axis: UP (knee → hip)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The ankle position is transformed to the knee reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is calculated using the anteroposterior and vertical components of
        the transformed ankle vector.
        The calculation uses rotation matrix indices that correspond to semantic axes:
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component
        Zero degrees corresponds to full knee extension (ankle directly below knee).
        Returns
        -------
        Signal1D
            Knee flexion/extension angle in degrees.
            Positive = flexion (bent knee)
            Range: [0°, 180°]
        See Also
        --------
        right_knee_flexionextension : Right knee flexion angle
        left_knee_varusvalgus : Left knee frontal plane alignment
        left_hip_flexionextension : Left hip flexion angle
        """
        try:
            knee = self.left_knee
            ankle = self.left_ankle
            rmat = self.left_knee_referenceframe.rotation_matrix
            # Ankle vector from knee origin
            ankle_vec = (ankle - knee).to_numpy()
            # Transform to knee reference frame
            ankle_rf = np.einsum("nij,nj->ni", rmat, ankle_vec)
            # Extract components in reference frame coordinates
            # Index [2] = anteroposterior_axis component (by ReferenceFrame construction)
            # Index [1] = vertical_axis component (by ReferenceFrame construction)
            # Knee flexion is measured from the extended position (ankle below knee, vertical negative)
            # arctan2 of anteroposterior and vertical components gives flexion/extension angle
            # At extended (ankle straight down): anteroposterior≈0, vertical<0 → angle ≈ 0°
            # At flexed (ankle forward): anteroposterior>0, vertical<0 → angle < 0°, so negate to get positive flexion
            flexion = -np.arctan2(ankle_rf[:, 2], -ankle_rf[:, 1]) * 180 / np.pi
            return Signal1D(data=flexion, index=knee.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_knee_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def left_knee_varusvalgus(self):
        """
        Calculate left knee varus/valgus angle in frontal plane.
        The angle represents the frontal plane alignment of the knee joint.
        Interpretation
        --------------
        - **Positive (+)**: Varus deformity (ginocchio varo, "a parentesi", bow-legged)
          The knee deviates laterally; the leg angle opens medially.
        - **Negative (-)**: Valgus deformity (ginocchio valgo, "a X", knock-knee)
          The knee deviates medially; the leg angle opens laterally.
        - **0°**: Neutral alignment (anca-ginocchio-caviglia collineari nel piano frontale)
        Calculation Method
        ------------------
        Uses left knee reference frame with:
        - Origin: Left knee center (midpoint of lateral and medial knee markers)
        - lateral_axis: LEFT (knee_lateral → knee_medial)
        - vertical_axis: UP (knee → hip)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The ankle position is transformed to the knee reference frame and projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is calculated using the lateral and vertical components of the
        transformed ankle vector.
        The calculation uses rotation matrix indices that correspond to semantic axes:
        - rotation_matrix[:, :, 0] = lateral_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component
        Positive lateral deviation (ankle lateral to knee) indicates varus alignment.
        Returns
        -------
        Signal1D
            Knee varus/valgus angle in degrees.
            Positive = varus (ginocchio varo)
            Negative = valgus (ginocchio valgo)
        """
        knee = self.left_knee
        ankle = self.left_ankle
        rmat = self.left_knee_referenceframe.rotation_matrix
        # Ankle vector from knee origin
        ankle_vec = (ankle - knee).to_numpy()
        # Transform to knee reference frame
        ankle_rf = np.einsum("nij,nj->ni", rmat, ankle_vec)
        # Extract components in reference frame coordinates
        # Index [0] = lateral_axis component (by ReferenceFrame construction)
        # Index [1] = vertical_axis component (by ReferenceFrame construction)
        # Varus/valgus is measured in the frontal plane (lateral-vertical)
        # arctan2 of lateral and vertical components gives varus/valgus angle
        # Positive lateral with vertical negative (down) = varus (knee out, bow-legged)
        # Negative lateral (medial) with vertical negative (down) = valgus (knee in, knock-knee)
        angle = np.arctan2(ankle_rf[:, 0], -ankle_rf[:, 1]) * 180 / np.pi
        return Signal1D(data=angle, index=knee.index, unit="°")

    @property
    def right_knee_flexionextension(self):
        """
        Calculate right knee flexion/extension angle in sagittal plane.
        The angle represents the bending or straightening of the knee joint,
        indicating flexion (bent knee) or extension (straight knee).
        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The knee is bent, bringing the heel toward the buttock.
          Common in squatting, running, jumping preparation.
        - **Negative (-)**: Extension (estensione)
          The knee is straightened beyond neutral.
          Rare; indicates hyperextension.
        - **0°**: Neutral position (fully straight knee)
        Calculation Method
        ------------------
        Uses right knee reference frame with:
        - Origin: Right knee center (midpoint of lateral and medial knee markers)
        - lateral_axis: RIGHT (knee_lateral → knee_medial)
        - vertical_axis: UP (knee → hip)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)
        The ankle position is transformed to the knee reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is calculated using the anteroposterior and vertical components of
        the transformed ankle vector.
        The calculation uses rotation matrix indices that correspond to semantic axes:
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component
        Zero degrees corresponds to full knee extension (ankle directly below knee).
        Returns
        -------
        Signal1D
            Knee flexion/extension angle in degrees.
            Positive = flexion (bent knee)
            Negative = extension (hyperextension)
        See Also
        --------
        left_knee_flexionextension : Left knee flexion angle
        right_knee_varusvalgus : Right knee frontal plane alignment
        right_hip_flexionextension : Right hip flexion angle
        """
        try:
            knee = self.right_knee
            ankle = self.right_ankle
            rmat = self.right_knee_referenceframe.rotation_matrix
            # Ankle vector from knee origin
            ankle_vec = (ankle - knee).to_numpy()
            # Transform to knee reference frame
            ankle_rf = np.einsum("nij,nj->ni", rmat, ankle_vec)
            # Extract components in reference frame coordinates
            # Index [2] = anteroposterior_axis component (by ReferenceFrame construction)
            # Index [1] = vertical_axis component (by ReferenceFrame construction)
            # Knee flexion is measured from the extended position (ankle below knee, vertical negative)
            # For left-handed frame: anteroposterior_axis is negated to point forward
            # arctan2 of anteroposterior and vertical components gives flexion/extension angle
            # At extended (ankle straight down): anteroposterior≈0, vertical<0 → angle ≈ 0°
            # At flexed (ankle forward): anteroposterior<0 (negated in frame), vertical<0 → angle > 0°, but we need to negate result
            flexion = -np.arctan2(ankle_rf[:, 2], -ankle_rf[:, 1]) * 180 / np.pi
            return Signal1D(data=flexion, index=knee.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_knee_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_knee_varusvalgus(self):
        """
        Calculate right knee varus/valgus angle in frontal plane.
        The angle represents the frontal plane alignment of the knee joint.
        Interpretation
        --------------
        - **Positive (+)**: Varus deformity (ginocchio varo, "a parentesi", bow-legged)
          The knee deviates laterally; the leg angle opens medially.
        - **Negative (-)**: Valgus deformity (ginocchio valgo, "a X", knock-knee)
          The knee deviates medially; the leg angle opens laterally.
        - **0°**: Neutral alignment (anca-ginocchio-caviglia collineari nel piano frontale)
        Calculation Method
        ------------------
        Uses right knee reference frame with:
        - Origin: Right knee center (midpoint of lateral and medial knee markers)
        - lateral_axis: RIGHT (knee_lateral → knee_medial)
        - vertical_axis: UP (knee → hip)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)
        The ankle position is transformed to the knee reference frame and projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is calculated using the lateral and vertical components of the
        transformed ankle vector.
        The calculation uses rotation matrix indices that correspond to semantic axes:
        - rotation_matrix[:, :, 0] = lateral_axis component
        - rotation_matrix[:, :, 1] = vertical_axis component
        Positive lateral deviation (ankle lateral to knee) indicates varus alignment.
        Returns
        -------
        Signal1D
            Knee varus/valgus angle in degrees.
            Positive = varus (ginocchio varo)
            Negative = valgus (ginocchio valgo)
        """
        knee = self.right_knee
        ankle = self.right_ankle
        rmat = self.right_knee_referenceframe.rotation_matrix
        # Ankle vector from knee origin
        ankle_vec = (ankle - knee).to_numpy()
        # Transform to knee reference frame
        ankle_rf = np.einsum("nij,nj->ni", rmat, ankle_vec)
        # Extract components in reference frame coordinates
        # Index [0] = lateral_axis component (by ReferenceFrame construction)
        # Index [1] = vertical_axis component (by ReferenceFrame construction)
        # Varus/valgus is measured in the frontal plane (lateral-vertical)
        # For left-handed frame: lateral_axis points RIGHT, already correct
        # arctan2 of lateral and vertical components gives varus/valgus angle
        # Positive lateral with vertical negative (down) = varus (knee out, bow-legged)
        # Negative lateral (medial) with vertical negative (down) = valgus (knee in, knock-knee)
        angle = np.arctan2(ankle_rf[:, 0], -ankle_rf[:, 1]) * 180 / np.pi
        return Signal1D(data=angle, index=knee.index, unit="°")
