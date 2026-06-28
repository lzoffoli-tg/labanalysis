"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D
from ....signalprocessing import gram_schmidt

class HipAnglesMixin:
    """HipAngles properties for WholeBody."""

    @property
    def left_hip_abductionadduction(self):
        """
        Calculate left hip abduction/adduction angle in frontal plane.
        The angle represents the lateral (outward) or medial (inward)
        movement of the thigh relative to the pelvis in the frontal plane.
        Interpretation
        --------------
        - **Positive (+)**: Abduction (abduzione)
          The thigh is moved laterally away from the body midline.
          Common in side-stepping, lateral movements.
        - **Negative (-)**: Adduction (adduzione)
          The thigh is moved medially toward or across the body midline.
          Common in crossover movements, cutting.
        - **0°**: Neutral position (thigh vertical, aligned with hip)
        Calculation Method
        ------------------
        Uses left hip reference frame (based on pelvis frame with origin at left hip):
        - Origin: Left hip joint center (De Leva 1996 regression)
        - lateral_axis: LEFT (from pelvis right midpoint → left midpoint)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The knee position is transformed to the hip reference frame and projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is calculated using the lateral and vertical components of the
        transformed knee vector.
        Zero degrees corresponds to neutral position (thigh vertical, knee directly
        below hip). The result is normalized to [0°, 360°] range.
        Returns
        -------
        Signal1D
            Hip abduction/adduction angle in degrees.
            Positive = abduction (thigh outward)
            Negative = adduction (thigh inward)
        See Also
        --------
        right_hip_abductionadduction : Right hip frontal plane motion
        left_hip_flexionextension : Left hip sagittal plane motion
        left_knee_varusvalgus : Left knee frontal plane alignment
        """
        try:
            # get points and reference frame
            hip = self.left_hip
            rmat = self.left_hip_referenceframe.rotation_matrix
            knee = self.left_knee
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                knee,
                hip,
                rmat,
                self.lateral_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            # Transform to anatomical convention: 0° = neutral (thigh vertical)
            # The reference frame vertical_axis points downward in the hip frame
            # (determined by the lateral_axis × vertical_axis orthonormalization)
            # For vertical thigh (knee below hip), arctan2 gives ~+90°
            # Subtract 90° to make neutral position = 0°, then normalize to [0°, 360°]
            result = (angle.to_numpy() - 90) % 360
            return Signal1D(data=result, index=angle.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_hip_abductionadduction: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def left_hip_flexionextension(self):
        """
        Calculate left hip flexion/extension angle in sagittal plane.
        The angle represents the forward (flexion) or backward (extension)
        movement of the thigh relative to the pelvis in the sagittal plane.
        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The thigh is brought forward (anteriorly) toward the torso.
          Common in running, climbing, sitting.
        - **Negative (-)**: Extension (estensione)
          The thigh is moved backward (posteriorly) behind the body.
          Common in push-off phase of gait, sprinting.
        - **0°**: Neutral position (standing upright, thigh vertical)
        Calculation Method
        ------------------
        Uses left hip reference frame (based on pelvis frame with origin at left hip):
        - Origin: Left hip joint center (De Leva 1996 regression)
        - lateral_axis: LEFT (from pelvis right midpoint → left midpoint)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The knee position is transformed to the hip reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is calculated using the anteroposterior and vertical components
        of the transformed knee vector.
        Zero degrees corresponds to neutral position (thigh vertical, knee directly
        below hip). The result is normalized to [0°, 360°] range.
        Returns
        -------
        Signal1D
            Hip flexion/extension angle in degrees.
            Positive = flexion (thigh forward)
            Negative = extension (thigh backward)
        See Also
        --------
        right_hip_flexionextension : Right hip flexion angle
        left_hip_abductionadduction : Left hip frontal plane motion
        left_knee_flexionextension : Left knee flexion angle
        """
        try:
            # get points and reference frame
            hip = self.left_hip
            rmat = self.left_hip_referenceframe.rotation_matrix
            knee = self.left_knee
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                knee,
                hip,
                rmat,
                self.anteroposterior_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            # Transform to anatomical convention: 0° = neutral (thigh vertical)
            # The reference frame vertical_axis points downward in the hip frame
            # (determined by the lateral_axis × vertical_axis orthonormalization)
            # For vertical thigh (knee below hip), arctan2 gives ~+90°
            # Subtract 90° to make neutral position = 0°, then normalize to [0°, 360°]
            result = (angle.to_numpy() - 90) % 360
            return Signal1D(data=result, index=angle.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_hip_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def left_hip_internalexternalrotation(self):
        """
        Calculate left hip internal/external rotation angle in transverse plane.
        The angle represents the rotational orientation of the thigh around
        its longitudinal axis, indicating internal (medial) or external
        (lateral) rotation.
        Interpretation
        --------------
        - **Positive (+)**: Internal rotation (rotazione interna)
          The thigh rotates medially, turning the knee and foot inward.
          Common in cutting maneuvers, toe-in gait.
        - **Negative (-)**: External rotation (rotazione esterna)
          The thigh rotates laterally, turning the knee and foot outward.
          Common in toe-out gait, dance movements.
        - **0°**: Neutral position (knee pointing straight ahead)
        Calculation Method
        ------------------
        Uses a custom thigh reference frame constructed from hip and knee positions:
        - lateral_axis: LEFT (left_hip → right_hip direction)
        - vertical_axis: thigh longitudinal axis (hip → knee direction)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The average of knee and ankle medial-lateral vectors is transformed to this
        thigh reference frame and projected onto the transverse plane (defined by
        lateral_axis and anteroposterior_axis).
        The calculation uses rotation matrix indices:
        - rotation_matrix[:, :, 0] = lateral_axis component
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        The angle is calculated as arctan2(anteroposterior, lateral) and negated to
        match clinical convention: positive = internal rotation, negative = external.
        Returns
        -------
        Signal1D
            Hip rotation angle in degrees.
            Positive = internal rotation (knee/foot inward)
            Negative = external rotation (knee/foot outward)
        See Also
        --------
        right_hip_internalexternalrotation : Right hip rotation angle
        left_hip_flexionextension : Left hip sagittal plane motion
        left_knee_varusvalgus : Left knee frontal plane alignment
        """
        # Get necessary parameters
        knee_lat = self._get_point("left_knee_lateral")
        knee_med = self._get_point("left_knee_medial")
        ankle_lat = self._get_point("left_ankle_lateral")
        ankle_med = self._get_point("left_ankle_medial")
        # Compute average vector from medial-lateral markers
        v1 = (knee_lat - knee_med).to_numpy()
        v2 = (ankle_lat - ankle_med).to_numpy()
        va = (v1 + v2) / 2
        # Determine reference frame rotation matrix
        i = (self.left_hip - self.right_hip).to_numpy()
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        k = (self.left_hip - self.left_knee).to_numpy()
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))
        # Align vector to new reference frame
        vr = np.einsum("nij,nj->ni", rmat, va)
        # Extract components in reference frame coordinates
        # Index [0] = lateral_axis component (by reference frame construction)
        # Index [2] = anteroposterior_axis component (by reference frame construction)
        lateral_component = vr[:, 0]
        ap_component = vr[:, 2]
        # Calculate angle of vector with respect to transverse plane
        angle = np.degrees(np.arctan2(ap_component, lateral_component))
        # Invert sign to match clinical convention: positive = internal rotation, negative = external rotation
        # Original implementation had opposite convention
        angle = -angle
        # Return angle
        return Signal1D(data=angle, index=knee_lat.index, unit="°")

    @property
    def right_hip_abductionadduction(self):
        """
        Calculate right hip abduction/adduction angle in frontal plane.
        The angle represents the lateral (outward) or medial (inward)
        movement of the thigh relative to the pelvis in the frontal plane.
        Interpretation
        --------------
        - **Positive (+)**: Abduction (abduzione)
          The thigh is moved laterally away from the body midline.
          Common in side-stepping, lateral movements.
        - **Negative (-)**: Adduction (adduzione)
          The thigh is moved medially toward or across the body midline.
          Common in crossover movements, cutting.
        - **0°**: Neutral position (thigh vertical, aligned with hip)
        Calculation Method
        ------------------
        Uses right hip reference frame (mirrored pelvis frame with origin at right hip):
        - Origin: Right hip joint center (De Leva 1996 regression)
        - lateral_axis: RIGHT (negated pelvis lateral_axis)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (same as pelvis, creating left-handed frame)
        The knee position is transformed to the hip reference frame and projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is calculated using the lateral and vertical components of the
        transformed knee vector.
        Zero degrees corresponds to neutral position (thigh vertical, knee directly
        below hip). The result is normalized to [0°, 360°] range.
        Returns
        -------
        Signal1D
            Hip abduction/adduction angle in degrees.
            Positive = abduction (thigh outward)
            Negative = adduction (thigh inward)
        See Also
        --------
        left_hip_abductionadduction : Left hip frontal plane motion
        right_hip_flexionextension : Right hip sagittal plane motion
        right_knee_varusvalgus : Right knee frontal plane alignment
        """
        try:
            hip = self.right_hip
            rmat = self.right_hip_referenceframe.rotation_matrix
            knee = self.right_knee
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                knee,
                hip,
                rmat,
                self.lateral_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            # Transform to anatomical convention: 0° = neutral (thigh vertical)
            # The reference frame vertical_axis points downward in the hip frame
            # (determined by the lateral_axis × vertical_axis orthonormalization)
            # For vertical thigh (knee below hip), arctan2 gives ~+90°
            # Subtract 90° to make neutral position = 0°, then normalize to [0°, 360°]
            result = (angle.to_numpy() - 90) % 360
            return Signal1D(data=result, index=angle.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_hip_abductionadduction: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_hip_flexionextension(self):
        """
        Calculate right hip flexion/extension angle in sagittal plane.
        The angle represents the forward (flexion) or backward (extension)
        movement of the thigh relative to the pelvis in the sagittal plane.
        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The thigh is brought forward (anteriorly) toward the torso.
          Common in running, climbing, sitting.
        - **Negative (-)**: Extension (estensione)
          The thigh is moved backward (posteriorly) behind the body.
          Common in push-off phase of gait, sprinting.
        - **0°**: Neutral position (standing upright, thigh vertical)
        Calculation Method
        ------------------
        Uses right hip reference frame (mirrored pelvis frame with origin at right hip):
        - Origin: Right hip joint center (De Leva 1996 regression)
        - lateral_axis: RIGHT (negated pelvis lateral_axis)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (same as pelvis, creating left-handed frame)
        The knee position is transformed to the hip reference frame and projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is calculated using the anteroposterior and vertical components
        of the transformed knee vector.
        Zero degrees corresponds to neutral position (thigh vertical, knee directly
        below hip). The result is normalized to [0°, 360°] range.
        Returns
        -------
        Signal1D
            Hip flexion/extension angle in degrees.
            Positive = flexion (thigh forward)
            Negative = extension (thigh backward)
        See Also
        --------
        left_hip_flexionextension : Left hip flexion angle
        right_hip_abductionadduction : Right hip frontal plane motion
        right_knee_flexionextension : Right knee flexion angle
        """
        try:
            # get points and reference frame
            hip = self.right_hip
            rmat = self.right_hip_referenceframe.rotation_matrix
            knee = self.right_knee
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                knee,
                hip,
                rmat,
                self.anteroposterior_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            # Transform to anatomical convention: 0° = neutral (thigh vertical)
            # The reference frame vertical_axis points downward in the hip frame
            # (determined by the lateral_axis × vertical_axis orthonormalization)
            # For vertical thigh (knee below hip), arctan2 gives ~+90°
            # Subtract 90° to make neutral position = 0°, then normalize to [0°, 360°]
            result = (angle.to_numpy() - 90) % 360
            return Signal1D(data=result, index=angle.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_hip_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_hip_internalexternalrotation(self):
        """
        Calculate right hip internal/external rotation angle in transverse plane.
        The angle represents the rotational orientation of the thigh around
        its longitudinal axis, indicating internal (medial) or external
        (lateral) rotation.
        Interpretation
        --------------
        - **Positive (+)**: Internal rotation (rotazione interna)
          The thigh rotates medially, turning the knee and foot inward.
          Common in cutting maneuvers, toe-in gait.
        - **Negative (-)**: External rotation (rotazione esterna)
          The thigh rotates laterally, turning the knee and foot outward.
          Common in toe-out gait, dance movements.
        - **0°**: Neutral position (knee pointing straight ahead)
        Calculation Method
        ------------------
        Uses a custom thigh reference frame constructed from hip and knee positions:
        - lateral_axis: RIGHT (right_hip → left_hip direction)
        - vertical_axis: thigh longitudinal axis (hip → knee direction)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The average of knee and ankle medial-lateral vectors is transformed to this
        thigh reference frame and projected onto the transverse plane (defined by
        lateral_axis and anteroposterior_axis).
        The calculation uses rotation matrix indices:
        - rotation_matrix[:, :, 0] = lateral_axis component
        - rotation_matrix[:, :, 2] = anteroposterior_axis component
        The angle is calculated as arctan2(anteroposterior, lateral) and negated to
        match clinical convention: positive = internal rotation, negative = external.
        Returns
        -------
        Signal1D
            Hip rotation angle in degrees.
            Positive = internal rotation (knee/foot inward)
            Negative = external rotation (knee/foot outward)
        See Also
        --------
        left_hip_internalexternalrotation : Left hip rotation angle
        right_hip_flexionextension : Right hip sagittal plane motion
        right_knee_varusvalgus : Right knee frontal plane alignment
        """
        # Get necessary parameters
        knee_lat = self._get_point("right_knee_lateral")
        knee_med = self._get_point("right_knee_medial")
        ankle_lat = self._get_point("right_ankle_lateral")
        ankle_med = self._get_point("right_ankle_medial")
        # Compute average vector from medial-lateral markers
        v1 = (knee_lat - knee_med).to_numpy()
        v2 = (ankle_lat - ankle_med).to_numpy()
        va = (v1 + v2) / 2
        # Determine reference frame rotation matrix
        # Fixed: Changed from left_hip - right_hip to right_hip - left_hip for correct symmetry
        i = (self.right_hip - self.left_hip).to_numpy()
        i = i / np.linalg.norm(i, axis=1, keepdims=True)
        # Fixed: Changed from left_hip - left_knee to right_hip - right_knee (was copy-paste error)
        k = (self.right_hip - self.right_knee).to_numpy()
        k = k / np.linalg.norm(k, axis=1, keepdims=True)
        j = np.cross(k, i)
        rmat = gram_schmidt(i, j, k).transpose((0, 2, 1))
        # Align vector to new reference frame
        vr = np.einsum("nij,nj->ni", rmat, va)
        # Extract components in reference frame coordinates
        # Index [0] = lateral_axis component (by reference frame construction)
        # Index [2] = anteroposterior_axis component (by reference frame construction)
        lateral_component = vr[:, 0]
        ap_component = vr[:, 2]
        # Calculate angle of vector with respect to transverse plane
        angle = np.degrees(np.arctan2(ap_component, lateral_component))
        # NOTE: Removed compensatory transformations (180 - angle, conditional - 360)
        # These were only needed to compensate for the wrong reference frame above
        # Invert sign to match clinical convention: positive = internal rotation, negative = external rotation
        # Original implementation had opposite convention
        angle = -angle
        # Return angle
        return Signal1D(data=angle, index=knee_lat.index, unit="°")
