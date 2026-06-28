"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Point3D, Signal1D

class ShoulderAnglesMixin:
    """ShoulderAngles properties for WholeBody."""

    @property
    def left_shoulder_abductionadduction(self):
        """
        Calculate left shoulder abduction/adduction angle in frontal plane.
        The angle represents the lateral (outward) or medial (inward)
        movement of the arm relative to the shoulder in the frontal plane.
        Interpretation
        --------------
        - **Positive (+)**: Abduction (abduzione)
          The arm is raised laterally away from the body.
          Common in lateral raises, overhead reaching.
        - **Negative (-)**: Adduction (adduzione)
          The arm is moved medially toward or across the body.
          Common in cross-body movements.
        - **0°**: Neutral position (arm hanging at side)
        Calculation Method
        ------------------
        Uses left shoulder reference frame with:
        - Origin: Left shoulder joint center
        - lateral_axis: LEFT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The elbow position is transformed to the shoulder reference frame and
        projected onto the frontal plane (defined by lateral_axis and vertical_axis).
        The calculation uses self.lateral_axis and self.vertical_axis to identify
        rotation matrix components. The result is normalized to [0°, 360°] with
        zero corresponding to the arm hanging vertically downward.
        Returns
        -------
        Signal1D
            Shoulder abduction/adduction angle in degrees.
            Positive = abduction (arm outward)
            Negative = adduction (arm inward)
        See Also
        --------
        right_shoulder_abductionadduction : Right shoulder frontal plane motion
        left_shoulder_flexionextension : Left shoulder sagittal plane motion
        left_scapular_protractionretraction : Left scapular position
        """
        try:
            # Get necessary parameters
            shoulder = self.left_shoulder
            rmat = self.left_shoulder_referenceframe.rotation_matrix
            elbow = self.left_elbow
            # Calculate arm orientation with respect to shoulder reference frame
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                elbow,
                shoulder,
                rmat,
                self.lateral_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            # Transform to anatomical convention: 0° = neutral (arm hanging vertically)
            # arctan2 gives ~-90° for vertical downward vector
            # Add 90° to make neutral = 0°, then normalize to [0°, 360°]
            result = (angle.to_numpy() + 90) % 360
            return Signal1D(data=result, index=angle.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_shoulder_abductionadduction: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def left_shoulder_elevationdepression(self):
        """
        Calculate left shoulder elevation/depression angle in frontal plane.
        The angle represents the vertical position of the left shoulder relative
        to the upper thoracic spine, indicating shoulder elevation (shrugging) or
        depression (dropping).
        Interpretation
        --------------
        - **Positive (+)**: Shoulder elevation (elevazione della spalla)
          The shoulder is elevated (shrugged upward).
          Common in upper trapezius tension, stress postures.
        - **Negative (-)**: Shoulder depression (depressione della spalla)
          The shoulder is depressed (pulled downward).
          Less common in static postures.
        - **0°**: Neutral position (shoulder aligned with thoracic reference)
        Calculation Method
        ------------------
        Uses neck reference frame with:
        - Origin: neck_base = midpoint(C7, sternoclavicular_junction)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (C7 → sternoclavicular_junction)
        - lateral_axis: LEFT (cross product vertical × anteroposterior, Gram-Schmidt)
        The left shoulder position is transformed to the neck reference frame and
        projected onto the frontal plane (defined by lateral_axis and vertical_axis).
        The calculation uses self.lateral_axis and self.vertical_axis to identify
        rotation matrix components. Positive values indicate elevation (shoulder up),
        negative values indicate depression (shoulder down).
        Returns
        -------
        Signal1D
            Shoulder elevation/depression angle in degrees.
            Positive = elevation (shoulder up)
            Negative = depression (shoulder down)
        See Also
        --------
        right_shoulder_elevationdepression : Right shoulder elevation/depression
        left_scapular_protractionretraction : Left scapular anterior/posterior position
        neck_referenceframe : Neck reference frame used for this calculation
        """
        shoulder = self.left_shoulder
        neck_base = self.neck_base
        rmat = self.neck_referenceframe.rotation_matrix
        # Calculate angle of left shoulder in neck reference frame's frontal plane
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            shoulder,
            neck_base,
            rmat,
            self.lateral_axis,
            self.vertical_axis,
        )
        # In neck frame, left shoulder has:
        # - lateral_axis > 0 (shoulder is left, +lateral_axis points LEFT)
        # - vertical_axis > 0 when elevated (shoulder is above, +vertical_axis points UP)
        # arctan2 of vertical and lateral components gives elevation/depression angle
        # For symmetric sign convention: positive = elevation
        angle_result = angle.to_numpy()
        return Signal1D(data=angle_result, index=shoulder.index, unit="°")

    @property
    def left_shoulder_flexionextension(self):
        """
        Calculate left shoulder flexion/extension angle in sagittal plane.
        The angle represents the forward (flexion) or backward (extension)
        movement of the arm relative to the shoulder in the sagittal plane.
        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The arm is raised forward (anteriorly).
          Common in reaching forward, throwing preparation.
        - **Negative (-)**: Extension (estensione)
          The arm is moved backward (posteriorly).
          Common in backswing, reaching behind.
        - **0°**: Neutral position (arm hanging at side)
        Calculation Method
        ------------------
        Uses left shoulder reference frame with:
        - Origin: Left shoulder joint center
        - lateral_axis: LEFT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The elbow position is transformed to the shoulder reference frame and
        projected onto the sagittal plane (defined by anteroposterior_axis and
        vertical_axis).
        The calculation uses self.anteroposterior_axis and self.vertical_axis to
        identify rotation matrix components. The result is normalized to [0°, 360°]
        with zero corresponding to the arm hanging vertically downward.
        Returns
        -------
        Signal1D
            Shoulder flexion/extension angle in degrees.
            Positive = flexion (arm forward)
            Negative = extension (arm backward)
        See Also
        --------
        right_shoulder_flexionextension : Right shoulder flexion angle
        left_shoulder_abductionadduction : Left shoulder frontal plane motion
        left_elbow_flexionextension : Left elbow flexion angle
        """
        try:
            # ottengo i parametri necessari
            shoulder = self.left_shoulder
            rmat = self.left_shoulder_referenceframe.rotation_matrix
            elbow = self.left_elbow
            # calcolo l'orientamento del braccio rispetto al sistema di riferimento
            # della spalla
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                elbow,
                shoulder,
                rmat,
                self.anteroposterior_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            # Transform to anatomical convention: 0° = neutral (arm hanging vertically)
            # The reference frame Y-axis points upward
            # For vertical arm (elbow below shoulder), arctan2 gives ~-90°
            # Add 90° to make neutral position = 0°, then normalize to [0°, 360°]
            angle_result = (angle.to_numpy() + 90) % 360
            return Signal1D(data=angle_result, index=angle.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_shoulder_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def left_shoulder_internalexternalrotation(self):
        """
        Calculate left shoulder internal/external rotation angle in transverse plane.
        The angle represents the rotational orientation of the upper arm around
        its longitudinal axis, measured by the orientation of the elbow reference
        frame's anteroposterior axis relative to the shoulder reference frame.
        Interpretation
        --------------
        - **Positive (+)**: Internal rotation (rotazione interna)
          The forearm rotates medially (toward the body).
          Common in throwing acceleration, reaching across body.
        - **Negative (-)**: External rotation (rotazione esterna)
          The forearm rotates laterally (away from the body).
          Common in throwing cocking phase, backhand motions.
        - **0°**: Neutral position (elbow anteroposterior axis parallel to shoulder anteroposterior axis)
        Calculation Method
        ------------------
        Uses left shoulder reference frame with:
        - Origin: Left shoulder joint center
        - lateral_axis: LEFT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The elbow reference frame's anteroposterior_axis (column 2 of its rotation
        matrix) is extracted and added to the elbow position to create a point in
        global space. This point is then transformed to the shoulder reference frame
        and projected onto the transverse plane (defined by lateral_axis and
        anteroposterior_axis).
        The calculation uses self.lateral_axis and self.anteroposterior_axis to
        identify rotation matrix components. The angle is calculated and adjusted
        so that positive values indicate internal rotation (forearm inward) and
        negative values indicate external rotation (forearm outward).
        Returns
        -------
        Signal1D
            Shoulder rotation angle in degrees.
            Positive = internal rotation (forearm inward)
            Negative = external rotation (forearm outward)
        See Also
        --------
        right_shoulder_internalexternalrotation : Right shoulder rotation angle
        left_shoulder_referenceframe : Shoulder reference frame used for calculation
        left_elbow_referenceframe : Elbow reference frame used for calculation
        left_shoulder_abductionadduction : Left shoulder frontal plane motion
        """
        # Get reference frames
        shoulder_rf = self.left_shoulder_referenceframe
        elbow_rf = self.left_elbow_referenceframe
        # Get elbow anteroposterior axis from rotation matrix (column 2)
        elbow_ap_axis = elbow_rf.rotation_matrix[:, :, 2]  # shape (N, 3)
        # Create a point at elbow origin + anteroposterior axis direction
        elbow = self.left_elbow
        elbow_ap_global = Point3D(
            data=elbow_rf.origin + elbow_ap_axis,
            index=elbow.index,
            columns=elbow.columns,
        )
        # Calculate the angle in shoulder reference frame's transverse plane
        shoulder = self.left_shoulder
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow_ap_global,
            shoulder,
            shoulder_rf.rotation_matrix,
            self.lateral_axis,
            self.anteroposterior_axis,
        )
        # Correct the angle (positive means internal rotation, negative external)
        angle = 90 - angle.to_numpy()
        return Signal1D(data=angle, index=elbow.index, unit="°")

    @property
    def right_shoulder_abductionadduction(self):
        """
        Calculate right shoulder abduction/adduction angle in frontal plane.
        The angle represents the lateral (outward) or medial (inward)
        movement of the arm relative to the shoulder in the frontal plane.
        Interpretation
        --------------
        - **Positive (+)**: Abduction (abduzione)
          The arm is raised laterally away from the body.
          Common in lateral raises, overhead reaching.
        - **Negative (-)**: Adduction (adduzione)
          The arm is moved medially toward or across the body.
          Common in cross-body movements.
        - **0°**: Neutral position (arm hanging at side)
        Calculation Method
        ------------------
        Uses right shoulder reference frame with:
        - Origin: Right shoulder joint center
        - lateral_axis: RIGHT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)
        The elbow position is transformed to the shoulder reference frame and
        projected onto the frontal plane (defined by lateral_axis and vertical_axis).
        The calculation uses self.lateral_axis and self.vertical_axis to identify
        rotation matrix components. The result is normalized to [0°, 360°] with
        zero corresponding to the arm hanging vertically downward.
        Returns
        -------
        Signal1D
            Shoulder abduction/adduction angle in degrees.
            Positive = abduction (arm outward)
            Negative = adduction (arm inward)
        See Also
        --------
        left_shoulder_abductionadduction : Left shoulder frontal plane motion
        right_shoulder_flexionextension : Right shoulder sagittal plane motion
        right_scapular_protractionretraction : Right scapular position
        """
        try:
            # Get necessary parameters
            shoulder = self.right_shoulder
            rmat = self.right_shoulder_referenceframe.rotation_matrix
            elbow = self.right_elbow
            # Calculate arm orientation with respect to shoulder reference frame
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                elbow,
                shoulder,
                rmat,
                self.lateral_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            # Transform to anatomical convention: 0° = neutral (arm hanging vertically)
            # arctan2 gives ~-90° for vertical downward vector
            # Add 90° to make neutral = 0°, then normalize to [0°, 360°]
            result = (angle.to_numpy() + 90) % 360
            return Signal1D(data=result, index=angle.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_shoulder_abductionadduction: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_shoulder_elevationdepression(self):
        """
        Calculate right shoulder elevation/depression angle in frontal plane.
        The angle represents the vertical position of the right shoulder relative
        to the upper thoracic spine, indicating shoulder elevation (shrugging) or
        depression (dropping).
        Interpretation
        --------------
        - **Positive (+)**: Shoulder elevation (elevazione della spalla)
          The shoulder is elevated (shrugged upward).
          Common in upper trapezius tension, stress postures.
        - **Negative (-)**: Shoulder depression (depressione della spalla)
          The shoulder is depressed (pulled downward).
          Less common in static postures.
        - **0°**: Neutral position (shoulder aligned with thoracic reference)
        Calculation Method
        ------------------
        Uses neck reference frame with:
        - Origin: neck_base = midpoint(C7, sternoclavicular_junction)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (C7 → sternoclavicular_junction)
        - lateral_axis: LEFT (cross product vertical × anteroposterior, Gram-Schmidt)
        The right shoulder position is transformed to the neck reference frame and
        projected onto the frontal plane (defined by lateral_axis and vertical_axis).
        The calculation uses self.lateral_axis and self.vertical_axis to identify
        rotation matrix components. The lateral component is negated for the right
        side to maintain consistent sign convention: positive = elevation (shoulder up),
        negative = depression (shoulder down).
        Returns
        -------
        Signal1D
            Shoulder elevation/depression angle in degrees.
            Positive = elevation (shoulder up)
            Negative = depression (shoulder down)
        See Also
        --------
        left_shoulder_elevationdepression : Left shoulder elevation/depression
        right_scapular_protractionretraction : Right scapular anterior/posterior position
        neck_referenceframe : Neck reference frame used for this calculation
        """
        shoulder = self.right_shoulder
        neck_base = self.neck_base
        rmat = self.neck_referenceframe.rotation_matrix
        # Calculate angle of right shoulder in neck reference frame's frontal plane
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            shoulder,
            neck_base,
            rmat,
            self.lateral_axis,
            self.vertical_axis,
        )
        # In neck frame, right shoulder has:
        # - lateral_axis < 0 (shoulder is right, opposite to +lateral_axis which points LEFT)
        # - vertical_axis > 0 when elevated (shoulder is above, +vertical_axis points UP)
        # arctan2 with lateral < 0 gives angle in wrong quadrant
        # Recalculate angle using negated lateral component to match left side convention
        angle_result = np.degrees(np.arctan2(y, -x))
        return Signal1D(data=angle_result, index=shoulder.index, unit="°")

    @property
    def right_shoulder_flexionextension(self):
        """
        Calculate right shoulder flexion/extension angle in sagittal plane.
        The angle represents the forward (flexion) or backward (extension)
        movement of the arm relative to the shoulder in the sagittal plane.
        Interpretation
        --------------
        - **Positive (+)**: Flexion (flessione)
          The arm is raised forward (anteriorly).
          Common in reaching forward, throwing preparation.
        - **Negative (-)**: Extension (estensione)
          The arm is moved backward (posteriorly).
          Common in backswing, reaching behind.
        - **0°**: Neutral position (arm hanging at side)
        Calculation Method
        ------------------
        Uses right shoulder reference frame with:
        - Origin: Right shoulder joint center
        - lateral_axis: RIGHT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)
        The elbow position is transformed to the shoulder reference frame and
        projected onto the sagittal plane (defined by anteroposterior_axis and
        vertical_axis).
        The calculation uses self.anteroposterior_axis and self.vertical_axis to
        identify rotation matrix components. The result is normalized to [0°, 360°]
        with zero corresponding to the arm hanging vertically downward.
        Returns
        -------
        Signal1D
            Shoulder flexion/extension angle in degrees.
            Positive = flexion (arm forward)
            Negative = extension (arm backward)
        See Also
        --------
        left_shoulder_flexionextension : Left shoulder flexion angle
        right_shoulder_abductionadduction : Right shoulder frontal plane motion
        right_elbow_flexionextension : Right elbow flexion angle
        """
        try:
            # ottengo i parametri necessari
            shoulder = self.right_shoulder
            rmat = self.right_shoulder_referenceframe.rotation_matrix
            elbow = self.right_elbow
            # calcolo l'orientamento del braccio rispetto al sistema di riferimento
            # della spalla
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                elbow,
                shoulder,
                rmat,
                self.anteroposterior_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            # Transform to anatomical convention: 0° = neutral (arm hanging vertically)
            # The reference frame Y-axis points upward
            # For vertical arm (elbow below shoulder), arctan2 gives ~-90°
            # Add 90° to make neutral position = 0°, then normalize to [0°, 360°]
            angle_result = (angle.to_numpy() + 90) % 360
            return Signal1D(data=angle_result, index=angle.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_shoulder_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_shoulder_internalexternalrotation(self):
        """
        Calculate right shoulder internal/external rotation angle in transverse plane.
        The angle represents the rotational orientation of the upper arm around
        its longitudinal axis, measured by the orientation of the elbow reference
        frame's anteroposterior axis relative to the shoulder reference frame.
        Interpretation
        --------------
        - **Positive (+)**: Internal rotation (rotazione interna)
          The forearm rotates medially (toward the body).
          Common in throwing acceleration, reaching across body.
        - **Negative (-)**: External rotation (rotazione esterna)
          The forearm rotates laterally (away from the body).
          Common in throwing cocking phase, backhand motions.
        - **0°**: Neutral position (elbow anteroposterior axis parallel to shoulder anteroposterior axis)
        Calculation Method
        ------------------
        Uses right shoulder reference frame with:
        - Origin: Right shoulder joint center
        - lateral_axis: RIGHT (neck_base → shoulder, pointing outward)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)
        The elbow reference frame's anteroposterior_axis (column 2 of its rotation
        matrix) is extracted and added to the elbow position to create a point in
        global space. This point is then transformed to the shoulder reference frame
        and projected onto the transverse plane (defined by lateral_axis and
        anteroposterior_axis).
        The calculation uses self.lateral_axis and self.anteroposterior_axis to
        identify rotation matrix components. The angle is calculated and adjusted
        so that positive values indicate internal rotation (forearm inward) and
        negative values indicate external rotation (forearm outward).
        Returns
        -------
        Signal1D
            Shoulder rotation angle in degrees.
            Positive = internal rotation (forearm inward)
            Negative = external rotation (forearm outward)
        See Also
        --------
        left_shoulder_internalexternalrotation : Left shoulder rotation angle
        right_shoulder_referenceframe : Shoulder reference frame used for calculation
        right_elbow_referenceframe : Elbow reference frame used for calculation
        right_shoulder_abductionadduction : Right shoulder frontal plane motion
        """
        # Get reference frames
        shoulder_rf = self.right_shoulder_referenceframe
        elbow_rf = self.right_elbow_referenceframe
        # Get elbow anteroposterior axis from rotation matrix (column 2)
        elbow_ap_axis = elbow_rf.rotation_matrix[:, :, 2]  # shape (N, 3)
        # Create a point at elbow origin + anteroposterior axis direction
        elbow = self.right_elbow
        elbow_ap_global = Point3D(
            data=elbow_rf.origin + elbow_ap_axis,
            index=elbow.index,
            columns=elbow.columns,
        )
        # Calculate the angle in shoulder reference frame's transverse plane
        shoulder = self.right_shoulder
        angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
            elbow_ap_global,
            shoulder,
            shoulder_rf.rotation_matrix,
            self.lateral_axis,
            self.anteroposterior_axis,
        )
        # Correct the angle (positive means internal rotation, negative external)
        # Right shoulder RF has negated anteroposterior_axis, so negate the angle to match left side
        angle = angle.to_numpy() - 90
        return Signal1D(data=angle, index=elbow.index, unit="°")
