"""Auto-generated mixin for WholeBody properties."""

import warnings

class AnkleAnglesMixin:
    """AnkleAngles properties for WholeBody."""

    @property
    def left_ankle_flexionextension(self):
        """
        Calculate left ankle dorsiflexion/plantarflexion angle in sagittal plane.
        The angle represents the orientation of the foot relative to the shank,
        indicating dorsiflexion (toe up) or plantarflexion (toe down).
        Interpretation
        --------------
        - **Positive (+)**: Dorsiflexion (flessione dorsale)
          The foot is angled upward relative to the shin.
          Common in landing, deceleration, squatting.
        - **Negative (-)**: Plantarflexion (flessione plantare)
          The foot is angled downward relative to the shin.
          Common in toe-off, jumping, pointing.
        - **0°**: Neutral position (foot perpendicular to shin at 90°)
        Calculation Method
        ------------------
        Uses left ankle reference frame with:
        - Origin: Left ankle center (midpoint of lateral and medial ankle markers)
        - lateral_axis: LEFT (ankle_lateral → ankle_medial)
        - vertical_axis: UP (ankle → knee)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The foot plane (defined by heel, toe, and metatarsal markers) is projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is measured from the projected foot orientation to the vertical axis.
        The calculation uses self.anteroposterior_axis and self.vertical_axis to
        identify the appropriate rotation matrix components for the sagittal plane
        projection.
        Returns
        -------
        Signal1D
            Ankle flexion/extension angle in degrees.
            Positive = dorsiflexion (foot up)
            Negative = plantarflexion (foot down)
        See Also
        --------
        right_ankle_flexionextension : Right ankle dorsiflexion/plantarflexion
        left_ankle_inversioneversion : Left ankle frontal plane motion
        left_knee_flexionextension : Left knee flexion angle
        """
        try:
            # get points and reference frame
            ankle = self.left_ankle
            rmat = self.left_ankle_referenceframe.rotation_matrix
            proj = self._get_projection_point_on_plane(
                ankle,
                self._left_foot_plane,
            )
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                proj,
                ankle,
                rmat,
                self.anteroposterior_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            # adjust the signs
            return angle + 90
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_ankle_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def left_ankle_inversioneversion(self):
        """
        Calculate left ankle inversion/eversion angle in frontal plane.
        The angle represents the tilting of the foot relative to the shank
        in the frontal (coronal) plane, indicating inversion (sole inward)
        or eversion (sole outward).
        Interpretation
        --------------
        - **Positive (+)**: Eversion (eversione)
          The sole of the foot is tilted outward (away from midline).
          Common in overpronation, pes planus (flat feet).
        - **Negative (-)**: Inversion (inversione)
          The sole of the foot is tilted inward (toward midline).
          Common in supination, ankle sprains (lateral).
        - **0°**: Neutral position (foot aligned with shin in frontal plane)
        Calculation Method
        ------------------
        Uses left ankle reference frame with:
        - Origin: Left ankle center (midpoint of lateral and medial ankle markers)
        - lateral_axis: LEFT (ankle_lateral → ankle_medial)
        - vertical_axis: UP (ankle → knee)
        - anteroposterior_axis: FORWARD (cross product, Gram-Schmidt)
        The foot plane (defined by heel, toe, and metatarsal markers) is projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is measured from the projected foot orientation in the frontal plane.
        The calculation uses self.lateral_axis and self.vertical_axis to identify
        the appropriate rotation matrix components for the frontal plane projection.
        Returns
        -------
        Signal1D
            Ankle inversion/eversion angle in degrees.
            Positive = eversion (sole out)
            Negative = inversion (sole in)
        See Also
        --------
        right_ankle_inversioneversion : Right ankle frontal plane motion
        left_ankle_flexionextension : Left ankle sagittal plane motion
        left_knee_varusvalgus : Left knee frontal plane alignment
        """
        try:
            ankle = self.left_ankle
            rmat = self.left_ankle_referenceframe.rotation_matrix
            proj = self._get_projection_point_on_plane(
                ankle,
                self._left_foot_plane,
            )
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                proj,
                ankle,
                rmat,
                self.lateral_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            return angle + 90
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_ankle_inversioneversion: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_ankle_flexionextension(self):
        """
        Calculate right ankle dorsiflexion/plantarflexion angle in sagittal plane.
        The angle represents the orientation of the foot relative to the shank,
        indicating dorsiflexion (toe up) or plantarflexion (toe down).
        Interpretation
        --------------
        - **Positive (+)**: Dorsiflexion (flessione dorsale)
          The foot is angled upward relative to the shin.
          Common in landing, deceleration, squatting.
        - **Negative (-)**: Plantarflexion (flessione plantare)
          The foot is angled downward relative to the shin.
          Common in toe-off, jumping, pointing.
        - **0°**: Neutral position (foot perpendicular to shin at 90°)
        Calculation Method
        ------------------
        Uses right ankle reference frame with:
        - Origin: Right ankle center (midpoint of lateral and medial ankle markers)
        - lateral_axis: RIGHT (ankle_lateral → ankle_medial)
        - vertical_axis: UP (ankle → knee)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)
        The foot plane (defined by heel, toe, and metatarsal markers) is projected
        onto the sagittal plane (defined by anteroposterior_axis and vertical_axis).
        The angle is measured from the projected foot orientation to the vertical axis.
        The calculation uses self.anteroposterior_axis and self.vertical_axis to
        identify the appropriate rotation matrix components for the sagittal plane
        projection.
        Returns
        -------
        Signal1D
            Ankle flexion/extension angle in degrees.
            Positive = dorsiflexion (foot up)
            Negative = plantarflexion (foot down)
        See Also
        --------
        left_ankle_flexionextension : Left ankle dorsiflexion/plantarflexion
        right_ankle_inversioneversion : Right ankle frontal plane motion
        right_knee_flexionextension : Right knee flexion angle
        """
        try:
            ankle = self.right_ankle
            rmat = self.right_ankle_referenceframe.rotation_matrix
            proj = self._get_projection_point_on_plane(
                ankle,
                self._right_foot_plane,
            )
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                proj,
                ankle,
                rmat,
                self.anteroposterior_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            return angle + 90
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_ankle_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_ankle_inversioneversion(self):
        """
        Calculate right ankle inversion/eversion angle in frontal plane.
        The angle represents the tilting of the foot relative to the shank
        in the frontal (coronal) plane, indicating inversion (sole inward)
        or eversion (sole outward).
        Interpretation
        --------------
        - **Positive (+)**: Eversion (eversione)
          The sole of the foot is tilted outward (away from midline).
          Common in overpronation, pes planus (flat feet).
        - **Negative (-)**: Inversion (inversione)
          The sole of the foot is tilted inward (toward midline).
          Common in supination, ankle sprains (lateral).
        - **0°**: Neutral position (foot aligned with shin in frontal plane)
        Calculation Method
        ------------------
        Uses right ankle reference frame with:
        - Origin: Right ankle center (midpoint of lateral and medial ankle markers)
        - lateral_axis: RIGHT (ankle_lateral → ankle_medial)
        - vertical_axis: UP (ankle → knee)
        - anteroposterior_axis: FORWARD (negated cross product for left-handed frame)
        The foot plane (defined by heel, toe, and metatarsal markers) is projected
        onto the frontal plane (defined by lateral_axis and vertical_axis).
        The angle is measured from the projected foot orientation in the frontal plane.
        The calculation uses self.lateral_axis and self.vertical_axis to identify
        the appropriate rotation matrix components for the frontal plane projection.
        Returns
        -------
        Signal1D
            Ankle inversion/eversion angle in degrees.
            Positive = eversion (sole out)
            Negative = inversion (sole in)
        See Also
        --------
        left_ankle_inversioneversion : Left ankle frontal plane motion
        right_ankle_flexionextension : Right ankle sagittal plane motion
        right_knee_varusvalgus : Right knee frontal plane alignment
        """
        try:
            ankle = self.right_ankle
            rmat = self.right_ankle_referenceframe.rotation_matrix
            proj = self._get_projection_point_on_plane(
                ankle,
                self._right_foot_plane,
            )
            angle, x, y = self._get_angle_by_point_on_reference_frame_and_plane(
                proj,
                ankle,
                rmat,
                self.lateral_axis,  # type: ignore
                self.vertical_axis,  # type: ignore
            )
            return -1 * (90 + angle)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_ankle_inversioneversion: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
