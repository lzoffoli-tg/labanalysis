"""Auto-generated mixin for WholeBody properties."""

import warnings
from ....timeseries import Signal1D

class NeckAnglesMixin:
    """NeckAngles properties for WholeBody."""

    @property
    def neck_flexionextension(self):
        """
        Calculate neck flexion/extension in sagittal plane of neck reference frame.
        The angle represents the forward or backward deviation of the head
        from vertical position, measured from neck_base to head_center.
        Interpretation
        --------------
        - **Positive (+)**: Forward flexion (flessione in avanti)
          The head moves forward; chin moves toward chest.
          Common in forward head posture, texting posture.
        - **Negative (-)**: Backward extension (estensione indietro)
          The head tilts backward; looking up.
          Common in looking up, backward head tilt.
        - **0°**: Neutral position (head vertical UP from neck_base, at 90° in sagittal plane)
        Calculation Method
        ------------------
        Uses neck reference frame with:
        - Origin: neck_base = midpoint(C7, sternoclavicular_junction)
        - vertical_axis: UP (pelvis_center → neck_base)
        - anteroposterior_axis: FORWARD (C7 → sternoclavicular_junction)
        - lateral_axis: LEFT (cross product vertical × anteroposterior, Gram-Schmidt)
        The head_center position is transformed to the neck reference frame and
        projected onto the sagittal plane (defined by anteroposterior_axis and
        vertical_axis).
        The angle is calculated as:
        - arctan2(vertical_component, anteroposterior_component) - 90°
        - Neutral position (head directly above neck_base) gives ~90° before
          correction, which becomes 0° after subtracting 90°.
        - Positive values indicate forward flexion (chin toward chest).
        Returns
        -------
        Signal1D
            Neck flexion/extension angle in degrees.
            Positive = forward flexion (chin to chest)
            Negative = backward extension (looking up)
        See Also
        --------
        neck_lateralflexion : Neck frontal plane lateral flexion
        trunk_rotation : Trunk transverse plane rotation
        """
        try:
            head = self.head_center
            neck_base = self.neck_base
            rmat = self.neck_referenceframe.rotation_matrix
            # Calculate angle in neck reference frame's sagittal plane
            # Sagittal plane: anteroposterior_axis (Z) and vertical_axis (Y)
            # Using arctan2(y, x) convention: arctan2(vertical, anteroposterior)
            angle, ap_comp, vertical_comp = (
                self._get_angle_by_point_on_reference_frame_and_plane(
                    head,
                    neck_base,
                    rmat,
                    self.anteroposterior_axis,  # First arg (x in arctan2)
                    self.vertical_axis,  # Second arg (y in arctan2)
                )
            )
            # The helper method returns arctan2(second_axis, first_axis)
            # So this gives arctan2(vertical, anteroposterior)
            # At neutral (head directly above): anteroposterior≈0, vertical>0 → angle≈90°
            # Apply zero correction: subtract 90° so that vertical = 0°
            # Positive = flexion (forward), Negative = extension (backward)
            flexionextension = 90 - angle.to_numpy()
            return Signal1D(data=flexionextension, index=head.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate neck_flexionextension: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def neck_lateralflexion(self):
        """
        Calculate neck lateral flexion in frontal plane of neck reference frame.
        The angle represents the lateral deviation of the head from vertical,
        measured relative to the neck reference frame.
        Interpretation
        --------------
        - **Positive (+)**: Right lateral flexion (flessione laterale destra)
          The head tilts toward the right shoulder.
        - **Negative (-)**: Left lateral flexion (flessione laterale sinistra)
          The head tilts toward the left shoulder.
        - **0°**: Neutral position (head centered above neck_base)
        Calculation Method
        ------------------
        Uses the head_center to neck_base vector transformed into the neck
        reference frame and projected onto the frontal plane (defined by
        lateral_axis and vertical_axis of the neck reference frame).
        The angle is calculated as:
        - arctan2(vertical_component, lateral_component) - 90°
        - Neutral position (head directly above neck_base) gives ~90° before
          correction, which becomes 0° after subtracting 90°.
        - Positive values indicate right lateral flexion.
        Returns
        -------
        Signal1D
            Neck lateral flexion angle in degrees.
            Positive = right lateral flexion (destra)
            Negative = left lateral flexion (sinistra)
        See Also
        --------
        neck_flexionextension : Neck sagittal plane flexion/extension
        neck_referenceframe : Reference frame used for this calculation
        """
        try:
            head = self.head_center
            neck_base = self.neck_base
            rmat = self.neck_referenceframe.rotation_matrix
            # Calculate angle in neck reference frame's frontal plane
            # Frontal plane: lateral_axis (X) and vertical_axis (Y)
            # Using arctan2(y, x) convention: arctan2(vertical, lateral)
            angle, lateral_comp, vertical_comp = (
                self._get_angle_by_point_on_reference_frame_and_plane(
                    head,
                    neck_base,
                    rmat,
                    self.lateral_axis,  # First arg (x in arctan2)
                    self.vertical_axis,  # Second arg (y in arctan2)
                )
            )
            # The helper method returns arctan2(second_axis, first_axis)
            # So this gives arctan2(vertical, lateral)
            # At neutral (head directly above): lateral≈0, vertical>0 → angle≈90°
            # Apply zero correction: subtract 90° so that vertical = 0°
            # Positive = right flexion, Negative = left flexion
            lateralflexion = angle.to_numpy() - 90
            return Signal1D(data=lateralflexion, index=head.index, unit="°")
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate neck_lateralflexion: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
