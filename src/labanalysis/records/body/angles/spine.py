"""Auto-generated mixin for WholeBody properties."""

import numpy as np
from ....timeseries import Point3D, Signal1D

class SpineAnglesMixin:
    """SpineAngles properties for WholeBody."""

    @property
    def dorsal_kyphosis(self):
        """
        Calculate thoracic (dorsal) kyphosis (curvatura cifotica toracica).
        The angle quantifies the posterior curvature of the thoracic spine.
        Interpretation
        --------------
        The angle represents the degree of thoracic curvature:
        - **Larger angles (>160°)**: Indicates straighter spine or reduced kyphosis
          (ipocifosi, "flat upper back")
        - **Smaller angles (<140°)**: Indicates increased kyphotic curvature
          (ipercifosi, "rounded back", "hunchback", excessive thoracic curve)
        - **Normal range**: Typically 140-160° (curvatura fisiologica)
        Note: This is the internal angle at T5. A more curved (kyphotic) spine
        produces a smaller angle because the vertebrae form a tighter curve.
        Calculation Method
        ------------------
        Measured as the angle at T5 vertex formed by three points:
        1. L2 (Second Lumbar vertebra)
        2. T5 (Fifth Thoracic vertebra) - vertex
        3. C7 (Seventh Cervical vertebra)
        This spans the thoracic region from lower thoracic (near lumbar junction)
        to upper thoracic (near cervical junction).
        Returns
        -------
        Signal1D
            Thoracic kyphosis angle in degrees.
            Smaller angle = greater kyphotic curvature (more curved/rounded)
            Larger angle = reduced kyphosis (flatter upper back)
        """
        l2 = self._get_point("l2")
        t5 = self._get_point("t5")
        c7 = self._get_point("c7")
        # Calculate 3-point angle: C7 - T5 - L2
        # Internal angle at T5 vertex (order: superior → vertex → inferior)
        angle = self._get_angle_between_three_points(c7, t5, l2)
        return Signal1D(data=angle, index=t5.index, unit="°")

    @property
    def lumbar_lordosis(self):
        """
        Calculate lumbar lordosis (curvatura lordotica lombare).
        The angle quantifies the anterior curvature of the lumbar spine.
        Interpretation
        --------------
        The angle represents the degree of lumbar curvature:
        - **Larger angles (>160°)**: Indicates straighter spine or reduced lordosis
          (ipolordosi, "flat back")
        - **Smaller angles (<140°)**: Indicates increased lordotic curvature
          (iperlordosi, "sway back", excessive lumbar curve)
        - **Normal range**: Typically 140-160° (curvatura fisiologica)
        Note: This is the internal angle at L2. A more curved (lordotic) spine
        produces a smaller angle because the vertebrae form a tighter curve.
        Calculation Method
        ------------------
        Measured as the angle at L2 vertex formed by three points:
        1. Midpoint of PSIS (Posterior Superior Iliac Spine) markers
        2. L2 (Second Lumbar vertebra) - vertex
        3. T5 (Fifth Thoracic vertebra)
        This represents the transition from lumbar to thoracic curvature.
        Returns
        -------
        Signal1D
            Lumbar lordosis angle in degrees.
            Smaller angle = greater lordotic curvature (more curved)
            Larger angle = reduced lordosis (flatter)
        """
        l_psis = self._get_point("left_psis")
        r_psis = self._get_point("right_psis")
        l2 = self._get_point("l2")
        t5 = self._get_point("t5")
        # Calculate PSIS midpoint (posterior pelvis reference)
        psis_mid_data = (l_psis.to_numpy() + r_psis.to_numpy()) / 2
        psis_mid_index = np.unique(
            np.concatenate([l_psis.index, r_psis.index])
        ).tolist()
        # Create a Point3D for psis_mid
        psis_mid = Point3D(
            data=psis_mid_data,
            index=psis_mid_index,
            columns=l_psis.columns,
        )
        # Calculate 3-point angle: T5 - L2 - PSIS_mid
        # Internal angle at L2 vertex (order: superior → vertex → inferior)
        angle = self._get_angle_between_three_points(t5, l2, psis_mid)
        return Signal1D(data=angle, index=l2.index, unit="°")
