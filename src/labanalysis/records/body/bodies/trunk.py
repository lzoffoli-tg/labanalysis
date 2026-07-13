"""trunk joint module"""

import numpy as np

from ....timeseries import Point3D, Signal1D, Signal3D
from .joint import Joint
from .left_hip import LeftHip
from .pelvis import Pelvis
from .right_hip import RightHip
from .segment import Segment

__all__ = ["Trunk"]


class Trunk(Joint, Segment):
    """Trunk Joint."""

    def __init__(
        self,
        c7: Point3D,
        sc: Point3D,
        l2: Point3D,
        t5: Point3D,
        pelvis: Pelvis,
        left_hip: LeftHip,
        right_hip: RightHip,
    ):

        # get the reference frame
        ML = left_hip.center - right_hip.center
        H = (left_hip.center + right_hip.center) / 2
        VT = pelvis.center - H

        # build the class
        super().__init__(
            pelvis.center,  # type: ignore
            ML,
            VT,
        )

        # upper markers
        self["c7"] = c7
        self["sc"] = sc
        self["l2"] = l2
        self["t5"] = t5
        self["pelvis"] = pelvis

    def _get_angle_between_three_points(
        self,
        p1: Point3D,
        p2: Point3D,
        p3: Point3D,
    ):
        """
        Calculate angle formed by three 3D points with vertex at the middle point.

        Computes the angle at point p2 formed by the segments p1-p2 and p2-p3
        using the law of cosines.

        Parameters
        ----------
        p1 : Point3D or np.ndarray
            First point (shape (N, 3)).
        p2 : Point3D or np.ndarray
            Vertex point (shape (N, 3)).
        p3 : Point3D or np.ndarray
            Third point (shape (N, 3)).

        Returns
        -------
        np.ndarray
            Angles in degrees at each time instant.

        Notes
        -----
        Uses law of cosines: cos(θ) = (AB² + BC² - AC²) / (2·AB·BC)
        where AB, BC, AC are segment lengths.
        """

        # Get segment lengths
        v1 = p1.to_numpy() if isinstance(p1, Point3D) else p1
        v2 = p2.to_numpy() if isinstance(p2, Point3D) else p2
        v3 = p3.to_numpy() if isinstance(p3, Point3D) else p3
        AB = ((v1 - v2) ** 2).sum(axis=1) ** 0.5
        BC = ((v2 - v3) ** 2).sum(axis=1) ** 0.5
        AC = ((v1 - v3) ** 2).sum(axis=1) ** 0.5

        cos_angle = np.clip((AC**2 - AB**2 - BC**2) / (-2 * AB * BC), -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return Signal1D(
            np.asarray(angle_deg, float),
            p1.index,
            "deg",
        )

    @property
    def _c7(self):
        """return c7 in the global reference frame"""
        out: Point3D = self["c7"]  # type: ignore
        return out

    @property
    def _sc(self):
        """return sc in the global reference frame"""
        out: Point3D = self["sc"]  # type: ignore
        return out

    @property
    def _l2(self):
        """return l2 in the global reference frame"""
        out: Point3D = self["l2"]  # type: ignore
        return out

    @property
    def _t5(self):
        """return t5 in the global reference frame"""
        out: Point3D = self["t5"]  # type: ignore
        return out

    @property
    def _pelvis(self):
        """return pelvis in the global reference frame"""
        out: Pelvis = self["pelvis"]  # type: ignore
        return out

    @property
    def _neck_base(self):
        """return neck base in the global reference frame"""
        out: Point3D = (self._c7 + self._sc) / 2  # type: ignore
        return out

    @property
    def flexionextension(self):
        """
        Calculate hip flexion-extension angle.

        Interpretation
        --------------
        Positive = flexion
        Negative = extension

        Returns
        -------
        Signal1D
            hip flexion/extension angle in degrees.
        """
        return self.get_angle_by_point(
            self.apply(self._neck_base),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.anteriorposterior_axis,  # type: ignore
        )

    @property
    def lateralflexion(self):
        """
        Calculate trunk adduction/abduction

        Interpretation
        --------------
        Positive = left
        Negative = right

        Returns
        -------
        Signal1D
            trunk lateral flexion angle in degrees.
        """
        return self.get_angle_by_point(
            self.apply(self._neck_base),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.lateral_axis,  # type: ignore
        )

    @property
    def rotation(self):
        """
        Calculate trunk rotation

        Interpretation
        --------------
        Positive = left
        Negative = right

        Returns
        -------
        Signal1D
            trunk rotation angle in degrees.
        """
        vec: Signal3D = self._sc - self._c7  # type: ignore
        vec.loc[vec.index, vec.vertical_axis] = 0
        return self.get_angle_by_point(
            self.apply(vec),  # type: ignore
            self.anteroposterior_axis,  # type: ignore
            self.lateral_axis,  # type: ignore
        )

    @property
    def sagittal_plane_tilt(self):
        """
        Calculate trunk tilt in the sagittal global frame coordinate system

        Interpretation
        --------------
        The angle represents the incline of the trunk plane in lateral view.

        Returns
        -------
        Signal1D
            trunk sagittal tilt angle in degrees.
        """
        return self.get_angle_by_point(
            self._neck_base - self.center,  # type: ignore
            self.anteriorposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

    @property
    def frontal_plane_tilt(self):
        """
        Calculate trunk tilt in the frontal global frame coordinate system

        Interpretation
        --------------
        The angle represents the incline of the trunk plane in frontal view.

        Returns
        -------
        Signal1D
            trunk frontal tilt angle in degrees.
        """
        return self.get_angle_by_point(
            self._neck_base - self.center,  # type: ignore
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

    @property
    def transverse_plane_tilt(self):
        """
        Calculate trunk tilt in the transverse global frame coordinate system

        Interpretation
        --------------
        The angle represents the incline of the trunk plane in transverse view.

        Returns
        -------
        Signal1D
            trunk transverse tilt angle in degrees.
        """
        return self.get_angle_by_point(
            self["c7"] - self["sc"],  # type: ignore
            self.lateral_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )
        return angle

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

        # Calculate 3-point angle: C7 - T5 - L2
        # Internal angle at T5 vertex (order: superior → vertex → inferior)
        return self._get_angle_between_three_points(
            self._c7,
            self._t5,
            self._l2,
        )

    @property
    def lumbar_lordosis(self):
        """
        Calculate lumbar lordosis as the angle defining the curvature of the
        lumbar spine.

        Returns
        -------
        Signal1D
            Lumbar lordosis angle in degrees.
            Smaller angle = greater lordotic curvature (more curved)
            Larger angle = reduced lordosis (flatter)
        """
        return self._get_angle_between_three_points(
            self._t5,
            self._l2,
            self._pelvis.psis_midpoint,
        )

    @property
    def length(self):
        """
        Calculate the length of the trunk.

        Returns
        -------
        Signal1D
            Trunk length in meters.
        """
        return self._get_distance(
            self._neck_base,
            self._pelvis.center,
        )
