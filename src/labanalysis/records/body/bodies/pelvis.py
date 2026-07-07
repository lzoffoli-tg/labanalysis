"""left hip joint module"""

from ....timeseries import Point3D
from .body_plane import BodyPlane
from .joint import Joint
from .segment import Segment

__all__ = ["Pelvis"]


class Pelvis(Joint, BodyPlane, Segment):
    """Pelvis"""

    def __init__(
        self,
        left_asis: Point3D,
        right_asis: Point3D,
        left_psis: Point3D,
        right_psis: Point3D,
    ):

        # get the projected pelvis points to reduce positioning errors
        plane = BodyPlane(
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
        )
        lasis = plane.get_projected_point(left_asis)
        rasis = plane.get_projected_point(right_asis)
        lpsis = plane.get_projected_point(left_psis)
        rpsis = plane.get_projected_point(right_psis)

        # setup the object reference frame and extract the versors
        L = (lasis + lpsis) / 2
        F = (lasis + rasis) / 2
        R = (rasis + rpsis) / 2
        B = (lpsis + rpsis) / 2
        ML: Point3D = L - R  # type: ignore
        AP: Point3D = F - B  # type: ignore
        O: Point3D = (lasis + lpsis + rasis + rpsis) / 4  # type: ignore

        super().__init__(
            center=O,
            lateral_vector=ML,
            anteroposterior_vector=AP,
            vertical_vector=None,
        )
        self["left_asis"] = left_asis
        self["right_asis"] = right_asis
        self["left_psis"] = left_psis
        self["right_psis"] = right_psis

    @property
    def left_midpoint(self):
        """return the mid point between left asis and psis"""
        out: Point3D = (self.left_asis + self.left_psis) * 0.5  # type: ignore
        return out

    @property
    def right_midpoint(self):
        """return the mid point between right asis and psis"""
        out: Point3D = (self.right_psis + self.right_asis) * 0.5  # type: ignore
        return out

    @property
    def asis_midpoint(self):
        """return the ASIS mid point"""
        out: Point3D = (self.left_asis + self.right_asis) * 0.5  # type: ignore
        return out

    @property
    def psis_midpoint(self):
        """return the PSIS mid point"""
        out: Point3D = (self.left_psis + self.right_psis) * 0.5  # type: ignore
        return out

    @property
    def pelvis_midpoint(self):
        """return the pelvis mid point"""
        out: Point3D = (self.left_asis + self.right_asis + self.left_psis + self.right_psis) * 0.25  # type: ignore
        return out

    @property
    def left_asis(self):
        """return the left asis"""
        out: Point3D = self["left_asis"]  # type: ignore
        return out

    @property
    def right_asis(self):
        """return the right asis"""
        out: Point3D = self["right_asis"]  # type: ignore
        return out

    @property
    def right_psis(self):
        """return the right psis"""
        out: Point3D = self["right_psis"]  # type: ignore
        return out

    @property
    def left_psis(self):
        """return the left psis"""
        out: Point3D = self["left_psis"]  # type: ignore
        return out

    @property
    def left_hip_approximated(self):
        """
        Left hip joint center.

        Estimated using Bell & Brand (1989/1990) regression equations based on
        pelvis dimensions. The hip joint center is offset from the
        trochanter according to pelvis width and height.

        Returns
        -------
        Point3D
            Hip joint center position.

        References
        ----------
        Bell, A. L., Pedersen, D. R., & Brand, R. A. (1989).
            Prediction of hip joint centre location from external landmarks.
            Human Movement Science, 8(1), 3–16.
            https://doi.org/10.1016/0167-9457(89)90020-1

        Bell, A. L., Pedersen, D. R., & Brand, R. A. (1990).
            A comparison of the accuracy of several hip center location
            prediction methods. Journal of Biomechanics, 23(6), 617–621.
            https://doi.org/10.1016/0021-9290(90)90054-7
        """

        # get the joint coordinates in local reference frame
        Og = self.asis_midpoint
        W = self.width
        Hl: Point3D = Og.copy() * 0  # type: ignore
        Hl.loc[Hl.index, Hl.lateral_axis] = 0.36 * W
        Hl.loc[Hl.index, Hl.vertical_axis] = -0.30 * W
        Hl.loc[Hl.index, Hl.anteroposterior_axis] = -0.19 * W

        # return the hip joint the global reference frame
        Hg: Point3D = self.apply_inverse(Hl)  # type: ignore
        return Hg

    @property
    def right_hip_approximated(self):
        """
        Right hip joint center.

        Estimated using Bell & Brand (1989/1990) regression equations based on
        pelvis dimensions. The hip joint center is offset from the
        trochanter according to pelvis width and height.

        Returns
        -------
        Point3D
            Hip joint center position.

        References
        ----------
        Bell, A. L., Pedersen, D. R., & Brand, R. A. (1989).
            Prediction of hip joint centre location from external landmarks.
            Human Movement Science, 8(1), 3–16.
            https://doi.org/10.1016/0167-9457(89)90020-1

        Bell, A. L., Pedersen, D. R., & Brand, R. A. (1990).
            A comparison of the accuracy of several hip center location
            prediction methods. Journal of Biomechanics, 23(6), 617–621.
            https://doi.org/10.1016/0021-9290(90)90054-7
        """

        # get the joint coordinates in local reference frame
        Og = self.asis_midpoint
        W = self.width
        Hl: Point3D = Og.copy() * 0  # type: ignore
        Hl.loc[Hl.index, Hl.lateral_axis] = -0.36 * W
        Hl.loc[Hl.index, Hl.vertical_axis] = -0.30 * W
        Hl.loc[Hl.index, Hl.anteroposterior_axis] = -0.19 * W

        # return the hip joint the global reference frame
        Hg: Point3D = self.apply_inverse(Hl)  # type: ignore
        return Hg

    @property
    def width(self):
        """return the asis width"""
        return float((((self.left_asis - self.right_asis) ** 2).sum(axis=1) ** 0.5).mean())  # type: ignore

    @property
    def depth(self):
        """return the pelvis depth"""
        return float((((self.asis_midpoint - self.psis_midpoint) ** 2).sum(axis=1) ** 0.5).mean())  # type: ignore

    @property
    def sagittal_plane_tilt(self):
        """
        Calculate pelvis tilt in the sagittal global frame coordinate system

        Interpretation
        --------------
        The angle represents the incline of the pelvis plane in lateral view.

        Returns
        -------
        Signal1D
            pelvis sagittal tilt angle in degrees.
        """
        return self.get_angle_by_point(
            self.asis_midpoint - self.psis_midpoint,  # type: ignore
            self.anteriorposterior_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

    @property
    def frontal_plane_tilt(self):
        """
        Calculate pelvis tilt in the frontal global frame coordinate system

        Interpretation
        --------------
        The angle represents the incline of the pelvis plane in frontal view.

        Returns
        -------
        Signal1D
            pelvis frontal tilt angle in degrees.
        """
        return self.get_angle_by_point(
            self.left_midpoint - self.right_midpoint,  # type: ignore
            self.lateral_axis,  # type: ignore
            self.vertical_axis,  # type: ignore
        )

    @property
    def transverse_plane_tilt(self):
        """
        Calculate pelvis tilt in the transverse global frame coordinate system

        Interpretation
        --------------
        The angle represents the incline of the pelvis plane in transverse view.

        Returns
        -------
        Signal1D
            pelvis transverse tilt angle in degrees.
        """
        return self.get_angle_by_point(
            self.left_midpoint - self.right_midpoint,  # type: ignore
            self.lateral_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )
