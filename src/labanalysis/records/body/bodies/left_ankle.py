"""left ankle joint module"""

from ....timeseries import Point3D
from .joint import Joint
from .left_foot import LeftFoot

__all__ = ["LeftAnkle"]


class LeftAnkle(Joint):
    """LeftAnkle Joint."""

    def __init__(
        self,
        left_ankle_lateral: Point3D,
        left_ankle_medial: Point3D,
        left_knee_lateral: Point3D,
        left_knee_medial: Point3D,
        left_foot: LeftFoot,
    ):

        # object with reference frame
        ori = (left_ankle_lateral + left_ankle_medial) / 2
        lax = left_ankle_medial - left_ankle_lateral
        kne = (left_knee_medial + left_knee_lateral) / 2
        vrt = ori - kne
        super().__init__(
            center=ori, # type: ignore
            lateral_vector=lax,  # type: ignore
            vertical_vector=vrt,  # type: ignore
            anteroposterior_vector=None,
        )  # type: ignore
        self["foot"] = left_foot

    @property
    def _foot(self):
        """return the foot_plane"""
        out: LeftFoot = self["foot"]  # type: ignore
        return out

    @property
    def flexionextension(self):
        """
        Calculate right ankle dorsiflexion/plantarflexion.

        Interpretation
        --------------
        - **Positive (+)**: Dorsiflexion (flessione dorsale)
          The foot is angled upward relative to the shin.
          Common in landing, deceleration, squatting.

        - **Negative (-)**: Plantarflexion (flessione plantare)
          The foot is angled downward relative to the shin.
          Common in toe-off, jumping, pointing.
        - **0°**: Neutral position (foot perpendicular to shin at 90°)

        Returns
        -------
        Signal1D
            Ankle flexion/extension angle in degrees.
            Positive = dorsiflexion (foot up)
            Negative = plantarflexion (foot down)
        """
        proj = self._foot.get_projected_point(self.center)
        return self.get_angle_by_point(
            self.apply(proj),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.anteroposterior_axis,  # type: ignore
        )

    @property
    def inversioneversion(self):
        """
        Calculate left ankle inversion/eversion angle.

        Interpretation
        --------------
        - **Positive (+)**: Eversion (eversione)
          The sole of the foot is tilted outward (away from midline).
          Common in overpronation, pes planus (flat feet).
        - **Negative (-)**: Inversion (inversione)
          The sole of the foot is tilted inward (toward midline).
          Common in supination, ankle sprains (lateral).
        - **0°**: Neutral position (foot aligned with shin in frontal plane)

        Returns
        -------
        Signal1D
            Ankle inversion/eversion angle in degrees.
            Positive = eversion (sole out)
            Negative = inversion (sole in)
        """
        proj = self._foot.get_projected_point(self.center)
        return self.get_angle_by_point(
            self.apply(proj),  # type: ignore
            self.vertical_axis,  # type: ignore
            self.lateral_axis,  # type: ignore
        )
