"""repeatedjump module"""

#! IMPORTS

from .bodies import WholeBody
from .records import ForcePlatform
from .timeseries import EMGSignal, Point3D, Signal1D, Signal3D

__all__ = ["UprightPosture", "PronePosture"]


#! CLASSES


class UprightPosture(WholeBody):
    """Represents an upright posture stance."""

    @property
    def side(self):
        """
        Returns which side(s) have force data.

        Returns
        -------
        str
            "bilateral", "left", or "right".
        """
        left_foot = self.get("left_foot_ground_reaction_force")
        right_foot = self.get("right_foot_ground_reaction_force")
        if left_foot is not None and right_foot is not None:
            return "bilateral"
        if left_foot is not None:
            return "left"
        if right_foot is not None:
            return "right"
        raise ValueError("both left_foot and right_foot are None")

    def __init__(
        self,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        # check the inputs
        forces = {}
        if left_foot_ground_reaction_force is not None:
            if not isinstance(left_foot_ground_reaction_force, ForcePlatform):
                raise ValueError(
                    "left_foot_ground_reaction_force must be a ForcePlatform"
                    + " instance or None."
                )
            forces["left_foot_ground_reaction_force"] = left_foot_ground_reaction_force
        if right_foot_ground_reaction_force is not None:
            if not isinstance(right_foot_ground_reaction_force, ForcePlatform):
                raise ValueError(
                    "right_foot_ground_reaction_force must be a ForcePlatform"
                    + " instance or None."
                )
            forces["right_foot_ground_reaction_force"] = (
                right_foot_ground_reaction_force
            )
        if len(forces) == 0:
            raise ValueError(
                "at least one of 'left_foot_ground_reaction_force' or"
                + "'right_foot_ground_reaction_force' must be ForcePlatform"
                + " instances."
            )

        # build
        super().__init__(**signals, **forces)

    @classmethod
    def from_tdf(
        cls,
        file: str,
        left_foot_ground_reaction_force: str | None = "left_foot",
        right_foot_ground_reaction_force: str | None = "right_foot",
    ):
        """
        Create a Jump object from a TDF file.

        Parameters
        ----------
        file : str
            Path to the TDF file.
        bodymass_kg : float or int
            The subject's body mass in kilograms.
        vertical_axis : str, optional
            Name of the vertical axis in the force data.
        anteroposterior_axis : str, optional
            Name of the anteroposterior axis in the force data.
        left_foot_ground_reaction_force : str or None, optional
            Key for left foot force data.
        right_foot_ground_reaction_force : str or None, optional
            Key for right foot force data.

        Returns
        -------
        Jump
            A Jump object created from the TDF file.
        """
        mandatory_labels = {
            "left_foot_ground_reaction_force": left_foot_ground_reaction_force,
            "right_foot_ground_reaction_force": right_foot_ground_reaction_force,
        }
        valid = {i: v for i, v in mandatory_labels.items() if v is not None}
        if len(valid) == 0:
            raise ValueError(
                f"at least one of left_ground_reaction_force or"
                + " right_ground_reaction_force must be provided."
            )
        record = WholeBody.from_tdf(file, **valid)
        if not any([record.get(i) for i in mandatory_labels.keys()]):
            raise ValueError(
                f"at least one of left_ground_reaction_force or"
                + " right_ground_reaction_force must be provided."
            )
        return cls(**record._data)  # type: ignore


class PronePosture(WholeBody):
    """Represents a prone (plank) posture stance."""

    def __init__(
        self,
        left_foot_ground_reaction_force: ForcePlatform,
        right_foot_ground_reaction_force: ForcePlatform,
        left_hand_ground_reaction_force: ForcePlatform,
        right_hand_ground_reaction_force: ForcePlatform,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        # check the inputs
        forces = {
            "left_foot_ground_reaction_force": left_foot_ground_reaction_force,
            "right_foot_ground_reaction_force": right_foot_ground_reaction_force,
            "left_hand_ground_reaction_force": left_hand_ground_reaction_force,
            "right_hand_ground_reaction_force": right_hand_ground_reaction_force,
        }
        for f in forces.values():
            if not isinstance(f, ForcePlatform):
                raise ValueError(
                    "left/right foot and hand must be ForcePlatform instances"
                )

        # build
        super().__init__(**signals, **forces)  # type: ignore

    @classmethod
    def from_tdf(
        cls,
        file: str,
        left_foot_ground_reaction_force: str = "left_foot",
        right_foot_ground_reaction_force: str = "right_foot",
        left_hand_ground_reaction_force: str = "left_hand",
        right_hand_ground_reaction_force: str = "right_hand",
    ):
        """read from file"""
        labels = {
            "left_foot_ground_reaction_force": left_foot_ground_reaction_force,
            "right_foot_ground_reaction_force": right_foot_ground_reaction_force,
            "left_hand_ground_reaction_force": left_hand_ground_reaction_force,
            "right_hand_ground_reaction_force": right_hand_ground_reaction_force,
        }
        if not all([isinstance(i, str) for i in labels.values()]):
            msg = "left/right foot and hand ground reaction force labels must "
            msg += "be provided as strings."
            raise ValueError(msg)
        record = super().from_tdf(
            file,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
        )
        for key in labels.keys():
            if record.get(key) is None:
                raise ValueError(f"{key} not found within the provided file.")
        return cls(**record._data)  # type: ignore
