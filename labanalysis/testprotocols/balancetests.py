"""balance test module"""

__all__ = ["UprightBalanceTest", "PlankBalanceTest"]


from typing import Literal

import numpy as np
import pandas as pd

from ..constants import G
from ..records import *
from ..modelling import Ellipse
from .normativedata import *
from .protocols import Participant, TestProtocol


class UprightBalanceTest(WholeBody, TestProtocol):

    _eyes: Literal["open", "closed"]

    @property
    def eyes(self):
        return self._eyes

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

    @property
    def bodymass_kg(self):
        """
        Returns the subject's body mass in kilograms.

        Returns
        -------
        float
            Body mass in kg.
        """
        grf = self.resultant_force
        if grf is None:
            return np.nan
        return float(grf.force[self.vertical_axis].to_numpy().mean() / G)

    @property
    def muscle_symmetry(self):
        """
        Returns coordination and balance metrics from EMG signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with coordination and balance metrics, or empty if not available.
        """

        # check if a bilateral jump was performed
        # (otherwise it makes no sense to test balance)
        if self.side != "bilateral":
            return pd.DataFrame()

        # get the muscle activations
        # (if there are no emg data return and empty dataframe)
        emgs = self.emgsignals
        if emgs.shape[1] == 0:
            return pd.DataFrame()

        # check the presence of left and right muscles
        muscles = {}
        for emg in emgs.values():
            if isinstance(emg, EMGSignal):
                name = emg.muscle_name
                side = emg.side
                if side not in ["left", "right"]:
                    continue
                if name not in list(muscles.keys()):
                    muscles[name] = {}

                # get the area under the curve of the muscle activation
                muscles[name][side] = emg.to_numpy().flatten()

        # remove those muscles not having both sides
        muscles = {i: v for i, v in muscles.items() if len(v) == 2}

        # calculate coordination and imbalance between left and right side
        out = {}
        for muscle, sides in muscles.items():
            params = self._get_symmetry(**sides)
            out.update(**{f"{muscle}_{i}": v for i, v in params.items()})

        return pd.DataFrame(pd.Series(out)).T

    @property
    def force_symmetry(self):
        """
        Returns coordination and balance metrics from force signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with force coordination and balance metrics, or empty if not available.
        """

        # get the forces from each foot and hand
        left_foot = self.get("left_foot_ground_reaction_force")
        right_foot = self.get("right_foot_ground_reaction_force")
        if (
            left_foot is None
            or right_foot is None
            or not isinstance(left_foot, ForcePlatform)
            or not isinstance(right_foot, ForcePlatform)
        ):
            return pd.DataFrame()
        left_vt = left_foot.copy().force[self.vertical_axis].to_numpy().flatten()  # type: ignore
        right_vt = right_foot.copy().force[self.vertical_axis].to_numpy().flatten()  # type: ignore

        # get the pairs to be tested
        pairs = {
            "lower_limbs": {"left_foot": left_vt, "right_foot": right_vt},
        }

        # calculate balance and coordination
        out = []
        unit = self.resultant_force
        if unit is None:
            return pd.DataFrame()
        unit = unit.force.unit
        for region, pair in pairs.items():
            left, right = list(pair.values())
            fit = self._get_symmetry(left, right)
            line = {f"force_{i}": float(v) for i, v in fit.items()}  # type: ignore
            line = pd.DataFrame(pd.Series(line)).T
            line.insert(0, "region", region)
            out += [line]

        return pd.concat(out, ignore_index=True)

    @property
    def area_of_stability_sqm(self):
        cop = self.resultant_force
        if cop is None:
            return np.nan
        cop = cop.origin.to_dataframe().dropna()
        horizontal_axes = [self.anteroposterior_axis, self.lateral_axis]
        ap, ml = cop[horizontal_axes].values.astype(float).T
        ellipse = Ellipse().fit(ap.astype(float), ml.astype(float))
        return ellipse.area

    def _get_symmetry(self, left: np.ndarray, right: np.ndarray):
        line = {"left_%": np.mean(left), "right_%": np.mean(right)}
        line = pd.DataFrame(pd.Series(line)).T
        norm = line.sum(axis=1).values.astype(float)
        line.loc[line.index, line.columns] = line.values.astype(float) / norm * 100
        return line

    def set_eyes(self, eyes: Literal["open", "closed"]):
        self._eyes = eyes

    def __init__(
        self,
        participant: Participant,
        eyes: Literal["open", "closed"],
        normative_data_path: str = UPRIGHTBALANCE_NORMATIVE_DATA_PATH,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):

        super().__init__(
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            **signals,  # type: ignore
        )
        self.set_participant(participant)
        self.set_normative_data_path(normative_data_path)
        self.set_eyes(eyes)

    def copy(self):
        return UprightBalanceTest(
            participant=self.participant.copy(),
            eyes=self.eyes,
            normative_data_path=self.normative_data_path,
            **{i: v.copy() for i, v in self.items()},  # type: ignore
        )

    @property
    def results(self):
        out = {
            "summary": pd.DataFrame(),
            "analytics": {"centre_of_pressure": pd.DataFrame()},
        }
        cop = self.resultant_force
        if cop is not None:
            summary = {
                "type": self.__class__.__name__,
                "eyes": self.eyes,
                "side": self.side,
                "bodymass_kg": self.bodymass_kg,
                "area_of_stability_mm2": self.area_of_stability_sqm * 1000000,
            }
            summary = pd.DataFrame(pd.Series(summary)).T
            summary = [summary, self.force_symmetry, self.muscle_symmetry]
            summary = pd.concat(summary, axis=1)
            cop = cop.copy()["origin"].to_dataframe().dropna()
            out["summary"] = summary
            out["analytics"]["centre_of_pressure"] = cop
        return out

    @classmethod
    def from_tdf(
        cls,
        filename: str,
        participant: Participant,
        eyes: Literal["open", "closed"],
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        normative_data_path: str = UPRIGHTBALANCE_NORMATIVE_DATA_PATH,
    ):
        tdf = WholeBody.from_tdf(
            filename=filename,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
        )
        mandatory = [
            "left_foot_ground_reaction_force",
            "right_foot_ground_reaction_force",
        ]
        mandatory = {i: tdf.get(i) for i in mandatory}
        signals = {i: v for i, v in tdf.items() if i not in mandatory}
        return cls(
            participant=participant,
            normative_data_path=normative_data_path,
            eyes=eyes,
            **mandatory,  # type: ignore
            **signals,  # type: ignore
        )


class PlankBalanceTest(UprightBalanceTest):

    def __init__(
        self,
        participant: Participant,
        left_foot_ground_reaction_force: ForcePlatform,
        right_foot_ground_reaction_force: ForcePlatform,
        left_hand_ground_reaction_force: ForcePlatform,
        right_hand_ground_reaction_force: ForcePlatform,
        eyes: Literal["open", "closed"],
        normative_data_path: str = PLANKBALANCE_NORMATIVE_DATA_PATH,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):

        super().__init__(
            participant=participant,
            normative_data_path=normative_data_path,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            eyes=eyes,
            **signals,  # type: ignore
        )

    def copy(self):
        return PlankBalanceTest(
            participant=self.participant.copy(),
            eyes=self.eyes,
            normative_data_path=self.normative_data_path,
            **{i: v.copy() for i, v in self.items()},  # type: ignore
        )

    @classmethod
    def from_tdf(
        cls,
        filename: str,
        participant: Participant,
        eyes: Literal["open", "closed"],
        left_foot_ground_reaction_force: str,
        right_foot_ground_reaction_force: str,
        left_hand_ground_reaction_force: str,
        right_hand_ground_reaction_force: str,
        normative_data_path: str = PLANKBALANCE_NORMATIVE_DATA_PATH,
    ):
        tdf = WholeBody.from_tdf(
            filename=filename,
            left_hand_ground_reaction_force=left_hand_ground_reaction_force,
            left_foot_ground_reaction_force=left_foot_ground_reaction_force,
            right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            right_foot_ground_reaction_force=right_foot_ground_reaction_force,
        )
        mandatory = [
            "left_foot_ground_reaction_force",
            "right_foot_ground_reaction_force",
            "left_hand_ground_reaction_force",
            "right_hand_ground_reaction_force",
        ]
        mandatory_dict = {i: tdf.get(i) for i in mandatory}
        if any([i is None for i in list(mandatory_dict.values())]):
            raise ValueError(
                "all the following ForcePlatform elements must be included "
                + "into the provided file"
            )
        signals = {i: v for i, v in tdf.items() if i not in mandatory}
        return cls(
            participant=participant,
            normative_data_path=normative_data_path,
            eyes=eyes,
            **mandatory_dict,  # type: ignore
            **signals,  # type: ignore
        )

    @property
    def force_symmetry(self):
        """
        Returns coordination and balance metrics from force signals.

        Returns
        -------
        pd.DataFrame
            DataFrame with force coordination and balance metrics,
            or empty if not available.
        """

        # get the forces from each foot and hand
        left_foot = self.get("left_foot_ground_reaction_force")
        right_foot = self.get("right_foot_ground_reaction_force")
        left_hand = self.get("left_hand_ground_reaction_force")
        right_hand = self.get("right_hand_ground_reaction_force")
        if (
            left_foot is None
            or right_foot is None
            or left_hand is None
            or right_hand is None
        ):
            return pd.DataFrame()
        left_foot = left_foot.copy()["force"][self.vertical_axis].to_numpy()
        right_foot = right_foot.copy()["force"][self.vertical_axis].to_numpy()
        left_hand = left_hand.copy()["force"][self.vertical_axis].to_numpy()
        right_hand = right_hand.copy()["force"][self.vertical_axis].to_numpy()

        # get the pairs to be tested
        pairs = {
            "lower_limbs": {"left_foot": left_foot, "right_foot": right_foot},
            "upper_limbs": {"left_hand": left_hand, "right_hand": right_hand},
            "whole_body": {
                "left": left_foot + left_hand,
                "right": right_foot + right_hand,
            },
        }

        # calculate balance and coordination
        out = []
        unit = self.resultant_force
        if unit is None:
            return pd.DataFrame()
        unit = unit.force.unit
        for region, pair in pairs.items():
            left, right = list(pair.values())
            fit = self._get_symmetry(left, right)
            line = {f"force_{i}": float(v.values[0]) for i, v in fit.items()}  # type: ignore
            line = pd.DataFrame(pd.Series(line)).T
            line.insert(0, "region", region)
            out += [line]

        return pd.concat(out, ignore_index=True)
