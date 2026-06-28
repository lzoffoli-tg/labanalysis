"""Plank balance test implementation."""

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from ...records import ForcePlatform, TimeseriesRecord
from ...exercises import PronePosture
from ...pipelines import get_default_processing_pipeline
from ...referenceframes import ReferenceFrame
from ..normativedata import plankbalance_normative_values
from ..participant import Participant
from ..test_protocol import TestProtocol


class PlankBalanceTest(TestProtocol):

    @property
    def eyes(self):
        return self._eyes

    def set_eyes(self, eyes: Literal["open", "closed"]):
        if eyes not in ["open", "closed"]:
            raise ValueError("eyes must be 'open' or 'closed'.")
        self._eyes = eyes

    def __init__(
        self,
        participant: Participant,
        exercise: PronePosture,
        eyes: Literal["open", "closed"],
        normative_data: pd.DataFrame = plankbalance_normative_values,
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_activation_threshold: float | int = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        super().__init__(
            participant=participant,
            normative_data=normative_data,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            relevant_muscle_map=relevant_muscle_map,
        )
        self.set_eyes(eyes)
        self.set_exercise(exercise)

    def copy(self):
        return PlankBalanceTest(
            participant=self.participant,
            exercise=self.exercise,
            eyes=self.eyes,  # type: ignore
            normative_data=self.normative_data,
            emg_normalization_references=self.emg_normalization_references,
            emg_activation_references=self.emg_activation_references,
            emg_activation_threshold=self.emg_activation_threshold,
            emg_normalization_function=self.emg_normalization_function,
            relevant_muscle_map=self.relevant_muscle_map,
        )

    def set_exercise(self, exercise: PronePosture):
        if not isinstance(exercise, PronePosture):
            raise ValueError("exercise must be a PronePosture instance.")
        self._exercise = exercise

    @property
    def exercise(self):
        return self._exercise

    @classmethod
    def from_files(
        cls,
        filename: str,
        participant: Participant,
        eyes: Literal["open", "closed"],
        left_foot_ground_reaction_force: str = "left_foot",
        right_foot_ground_reaction_force: str = "right_foot",
        left_hand_ground_reaction_force: str = "left_hand",
        right_hand_ground_reaction_force: str = "right_hand",
        normative_data: pd.DataFrame = plankbalance_normative_values,
        emg_normalization_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_normalization_function: Callable = np.mean,
        emg_activation_references: (
            TimeseriesRecord | str | Literal["self"]
        ) = TimeseriesRecord(),
        emg_activation_threshold: float | int = 3,
        relevant_muscle_map: list[str] | None = None,
    ):
        return cls(
            participant=participant,
            normative_data=normative_data,
            eyes=eyes,
            emg_activation_references=emg_activation_references,
            emg_activation_threshold=emg_activation_threshold,
            emg_normalization_references=emg_normalization_references,
            emg_normalization_function=emg_normalization_function,
            relevant_muscle_map=relevant_muscle_map,
            exercise=PronePosture.from_tdf(
                file=filename,
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                left_hand_ground_reaction_force=left_hand_ground_reaction_force,
                right_hand_ground_reaction_force=right_hand_ground_reaction_force,
            ),
        )

    @property
    def processed_data(self):

        # apply the pipeline to the test data
        exe = self.processing_pipeline(self.exercise, inplace=False)
        if not isinstance(exe, PronePosture):
            raise ValueError("Something went wrong during data processing.")

        # normalize emg data and remove non-relevant muscles
        norms = self.emg_normalization_values
        to_remove: list[str] = []
        for k, m in exe.emgsignals.items():

            # remove if non relevant
            if self.relevant_muscle_map is not None:
                if not any([i.lower() in k.lower() for i in self.relevant_muscle_map]):
                    to_remove.append(k)
                    continue

            # normalize
            if isinstance(m, EMGSignal):
                for (name, side), val in norms.items():
                    if m.muscle_name == name and m.side == side:
                        exe[k] = m / val * 100
                        exe[k].set_unit("%")  # type: ignore
                        break
        if len(to_remove) > 0:
            exe.drop(to_remove, True)

        # align the reference frame
        def extract_cop(force: Any):
            if not isinstance(force, ForcePlatform):
                raise ValueError("force must be a ForcePlatform instance.")
            cop = force.origin
            if not isinstance(cop, Point3D):
                raise ValueError("force must be a ForcePlatform instance.")
            cop = cop.copy()
            return cop.to_numpy().astype(float).mean(axis=0)

        # on bilateral test, we rotate the system of forces to a
        rf = extract_cop(exe.right_foot_ground_reaction_force)
        lf = extract_cop(exe.left_foot_ground_reaction_force)
        rh = extract_cop(exe.right_hand_ground_reaction_force)
        lh = extract_cop(exe.left_hand_ground_reaction_force)

        def norm(arr):
            return arr / np.sum(arr**2) ** 0.5

        ml = norm((lf + lh) / 2 - (rf + rh) / 2)
        vt = np.array([0, 1, 0])
        ap = np.cross(ml, vt)
        origin = (rf + lf + rh + lh) / 4
        ref_frame = ReferenceFrame(origin, ml, vt, ap)
        exe = exe.apply(ref_frame, inplace=False
        )
        if exe is None:
            raise ValueError("reference frame alignment returned None")

        # return processed data
        out = self.copy()
        out.set_exercise(exe)  # type: ignore
        return out

    @property
    def processing_pipeline(self):
        return get_default_processing_pipeline()

    def get_results(self, include_emg: bool = True):
        return PlankBalanceTestResults(
            self.processed_data,
            include_emg,
        )


__all__ = ["PlankBalanceTest"]
