"""Upright balance test implementation."""

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from ...records import ForcePlatform, TimeseriesRecord
from ...exercises import UprightPosture
from ...pipelines import get_default_processing_pipeline
from ...referenceframes import ReferenceFrame
from ..normativedata import uprightbalance_normative_values
from ..participant import Participant
from ..test_protocol import TestProtocol


class UprightBalanceTest(TestProtocol):

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
        return self.exercise.side

    def set_eyes(self, eyes: Literal["open", "closed"]):
        if eyes not in ["open", "closed"]:
            raise ValueError("eyes must be 'open' or 'closed'.")
        self._eyes = eyes

    def __init__(
        self,
        participant: Participant,
        exercise: UprightPosture,
        eyes: Literal["open", "closed"],
        normative_data: pd.DataFrame = uprightbalance_normative_values,
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

    @classmethod
    def from_files(
        cls,
        filename: str,
        participant: Participant,
        eyes: Literal["open", "closed"],
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        normative_data: pd.DataFrame = uprightbalance_normative_values,
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
            relevant_muscle_map=relevant_muscle_map,
            emg_normalization_function=emg_normalization_function,
            exercise=UprightPosture.from_tdf(
                file=filename,
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
            ),
        )

    def set_exercise(self, exercise: UprightPosture):
        if not isinstance(exercise, UprightPosture):
            raise ValueError("exercise must be an UprightPosture instance.")
        self._exercise = exercise

    @property
    def exercise(self):
        return self._exercise

    def copy(self):
        return UprightBalanceTest(
            participant=self.participant,
            exercise=self.exercise,
            eyes=self.eyes,  # type: ignore
            normative_data=self.normative_data,
            emg_normalization_references=self.emg_normalization_references,
            emg_normalization_function=self.emg_normalization_function,
            emg_activation_references=self.emg_activation_references,
            emg_activation_threshold=self.emg_activation_threshold,
            relevant_muscle_map=self.relevant_muscle_map,
        )

    @property
    def processed_data(self):

        # apply the pipeline to the test data
        exe = self.processing_pipeline(self.exercise, inplace=False)
        if not isinstance(exe, TimeseriesRecord):
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
        if self.side not in ["right", "left"]:

            def extract_cop(force: Any):
                if not isinstance(force, ForcePlatform):
                    raise ValueError("force must be a ForcePlatform instance.")
                cop = force.origin
                if not isinstance(cop, Point3D):
                    raise ValueError("force must be a ForcePlatform instance.")
                cop = cop.copy()
                return cop.to_numpy().astype(float).mean(axis=0)

            # on bilateral test, we rotate the system of forces to a
            rt = extract_cop(exe.right_foot_ground_reaction_force)
            lt = extract_cop(exe.left_foot_ground_reaction_force)

            def norm(arr):
                return arr / np.sum(arr**2) ** 0.5

            ml = norm(lt - rt)
            vt = np.array([0, 1, 0])
            ap = np.cross(ml, vt)
            origin = (rt + lt) / 2
            ref_frame = ReferenceFrame(origin, ml, vt, ap)
            exe.apply(ref_frame, inplace=True)
            if exe is None:
                raise ValueError("reference frame alignment returned None")

        # return processed data
        out = self.copy()
        if not isinstance(exe, UprightPosture):
            raise ValueError("Something went wrong during data processing.")
        out.set_exercise(exe)
        return out

    @property
    def processing_pipeline(self):
        return get_default_processing_pipeline()

    def get_results(self, include_emg: bool = True):
        return UprightBalanceTestResults(
            self.processed_data,
            include_emg,
        )


__all__ = ["UprightBalanceTest"]
