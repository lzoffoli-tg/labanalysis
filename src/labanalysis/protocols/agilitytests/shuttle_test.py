"""Shuttle test implementation."""

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from ...constants import MINIMUM_CONTACT_FORCE_N
from ...records import ForcePlatform, TimeseriesRecord
from ...exercises import ChangeOfDirectionExercise
from ...pipelines import get_default_processing_pipeline
from ...signalprocessing import butterworth_filt, fillna
from ...timeseries import Point3D
from ..participant import Participant
from ..test_protocol import TestProtocol
from .shuttle_test_results import ShuttleTestResults


class ShuttleTest(TestProtocol):

    def __init__(
        self,
        participant: Participant,
        change_of_direction_exercises: list[ChangeOfDirectionExercise],
        normative_data: pd.DataFrame = pd.DataFrame(),
    ):
        super().__init__(
            participant,
            normative_data,
        )
        self.set_change_of_direction_exercise(change_of_direction_exercises)

    def set_change_of_direction_exercise(
        self, records: list[ChangeOfDirectionExercise]
    ):
        if (not isinstance(records, list)) or (
            not all(isinstance(record, ChangeOfDirectionExercise) for record in records)
        ):
            raise ValueError(
                "recorda must be a list of ChangeOfDirectionExercise instances."
            )
        self._change_of_direction_exercises = records

    @property
    def change_of_direction_exercises(self):
        return self._change_of_direction_exercises

    def copy(self):
        return ShuttleTest(
            participant=self.participant.copy(),
            normative_data=self.normative_data,
            change_of_direction_exercises=[
                i.copy() for i in self.change_of_direction_exercises
            ],
        )

    @classmethod
    def from_files(
        cls,
        filenames: list[str],
        participant: Participant,
        normative_data: pd.DataFrame = pd.DataFrame(),
        left_foot_ground_reaction_force: str | None = "left_foot",
        right_foot_ground_reaction_force: str | None = "right_foot",
        s2: str | None = "s2",
    ):
        if not isinstance(filenames, list):
            raise ValueError("filename must be a list")
        return cls(
            participant=participant,
            normative_data=normative_data,
            change_of_direction_exercises=[
                ChangeOfDirectionExercise.from_tdf(
                    file=filename,
                    left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                    right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                    s2=s2,
                )
                for filename in filenames
            ],
        )

    def get_results(self):
        return ShuttleTestResults(self.processed_data)

    @property
    def processed_data(self):
        out = self.copy()
        pipeline = self.processing_pipeline
        for i in range(len(out.change_of_direction_exercises)):
            pipeline(out.change_of_direction_exercises[i], inplace=True)
        return out

    @property
    def processing_pipeline(self):
        pipeline = get_default_processing_pipeline()

        def get_point3d_processing_func(point: Point3D):
            point.strip(inplace=True)
            point.fillna(inplace=True)
            fsamp = float(1 / np.mean(np.diff(point.index)))
            point.apply(
                butterworth_filt,
                fcut=6,
                fsamp=fsamp,
                order=4,
                ftype="lowpass",
                phase_corrected=True,
                inplace=True,
            )

        def get_forceplatform_processing_func(fp: ForcePlatform):

            # ensure force below minimum contact are set to NaN
            vals = fp.force.copy().to_numpy()
            module = fp.force.copy().module.to_numpy().flatten()  # type: ignore
            idxs = module < MINIMUM_CONTACT_FORCE_N
            for i in ["origin", "force", "torque"]:
                vals = fp[i].copy().to_numpy()
                vals[idxs, :] = np.nan
                fp[i][:, :] = vals

            # strip nans from the ends
            fp.strip(inplace=True)

            # fill remaining force nans with zeros
            fp.force[:, :] = fillna(fp.force.to_numpy(), value=0, inplace=False)

            # fill remaining position nans via cubic spline
            fp.origin[:, :] = fillna(fp.origin.to_numpy(), inplace=False)

            # lowpass filter both origin and force
            fsamp = float(1 / np.mean(np.diff(fp.index)))
            filt_fun = lambda x: butterworth_filt(
                x,
                fcut=10,
                fsamp=fsamp,  # type: ignore
                order=4,
                ftype="lowpass",
                phase_corrected=True,
            )
            fp.origin.apply(filt_fun, axis=0, inplace=True)
            fp.force.apply(filt_fun, axis=0, inplace=True)

            # update moments
            fp.update_moments(inplace=True)

            # set moments corresponding to the very low vertical force to zero
            module = fp.force.copy().module.to_numpy().flatten()  # type: ignore
            idxs = module < MINIMUM_CONTACT_FORCE_N
            vals = fp.torque.copy().to_numpy()
            vals[idxs, :] = 0
            fp.torque[:, :] = vals

        pipeline["Point3D"] = [get_point3d_processing_func]
        pipeline["ForcePlatform"] = [get_forceplatform_processing_func]

        return pipeline


__all__ = ["ShuttleTest"]
