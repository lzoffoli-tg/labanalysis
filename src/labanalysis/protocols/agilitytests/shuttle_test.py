"""Shuttle test implementation."""

import numpy as np
import pandas as pd

from ...constants import MINIMUM_CONTACT_FORCE_N
from ...exercises import ChangeOfDirectionExercise
from ...pipelines import get_default_processing_pipeline
from ...records import ForcePlatform
from ...signalprocessing import butterworth_filt, fillna
from ...timeseries import Point3D
from ..participant import Participant
from ..test_protocol import TestProtocol
from .shuttle_test_results import ShuttleTestResults


class ShuttleTest(TestProtocol):
    """
    Shuttle test protocol for assessing change of direction ability and agility.

    This class implements a shuttle test protocol where participants perform multiple
    change of direction exercises. The test evaluates agility metrics such as contact
    time, force production, and center of pressure movement during directional changes.

    Parameters
    ----------
    participant : Participant
        Participant information object containing demographic and anthropometric data.
    change_of_direction_exercises : list of ChangeOfDirectionExercise
        List of change of direction exercises performed during the shuttle test.
    normative_data : pandas.DataFrame, optional
        Normative data for comparison with test results (default is empty DataFrame).

    Attributes
    ----------
    participant : Participant
        The participant who performed the test.
    change_of_direction_exercises : list of ChangeOfDirectionExercise
        The recorded change of direction exercises.
    normative_data : pandas.DataFrame
        Reference data for normative comparisons.

    Methods
    -------
    from_files(filenames, participant, normative_data,
        left_foot_ground_reaction_force, right_foot_ground_reaction_force, s2)
        Create ShuttleTest instance from TDF files.
    get_results()
        Generate ShuttleTestResults from processed data.
    copy()
        Create a deep copy of the test object.
    set_change_of_direction_exercise(records)
        Set the list of change of direction exercises.

    Properties
    ----------
    change_of_direction_exercises : list of ChangeOfDirectionExercise
        Access to the recorded exercises.
    processed_data : ShuttleTest
        Returns a processed copy of the test data.
    processing_pipeline : ProcessingPipeline
        Default signal processing pipeline for shuttle test data.

    Examples
    --------
    >>> from labanalysis import Participant, ShuttleTest
    >>> participant = Participant(name="John", surname="Doe", weight=75, height=180)
    >>> shuttle = ShuttleTest.from_files(
    ...     filenames=["trial1.tdf", "trial2.tdf"],
    ...     participant=participant
    ... )
    >>> results = shuttle.get_results()
    >>> print(results.summary)

    See Also
    --------
    ChangeOfDirectionExercise : Single change of direction exercise data.
    ShuttleTestResults : Analysis results for shuttle tests.

    Notes
    -----
    The shuttle test protocol applies default signal processing including:
    - Ground reaction force filtering (10 Hz lowpass, 4th order Butterworth)
    - Force threshold detection (30 N minimum contact force)
    - Point3D marker filtering (6 Hz lowpass, 4th order Butterworth)
    """

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
        """
        Set the change of direction exercises for this shuttle test.

        Parameters
        ----------
        records : list of ChangeOfDirectionExercise
            List of change of direction exercise objects to assign to this test.

        Raises
        ------
        ValueError
            If records is not a list of ChangeOfDirectionExercise instances.
        """
        if (not isinstance(records, list)) or (
            not all(isinstance(record, ChangeOfDirectionExercise) for record in records)
        ):
            raise ValueError(
                "recorda must be a list of ChangeOfDirectionExercise instances."
            )
        self._change_of_direction_exercises = records

    @property
    def change_of_direction_exercises(self):
        """
        Get the list of change of direction exercises.

        Returns
        -------
        list of ChangeOfDirectionExercise
            The recorded change of direction exercises for this shuttle test.
        """
        return self._change_of_direction_exercises

    def copy(self):
        """
        Create a deep copy of the ShuttleTest object.

        Returns
        -------
        ShuttleTest
            A new ShuttleTest instance with copied participant data, normative data,
            and change of direction exercises.
        """
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
        """
        Create a ShuttleTest instance from TDF files.

        Parameters
        ----------
        filenames : list of str
            List of file paths to TDF files containing change of direction exercise data.
        participant : Participant
            Participant information object.
        normative_data : pandas.DataFrame, optional
            Normative reference data for comparison (default is empty DataFrame).
        left_foot_ground_reaction_force : str or None, optional
            Name of the force platform measuring left foot ground reaction force
            (default is "left_foot").
        right_foot_ground_reaction_force : str or None, optional
            Name of the force platform measuring right foot ground reaction force
            (default is "right_foot").
        s2 : str or None, optional
            Name of the sternum marker (default is "s2").

        Returns
        -------
        ShuttleTest
            A new ShuttleTest instance with exercises loaded from the provided files.

        Raises
        ------
        ValueError
            If filenames is not a list.

        Examples
        --------
        >>> from labanalysis import Participant, ShuttleTest
        >>> p = Participant(name="John", surname="Doe", weight=75, height=180)
        >>> shuttle = ShuttleTest.from_files(
        ...     filenames=["trial1.tdf", "trial2.tdf", "trial3.tdf"],
        ...     participant=p
        ... )
        """
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
        """
        Generate ShuttleTestResults from processed test data.

        Returns
        -------
        ShuttleTestResults
            Analysis results object containing summary statistics, analytics,
            and visualization figures for the shuttle test.
        """
        return ShuttleTestResults(self.processed_data)

    @property
    def processed_data(self):
        """
        Get a processed copy of the shuttle test data.

        Applies the default signal processing pipeline to all change of direction
        exercises. The processing includes force platform filtering, marker trajectory
        smoothing, and contact detection.

        Returns
        -------
        ShuttleTest
            A copy of the test with all exercises processed according to the
            default processing pipeline.
        """
        out = self.copy()
        pipeline = self.processing_pipeline
        for i in range(len(out.change_of_direction_exercises)):
            pipeline(out.change_of_direction_exercises[i], inplace=True)
        return out

    @property
    def processing_pipeline(self):
        """
        Get the default signal processing pipeline for shuttle test data.

        The pipeline includes:
        - Point3D processing: strip NaNs, cubic spline interpolation, 6 Hz lowpass filter
        - ForcePlatform processing: contact detection (30 N threshold), gap filling,
          10 Hz lowpass filter, moment updates

        Returns
        -------
        ProcessingPipeline
            Configured processing pipeline with default signal processing functions
            for Point3D and ForcePlatform data types.
        """
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
                vals = fp[i].copy().to_numpy()  # type: ignore
                vals[idxs, :] = np.nan
                fp[i][:, :] = vals  # type: ignore

            # strip nans from the ends
            fp.strip(inplace=True)

            # fill remaining force nans with zeros
            fp.force[:, :] = fillna(fp.force.to_numpy(), value=0, inplace=False)  # type: ignore

            # fill remaining position nans via cubic spline
            fp.origin[:, :] = fillna(fp.origin.to_numpy(), inplace=False)  # type: ignore

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
