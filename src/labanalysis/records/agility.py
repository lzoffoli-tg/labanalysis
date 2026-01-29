"""agility module"""

#! IMPORTS

import numpy as np

from ..records import EMGSignal, Signal1D, Signal3D, Point3D, ForcePlatform, TimeseriesRecord


from .bodies import WholeBody

__all__ = ["ChangeOfDirectionExercise"]


#! CLASSES


class ChangeOfDirectionExercise(WholeBody):
    """
    Represents a single step on forceplatform during a change of direction.

    Parameters
    ----------
    left_foot_ground_reaction_force : ForcePlatform, optional
        ForcePlatform object for the left foot.
    right_foot_ground_reaction_force : ForcePlatform, optional
        ForcePlatform object for the right foot.
    s2: Point3D, optional
        Marker reflecting the position of s2 on the body.
    vertical_axis : str, optional
        Name of the vertical axis in the force data (default "Y").
    anteroposterior_axis : str, optional
        Name of the anteroposterior axis in the force data (default "X").
    **signals : Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
        Additional signals to include in the record.
    """
    _inversion_time:float | None = None

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
    def contact_phase(self):
        """
        Returns the concentric phase of the jump.

        Returns
        -------
        TimeseriesRecord
            Data for the concentric phase.

        Procedure
        ---------
            1. get the longest countinuous batch with positive acceleration
            of S2 occurring before con_end.
            2. define 'con_start' as the last local minima in the vertical grf
            occurring before the beginning of the batch defined in 2.
            3. define 'con_end' as the end of the concentric phase as the time
            instant immediately before the flight phase. Please look at the
            concentric_phase documentation to have a detailed view about how
            it is detected.
        """
        # get the longest batch with grf lower than 30N
        rf = self.resultant_force.strip()
        start = rf.index[0]
        stop = rf.index[-1]

        signals = {k: v.copy().loc(start, stop) for k, v in self.items()}
        return WholeBody(**signals)  # type: ignore

    @property
    def contact_time (self):
        '''
        Return the contact time in seconds
        '''
        index = self.contact_phase.index
        return float(index[-1] - index[0])

    @property
    def velocity (self):
        '''
        Return the movement velocity of s2 marker.
        '''
        if not any([i == 's2' for i in self.points3d.keys()]):
            data = []
            index = []
        else:
            s2 = self.s2.copy()
            data = s2.to_numpy()
            index = s2.index
            data = np.gradient(data, index, axis = 0)
        return Signal3D(
            data = data,
            index = index,
            vertical_axis = self.vertical_axis,
            anteroposterior_axis = self.anteroposterior_axis,
            unit = 'm/s',
        )
    
    @property
    def inversion_time(self):
        "time instant of the end of the loading phase"
        if self._inversion_time is None:
            s2 = self.s2.copy()
            s2z = s2[self.anteroposterior_axis].to_numpy().flatten()
            self._inversion_time = s2.index[np.argmax(s2z)]
        return self._inversion_time

    @property
    def loading_phase (self):
        '''
        Return loading phase of the step
        '''   
        loading_phase = self.contact_phase[self.index[0]:self.inversion_time]
        if not isinstance(loading_phase, WholeBody): 
            raise RuntimeError('Loading phase should be a WholeBody instance')
        return loading_phase

    @property
    def loading_time(self):
        '''
        Return the loading phase duration in seconds.
        '''
        idx_loading_phase = self.loading_phase.index
        return float(idx_loading_phase[-1] - idx_loading_phase[0])
        
    @property
    def propulsion_phase (self):
        '''
        Return propulsion phase of the step
        ''' 
        contact_phase = self.contact_phase
        propulsion_phase = contact_phase[self.inversion_time:contact_phase.index[-1]]
        if not isinstance(propulsion_phase, WholeBody): 
            raise RuntimeError('Propulsion phase should be a WholeBody instance')
        return propulsion_phase
    
    @property
    def propulsion_time(self):
        '''
        Return the propulsion phase duration in seconds.
        '''
        idx_propulsion_phase = self.propulsion_phase.index
        return float(idx_propulsion_phase[-1] - idx_propulsion_phase[0])

    def __init__(
        self,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        s2: Point3D | None = None,
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
        points = {}
        if s2 is not None:
            if not isinstance(s2, Point3D):
                raise ValueError("s2 must be a Point3D instance or None.")
            points['s2'] = s2

        # build
        super().__init__(**points, **signals, **forces)

    @classmethod
    def from_tdf(
        cls,
        file: str,
        left_foot_ground_reaction_force: str | None = "left_foot",
        right_foot_ground_reaction_force: str | None = "right_foot",
        s2: str | None = 's2',
    ):
        """
        Create a Jump object from a TDF file.

        Parameters
        ----------
        file : str
            Path to the TDF file.
        left_foot_ground_reaction_force : str or None, optional
            Key for left foot force data.
        right_foot_ground_reaction_force : str or None, optional
            Key for right foot force data.
        s2 : str or None, optionale
            Marker reflecting the position of s2 on the body.

        Returns
        -------
        Jump
            A Jump object created from the TDF file.
        """
        mandatory_labels = {
            "left_foot_ground_reaction_force": left_foot_ground_reaction_force,
            "right_foot_ground_reaction_force": right_foot_ground_reaction_force,
        }
        if all([i is None for i in mandatory_labels.values()]):
            raise ValueError(
                "at least one of left_ground_reaction_force or"
                + " right_ground_reaction_force must be provided."
            )
        record = TimeseriesRecord.from_tdf(file)
        mandatory = {}
        for key, lbl in mandatory_labels.items():
            if lbl is not None:
                mandatory[key] = record.get(lbl)
        if all(i is None for i in mandatory.values()):
            raise ValueError(
                "at least one foot ground reaction force must be "
                "found on the provided file."
            )
        to_exclude = list(mandatory_labels.values())
        if s2 is not None:
            if not any([i == s2 for i in record.keys()]):
                raise ValueError(f"{s2} not found in the provided file.")
            else:
                s2_point = record[s2]
            if not isinstance(s2_point, Point3D):
                raise ValueError(f"{s2} must be a Point3D object.")
            to_exclude += [s2]
        else:
            s2_point = None

        signals = {i: v for i, v in record.items() if i not in to_exclude}
        return cls(
            s2 = s2_point,
            **signals,  # type: ignore
            **mandatory,  # type: ignore
        )

    def copy(self):
        return ChangeOfDirectionExercise(
            **{i: v.copy() for i, v in self.items()},  # type: ignore
        )
