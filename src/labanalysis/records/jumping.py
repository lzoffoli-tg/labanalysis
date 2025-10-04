"""repeatedjump module"""

#! IMPORTS


import numpy as np
import pandas as pd

from ..constants import MINIMUM_CONTACT_FORCE_N, MINIMUM_FLIGHT_TIME_S, G
from ..signalprocessing import continuous_batches
from .timeseries import *
from .bodies import WholeBody
from .records import *

__all__ = ["JumpExercise", "SingleJump"]


#! CLASSES


class SingleJump(WholeBody):
    """
    Represents a single jump trial, providing methods and properties to analyze
    phases, forces, and performance metrics of the jump.

    Parameters
    ----------
    bodymass_kg : float
        The subject's body mass in kilograms.
    left_foot_ground_reaction_force : ForcePlatform, optional
        ForcePlatform object for the left foot.
    right_foot_ground_reaction_force : ForcePlatform, optional
        ForcePlatform object for the right foot.
    vertical_axis : str, optional
        Name of the vertical axis in the force data (default "Y").
    anteroposterior_axis : str, optional
        Name of the anteroposterior axis in the force data (default "X").
    **signals : Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
        Additional signals to include in the record.

    Attributes
    ----------
    _bodymass_kg : float
        The subject's body mass in kilograms.
    _vertical_axis : str
        Name of the vertical axis.
    _antpos_axis : str
        Name of the anteroposterior axis.

    Properties
    ----------
    vertical_axis : str
        The vertical axis label.
    anteroposterior_axis : str
        The anteroposterior axis label.
    lateral_axis : str
        The lateral axis label.
    vertical_force : np.ndarray
        The mean vertical ground reaction force across both feet.
    side : str
        "bilateral", "left", or "right" depending on available force data.
    bodymass_kg : float
        The subject's body mass in kilograms.
    eccentric_phase : TimeseriesRecord
        Data for the eccentric phase of the jump.
    concentric_phase : TimeseriesRecord
        Data for the concentric phase of the jump.
    flight_phase : TimeseriesRecord
        Data for the flight phase of the jump.
    contact_time_s : float
        Duration of the contact phase (s).
    flight_time_s : float
        Duration of the flight phase (s).
    takeoff_velocity_ms : float
        Takeoff velocity at the end of the concentric phase (m/s).
    elevation_cm : float
        Jump elevation (cm) calculated from flight time.
    muscle_coordination_and_balance : pd.DataFrame
        Coordination and balance metrics from EMG signals (if available).
    force_coordination_and_balance : pd.DataFrame
        Coordination and balance metrics from force signals.
    output_metrics : pd.DataFrame
        Summary metrics for the jump.

    Methods
    -------
    __init__(...)
        Initialize a Jump object.
    from_tdf(...)
        Create a Jump object from a TDF file.
    """

    _bodymass_kg: float

    def _get_symmetry(self, left: np.ndarray, right: np.ndarray):
        left_val = np.mean(left)
        right_val = np.mean(right)
        vals = left_val + right_val
        return {
            "left_%": left_val / vals * 100,
            "right_%": right_val / vals * 100,
        }

    def _get_muscle_symmetry(self):
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
        emgs = self.contact_phase.emgsignals
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
            muscles[name][side] = emg.copy().to_numpy().flatten()

        # remove those muscles not having both sides
        muscles = {i: v for i, v in muscles.items() if len(v) == 2}

        # calculate coordination and imbalance between left and right side
        out = {}
        for muscle, sides in muscles.items():
            params = self._get_symmetry(**sides)
            out.update(**{f"{muscle}_{i}": v for i, v in params.items()})

        return pd.DataFrame(pd.Series(out)).T

    def _get_force_symmetry(self):
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
        if left_foot is None or right_foot is None:
            return pd.DataFrame()
        left_foot = left_foot["force"].copy()[self.vertical_axis].to_numpy().flatten()
        right_foot = right_foot["force"].copy()[self.vertical_axis].to_numpy().flatten()

        # get the pairs to be tested
        pairs = {"lower_limbs": {"left_foot": left_foot, "right_foot": right_foot}}

        # calculate balance and coordination
        out = []
        for region, pair in pairs.items():
            left, right = list(pair.values())
            fit = self._get_symmetry(left, right)
            line = {f"force_{i}": v for i, v in fit.items()}
            line = pd.DataFrame(pd.Series(line)).T
            line.insert(0, "region", region)
            out += [line]

        return pd.concat(out, ignore_index=True)

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
        return self._bodymass_kg

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
        out = WholeBody()

        # get the longest batch with grf lower than 30N
        vgrf = self.resultant_force
        if vgrf is None:
            return out
        grfy = vgrf.copy().force[self.vertical_axis].to_numpy().flatten()
        grft = self.index
        batches = continuous_batches(
            (grfy > MINIMUM_CONTACT_FORCE_N) & (grft < self.flight_phase.index[0])
        )
        if len(batches) == 0:
            return out
        batch = batches[-1]

        sliced = self.copy()[grft[batch[0]] : grft[batch[-1]]]
        if sliced is not None:
            if isinstance(sliced, WholeBody):
                for key, value in sliced.items():
                    out[key] = value
        return out

    @property
    def flight_phase(self):
        """
        Returns the flight phase of the jump.

        Returns
        -------
        TimeseriesRecord
            Data for the flight phase.

        Procedure
        ---------
            1. get the longest batch with grf lower than 30N.
            2. define 'flight_start' as the first local minima occurring after
            the start of the detected batch.
            3. define 'flight_end' as the last local minima occurring before the
            end of the detected batch.
        """

        # get the longest batch with grf lower than 30N
        vgrf = self.resultant_force
        if vgrf is None:
            return TimeseriesRecord()
        grfy = vgrf.force.copy()[self.vertical_axis].to_numpy().flatten()
        grft = self.index
        batches = continuous_batches(grfy <= MINIMUM_CONTACT_FORCE_N)
        msg = "No flight phase found."
        if len(batches) == 0:
            raise RuntimeError(msg)
        batch = batches[np.argmax([len(i) for i in batches])]

        # check the length of the batch is at minimum 2 samples
        if len(batch) < 2:
            raise RuntimeError(msg)

        # # get the time samples corresponding to the start and end of each
        # batch
        time_start = float(np.round(grft[batch[0]], 3))
        time_stop = float(np.round(grft[batch[-1]], 3))

        # return a slice of the available data
        sliced = self.copy()[time_start:time_stop]
        if sliced is None:
            raise RuntimeError(msg)
        out = WholeBody()
        if isinstance(sliced, WholeBody):
            for key, value in sliced.items():
                out[key] = value
        return out

    @property
    def contact_time_s(self):
        """
        Returns the duration of the contact phase (eccentric + concentric).

        Returns
        -------
        float
            Contact time in seconds.
        """
        time = self.contact_phase.index
        return float(time[-1] - time[0])

    @property
    def flight_time_s(self):
        """
        Returns the duration of the flight phase.

        Returns
        -------
        float
            Flight time in seconds.
        """
        time = self.flight_phase.index
        return time[-1] - time[0]

    @property
    def takeoff_velocity_ms(self):
        """
        Returns the takeoff velocity at the end of the concentric phase.

        Returns
        -------
        float
            Takeoff velocity in m/s.
        """

        # get the ground reaction force during the concentric phase
        con = self.contact_phase.resultant_force
        if con is None:
            return np.nan
        grf = con.copy().force[self.vertical_axis].to_numpy().flatten()
        time = con.index

        # get the output velocity
        net_grf = grf - self.bodymass_kg * G
        return float(np.trapezoid(net_grf, time) / self.bodymass_kg)

    @property
    def elevation_cm(self):
        """
        Returns the jump elevation in centimeters, calculated from flight time.

        Returns
        -------
        float
            Jump elevation in cm.
        """

        # from flight time
        flight_time = self.flight_phase.index
        flight_time = flight_time[-1] - flight_time[0]
        elevation_from_time = (flight_time**2) * G / 8 * 100

        # from force impulse
        elevation_from_velocity = (self.takeoff_velocity_ms**2) / (2 * G) * 100

        # return the lower of the two
        return min(elevation_from_time, elevation_from_velocity)

    @property
    def output_metrics(self):
        """
        Returns summary metrics for the jump.

        Returns
        -------
        pd.DataFrame
            DataFrame with summary metrics for the jump.
        """
        new = {
            "type": self.__class__.__name__,
            "side": self.side,
            "elevation_cm": self.elevation_cm,
            "takeoff_velocity_m/s": self.takeoff_velocity_ms,
            "contact_time_ms": self.contact_time_s * 1000,
            "flight_time_ms": self.flight_time_s * 1000,
        }
        ftime = float(new["flight_time_ms"])
        ctime = float(new["contact_time_ms"])
        new["flight_to_contact_ratio"] = ftime / ctime
        new = pd.DataFrame(pd.Series(new)).T
        return pd.concat(
            objs=[new, self._get_force_symmetry(), self._get_muscle_symmetry()],
            axis=1,
        )

    def __init__(
        self,
        bodymass_kg: float,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a Jump object.

        Parameters
        ----------
        bodymass_kg : float
            The subject's body mass in kilograms.
        left_foot_ground_reaction_force : ForcePlatform, optional
            ForcePlatform object for the left foot.
        right_foot_ground_reaction_force : ForcePlatform, optional
            ForcePlatform object for the right foot.
        vertical_axis : str, optional
            Name of the vertical axis in the force data (default "Y").
        anteroposterior_axis : str, optional
            Name of the anteroposterior axis in the force data (default "X").
        **signals : Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
            Additional signals to include in the record.

        Raises
        ------
        TypeError
            If left_foot or right_foot is not a ForcePlatform.
        ValueError
            If axes are not valid or bodymass_kg is not a float or int.
        """

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

        # check the inputs
        try:
            self._bodymass_kg = float(bodymass_kg)
        except Exception as exc:
            raise ValueError("bodymass_kg must be a float or int")

    @classmethod
    def from_tdf(
        cls,
        file: str,
        bodymass_kg: float | int,
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
        if all([i is None for i in mandatory_labels.values()]):
            raise ValueError(
                f"at least one of left_ground_reaction_force or"
                + " right_ground_reaction_force must be provided."
            )
        record = TimeseriesRecord.from_tdf(file)
        mandatory = {}
        for key, lbl in mandatory_labels.items():
            if lbl is not None:
                mandatory[key] = record.get(lbl)
                if mandatory[key] is None:
                    raise ValueError(f"{lbl} not found in the provided file.")
        signals = {i: v for i, v in record.items() if i not in list(mandatory.keys())}
        return cls(
            bodymass_kg=bodymass_kg,
            **signals,  # type: ignore
            **mandatory,  # type: ignore
        )


class JumpExercise(WholeBody):

    _bodymass_kg: float

    @property
    def side(self):
        """
        Returns which side(s) have force data.

        Returns
        -------
        str
            "bilateral", "left", or "right".
        """
        left_foot = self.get("ground_reaction_force_left_foot")
        right_foot = self.get("ground_reaction_force_left_foot")
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
        return self._bodymass_kg

    @property
    def jumps(self):
        jumps: list[SingleJump] = []
        vgrf = self.resultant_force
        if vgrf is None:
            raise ValueError("No resultant force was found")
        time = np.array(self.index)
        vgrf = vgrf.force.copy()[self.vertical_axis].to_numpy().flatten()

        # get the batches with grf lower than 30N (i.e flight phases)
        batches = continuous_batches(vgrf <= float(MINIMUM_CONTACT_FORCE_N))

        # remove those batches resulting in too short flight phases
        # a jump is assumed valid if the elevation is higher than 5 cm
        # (i.e. ~0.2s flight time)
        fsamp = 1 / np.mean(np.diff(time))
        min_samples = int(round(MINIMUM_FLIGHT_TIME_S * fsamp))
        batches = [i for i in batches if len(i) >= min_samples]

        # ensure that the first jump does not start with a flight
        if batches[0][0] == 0:
            batches = batches[1:]

        # ensure that the last jump does not end in flight
        if batches[-1][-1] == len(vgrf) - 1:
            batches = batches[:-1]

        # separate each jump
        flights_idx = np.where(vgrf < MINIMUM_CONTACT_FORCE_N)[0]
        for batch in batches:
            start_idx = flights_idx[flights_idx < batch[0]]
            start_idx = int(0 if len(start_idx) == 0 else (start_idx[-1] + 1))
            start = float(time[start_idx])
            stop = float(time[batch[-1] + 2])
            sliced = self.copy()[start:stop]
            if sliced is not None:
                if isinstance(sliced, TimeseriesRecord):
                    new_jump = SingleJump(
                        bodymass_kg=self.bodymass_kg,
                        **{i: v for i, v in sliced.items()},  # type: ignore
                    )
                    jumps += [new_jump]

        return jumps

    def __init__(
        self,
        bodymass_kg: float,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        **signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        """
        Initialize a Jump object.

        Parameters
        ----------
        bodymass_kg : float
            The subject's body mass in kilograms.
        left_foot_ground_reaction_force : ForcePlatform, optional
            ForcePlatform object for the left foot.
        right_foot_ground_reaction_force : ForcePlatform, optional
            ForcePlatform object for the right foot.
        vertical_axis : str, optional
            Name of the vertical axis in the force data (default "Y").
        anteroposterior_axis : str, optional
            Name of the anteroposterior axis in the force data (default "X").
        **signals : Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform
            Additional signals to include in the record.

        Raises
        ------
        TypeError
            If left_foot or right_foot is not a ForcePlatform.
        ValueError
            If axes are not valid or bodymass_kg is not a float or int.
        """

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

        # build the object
        super().__init__(
            **signals,
            **forces,
        )

        # check the inputs
        try:
            self._bodymass_kg = float(bodymass_kg)
        except Exception as exc:
            raise ValueError("bodymass_kg must be a float or int")

    @classmethod
    def from_tdf(
        cls,
        file: str,
        bodymass_kg: float | int,
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
        if all([i is None for i in mandatory_labels.values()]):
            raise ValueError(
                f"at least one of left_ground_reaction_force or"
                + " right_ground_reaction_force must be provided."
            )
        record = TimeseriesRecord.from_tdf(file)
        mandatory = {}
        for key, lbl in mandatory_labels.items():
            if lbl is not None:
                mandatory[key] = record.get(lbl)
                if mandatory[key] is None:
                    raise ValueError(f"{lbl} not found in the provided file.")
        signals = {i: v for i, v in record.items() if i not in list(mandatory.keys())}
        return cls(
            bodymass_kg=bodymass_kg,
            **mandatory,  # type: ignore
            **signals,  # type: ignore
        )
