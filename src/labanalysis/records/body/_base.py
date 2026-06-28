"""Base class for WholeBody with initialization and TDF loading."""

import warnings

import numpy as np

from ..timeseriesrecord import TimeseriesRecord
from ..forceplatform import ForcePlatform
from ...timeseries import Signal1D, Signal3D, EMGSignal, Point3D

__all__ = ["WholeBodyBase"]


class WholeBodyBase(TimeseriesRecord):
    """
    Base class for full body biomechanical model.

    Provides initialization and TDF loading functionality.
    All property calculations are implemented in mixin classes.
    """

    _angular_measures: list[str] = []

    def __init__(
        self,
        left_hand_ground_reaction_force: ForcePlatform | None = None,
        right_hand_ground_reaction_force: ForcePlatform | None = None,
        left_foot_ground_reaction_force: ForcePlatform | None = None,
        right_foot_ground_reaction_force: ForcePlatform | None = None,
        left_heel: Point3D | None = None,
        right_heel: Point3D | None = None,
        left_toe: Point3D | None = None,
        right_toe: Point3D | None = None,
        left_first_metatarsal_head: Point3D | None = None,
        left_fifth_metatarsal_head: Point3D | None = None,
        right_first_metatarsal_head: Point3D | None = None,
        right_fifth_metatarsal_head: Point3D | None = None,
        left_ankle_medial: Point3D | None = None,
        left_ankle_lateral: Point3D | None = None,
        right_ankle_medial: Point3D | None = None,
        right_ankle_lateral: Point3D | None = None,
        left_knee_medial: Point3D | None = None,
        left_knee_lateral: Point3D | None = None,
        right_knee_medial: Point3D | None = None,
        right_knee_lateral: Point3D | None = None,
        right_trochanter: Point3D | None = None,
        left_trochanter: Point3D | None = None,
        left_asis: Point3D | None = None,
        right_asis: Point3D | None = None,
        left_psis: Point3D | None = None,
        right_psis: Point3D | None = None,
        left_shoulder_anterior: Point3D | None = None,
        left_shoulder_posterior: Point3D | None = None,
        left_acromion: Point3D | None = None,
        right_shoulder_anterior: Point3D | None = None,
        right_shoulder_posterior: Point3D | None = None,
        right_acromion: Point3D | None = None,
        left_elbow_medial: Point3D | None = None,
        left_elbow_lateral: Point3D | None = None,
        right_elbow_medial: Point3D | None = None,
        right_elbow_lateral: Point3D | None = None,
        left_wrist_medial: Point3D | None = None,
        left_wrist_lateral: Point3D | None = None,
        right_wrist_medial: Point3D | None = None,
        right_wrist_lateral: Point3D | None = None,
        s2: Point3D | None = None,
        l2: Point3D | None = None,
        c7: Point3D | None = None,
        t5: Point3D | None = None,
        sc: Point3D | None = None,
        head_anterior: Point3D | None = None,
        head_posterior: Point3D | None = None,
        head_left: Point3D | None = None,
        head_right: Point3D | None = None,
        **extra_signals: Signal1D | Signal3D | EMGSignal | Point3D | ForcePlatform,
    ):
        signals = {
            **extra_signals,
            **dict(
                left_hand_ground_reaction_force=left_hand_ground_reaction_force,
                right_hand_ground_reaction_force=right_hand_ground_reaction_force,
                left_foot_ground_reaction_force=left_foot_ground_reaction_force,
                right_foot_ground_reaction_force=right_foot_ground_reaction_force,
                left_heel=left_heel,
                right_heel=right_heel,
                left_toe=left_toe,
                right_toe=right_toe,
                left_first_metatarsal_head=left_first_metatarsal_head,
                left_fifth_metatarsal_head=left_fifth_metatarsal_head,
                right_first_metatarsal_head=right_first_metatarsal_head,
                right_fifth_metatarsal_head=right_fifth_metatarsal_head,
                left_ankle_medial=left_ankle_medial,
                left_ankle_lateral=left_ankle_lateral,
                right_ankle_medial=right_ankle_medial,
                right_ankle_lateral=right_ankle_lateral,
                left_knee_medial=left_knee_medial,
                left_knee_lateral=left_knee_lateral,
                right_knee_medial=right_knee_medial,
                right_knee_lateral=right_knee_lateral,
                left_trochanter=left_trochanter,
                right_trochanter=right_trochanter,
                left_asis=left_asis,
                right_asis=right_asis,
                left_psis=left_psis,
                right_psis=right_psis,
                left_shoulder_anterior=left_shoulder_anterior,
                left_shoulder_posterior=left_shoulder_posterior,
                left_acromion=left_acromion,
                right_shoulder_anterior=right_shoulder_anterior,
                right_shoulder_posterior=right_shoulder_posterior,
                right_acromion=right_acromion,
                left_elbow_medial=left_elbow_medial,
                left_elbow_lateral=left_elbow_lateral,
                right_elbow_medial=right_elbow_medial,
                right_elbow_lateral=right_elbow_lateral,
                left_wrist_medial=left_wrist_medial,
                left_wrist_lateral=left_wrist_lateral,
                right_wrist_medial=right_wrist_medial,
                right_wrist_lateral=right_wrist_lateral,
                s2=s2,
                c7=c7,
                t5=t5,
                sc=sc,
                l2=l2,
                head_anterior=head_anterior,
                head_posterior=head_posterior,
                head_left=head_left,
                head_right=head_right,
            ),
        }
        super().__init__(**{i: v for i, v in signals.items() if v is not None})

    @classmethod
    def from_tdf(
        cls,
        filename: str,
        strict_labels: bool = False,
        ground_reaction_force: str | None = None,
        left_hand_ground_reaction_force: str | None = None,
        right_hand_ground_reaction_force: str | None = None,
        left_foot_ground_reaction_force: str | None = None,
        right_foot_ground_reaction_force: str | None = None,
        left_heel: str | None = None,
        right_heel: str | None = None,
        left_toe: str | None = None,
        right_toe: str | None = None,
        left_first_metatarsal_head: str | None = None,
        left_fifth_metatarsal_head: str | None = None,
        right_first_metatarsal_head: str | None = None,
        right_fifth_metatarsal_head: str | None = None,
        left_ankle_medial: str | None = None,
        left_ankle_lateral: str | None = None,
        right_ankle_medial: str | None = None,
        right_ankle_lateral: str | None = None,
        left_knee_medial: str | None = None,
        left_knee_lateral: str | None = None,
        right_knee_medial: str | None = None,
        right_knee_lateral: str | None = None,
        right_trochanter: str | None = None,
        left_trochanter: str | None = None,
        left_asis: str | None = None,
        right_asis: str | None = None,
        left_psis: str | None = None,
        right_psis: str | None = None,
        left_shoulder_anterior: str | None = None,
        left_shoulder_posterior: str | None = None,
        left_acromion: str | None = None,
        right_shoulder_anterior: str | None = None,
        right_shoulder_posterior: str | None = None,
        right_acromion: str | None = None,
        left_elbow_medial: str | None = None,
        left_elbow_lateral: str | None = None,
        right_elbow_medial: str | None = None,
        right_elbow_lateral: str | None = None,
        left_wrist_medial: str | None = None,
        left_wrist_lateral: str | None = None,
        right_wrist_medial: str | None = None,
        right_wrist_lateral: str | None = None,
        s2: str | None = None,
        l2: str | None = None,
        c7: str | None = None,
        t5: str | None = None,
        sc: str | None = None,
        head_anterior: str | None = None,
        head_posterior: str | None = None,
        head_left: str | None = None,
        head_right: str | None = None,
    ):
        """
        Create a WholeBody from a TDF file.

        Parameters
        ----------
        filename : str
            Path to TDF file
        strict_labels : bool, optional
            If True, raise ValueError when a provided label is not found in TDF.
            If False (default), emit UserWarning and skip missing labels.
        ground_reaction_force : str, optional
            Label for ground reaction force in TDF
        left_hand_ground_reaction_force : str, optional
            Label for left hand GRF in TDF
        right_hand_ground_reaction_force : str, optional
            Label for right hand GRF in TDF
        left_foot_ground_reaction_force : str, optional
            Label for left foot GRF in TDF
        right_foot_ground_reaction_force : str, optional
            Label for right foot GRF in TDF
        left_heel : str, optional
            Label for left heel marker in TDF
        right_heel : str, optional
            Label for right heel marker in TDF
        left_toe : str, optional
            Label for left toe marker in TDF
        right_toe : str, optional
            Label for right toe marker in TDF
        left_first_metatarsal_head : str, optional
            Label for left first metatarsal head marker in TDF
        left_fifth_metatarsal_head : str, optional
            Label for left fifth metatarsal head marker in TDF
        right_first_metatarsal_head : str, optional
            Label for right first metatarsal head marker in TDF
        right_fifth_metatarsal_head : str, optional
            Label for right fifth metatarsal head marker in TDF
        left_ankle_medial : str, optional
            Label for left medial malleolus marker in TDF
        left_ankle_lateral : str, optional
            Label for left lateral malleolus marker in TDF
        right_ankle_medial : str, optional
            Label for right medial malleolus marker in TDF
        right_ankle_lateral : str, optional
            Label for right lateral malleolus marker in TDF
        left_knee_medial : str, optional
            Label for left medial femoral epicondyle marker in TDF
        left_knee_lateral : str, optional
            Label for left lateral femoral epicondyle marker in TDF
        right_knee_medial : str, optional
            Label for right medial femoral epicondyle marker in TDF
        right_knee_lateral : str, optional
            Label for right lateral femoral epicondyle marker in TDF
        left_trochanter : str, optional
            Label for left greater trochanter marker in TDF
        right_trochanter : str, optional
            Label for right greater trochanter marker in TDF
        left_asis : str, optional
            Label for left ASIS marker in TDF
        right_asis : str, optional
            Label for right ASIS marker in TDF
        left_psis : str, optional
            Label for left PSIS marker in TDF
        right_psis : str, optional
            Label for right PSIS marker in TDF
        left_shoulder_anterior : str, optional
            Label for left anterior shoulder marker in TDF
        left_shoulder_posterior : str, optional
            Label for left posterior shoulder marker in TDF
        left_acromion : str, optional
            Label for left acromion marker in TDF
        right_shoulder_anterior : str, optional
            Label for right anterior shoulder marker in TDF
        right_shoulder_posterior : str, optional
            Label for right posterior shoulder marker in TDF
        right_acromion : str, optional
            Label for right acromion marker in TDF
        left_elbow_medial : str, optional
            Label for left medial epicondyle marker in TDF
        left_elbow_lateral : str, optional
            Label for left lateral epicondyle marker in TDF
        right_elbow_medial : str, optional
            Label for right medial epicondyle marker in TDF
        right_elbow_lateral : str, optional
            Label for right lateral epicondyle marker in TDF
        left_wrist_medial : str, optional
            Label for left medial wrist marker in TDF
        left_wrist_lateral : str, optional
            Label for left lateral wrist marker in TDF
        right_wrist_medial : str, optional
            Label for right medial wrist marker in TDF
        right_wrist_lateral : str, optional
            Label for right lateral wrist marker in TDF
        s2 : str, optional
            Label for second sacral vertebra marker in TDF
        l2 : str, optional
            Label for second lumbar vertebra marker in TDF
        c7 : str, optional
            Label for seventh cervical vertebra marker in TDF
        t5 : str, optional
            Label for fifth thoracic vertebra marker in TDF
        sc : str, optional
            Label for sternoclavicular joint marker in TDF
        head_anterior : str, optional
            Label for anterior cranium marker in TDF
        head_posterior : str, optional
            Label for posterior cranium marker in TDF
        head_left : str, optional
            Label for left cranium marker in TDF
        head_right : str, optional
            Label for right cranium marker in TDF

        Returns
        -------
        WholeBody
            Instance created from TDF data
        """
        tdf = TimeseriesRecord.from_tdf(filename)

        points = {
            "left_heel": left_heel,
            "right_heel": right_heel,
            "left_toe": left_toe,
            "right_toe": right_toe,
            "left_first_metatarsal_head": left_first_metatarsal_head,
            "left_fifth_metatarsal_head": left_fifth_metatarsal_head,
            "right_first_metatarsal_head": right_first_metatarsal_head,
            "right_fifth_metatarsal_head": right_fifth_metatarsal_head,
            "left_ankle_medial": left_ankle_medial,
            "left_ankle_lateral": left_ankle_lateral,
            "right_ankle_medial": right_ankle_medial,
            "right_ankle_lateral": right_ankle_lateral,
            "left_knee_medial": left_knee_medial,
            "left_knee_lateral": left_knee_lateral,
            "right_knee_medial": right_knee_medial,
            "right_knee_lateral": right_knee_lateral,
            "right_trochanter": right_trochanter,
            "left_trochanter": left_trochanter,
            "left_asis": left_asis,
            "right_asis": right_asis,
            "left_psis": left_psis,
            "right_psis": right_psis,
            "left_shoulder_anterior": left_shoulder_anterior,
            "left_shoulder_posterior": left_shoulder_posterior,
            "left_acromion": left_acromion,
            "right_shoulder_anterior": right_shoulder_anterior,
            "right_shoulder_posterior": right_shoulder_posterior,
            "right_acromion": right_acromion,
            "left_elbow_medial": left_elbow_medial,
            "left_elbow_lateral": left_elbow_lateral,
            "right_elbow_medial": right_elbow_medial,
            "right_elbow_lateral": right_elbow_lateral,
            "left_wrist_medial": left_wrist_medial,
            "left_wrist_lateral": left_wrist_lateral,
            "right_wrist_medial": right_wrist_medial,
            "right_wrist_lateral": right_wrist_lateral,
            "s2": s2,
            "c7": c7,
            "l2": l2,
            "t5": t5,
            "sc": sc,
            "head_anterior": head_anterior,
            "head_posterior": head_posterior,
            "head_left": head_left,
            "head_right": head_right,
        }
        forces = {
            "ground_reaction_force": ground_reaction_force,
            "left_hand_ground_reaction_force": left_hand_ground_reaction_force,
            "right_hand_ground_reaction_force": right_hand_ground_reaction_force,
            "left_foot_ground_reaction_force": left_foot_ground_reaction_force,
            "right_foot_ground_reaction_force": right_foot_ground_reaction_force,
        }
        keys = tdf.keys()
        mandatory = {}
        for key, lbl in forces.items():
            if lbl is not None:
                if lbl not in keys:
                    if strict_labels:
                        raise ValueError(f"{lbl} not found.")
                    else:
                        warnings.warn(f"Marker '{lbl}' not found in TDF file. Skipping.", UserWarning)
                        continue
                if not isinstance(tdf[lbl], ForcePlatform):
                    raise ValueError(f"{lbl} must be a ForcePlatform instance.")
                mandatory[key] = tdf[lbl]
                tdf.drop(lbl, True)
        for key, lbl in points.items():
            if lbl is not None:
                if lbl not in keys:
                    if strict_labels:
                        raise ValueError(f"{lbl} not found.")
                    else:
                        warnings.warn(f"Marker '{lbl}' not found in TDF file. Skipping.", UserWarning)
                        continue
                if not isinstance(tdf[lbl], Point3D):
                    raise ValueError(f"{lbl} must be a Point3D instance.")
                mandatory[key] = tdf[lbl]
                tdf.drop(lbl, inplace=True)
        extras = {i: v for i, v in tdf.items() if i not in list(mandatory.keys())}

        return cls(**mandatory, **extras)

    def _get_point(self, label: str):
        """
        Get a Point3D marker by label with graceful degradation.

        Parameters
        ----------
        label : str
            Marker label

        Returns
        -------
        Point3D or None
            Point3D if found and is correct type, None otherwise
        """
        element = self.get(label)
        if element is None:
            warnings.warn(f"Marker '{label}' not found.", UserWarning)
            return None
        if not isinstance(element, Point3D):
            raise ValueError(f"{label} is not a Point3D.")
        return element

    def _find_any_valid_marker(self):
        """
        Find any valid Point3D marker in the body to use as reference for index.

        Returns
        -------
        Point3D or None
            First Point3D found in the body data, or None if no markers exist.
        """
        for key, value in self._data.items():
            if isinstance(value, Point3D):
                return value
        return None

    def _create_nan_point3d(self, reference_point=None):
        """
        Create a Point3D filled with NaN values.

        Parameters
        ----------
        reference_point : Point3D, optional
            Reference point to match index and shape. If None, uses a default single-sample.

        Returns
        -------
        Point3D
            Point3D with same shape as reference_point but filled with NaN.
        """
        if reference_point is not None:
            n_samples = len(reference_point.index)
            index = reference_point.index
        else:
            n_samples = 1
            index = [0.0]

        return Point3D(
            data=np.full((n_samples, 3), np.nan),
            index=index,
            columns=["X", "Y", "Z"],
            unit="mm"
        )

    def _create_nan_signal1d(self, reference_signal=None):
        """
        Create a Signal1D filled with NaN values.

        Parameters
        ----------
        reference_signal : Timeseries, optional
            Reference signal to match index and shape. If None, uses a default single-sample.

        Returns
        -------
        Signal1D
            Signal1D with same shape as reference_signal but filled with NaN.
        """
        if reference_signal is not None:
            n_samples = len(reference_signal.index)
            index = reference_signal.index
        else:
            n_samples = 1
            index = [0.0]

        return Signal1D(
            data=np.full(n_samples, np.nan),
            index=index,
            unit="deg"
        )
