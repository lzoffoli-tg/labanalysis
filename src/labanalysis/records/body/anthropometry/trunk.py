"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D

class TrunkMeasuresMixin:
    """TrunkMeasures properties for WholeBody."""

    @property
    def hip_width(self):
        """
        Calculate hip width as distance between left and right trochanters.
        Returns
        -------
        Signal1D
            Distance in meters between greater trochanter markers.
        """
        l_troch = self._get_point("left_trochanter")
        r_troch = self._get_point("right_trochanter")
        if l_troch is None or r_troch is None:
            warnings.warn(
                "Cannot calculate hip_width: missing markers ['left_trochanter' or 'right_trochanter']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([l_troch.index, r_troch.index])).tolist()
        data = np.asarray(l_troch - r_troch)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=l_troch.unit)

    @property
    def shoulder_width(self):
        """
        Calculate shoulder width as distance between left and right shoulders.
        Returns
        -------
        Signal1D
            Distance in meters between shoulder joint centers.
        """
        try:
            l_shoulder = self.left_shoulder
            r_shoulder = self.right_shoulder
            index = np.unique(np.concatenate([l_shoulder.index, r_shoulder.index])).tolist()
            data = np.asarray(l_shoulder - r_shoulder)
            data = np.sum(data**2, axis=1) ** 0.5
            return Signal1D(data=data, index=index, unit=l_shoulder.unit)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate shoulder_width: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def trunk_length(self):
        """
        Calculate trunk length as vertical distance from pelvis center to neck base.
        The trunk length is computed as the Euclidean distance between the pelvis center
        (average of ASIS markers) and the neck base (midpoint between C7 and SC markers).
        Returns
        -------
        Signal1D
            Distance in meters from pelvis to neck base.
        Notes
        -----
        Requires pelvis markers (left_asis, right_asis) and neck markers (c7, sc).
        """
        try:
            # Get pelvis center from ASIS markers
            pelvis_center = self.pelvis_center
            # Get neck base from neck markers
            neck_base = self.neck_base
            # Calculate distance
            index = np.unique(
                np.concatenate([pelvis_center.index, neck_base.index])
            ).tolist()
            data = np.asarray(neck_base - pelvis_center)
            data = np.sum(data**2, axis=1) ** 0.5
            return Signal1D(data=data, index=index, unit=pelvis_center.unit)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate trunk_length: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
