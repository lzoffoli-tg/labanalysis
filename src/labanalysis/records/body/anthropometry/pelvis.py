"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D

class PelvisMeasuresMixin:
    """PelvisMeasures properties for WholeBody."""

    @property
    def pelvis_height(self):
        """
        Pelvis height (perpendicular distance from pelvis plane to trochanter).
        Calculated as the average distance from left and right trochanters to the
        pelvis plane (defined by ASIS and PSIS markers). If only one trochanter is
        available, returns that distance.
        Returns
        -------
        Signal1D
            Distance in meters from pelvis plane to trochanter markers.
            Returns NaN if both trochanter markers are missing.
        See Also
        --------
        pelvis_width : Pelvis width (ASIS to ASIS distance)
        _pelvis_plane : Pelvis plane coefficients
        """
        pelvis_plane = self._pelvis_plane
        l_troch = self._get_point("left_trochanter")
        r_troch = self._get_point("right_trochanter")
        l_height = None
        r_height = None
        if l_troch is not None:
            try:
                l_height = self._get_point_to_plane_distance(l_troch, pelvis_plane)
            except Exception:
                pass
        if r_troch is not None:
            try:
                r_height = self._get_point_to_plane_distance(r_troch, pelvis_plane)
            except Exception:
                pass
        if l_height and r_height:
            return Signal1D(
                data=(l_height + r_height).to_numpy() / 2,
                index=pelvis_plane.index,
                unit=l_height.unit,
            )
        if l_height:
            return l_height
        if r_height:
            return r_height
        warnings.warn(
            "Cannot calculate pelvis_height: missing markers ['left_trochanter', 'right_trochanter']. Returning NaN.",
            UserWarning
        )
        ref = self._find_any_valid_marker()
        return self._create_nan_signal1d(ref)

    @property
    def pelvis_width(self):
        """
        Pelvis width (distance between left and right ASIS markers).
        Returns
        -------
        Signal1D
            Distance in meters between left and right anterior superior iliac spine markers.
            Returns NaN if either marker is missing.
        See Also
        --------
        pelvis_height : Pelvis height (ASIS plane to trochanter distance)
        hip_width : Hip width (trochanter to trochanter distance)
        """
        l_asis = self._get_point("left_asis")
        r_asis = self._get_point("right_asis")
        if l_asis is None or r_asis is None:
            warnings.warn(
                "Cannot calculate pelvis_width: missing markers ['left_asis' or 'right_asis']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = [l_asis.index, r_asis.index]
        index = np.unique(np.concatenate(index)).tolist()
        data = np.asarray(l_asis - r_asis)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=l_asis.unit)
