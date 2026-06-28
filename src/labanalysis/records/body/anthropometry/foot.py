"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D

class FootMeasuresMixin:
    """FootMeasures properties for WholeBody."""

    @property
    def left_foot_height(self):
        """
        Calculate left foot height as perpendicular distance from ankle to foot plane.
        Returns
        -------
        Signal1D
            Distance in meters from ankle joint to foot plane.
        """
        try:
            ankle = self.left_ankle
            foot_plane = self._left_foot_plane
            return self._get_point_to_plane_distance(ankle, foot_plane)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_foot_height: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def left_foot_length(self):
        """
        Calculate left foot length as distance from heel to toe.
        Returns
        -------
        Signal1D
            Distance in meters from heel to toe marker.
        """
        heel = self._get_point("left_heel")
        toe = self._get_point("left_toe")
        if heel is None or toe is None:
            warnings.warn(
                "Cannot calculate left_foot_length: missing markers ['left_heel' or 'left_toe']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([heel.index, toe.index])).tolist()
        data = np.asarray(toe - heel)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=heel.unit)

    @property
    def left_foot_width(self):
        """
        Calculate left foot width as distance between first and fifth metatarsal heads.
        Returns
        -------
        Signal1D
            Distance in meters between first and fifth metatarsal heads.
        """
        first = self._get_point("left_first_metatarsal_head")
        fifth = self._get_point("left_fifth_metatarsal_head")
        if first is None or fifth is None:
            warnings.warn(
                "Cannot calculate left_foot_width: missing markers ['left_first_metatarsal_head' or 'left_fifth_metatarsal_head']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([first.index, fifth.index])).tolist()
        data = np.asarray(fifth - first)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=first.unit)

    @property
    def right_foot_height(self):
        """
        Calculate right foot height as perpendicular distance from ankle to foot plane.
        Returns
        -------
        Signal1D
            Distance in meters from ankle joint to foot plane.
        """
        try:
            ankle = self.right_ankle
            foot_plane = self._right_foot_plane
            return self._get_point_to_plane_distance(ankle, foot_plane)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_foot_height: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_foot_length(self):
        """
        Calculate right foot length as distance from heel to toe.
        Returns
        -------
        Signal1D
            Distance in meters from heel to toe marker.
        """
        heel = self._get_point("right_heel")
        toe = self._get_point("right_toe")
        if heel is None or toe is None:
            warnings.warn(
                "Cannot calculate right_foot_length: missing markers ['right_heel' or 'right_toe']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([heel.index, toe.index])).tolist()
        data = np.asarray(toe - heel)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=heel.unit)

    @property
    def right_foot_width(self):
        """
        Calculate right foot width as distance between first and fifth metatarsal heads.
        Returns
        -------
        Signal1D
            Distance in meters between first and fifth metatarsal heads.
        """
        first = self._get_point("right_first_metatarsal_head")
        fifth = self._get_point("right_fifth_metatarsal_head")
        if first is None or fifth is None:
            warnings.warn(
                "Cannot calculate right_foot_width: missing markers ['right_first_metatarsal_head' or 'right_fifth_metatarsal_head']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([first.index, fifth.index])).tolist()
        data = np.asarray(fifth - first)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=first.unit)
