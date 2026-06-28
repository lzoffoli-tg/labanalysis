"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Timeseries

class AnglesHelpersMixin:
    """AnglesHelpers properties for WholeBody."""

    @property
    def _left_foot_plane(self):
        """
        Calculate left foot plane from toe, heel, and metatarsal markers.
        The plane is defined by least-squares fitting through four points:
        left_toe, left_heel, left_first_metatarsal_head, and left_fifth_metatarsal_head.
        Returns
        -------
        Timeseries
            Plane coefficients [a, b, c, d] defining the equation ax + by + cz + d = 0.
            Unit: dimensionless (a.u.).
        Notes
        -----
        The plane normal vector (a, b, c) is normalized to unit length.
        Used by left_foot_height and ankle angle calculations.
        """
        toe = self._get_point("left_toe")
        first_meta = self._get_point("left_first_metatarsal_head")
        fifth_meta = self._get_point("left_fifth_metatarsal_head")
        heel = self._get_point("left_heel")
        points = []
        if toe is not None:
            points.append(toe)
        if heel is not None:
            points.append(heel)
        if first_meta is not None:
            points.append(first_meta)
        if fifth_meta is not None:
            points.append(fifth_meta)
        if len(points) < 3:
            marker_names = ["left_toe", "left_heel", "left_first_metatarsal_head", "left_fifth_metatarsal_head"]
            all_markers = [toe, heel, first_meta, fifth_meta]
            missing = [marker_names[i] for i, m in enumerate(all_markers) if m is None]
            warnings.warn(
                f"Cannot calculate _left_foot_plane: not enough markers (missing {missing}). Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            if ref is not None:
                n_samples = len(ref.index)
                index = ref.index
            else:
                n_samples = 1
                index = [0.0]
            return Timeseries(
                data=np.full((n_samples, 4), np.nan),
                index=index,
                columns=["a", "b", "c", "d"],
                unit="a.u.",
            )
        return Timeseries(
            data=self._get_least_squares_plane_coefs(*points),
            index=points[0].index,
            columns=["a", "b", "c", "d"],
            unit="a.u.",
        )

    @property
    def _pelvis_plane(self):
        """
        Calculate pelvis plane using least squares method.
        This avoids circular dependency: pelvis_referenceframe needs hip_center,
        hip_center needs left/right_hip, which need pelvis_height, which needs
        _pelvis_plane. We use least squares plane fitting instead.
        """
        # get the pelvis references
        l_asis = self._get_point("left_asis")
        r_asis = self._get_point("right_asis")
        l_psis = self._get_point("left_psis")
        r_psis = self._get_point("right_psis")
        markers = [l_asis, r_asis, l_psis, r_psis]
        marker_names = ["left_asis", "right_asis", "left_psis", "right_psis"]
        valid_markers = [m for m in markers if m is not None]
        if len(valid_markers) < 3:
            missing = [marker_names[i] for i, m in enumerate(markers) if m is None]
            warnings.warn(
                f"Cannot calculate _pelvis_plane: not enough markers (need 3, missing {missing}). Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            if ref is not None:
                n_samples = len(ref.index)
                index = ref.index
            else:
                n_samples = 1
                index = [0.0]
            return Timeseries(
                data=np.full((n_samples, 4), np.nan),
                index=index,
                columns=["a", "b", "c", "d"],
                unit="a.u.",
            )
        # generate the pelvis plane by least squares
        plane = self._get_least_squares_plane_coefs(*valid_markers)
        # return the timeseries
        return Timeseries(
            data=plane,
            index=valid_markers[0].index,
            columns=["a", "b", "c", "d"],
            unit="a.u.",
        )

    @property
    def _right_foot_plane(self):
        """
        Calculate right foot plane from toe, heel, and metatarsal markers.
        The plane is defined by least-squares fitting through four points:
        right_toe, right_heel, right_first_metatarsal_head, and right_fifth_metatarsal_head.
        Returns
        -------
        Timeseries
            Plane coefficients [a, b, c, d] defining the equation ax + by + cz + d = 0.
            Unit: dimensionless (a.u.).
        Notes
        -----
        The plane normal vector (a, b, c) is normalized to unit length.
        Used by right_foot_height and ankle angle calculations.
        """
        toe = self._get_point("right_toe")
        first_meta = self._get_point("right_first_metatarsal_head")
        fifth_meta = self._get_point("right_fifth_metatarsal_head")
        heel = self._get_point("right_heel")
        points = []
        if toe is not None:
            points.append(toe)
        if heel is not None:
            points.append(heel)
        if first_meta is not None:
            points.append(first_meta)
        if fifth_meta is not None:
            points.append(fifth_meta)
        if len(points) < 3:
            marker_names = ["right_toe", "right_heel", "right_first_metatarsal_head", "right_fifth_metatarsal_head"]
            all_markers = [toe, heel, first_meta, fifth_meta]
            missing = [marker_names[i] for i, m in enumerate(all_markers) if m is None]
            warnings.warn(
                f"Cannot calculate _right_foot_plane: not enough markers (missing {missing}). Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            if ref is not None:
                n_samples = len(ref.index)
                index = ref.index
            else:
                n_samples = 1
                index = [0.0]
            return Timeseries(
                data=np.full((n_samples, 4), np.nan),
                index=index,
                columns=["a", "b", "c", "d"],
                unit="a.u.",
            )
        return Timeseries(
            data=self._get_least_squares_plane_coefs(*points),
            index=points[0].index,
            columns=["a", "b", "c", "d"],
            unit="a.u.",
        )
