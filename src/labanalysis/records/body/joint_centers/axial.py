"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Point3D
from ....referenceframes import ReferenceFrame

class AxialJointsMixin:
    """AxialJoints properties for WholeBody."""

    @property
    def head_center(self):
        """
        Calculate head center as centroid of 4 cranial markers.
        Returns
        -------
        Point3D
            Head center point (average of anterior, posterior, left, right markers).
        """
        h_ant = self._get_point("head_anterior")
        h_post = self._get_point("head_posterior")
        h_left = self._get_point("head_left")
        h_right = self._get_point("head_right")
        markers = [h_ant, h_post, h_left, h_right]
        marker_names = ["head_anterior", "head_posterior", "head_left", "head_right"]
        valid_markers = [m for m in markers if m is not None]
        if len(valid_markers) == 0:
            warnings.warn(
                f"Cannot calculate head_center: missing markers {marker_names}. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_point3d(ref)
        elif len(valid_markers) < 4:
            missing = [marker_names[i] for i, m in enumerate(markers) if m is None]
            warnings.warn(
                f"Calculating head_center with partial markers (missing {missing}).",
                UserWarning
            )
        # Calculate centroid
        data = sum(m.to_numpy() for m in valid_markers) / len(valid_markers)
        # Merge indices from all valid markers
        index = np.unique(
            np.concatenate([m.index for m in valid_markers])
        ).tolist()
        return Point3D(
            data=data,
            index=index,
            columns=h_ant.columns,
        )

    @property
    def neck_base(self):
        """
        Calculate neck base as midpoint between sternoclavicular joint (sc) and C7.
        Returns
        -------
        Point3D
            Neck3D base point (average of sc and c7 markers).
        """
        sc = self._get_point("sc")
        c7 = self._get_point("c7")
        # Calculate midpoint
        data = (sc.to_numpy() + c7.to_numpy()) / 2
        # Merge indices
        index = np.unique(np.concatenate([sc.index, c7.index])).tolist()
        return Point3D(
            data=data,
            index=index,
            columns=sc.columns,
        )

    @property
    def neck_referenceframe(self):
        """
        Neck reference frame for angular measurements.
        Reference Frame
        --------------
        The reference frame has three semantic axes constructed from anatomical landmarks:
        - **lateral_axis**: Mediolateral direction (construction details in code below)
        - **vertical_axis**: Superior-inferior direction (construction details in code below)
        - **anteroposterior_axis**: Anterior-posterior direction (construction details in code below)
        Note: The rotation matrix columns [0], [1], [2] correspond to lateral_axis, vertical_axis,
        and anteroposterior_axis respectively. These semantic meanings are fixed by construction,
        independent of global coordinate system configuration.
        Origin
        ------
        Neck base (use `self.neck_base` property)
        Construction
        ------------
        1. vertical_axis: UP (pelvis_center → neck_base)
        2. anteroposterior_axis: FORWARD (C7 → sc)
        3. lateral_axis: LEFT (cross product vertical × anteroposterior)
        4. Apply Gram-Schmidt orthonormalization
        Returns
        -------
        ReferenceFrame
            Reference frame with origin at neck base and orthonormal axes.
        See Also
        --------
        neck_base : Neck base (origin of this frame)
        pelvis_center : Pelvis center (used for Y-axis)
        neck_flexionextension : Neck flexion angle using this frame
        neck_lateralflexion : Neck lateral flexion using this frame
        pelvis_rotation : Pelvis rotation using this frame
        """
        neck_base = self.neck_base
        pelvis_center = self.pelvis_center
        # Construct vertical_axis: UP (pelvis_center to neck_base)
        axis_y = (neck_base - pelvis_center).to_numpy()
        axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)
        # Construct anteroposterior_axis: FORWARD (C7 to sternoclavicular junction)
        c7 = self._get_point("c7")
        sc = self._get_point("sc")
        axis_z = (sc - c7).to_numpy()
        axis_z = axis_z / np.linalg.norm(axis_z, axis=1, keepdims=True)
        # Construct lateral_axis: LEFT (cross product vertical × anteroposterior, to point left in right-handed system)
        axis_x = np.cross(axis_y, axis_z)
        return ReferenceFrame(
            origin=neck_base,
            lateral_axis=axis_x,
            vertical_axis=axis_y,
            anteroposterior_axis=axis_z,
        )
