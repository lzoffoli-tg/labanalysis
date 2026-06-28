"""Reference frame transformations module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..signalprocessing import gram_schmidt
from ..timeseries import Timeseries

if TYPE_CHECKING:
    from ..records import ForcePlatform, Record

__all__ = ["ReferenceFrame"]


class ReferenceFrame:
    """
    A reference frame defined by an origin and three orthonormal axes.

    This class encapsulates coordinate system transformations, allowing you to
    define a reference frame once and apply it to various data types including
    numpy arrays, pandas DataFrames, TimeSeries objects, and ForcePlatform objects.

    The axes are automatically orthonormalized using Gram-Schmidt process during
    construction. Unlike coordinate-specific naming (X/Y/Z), this class uses
    anatomically meaningful axis names (lateral/vertical/anteroposterior) to make
    the code independent of the user's coordinate system configuration.

    Parameters
    ----------
    origin : np.ndarray or list or tuple or Timeseries
        Origin point of the reference frame. Shape (N, 3) or (3,).

        If Timeseries (Point3D, Signal3D): ._data is extracted automatically.
        Must have exactly 3 columns.
    lateral_axis : np.ndarray or list or tuple or Timeseries
        First axis direction (lateral). Defines the mediolateral direction of the
        reference frame. Shape (N, 3) or (3,).

        If Timeseries (Point3D, Signal3D): ._data is extracted automatically.
        Must have exactly 3 columns.
    vertical_axis : np.ndarray or list or tuple or Timeseries
        Second axis direction (vertical). Defines the superior-inferior direction
        of the reference frame. Shape (N, 3) or (3,).

        If Timeseries (Point3D, Signal3D): ._data is extracted automatically.
        Must have exactly 3 columns.
    anteroposterior_axis : np.ndarray or list or tuple or Timeseries or None, optional
        Third axis direction (anteroposterior). Defines the forward-backward direction.
        If None, computed as cross product of lateral_axis and vertical_axis.
        Shape (N, 3) or (3,).

        If Timeseries (Point3D, Signal3D): ._data is extracted automatically.
        Must have exactly 3 columns.

    Attributes
    ----------
    origin : np.ndarray
        Origin point, shape (N, 3).
    lateral_axis : np.ndarray
        Lateral axis direction (mediolateral), shape (N, 3).
    vertical_axis : np.ndarray
        Vertical axis direction (superior-inferior), shape (N, 3).
    anteroposterior_axis : np.ndarray or None
        Anteroposterior axis direction (forward-backward), shape (N, 3) or None.
    rotation_matrix : np.ndarray
        Orthonormalized rotation matrix, shape (N, 3, 3).
        Column 0 = lateral_axis, Column 1 = vertical_axis, Column 2 = anteroposterior_axis.

    Examples
    --------
    >>> # Example 1: Define a single reference frame with lists
    >>> origin = [0.5, 1.0, 0.2]
    >>> lateral_axis = [1.0, 0.0, 0.0]   # Points laterally (e.g., left)
    >>> vertical_axis = [0.0, 1.0, 0.0]  # Points vertically (e.g., up)
    >>> ref_frame = ReferenceFrame(origin, lateral_axis, vertical_axis)

    >>> # Example 2: Apply to numpy array (broadcasting)
    >>> data = np.random.rand(100, 3)
    >>> transformed = ref_frame.apply(data)

    >>> # Example 3: Using Point3D objects directly (new in v206)
    >>> hip = body.left_hip          # Point3D with shape (1000, 3)
    >>> knee = body.left_knee        # Point3D
    >>> lateral_vec = body.right_hip - body.left_hip  # Point3D arithmetic
    >>>
    >>> ref_frame = ReferenceFrame(
    ...     origin=hip,                # Point3D accepted directly
    ...     lateral_axis=lateral_vec,  # Point3D from subtraction
    ...     vertical_axis=knee - hip   # Point3D expression
    ... )
    >>> # No .to_numpy() needed! Arrays extracted automatically.

    >>> # Example 4: Apply to Point3D
    >>> marker = Point3D(...)
    >>> transformed_marker = ref_frame(marker)  # Callable interface

    Notes
    -----
    The semantic axis naming (lateral/vertical/anteroposterior) makes the code
    independent of specific coordinate conventions. Whether the user's coordinate
    system uses X=lateral or X=vertical, the ReferenceFrame always knows which
    axis represents which anatomical direction.
    """

    @staticmethod
    def _extract_numpy_array(param, param_name: str) -> np.ndarray:
        """
        Extract numpy array from parameter.

        Accepts:
        - np.ndarray: returned as-is
        - list/tuple: converted to array
        - Timeseries (including Point3D, Signal3D): extracts ._data

        Parameters
        ----------
        param : np.ndarray | list | tuple | Timeseries
            Input parameter to convert
        param_name : str
            Name of parameter (for error messages)

        Returns
        -------
        np.ndarray
            Numpy array extracted from input

        Raises
        ------
        TypeError
            If parameter type is not supported
        """
        from ..timeseries import Timeseries

        if isinstance(param, Timeseries):
            # Extract ._data from Timeseries objects
            # Shape validation happens later in __init__
            return param._data
        elif isinstance(param, (np.ndarray, list, tuple)):
            # Pass through to existing conversion logic
            return np.asarray(param, dtype=float)
        else:
            raise TypeError(
                f"{param_name} must be np.ndarray, list, tuple, or Timeseries "
                f"(Point3D/Signal3D), got {type(param).__name__}"
            )

    def __init__(
        self,
        origin: "np.ndarray | list | tuple | Timeseries",
        lateral_axis: "np.ndarray | list | tuple | Timeseries",
        vertical_axis: "np.ndarray | list | tuple | Timeseries",
        anteroposterior_axis: "np.ndarray | list | tuple | Timeseries | None" = None,
    ):
        """
        Initialize the ReferenceFrame with anatomically meaningful axes.

        Parameters
        ----------
        origin : np.ndarray or list or tuple
            Origin point of the reference frame. Shape (N, 3) or (3,) where N is
            the number of time samples. All inputs must have the same N.
        lateral_axis : np.ndarray or list or tuple
            Direction vector for the lateral (mediolateral) axis. Shape (N, 3) or (3,).
            This will be the first column of the rotation matrix after orthonormalization.
        vertical_axis : np.ndarray or list or tuple
            Direction vector for the vertical (superior-inferior) axis. Shape (N, 3) or (3,).
            This will be the second column of the rotation matrix after orthonormalization.
        anteroposterior_axis : np.ndarray or list or tuple or None, optional
            Direction vector for the anteroposterior (forward-backward) axis.
            Shape (N, 3) or (3,). If None, computed as the cross product of
            lateral_axis × vertical_axis. Default is None.

        Raises
        ------
        ValueError
            If any input does not have exactly 3 columns.
            If inputs have mismatched number of rows (time samples).

        Notes
        -----
        The input axes are automatically orthonormalized using the Gram-Schmidt process:
        1. lateral_axis is normalized to unit length
        2. anteroposterior_axis is made perpendicular to lateral_axis
        3. vertical_axis is recomputed as perpendicular to both lateral and anteroposterior

        The resulting rotation_matrix has shape (N, 3, 3) where:
        - rotation_matrix[:, :, 0] = orthonormalized lateral_axis
        - rotation_matrix[:, :, 1] = orthonormalized vertical_axis
        - rotation_matrix[:, :, 2] = orthonormalized anteroposterior_axis

        This ensures the reference frame always has perpendicular unit vectors, even if
        the input axes are not perfectly orthogonal.
        """
        # Extract numpy arrays from parameters (handles Timeseries automatically)
        origin = self._extract_numpy_array(origin, "origin")
        lateral_axis = self._extract_numpy_array(lateral_axis, "lateral_axis")
        vertical_axis = self._extract_numpy_array(vertical_axis, "vertical_axis")

        if anteroposterior_axis is not None:
            anteroposterior_axis = self._extract_numpy_array(
                anteroposterior_axis, "anteroposterior_axis"
            )

        # Convert to 2D arrays (preserves existing behavior for lists/tuples)
        origin = np.atleast_2d(origin)
        lateral_axis = np.atleast_2d(lateral_axis)
        vertical_axis = np.atleast_2d(vertical_axis)
        if anteroposterior_axis is not None:
            anteroposterior_axis = np.atleast_2d(anteroposterior_axis)

        # Validate shapes - all must have 3 columns
        if origin.shape[1] != 3:
            raise ValueError(
                f"origin must have 3 columns, got {origin.shape[1]} columns"
            )
        if lateral_axis.shape[1] != 3:
            raise ValueError(
                f"lateral_axis must have 3 columns, got {lateral_axis.shape[1]} columns"
            )
        if vertical_axis.shape[1] != 3:
            raise ValueError(
                f"vertical_axis must have 3 columns, got {vertical_axis.shape[1]} columns"
            )
        if anteroposterior_axis is not None and anteroposterior_axis.shape[1] != 3:
            raise ValueError(
                f"anteroposterior_axis must have 3 columns, got {anteroposterior_axis.shape[1]} columns"
            )

        # Validate all inputs have same number of rows
        n_samples = origin.shape[0]
        if lateral_axis.shape[0] != n_samples:
            raise ValueError(
                f"All inputs must have same number of rows. "
                f"origin has {n_samples} rows, lateral_axis has {lateral_axis.shape[0]} rows"
            )
        if vertical_axis.shape[0] != n_samples:
            raise ValueError(
                f"All inputs must have same number of rows. "
                f"origin has {n_samples} rows, vertical_axis has {vertical_axis.shape[0]} rows"
            )
        if anteroposterior_axis is not None and anteroposterior_axis.shape[0] != n_samples:
            raise ValueError(
                f"All inputs must have same number of rows. "
                f"origin has {n_samples} rows, anteroposterior_axis has {anteroposterior_axis.shape[0]} rows"
            )

        # Store inputs
        self._origin = origin
        self._lateral_axis = lateral_axis
        self._vertical_axis = vertical_axis
        self._anteroposterior_axis = anteroposterior_axis

        # Determine if single frame
        self._n_samples = n_samples
        self._is_single_frame = n_samples == 1

        # Build rotation matrix using gram_schmidt
        rotation_matrix = gram_schmidt(lateral_axis, vertical_axis, anteroposterior_axis)

        # Transpose to get row-based rotation matrix for transformation
        self._rotation_matrix = rotation_matrix.transpose([0, 2, 1])

    @property
    def origin(self):
        """Return the origin array."""
        return self._origin

    @property
    def lateral_axis(self):
        """Return the lateral axis array."""
        return self._lateral_axis

    @property
    def vertical_axis(self):
        """Return the vertical axis array."""
        return self._vertical_axis

    @property
    def anteroposterior_axis(self):
        """Return the anteroposterior axis array (or None if computed)."""
        return self._anteroposterior_axis

    @property
    def rotation_matrix(self):
        """Return the rotation matrix (N, 3, 3)."""
        return self._rotation_matrix

    def __call__(self, obj, inplace: bool = False):
        """Make ReferenceFrame callable - delegates to apply()."""
        return self.apply(obj, inplace=inplace)

    def apply_inverse(
        self,
        obj: np.ndarray | pd.DataFrame | Timeseries | Record,
        inplace: bool = False,
    ):
        """
        Apply the inverse reference frame transformation to an object.

        This method reverses the transformation applied by `apply()`. For any object,
        `apply_inverse(apply(obj))` should return the original object (within numerical precision).

        The inverse transformation is:
            old = R^T @ new + origin

        where R^T is the transpose of the rotation matrix (which equals its inverse
        for orthonormal matrices).

        Parameters
        ----------
        obj : np.ndarray or pd.DataFrame or Timeseries or Record
            Object to transform back. Must represent 3D data.
        inplace : bool, optional
            If True, modify the object in place and return None.
            If False, return a transformed copy. Default is False.

        Returns
        -------
        np.ndarray or pd.DataFrame or Timeseries or Record or None
            Inverse-transformed object (if inplace=False) or None (if inplace=True).

        Raises
        ------
        ValueError
            If object has incompatible shape or type.
        TypeError
            If object type is not supported.

        Examples
        --------
        >>> # Round-trip transformation
        >>> rf = ReferenceFrame([0, 0, 0], [1, 0, 0], [0, 1, 0])
        >>> data = np.array([[1.0, 2.0, 3.0]])
        >>> transformed = rf.apply(data)
        >>> recovered = rf.apply_inverse(transformed)
        >>> np.allclose(recovered, data)
        True
        """
        if isinstance(obj, np.ndarray):
            return self._apply_inverse_to_numpy(obj)

        if isinstance(obj, pd.DataFrame):
            return self._apply_inverse_to_dataframe(obj, inplace)

        # Import at runtime to avoid circular import
        from ..records import ForcePlatform, Record

        if isinstance(obj, ForcePlatform):
            return self._apply_inverse_to_forceplatform(obj, inplace)

        if isinstance(obj, Timeseries):
            return self._apply_inverse_to_timeseries(obj, inplace)

        if isinstance(obj, Record):
            return self._apply_inverse_to_record(obj, inplace)

        raise TypeError(
            f"Unsupported type {type(obj).__name__}. "
            f"Supported types: np.ndarray, pd.DataFrame, Timeseries, ForcePlatform, Record"
        )

    def apply(
        self,
        obj: np.ndarray | pd.DataFrame | Timeseries | Record,
        inplace: bool = False,
    ):
        """
        Apply the reference frame transformation to an object.

        Parameters
        ----------
        obj : np.ndarray or pd.DataFrame or Timeseries or Record
            Object to transform. Must represent 3D data.
        inplace : bool, optional
            If True, modify the object in place and return None.
            If False, return a transformed copy. Default is False.

        Returns
        -------
        np.ndarray or pd.DataFrame or Timeseries or Record or None
            Transformed object (if inplace=False) or None (if inplace=True).

        Raises
        ------
        ValueError
            If object has incompatible shape or type.
        TypeError
            If object type is not supported.
        """
        if isinstance(obj, np.ndarray):
            return self._apply_to_numpy(obj)

        if isinstance(obj, pd.DataFrame):
            return self._apply_to_dataframe(obj, inplace)

        # Import at runtime to avoid circular import
        from ..records import ForcePlatform, Record

        if isinstance(obj, ForcePlatform):
            return self._apply_to_forceplatform(obj, inplace)

        if isinstance(obj, Timeseries):
            return self._apply_to_timeseries(obj, inplace)

        if isinstance(obj, Record):
            return self._apply_to_record(obj, inplace)

        raise TypeError(
            f"Unsupported type {type(obj).__name__}. "
            f"Supported types: np.ndarray, pd.DataFrame, Timeseries, ForcePlatform, Record"
        )

    def _apply_to_numpy(self, data: np.ndarray):
        """Apply transformation to numpy array."""
        # Track if input was 1D
        was_1d = data.ndim == 1

        # Ensure 2D
        data = np.atleast_2d(data)

        # Validate shape
        if data.shape[1] != 3:
            raise ValueError(f"Data must have 3 columns, got {data.shape[1]} columns")

        # Handle broadcasting
        n_data_samples = data.shape[0]

        if self._is_single_frame:
            # Broadcast single frame to all samples
            rmat = np.broadcast_to(self._rotation_matrix, (n_data_samples, 3, 3))
            origin = np.broadcast_to(self._origin, (n_data_samples, 3))
        else:
            # Multiple frames - must match exactly
            if n_data_samples != self._n_samples:
                raise ValueError(
                    f"Shape mismatch: data has {n_data_samples} samples but "
                    f"ReferenceFrame has {self._n_samples} samples. "
                    f"Use a single-sample ReferenceFrame for broadcasting."
                )
            rmat = self._rotation_matrix
            origin = self._origin

        # Apply transformation: new = R @ (old - origin)
        centered = data - origin
        transformed = np.einsum("nij,nj->ni", rmat, centered)

        # Preserve input dimensionality
        return transformed[0] if was_1d else transformed

    def _apply_to_dataframe(
        self, df: pd.DataFrame, inplace: bool
    ):
        """Apply transformation to pandas DataFrame."""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Validate exactly 3 numeric columns
        if len(numeric_cols) != 3:
            raise ValueError(
                f"DataFrame must have exactly 3 numeric columns, got {len(numeric_cols)} columns"
            )

        # Extract data
        data = df[numeric_cols].to_numpy()

        # Transform
        transformed = self._apply_to_numpy(data)

        # Update DataFrame
        if inplace:
            df.loc[:, numeric_cols] = transformed
            return None
        else:
            result = df.copy()
            result.loc[:, numeric_cols] = transformed
            return result

    def _apply_to_timeseries(
        self, ts: Timeseries, inplace: bool
    ):
        """Apply transformation to Timeseries object."""
        # Validate 3D data
        if ts._data.shape[1] != 3:
            raise ValueError(
                f"Timeseries must have 3 columns (3D data), got {ts._data.shape[1]} columns. "
                f"Cannot apply 3D reference frame to {ts._data.shape[1]}D signal."
            )

        # Transform data
        transformed = self._apply_to_numpy(ts._data)

        # Update Timeseries
        out = ts if inplace else ts.copy()
        out[:, :] = transformed

        if not inplace:
            return out

    def _apply_to_forceplatform(
        self, fp: ForcePlatform, inplace: bool
    ):
        """Apply transformation to ForcePlatform object."""
        # Get free moment before transformation
        free_moment = fp.free_moment

        # Create output
        out = fp if inplace else fp.copy()

        # Transform origin (Point3D) with full transformation
        out.origin[:, :] = self._apply_to_numpy(fp.origin._data)

        # Transform force (Signal3D) with rotation only (no translation)
        force_data = fp.force._data
        n_force_samples = force_data.shape[0]

        if self._is_single_frame:
            # Broadcast rotation matrix
            rmat = np.broadcast_to(self._rotation_matrix, (n_force_samples, 3, 3))
        else:
            if n_force_samples != self._n_samples:
                raise ValueError(
                    f"Shape mismatch: ForcePlatform has {n_force_samples} samples but "
                    f"ReferenceFrame has {self._n_samples} samples"
                )
            rmat = self._rotation_matrix

        transformed_force = np.einsum("nij,nj->ni", rmat, force_data)
        out.force[:, :] = transformed_force

        # Recalculate torque from free moment and transformed origin/force
        origin_broadcast = (
            np.broadcast_to(self._origin, (fp.origin.shape[0], 3))
            if self._is_single_frame
            else self._origin
        )

        out.torque[:, :] = free_moment.to_numpy() + np.cross(
            out.origin.to_numpy() - origin_broadcast, out.force.to_numpy()
        )

        if not inplace:
            return out

    def _apply_to_record(self, record: Record, inplace: bool):
        """Apply transformation to Record object recursively."""
        # Create output
        out = record if inplace else record.copy()

        # Recursively apply to all Timeseries and ForcePlatform values
        for value in out._data.values():
            if isinstance(value, (Timeseries, ForcePlatform)):
                self.apply(value, inplace=True)

        if not inplace:
            return out

    def _apply_inverse_to_numpy(self, data: np.ndarray):
        """Apply inverse transformation to numpy array."""
        # Track if input was 1D
        was_1d = data.ndim == 1

        # Ensure 2D
        data = np.atleast_2d(data)

        # Validate shape
        if data.shape[1] != 3:
            raise ValueError(f"Data must have 3 columns, got {data.shape[1]} columns")

        # Handle broadcasting
        n_data_samples = data.shape[0]

        if self._is_single_frame:
            # Broadcast single frame to all samples
            rmat = np.broadcast_to(self._rotation_matrix, (n_data_samples, 3, 3))
            origin = np.broadcast_to(self._origin, (n_data_samples, 3))
        else:
            # Multiple frames - must match exactly
            if n_data_samples != self._n_samples:
                raise ValueError(
                    f"Shape mismatch: data has {n_data_samples} samples but "
                    f"ReferenceFrame has {self._n_samples} samples. "
                    f"Use a single-sample ReferenceFrame for broadcasting."
                )
            rmat = self._rotation_matrix
            origin = self._origin

        # Apply inverse transformation: old = R^T @ new + origin
        # R^T is the transpose of rotation matrix (inverse for orthonormal matrices)
        rmat_transpose = rmat.transpose([0, 2, 1])
        transformed = np.einsum("nij,nj->ni", rmat_transpose, data) + origin

        # Preserve input dimensionality
        return transformed[0] if was_1d else transformed

    def _apply_inverse_to_dataframe(
        self, df: pd.DataFrame, inplace: bool
    ):
        """Apply inverse transformation to pandas DataFrame."""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Validate exactly 3 numeric columns
        if len(numeric_cols) != 3:
            raise ValueError(
                f"DataFrame must have exactly 3 numeric columns, got {len(numeric_cols)} columns"
            )

        # Extract data
        data = df[numeric_cols].to_numpy()

        # Transform
        transformed = self._apply_inverse_to_numpy(data)

        # Update DataFrame
        if inplace:
            df.loc[:, numeric_cols] = transformed
            return None
        else:
            result = df.copy()
            result.loc[:, numeric_cols] = transformed
            return result

    def _apply_inverse_to_timeseries(
        self, ts: Timeseries, inplace: bool
    ):
        """Apply inverse transformation to Timeseries object."""
        # Validate 3D data
        if ts._data.shape[1] != 3:
            raise ValueError(
                f"Timeseries must have 3 columns (3D data), got {ts._data.shape[1]} columns. "
                f"Cannot apply 3D reference frame to {ts._data.shape[1]}D signal."
            )

        # Transform data
        transformed = self._apply_inverse_to_numpy(ts._data)

        # Update Timeseries
        out = ts if inplace else ts.copy()
        out[:, :] = transformed

        if not inplace:
            return out

    def _apply_inverse_to_forceplatform(
        self, fp: ForcePlatform, inplace: bool
    ):
        """Apply inverse transformation to ForcePlatform object."""
        # Get free moment before transformation
        free_moment = fp.free_moment

        # Create output
        out = fp if inplace else fp.copy()

        # Inverse transform origin (Point3D)
        out.origin[:, :] = self._apply_inverse_to_numpy(fp.origin._data)

        # Inverse transform force (Signal3D) - rotation only
        force_data = fp.force._data
        n_force_samples = force_data.shape[0]

        if self._is_single_frame:
            # Broadcast rotation matrix transpose
            rmat_transpose = np.broadcast_to(
                self._rotation_matrix.transpose([0, 2, 1]), (n_force_samples, 3, 3)
            )
        else:
            if n_force_samples != self._n_samples:
                raise ValueError(
                    f"Shape mismatch: ForcePlatform has {n_force_samples} samples but "
                    f"ReferenceFrame has {self._n_samples} samples"
                )
            rmat_transpose = self._rotation_matrix.transpose([0, 2, 1])

        transformed_force = np.einsum("nij,nj->ni", rmat_transpose, force_data)
        out.force[:, :] = transformed_force

        # Recalculate torque from free moment and inverse-transformed origin/force
        origin_broadcast = (
            np.broadcast_to(self._origin, (fp.origin.shape[0], 3))
            if self._is_single_frame
            else self._origin
        )

        out.torque[:, :] = free_moment.to_numpy() + np.cross(
            out.origin.to_numpy() - origin_broadcast, out.force.to_numpy()
        )

        if not inplace:
            return out

    def _apply_inverse_to_record(self, record: Record, inplace: bool):
        """Apply inverse transformation to Record object recursively."""
        # Create output
        out = record if inplace else record.copy()

        # Recursively apply inverse to all Timeseries and ForcePlatform values
        for value in out._data.values():
            if isinstance(value, (Timeseries, ForcePlatform)):
                self.apply_inverse(value, inplace=True)

        if not inplace:
            return out

    def copy(self):
        """
        Create a deep copy of the ReferenceFrame.

        Returns
        -------
        ReferenceFrame
            A new ReferenceFrame instance with copies of all arrays.

        Notes
        -----
        All internal numpy arrays (origin, axes, rotation_matrix) are deep copied.
        """
        return ReferenceFrame(
            origin=self._origin.copy(),
            lateral_axis=self._lateral_axis.copy(),
            vertical_axis=self._vertical_axis.copy(),
            anteroposterior_axis=self._anteroposterior_axis.copy() if self._anteroposterior_axis is not None else None,
        )
