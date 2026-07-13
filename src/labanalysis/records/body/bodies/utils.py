"""rigid joint estimator module"""

import numpy as np
from scipy.optimize import least_squares

from ....timeseries.point3d import Point3D
from .joint import Joint

__all__ = ["estimate_rigid_joint_center"]


def _normalize(v: np.ndarray):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return v / norm


def _get_bounds(bounds: list | None):
    if bounds is None:
        return (
            np.full(3, -np.inf),
            np.full(3, np.inf),
        )

    lower = np.asarray([b[0] for b in bounds], dtype=float)
    upper = np.asarray([b[1] for b in bounds], dtype=float)

    return lower, upper


def _local_to_global(
    local_point: np.ndarray,
    origin: np.ndarray,
    rotation_matrix: np.ndarray,
):
    local = np.tile(local_point, (origin.shape[0], 1))
    rmat_t = rotation_matrix.transpose([0, 2, 1])

    return np.einsum("nij,nj->ni", rmat_t, local) + origin


def _estimate_motion_score(
    distal_marker: np.ndarray,
    reference_marker: np.ndarray | None,
):
    if reference_marker is None:
        return 10.0

    segment = _normalize(distal_marker - reference_marker)

    mean_segment = np.mean(segment, axis=0)
    mean_segment /= np.linalg.norm(mean_segment)

    ang = np.degrees(np.arccos(np.clip(segment @ mean_segment, -1, 1)))

    return float(np.std(ang))


def _estimate_sigmas(
    prior_global: np.ndarray,
    distal_marker: np.ndarray,
    reference_marker: np.ndarray | None,
):
    motion_score = _estimate_motion_score(
        distal_marker,
        reference_marker,
    )

    sigma_prior = np.clip(30.0 - 0.6 * motion_score, 5.0, 30.0)
    d0 = np.linalg.norm(prior_global - distal_marker, axis=1)
    sigma_distal = np.clip(np.std(d0), 3.0, 20.0)
    sigma_reference = None
    sigma_angle = None
    if reference_marker is not None:
        d1 = np.linalg.norm(prior_global - reference_marker, axis=1)
        sigma_reference = np.clip(np.std(d1), 3.0, 20.0)
        v1 = _normalize(distal_marker - prior_global)
        v2 = _normalize(reference_marker - prior_global)
        theta = np.sum(v1 * v2, axis=1)
        ang = np.degrees(np.arccos(np.clip(theta, -1, 1)))
        sigma_angle = np.clip(np.std(ang), 2.0, 15.0)

    return sigma_prior, sigma_distal, sigma_reference, sigma_angle


def estimate_rigid_joint_center(
    joint: Joint,
    distal_marker: Point3D,
    reference_marker: Point3D | None = None,
    prior_global: Point3D | None = None,
    bounds: list | None = None,
):
    """
    Estimate a functional joint center from a dynamic trial.

    The algorithm searches for the joint center that maximizes the
    rigidity of the distal segment while remaining consistent with
    an anatomical prior estimate.

    The optimization is performed in the LOCAL coordinate system
    defined by the supplied ``joint``. The estimated center is
    returned as a GLOBAL trajectory.

    Residuals are automatically normalized using variability
    estimated from the supplied trial. No manual weighting of
    the objective terms is required.

    Typical applications
    --------------------

    Hip
        joint
            Pelvis anatomical reference frame.

        distal_marker
            Knee center.

        reference_marker
            Greater trochanter.

        prior_global
            Hip joint center estimated
    Shoulder
        joint
            Thorax anatomical reference frame.

        distal_marker
            Elbow center.

        reference_marker
            Acromion.

        prior_global
            Shoulder joint center estimated

    Optimization criteria
    ---------------------
    The estimated joint center minimizes a combination of:

    - Deviation from the anatomical prior.
    - Variability of the distance between the joint center and the
      distal marker.
    - Variability of the distance between the joint center and the
      reference marker (if provided).
    - Variability of the angle formed by joint center, distal marker
      and reference marker (if provided).

    Parameters
    ----------
    joint
        Anatomical joint defining the local reference frame in which
        the optimization is performed.

        The frame should be constructed exclusively from observable
        anatomical landmarks and should not depend on the unknown
        joint center being estimated.

    distal_marker
        Landmark belonging to the distal segment.

        Examples:
            - Knee center for hip joint center estimation.
            - Elbow center for shoulder joint center estimation.

    reference_marker
        Optional landmark located close to the joint.

        When provided, additional rigidity constraints are included
        in the optimization.

        Examples:
            - Greater trochanter.
            - Acromion.

    prior_global
        Anatomical estimate of the joint center expressed in global
        coordinates.

        This prior is used both as the initial guess for the optimizer
        and as a regularization term.

    bounds
        Optional bounds applied to the estimated joint center
        expressed in the LOCAL coordinates of ``joint``.

        Format::

            [
                (xmin, xmax),
                (ymin, ymax),
                (zmin, zmax),
            ]

        Example::

            [
                (-200,  50),
                (-200, 200),
                (-200, 200),
            ]

        If ``None``, the optimization is unconstrained.

    Returns
    -------
    Point3D
        Estimated joint center trajectory expressed in GLOBAL
        coordinates.

        The returned trajectory has the same index, unit and axis
        conventions as ``prior_global``.

    Raises
    ------
    ValueError
        If ``prior_global`` is not provided.

    Notes
    -----
    The estimated joint center is represented internally as a single
    fixed point in the local coordinate system of ``joint``. The
    returned global trajectory corresponds to the frame-by-frame
    transformation of this local point into the laboratory reference
    frame.
    """

    if prior_global is None:
        raise ValueError("prior_global must be provided.")

    rf = joint.reference_frame
    origin = rf.origin
    rotation_matrix = rf.rotation_matrix
    distal = distal_marker.to_numpy()
    reference = None if reference_marker is None else reference_marker.to_numpy()
    prior_global_np = prior_global.to_numpy()
    prior_local_ts = joint.apply(prior_global).to_numpy()  # type: ignore
    prior_local = np.median(prior_local_ts, axis=0)

    (
        sigma_prior,
        sigma_distal,
        sigma_reference,
        sigma_angle,
    ) = _estimate_sigmas(
        prior_global_np,
        distal,
        reference,
    )

    def residuals(joint_local):

        joint_global = _local_to_global(
            joint_local,
            origin,
            rotation_matrix,
        )
        r = []
        r.extend((joint_local - prior_local) / sigma_prior)
        d = np.linalg.norm(joint_global - distal, axis=1)
        r.extend((d - np.mean(d)) / sigma_distal)

        if reference is not None:
            d = np.linalg.norm(joint_global - reference, axis=1)
            r.extend((d - np.mean(d)) / sigma_reference)
            v1 = _normalize(distal - joint_global)
            v2 = _normalize(reference - joint_global)
            theta = np.sum(v1 * v2, axis=1)
            ang = np.degrees(np.arccos(np.clip(theta, -1, 1)))
            r.extend((ang - np.mean(ang)) / sigma_angle)

        return np.asarray(r, dtype=float)

    lb, ub = _get_bounds(bounds)
    result = least_squares(residuals, x0=prior_local, bounds=(lb, ub), method="lm")
    joint_center_local = result.x
    joint_center_global = _local_to_global(
        joint_center_local,
        origin,
        rotation_matrix,
    )

    return Point3D(
        data=joint_center_global,
        index=prior_global.index,
        unit=prior_global.unit,
        vertical_axis=prior_global.vertical_axis,
        anteroposterior_axis=prior_global.anteroposterior_axis,
    )
