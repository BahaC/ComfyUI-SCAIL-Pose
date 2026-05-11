import logging

import numpy as np
import torch
from scipy.optimize import minimize


def fit_intrinsic_to_canvas(intrinsic_matrix, pose_input, height, width, margin_px=12):
    """Adjust intrinsics so the projected 3D pose AABB fits in the canvas.

    Used by ``RenderNLFPoses`` to prevent head/feet clipping when
    ``solve_new_camera_params_*`` picks a camera that zooms in past the
    canvas edges. Projects every valid 3D joint across every frame with
    the current ``intrinsic_matrix``, computes the resulting 2D AABB,
    and (only when the AABB exceeds ``[margin, width-margin] x
    [margin, height-margin]``) returns an intrinsic matrix with focal
    length scaled by a uniform ``s_fit <= 1`` and a principal point
    shifted so the scaled AABB lands centered in the canvas. The
    transform is purely a change of intrinsics; 3D points are not
    modified, so feeding the returned matrix into
    ``shift_dwpose_according_to_nlf`` keeps the 2D face/hand overlay
    aligned with the 3D-rendered pose.

    Args:
        intrinsic_matrix: ``3x3`` numpy array. Treated as immutable; a
            new array is returned when a fit is required, otherwise
            ``intrinsic_matrix`` is returned unchanged.
        pose_input: iterable of per-frame tensors / arrays of shape
            ``(N_persons, 24, 3)``. Joints with magnitude near zero
            (missing-person sentinel) or ``Z <= 1e-3`` (behind the
            camera) are ignored.
        height: canvas height in pixels.
        width: canvas width in pixels.
        margin_px: minimum pixel margin between the projected AABB and
            every canvas edge after fitting. Should be at least the
            cylinder radius in pixels so the limb thickness doesn't
            touch the edge.

    Returns:
        A ``3x3`` numpy intrinsic matrix.
    """
    K = np.asarray(intrinsic_matrix, dtype=np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx_old = float(K[0, 2])
    cy_old = float(K[1, 2])

    u_min = np.inf
    u_max = -np.inf
    v_min = np.inf
    v_max = -np.inf

    for frame in pose_input:
        if isinstance(frame, torch.Tensor):
            arr = frame.detach().cpu().numpy()
        else:
            arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            continue
        flat = arr.reshape(-1, 3)
        valid = np.any(np.abs(flat) > 0.01, axis=1)
        flat = flat[valid]
        if flat.size == 0:
            continue
        z = flat[:, 2]
        zmask = z > 1e-3
        flat = flat[zmask]
        if flat.size == 0:
            continue
        u = fx * flat[:, 0] / flat[:, 2] + cx_old
        v = fy * flat[:, 1] / flat[:, 2] + cy_old
        u_min = min(u_min, float(u.min()))
        u_max = max(u_max, float(u.max()))
        v_min = min(v_min, float(v.min()))
        v_max = max(v_max, float(v.max()))

    if not np.isfinite([u_min, u_max, v_min, v_max]).all():
        return intrinsic_matrix

    margin = float(margin_px)
    fits = (
        u_min >= margin
        and u_max <= float(width) - margin
        and v_min >= margin
        and v_max <= float(height) - margin
    )
    if fits:
        return intrinsic_matrix

    aabb_w = max(u_max - u_min, 1.0)
    aabb_h = max(v_max - v_min, 1.0)
    s_fit_x = (float(width) - 2.0 * margin) / aabb_w
    s_fit_y = (float(height) - 2.0 * margin) / aabb_h
    s_fit = min(1.0, s_fit_x, s_fit_y)

    u_center_old = 0.5 * (u_min + u_max)
    v_center_old = 0.5 * (v_min + v_max)
    cx_new = 0.5 * float(width) - s_fit * (u_center_old - cx_old)
    cy_new = 0.5 * float(height) - s_fit * (v_center_old - cy_old)

    K_new = K.copy()
    K_new[0, 0] = s_fit * fx
    K_new[1, 1] = s_fit * fy
    K_new[0, 2] = cx_new
    K_new[1, 2] = cy_new

    logging.info(
        "fit_intrinsic_to_canvas: AABB (%.1f,%.1f)-(%.1f,%.1f) on %dx%d canvas "
        "overflowed margin %.1f; scaled focal by %.3f, principal point "
        "(%.1f,%.1f) -> (%.1f,%.1f).",
        u_min,
        v_min,
        u_max,
        v_max,
        int(width),
        int(height),
        margin,
        s_fit,
        cx_old,
        cy_old,
        cx_new,
        cy_new,
    )
    return K_new


def solve_new_camera_params_central(three_d_points, focal_length, imshape, new_2d_points):
    """
    Solve for new camera parameters by minimizing the error between the original 2D projection points and the new 2D projection points.

    Args:
        three_d_points (torch.Tensor): N*3 3D points
        focal_length (float): Focal length of the original camera
        imshape (tuple): Image size, e.g., [512, 896]
        original_2d_points (torch.Tensor): N*2 original 2D projection points
        new_2d_points (torch.Tensor): N*2 new 2D projection points

    Returns:
        m, n, p, q: Parameters in the new camera intrinsic matrix
    """


    # Objective function: minimize the error between the original projection points and the new projection points
    def objective(params):
        m, s, p, q = params
        # Construct the new camera intrinsic matrix
        K_new = np.array([
            [focal_length * m , 0, imshape[1] / 2 + p],
            [0, focal_length * m * s, imshape[0] / 2 + q],
            [0, 0, 1]
        ])

        # Compute the new 2D projection points
        new_projections = []
        for point in three_d_points:
            X, Y, Z = point
            u = (K_new[0, 0] * X / Z) + K_new[0, 2]
            v = (K_new[1, 1] * Y / Z) + K_new[1, 2]
            new_projections.append([u, v])
        new_projections = np.array(new_projections)

        # Calculate the error between the original 2D projection points and the new projection points
        # Special handling for the 0th projection point
        error0 = np.sum((new_2d_points[:1] - new_projections[:1]) ** 2)
        error = np.sum((new_2d_points[1:] - new_projections[1:]) ** 2)
        return error0 * 8 + error

    # Initialize parameters m, beta, p, q
    initial_params = [1.0, 1.0, 0.0, 0.0]  # Initial values

    # Use least squares to solve for p, q
    result = minimize(objective, initial_params, bounds=[(0.7, 1.4), (0.8, 1.15), (-imshape[1], imshape[1]), (-imshape[0], imshape[0])])

    # Output the solution result
    m, s, p, q = result.x
    print(f"debug: solved camera params m={m}, s={s}, p={p}, q={q}")

    K_final = np.array([
        [focal_length * m, 0, imshape[1] / 2 + p],
        [0, focal_length * m * s, imshape[0] / 2 + q],
        [0, 0, 1]
    ])


    return K_final, m, s


def solve_new_camera_params_down(three_d_points, focal_length, imshape, new_2d_points):
    """
    Solve for new camera parameters by minimizing the error between the original 2D projection points and the new 2D projection points.

    Args:
        three_d_points (torch.Tensor): N*3 3D points
        focal_length (float): Focal length of the original camera
        imshape (tuple): Image size, e.g., [512, 896]
        original_2d_points (torch.Tensor): N*2 original 2D projection points
        new_2d_points (torch.Tensor): N*2 new 2D projection points

    Returns:
        m, n, p, q: Parameters in the new camera intrinsic matrix
    """

    # Objective function: minimize the error between the original projection points and the new projection points
    def objective(params):
        m, s, p, q = params
        # Construct the new camera intrinsic matrix
        K_new = np.array([
            [focal_length * m , 0, imshape[1] / 2 + p],
            [0, focal_length * m * s, imshape[0] / 2 + q],
            [0, 0, 1]
        ])

        # Compute the new 2D projection points
        new_projections = []
        for point in three_d_points:
            X, Y, Z = point
            u = (K_new[0, 0] * X / Z) + K_new[0, 2]
            v = (K_new[1, 1] * Y / Z) + K_new[1, 2]
            new_projections.append([u, v])
        new_projections = np.array(new_projections)

        # Calculate the error between the original 2D projection points and the new projection points
        # Special handling for the 0th projection point
        error0 = np.sum((new_2d_points[:1] - new_projections[:1]) ** 2)
        error = np.sum((new_2d_points[1:] - new_projections[1:]) ** 2)
        return error0 + error * 4

    # Initialize parameters m, beta, p, q
    initial_params = [1.0, 1.0, 0.0, 0.0]  # Initial values

    # Use least squares to solve for p, q
    result = minimize(objective, initial_params, bounds=[(0.7, 1.4), (0.8, 1.15), (-imshape[1], imshape[1]), (-imshape[0], imshape[0])])

    # Output the solution result
    m, s, p, q = result.x
    print(f"debug: solved camera params m={m}, s={s}, p={p}, q={q}")

    K_final = np.array([
        [focal_length * m, 0, imshape[1] / 2 + p],
        [0, focal_length * m * s, imshape[0] / 2 + q],
        [0, 0, 1]
    ])


    return K_final, m, s
