"""Project metric MuJoCo perspective depth into the Panda base frame."""

import numpy as np


def depth_to_base_points(depth, mask, fovy_deg, camera_pos, camera_rotation,
                         base_pos, base_rotation):
    """Return one XYZ point per selected pixel, in row-major pixel order.

    Camera and base poses are in MuJoCo world coordinates. Rotation matrices
    map local coordinates to world coordinates. MuJoCo cameras look along -Z,
    and image rows increase downward.
    """
    depth = np.asarray(depth, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth and mask must have the same 2D shape")
    if not 0 < fovy_deg < 180:
        raise ValueError("fovy_deg must be between 0 and 180 degrees")

    height, width = depth.shape
    rows, columns = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[rows, columns]
    focal = height / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
    camera_points = np.column_stack((
        (columns + 0.5 - width / 2) * z / focal,
        (height / 2 - rows - 0.5) * z / focal,
        -z,
    ))
    world_points = (
        camera_points @ np.asarray(camera_rotation).reshape(3, 3).T
        + np.asarray(camera_pos)
    )
    return (
        (world_points - np.asarray(base_pos))
        @ np.asarray(base_rotation).reshape(3, 3)
    )
