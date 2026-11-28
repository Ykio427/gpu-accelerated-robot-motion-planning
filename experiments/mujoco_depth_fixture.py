"""Validate metric depth and base-frame reconstruction with a known box."""

import json

import mujoco
import numpy as np

from depth_projection import depth_to_base_points


SCENE = """
<mujoco>
  <visual><global offwidth="320" offheight="240"/></visual>
  <worldbody>
    <body name="base_frame" pos="0.1 -0.2 0.03" euler="0 0 30">
      <camera name="depth" pos="0 0 1.25" fovy="45"/>
      <geom name="blocker" type="box" pos="0.18 0.12 0.2"
            size="0.06 0.04 0.1"/>
      <geom type="plane" size="1 1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera_id = model.camera("depth").id
    base_id = model.body("base_frame").id
    box_id = model.geom("blocker").id
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="depth")
        depth = renderer.render()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="depth")
        segments = renderer.render()

    # Simulator geom IDs supply a known mask for this calibration fixture.
    mask = (segments[..., 0] == box_id) & (
        segments[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    )
    points = depth_to_base_points(
        depth, mask, model.cam_fovy[camera_id],
        data.cam_xpos[camera_id], data.cam_xmat[camera_id],
        data.xpos[base_id], data.xmat[base_id],
    )
    if len(points) < 20:
        raise RuntimeError("Too few depth pixels on the calibration box")

    # From above, measure the top footprint and use the known box height.
    half_height = 0.1
    top = points[np.abs(points[:, 2] - points[:, 2].max()) < 0.001]
    if len(top) < 20:
        raise RuntimeError("Too few points on the calibration box top")
    lower_xy = top[:, :2].min(axis=0)
    upper_xy = top[:, :2].max(axis=0)
    center = np.r_[(lower_xy + upper_xy) / 2, np.median(top[:, 2]) - half_height]
    error = float(np.linalg.norm(center - model.geom_pos[box_id]))
    print(json.dumps({
        "depth_points": len(points),
        "top_surface_points": len(top),
        "estimated_center_base_m": center.tolist(),
        "center_error_m": error,
    }, indent=2))
    if error > 0.005:
        raise RuntimeError("Reconstructed center differs by more than 5 mm")


if __name__ == "__main__":
    main()
