"""MuJoCo native viewer debug visualizer implementation."""

from __future__ import annotations

import mujoco
import numpy as np
from typing_extensions import override

from mjlab.viewer.debug_visualizer import DebugVisualizer


class MujocoNativeDebugVisualizer(DebugVisualizer):
  """Debug visualizer for MuJoCo's native viewer.

  This implementation directly adds geometry to the MuJoCo scene using mjv_addGeoms
  and other MuJoCo visualization primitives.
  """

  def __init__(
    self,
    scn: mujoco.MjvScene,
    mj_model: mujoco.MjModel,
    env_idx: int,
    show_all_envs: bool = False,
  ):
    """Initialize the MuJoCo native visualizer.

    Args:
      scn: MuJoCo scene to add visualizations to
      mj_model: MuJoCo model for creating visualization data
      env_idx: Index of the environment being visualized
      show_all_envs: If True, visualize all environments instead of just env_idx
    """
    self.scn = scn
    self.mj_model = mj_model
    self.env_idx = env_idx
    self.show_all_envs = show_all_envs
    self._initial_geom_count = scn.ngeom
    self._meansize: float = mj_model.stat.meansize

    self._vopt = mujoco.MjvOption()
    self._vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    self._pert = mujoco.MjvPerturb()
    self._viz_data = mujoco.MjData(mj_model)

  @property
  @override
  def meansize(self) -> float:
    return self._meansize

  @override
  def add_arrow(
    self,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    """Add an arrow visualization using MuJoCo's arrow geometry."""
    del label  # Unused.

    self.scn.ngeom += 1
    geom = self.scn.geoms[self.scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    mujoco.mjv_initGeom(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      width=width,
      from_=start,
      to=end,
    )

  @override
  def add_ghost_mesh(
    self,
    qpos: np.ndarray,
    model: mujoco.MjModel,
    mocap_pos: np.ndarray | None = None,
    mocap_quat: np.ndarray | None = None,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    """Add a ghost mesh by rendering the robot at a different pose.

    This creates a semi-transparent copy of the robot geometry at the target pose.

    Args:
      qpos: Joint positions for the ghost pose
      model: MuJoCo model with pre-configured appearance (geom_rgba for colors)
      mocap_pos: Optional mocap position(s) for fixed-base entities
      mocap_quat: Optional mocap quaternion(s) for fixed-base entities
      alpha: Transparency override (not used in MuJoCo implementation)
      label: Optional label (not used in MuJoCo implementation)
    """
    del alpha, label  # Unused.

    self._viz_data.qpos[:] = qpos
    if mocap_pos is not None and model.nmocap > 0:
      mocap_pos_arr = np.asarray(mocap_pos)
      if mocap_pos_arr.ndim == 1:
        self._viz_data.mocap_pos[0] = mocap_pos_arr
      else:
        self._viz_data.mocap_pos[:] = mocap_pos_arr
    if mocap_quat is not None and model.nmocap > 0:
      mocap_quat_arr = np.asarray(mocap_quat)
      if mocap_quat_arr.ndim == 1:
        self._viz_data.mocap_quat[0] = mocap_quat_arr
      else:
        self._viz_data.mocap_quat[:] = mocap_quat_arr
    mujoco.mj_forward(model, self._viz_data)

    mujoco.mjv_addGeoms(
      model,
      self._viz_data,
      self._vopt,
      self._pert,
      mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
      self.scn,
    )

  @override
  def add_rectangle(
    self,
    center: np.ndarray,
    width: float,
    height: float,
    normal: np.ndarray | None = None,
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.5),
    edge_radius: float = 0.005,
    label: str | None = None,
  ) -> None:
    """Add a flat rectangle visualization with filled area and edge borders.

    Args:
      center: Center position of the rectangle (3D vector).
      width: Width of the rectangle (along local X axis).
      height: Height of the rectangle (along local Y axis).
      normal: Normal vector defining rectangle orientation (default: Z-up).
        The rectangle lies in the plane perpendicular to this normal.
      color: RGBA color for the rectangle fill and edges (values 0-1).
      edge_radius: Radius/thickness of the edge cylinders.
      label: Optional label for this rectangle.
    """
    del label  # Unused.

    center = np.asarray(center, dtype=np.float32).flatten()

    if normal is None:
      normal = np.array([0.0, 0.0, 1.0])
    normal = np.asarray(normal, dtype=np.float32).flatten()
    normal = normal / np.linalg.norm(normal)

    # Create local coordinate frame for the rectangle
    # Choose an arbitrary perpendicular vector
    if abs(normal[2]) < 0.9:
      up = np.array([0.0, 0.0, 1.0])
    else:
      up = np.array([1.0, 0.0, 0.0])

    local_x = np.cross(up, normal)
    local_x = local_x / np.linalg.norm(local_x)
    local_y = np.cross(normal, local_x)
    local_y = local_y / np.linalg.norm(local_y)

    # Build rotation matrix from local frame
    # Columns are the local axes
    rot_mat = np.column_stack([local_x, local_y, normal])

    # Add filled box geometry (very thin to appear as a plane)
    self.scn.ngeom += 1
    geom = self.scn.geoms[self.scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    # Size: [half-width, half-height, half-thickness]
    thickness = 0.001
    size = np.array([width / 2, height / 2, thickness])

    mujoco.mjv_initGeom(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_BOX.value,
      size=size,
      pos=center,
      mat=rot_mat.flatten(),
      rgba=np.asarray(color, dtype=np.float32),
    )

    # Calculate the 4 corners for edges
    half_width = width / 2
    half_height = height / 2

    corners = [
      center + local_x * half_width + local_y * half_height,  # top-right
      center - local_x * half_width + local_y * half_height,  # top-left
      center - local_x * half_width - local_y * half_height,  # bottom-left
      center + local_x * half_width - local_y * half_height,  # bottom-right
    ]

    # Draw 4 edges as cylinders for borders
    edge_color = (color[0], color[1], color[2], min(1.0, color[3] * 1.2))
    for i in range(4):
      start = corners[i]
      end = corners[(i + 1) % 4]
      self.add_cylinder(
        start=start,
        end=end,
        radius=edge_radius,
        color=edge_color,
        label=None,
      )


  @override
  def add_frame(
    self,
    position: np.ndarray,
    rotation_matrix: np.ndarray,
    scale: float = 0.3,
    label: str | None = None,
    axis_radius: float = 0.01,
    alpha: float = 1.0,
    axis_colors: tuple[tuple[float, float, float], ...] | None = None,
  ) -> None:
    """Add a coordinate frame visualization with RGB-colored axes."""
    del label  # Unused.

    default_colors = [(0.9, 0, 0), (0, 0.9, 0.0), (0.0, 0.0, 0.9)]
    colors = axis_colors if axis_colors is not None else default_colors

    for axis_idx in range(3):
      axis_direction = rotation_matrix[:, axis_idx]
      end_position = position + axis_direction * scale
      rgb = colors[axis_idx]
      color_rgba = (rgb[0], rgb[1], rgb[2], alpha)
      self.add_arrow(
        start=position,
        end=end_position,
        color=color_rgba,
        width=axis_radius,
      )

  @override
  def add_sphere(
    self,
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float, float],
    label: str | None = None,
  ) -> None:
    """Add a sphere visualization using MuJoCo's sphere geometry."""
    del label  # Unused.

    self.scn.ngeom += 1
    geom = self.scn.geoms[self.scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    mujoco.mjv_initGeom(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
      size=np.array([radius, 0, 0]),
      pos=center,
      mat=np.eye(3).flatten(),
      rgba=np.asarray(color, dtype=np.float32),
    )

  @override
  def add_cylinder(
    self,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: tuple[float, float, float, float],
    label: str | None = None,
  ) -> None:
    """Add a cylinder visualization using MuJoCo's cylinder connector."""
    del label  # Unused.

    self.scn.ngeom += 1
    geom = self.scn.geoms[self.scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    mujoco.mjv_initGeom(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_CYLINDER.value,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_CYLINDER.value,
      width=radius,
      from_=start,
      to=end,
    )

  @override
  def add_ellipsoid(
    self,
    center: np.ndarray,
    size: np.ndarray,
    mat: np.ndarray,
    color: tuple[float, float, float, float],
    label: str | None = None,
  ) -> None:
    """Add an ellipsoid visualization using MuJoCo's ellipsoid geometry."""
    del label  # Unused.

    self.scn.ngeom += 1
    geom = self.scn.geoms[self.scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    mujoco.mjv_initGeom(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_ELLIPSOID.value,
      size=np.asarray(size, dtype=np.float64),
      pos=np.asarray(center, dtype=np.float64),
      mat=np.asarray(mat, dtype=np.float64).flatten(),
      rgba=np.asarray(color, dtype=np.float32),
    )

  @override
  def add_box(
    self,
    center: np.ndarray,
    size: np.ndarray,
    mat: np.ndarray,
    color: tuple[float, float, float, float],
    label: str | None = None,
  ) -> None:
    """Add a box visualization using MuJoCo's box geometry."""
    del label  # Unused.

    self.scn.ngeom += 1
    geom = self.scn.geoms[self.scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    mujoco.mjv_initGeom(
      geom=geom,
      type=mujoco.mjtGeom.mjGEOM_BOX.value,
      size=np.asarray(size, dtype=np.float64),
      pos=np.asarray(center, dtype=np.float64),
      mat=np.asarray(mat, dtype=np.float64).flatten(),
      rgba=np.asarray(color, dtype=np.float32),
    )

  @override
  def clear(self) -> None:
    """Clear debug visualizations by resetting geom count."""
    self.scn.ngeom = self._initial_geom_count
