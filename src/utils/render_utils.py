"""Offscreen rendering utilities using pyrender."""

from __future__ import annotations

import numpy as np
import trimesh


def render_depth_isometric(
    mesh: trimesh.Trimesh,
    resolution: int = 1024,
    elevation_angle: float = 45.0,
    azimuth_angle: float = 45.0,
) -> np.ndarray:
    """Render an isometric depth map of a mesh using pyrender.

    Args:
        mesh: Input triangle mesh.
        resolution: Output image resolution (square).
        elevation_angle: Camera elevation in degrees.
        azimuth_angle: Camera azimuth in degrees.

    Returns:
        Depth map as float32 numpy array.
    """
    import pyrender

    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    # Add mesh
    py_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(py_mesh)

    # Compute camera parameters from mesh bounds
    bounds = mesh.bounds  # (2, 3): min, max
    center = mesh.centroid
    extent = np.max(bounds[1] - bounds[0])
    half_extent = extent * 0.6  # Add some padding

    # Orthographic camera
    camera = pyrender.OrthographicCamera(xmag=half_extent, ymag=half_extent)

    # Camera pose: looking down at elevation_angle from azimuth_angle
    elev_rad = np.radians(elevation_angle)
    azim_rad = np.radians(azimuth_angle)

    dist = extent * 2
    eye = np.array([
        dist * np.cos(elev_rad) * np.sin(azim_rad),
        dist * np.sin(elev_rad),
        dist * np.cos(elev_rad) * np.cos(azim_rad),
    ]) + center

    camera_pose = _look_at(eye, center, up=np.array([0, 1, 0]))
    scene.add(camera, pose=camera_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    try:
        _color, depth = renderer.render(scene)
    finally:
        renderer.delete()

    return depth


def render_object_front(
    mesh: trimesh.Trimesh,
    resolution: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a mesh from the front view. Returns (color, depth)."""
    import pyrender

    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])

    # Add mesh
    py_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(py_mesh)

    # Compute camera pose
    center = mesh.centroid
    extent = np.max(mesh.bounding_box.extents)

    camera = pyrender.OrthographicCamera(xmag=extent * 0.6, ymag=extent * 0.6)

    eye = center + np.array([0, 0, extent * 2])
    camera_pose = _look_at(eye, center, up=np.array([0, 1, 0]))
    scene.add(camera, pose=camera_pose)

    # Light
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    try:
        color, depth = renderer.render(scene)
    finally:
        renderer.delete()

    return color, depth


def render_multiview_depths(
    mesh: trimesh.Trimesh,
    num_side_views: int = 8,
    resolution: int = 512,
) -> list[np.ndarray]:
    """Render depth maps from multiple viewpoints around a mesh.

    Returns depth maps for: [front, side_0, ..., side_N, top]
    """
    depths = []
    center = mesh.centroid
    extent = np.max(mesh.bounding_box.extents)

    # Front view (azimuth=0)
    depths.append(render_depth_isometric(mesh, resolution, elevation_angle=0, azimuth_angle=0))

    # Side views at equal intervals
    for i in range(num_side_views):
        azimuth = (i + 1) * (360.0 / (num_side_views + 1))
        depths.append(
            render_depth_isometric(mesh, resolution, elevation_angle=0, azimuth_angle=azimuth)
        )

    # Top view
    depths.append(render_depth_isometric(mesh, resolution, elevation_angle=89, azimuth_angle=0))

    return depths


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a 4x4 camera-to-world matrix (OpenGL convention)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Forward is parallel to up; pick an arbitrary right vector
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm

    true_up = np.cross(right, forward)

    # OpenGL convention: camera looks along -Z
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye

    return mat
