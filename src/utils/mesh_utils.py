"""Common mesh utilities wrapping trimesh operations."""

from __future__ import annotations

import numpy as np
import trimesh


def heightmap_to_mesh(
    heightmap: np.ndarray,
    extent: float = 50.0,
    center: bool = True,
) -> trimesh.Trimesh:
    """Convert a 2D heightmap array to a triangle mesh.

    Args:
        heightmap: 2D numpy array of elevation values.
        extent: Physical size of the terrain in meters (both X and Z).
        center: If True, center the mesh at origin.

    Returns:
        Triangle mesh with vertices at heightmap grid positions.
    """
    rows, cols = heightmap.shape
    x = np.linspace(0, extent, cols)
    z = np.linspace(0, extent, rows)
    xx, zz = np.meshgrid(x, z)

    vertices = np.column_stack([
        xx.ravel(),
        heightmap.ravel(),
        zz.ravel(),
    ])

    # Generate triangle faces from grid quads
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            # Two triangles per quad
            faces.append([idx, idx + cols, idx + 1])
            faces.append([idx + 1, idx + cols, idx + cols + 1])

    faces = np.array(faces, dtype=np.int64)

    if center:
        vertices[:, 0] -= extent / 2
        vertices[:, 2] -= extent / 2

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    return mesh


def create_box_proxy(
    position: np.ndarray,
    size: np.ndarray,
    rotation_y: float = 0.0,
) -> trimesh.Trimesh:
    """Create a box proxy mesh at a given position.

    Args:
        position: [x, y, z] center of the box base.
        size: [width, height, depth] of the box.
        rotation_y: Rotation around Y axis in degrees.

    Returns:
        Box mesh positioned and rotated.
    """
    box = trimesh.creation.box(extents=size)

    # Shift so base is at y=0
    box.apply_translation([0, size[1] / 2, 0])

    # Apply rotation
    if rotation_y != 0:
        angle_rad = np.radians(rotation_y)
        rot_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])
        box.apply_transform(rot_matrix)

    # Move to position
    box.apply_translation(position)

    return box


def weld_vertices(mesh: trimesh.Trimesh, epsilon: float = 0.001) -> trimesh.Trimesh:
    """Merge vertices within epsilon distance to ensure topological connectivity."""
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def compute_face_normals_up_ratio(mesh: trimesh.Trimesh) -> float:
    """Compute the ratio of faces whose normals point upward (Y+)."""
    normals = mesh.face_normals
    up = np.array([0, 1, 0])
    dots = np.dot(normals, up)
    return float(np.mean(dots > 0.7))


def sample_height_at_xz(
    heightmap: np.ndarray, x: float, z: float, extent: float = 50.0
) -> float:
    """Sample terrain height at a given XZ position from heightmap.

    Args:
        heightmap: 2D numpy array.
        x, z: World coordinates (centered at origin).
        extent: Physical size of terrain.

    Returns:
        Interpolated height value.
    """
    rows, cols = heightmap.shape
    # Convert world coords to grid coords
    gx = (x + extent / 2) / extent * (cols - 1)
    gz = (z + extent / 2) / extent * (rows - 1)

    # Bilinear interpolation
    gx = np.clip(gx, 0, cols - 2)
    gz = np.clip(gz, 0, rows - 2)

    ix, iz = int(gx), int(gz)
    fx, fz = gx - ix, gz - iz

    h00 = heightmap[iz, ix]
    h10 = heightmap[iz, ix + 1]
    h01 = heightmap[iz + 1, ix]
    h11 = heightmap[iz + 1, ix + 1]

    h = h00 * (1 - fx) * (1 - fz) + h10 * fx * (1 - fz) + h01 * (1 - fx) * fz + h11 * fx * fz
    return float(h)
