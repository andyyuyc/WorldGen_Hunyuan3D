"""Mesh splitting for large fused components using DBSCAN clustering."""

from __future__ import annotations

import logging
import time

import numpy as np
import trimesh

from src.config import Stage3Config

logger = logging.getLogger(__name__)


def split_mesh(
    components: list[trimesh.Trimesh],
    ground_idx: int,
    config: Stage3Config,
) -> list[trimesh.Trimesh]:
    """Split components into individual objects, excluding ground.

    When the mesh is dominated by a single component (common with image-to-3D
    backends like Hunyuan3D), uses height-based splitting to separate
    ground from objects.
    """
    total_faces = sum(len(c.faces) for c in components)
    dominant = components[ground_idx]
    dominant_ratio = len(dominant.faces) / total_faces if total_faces > 0 else 0

    if dominant_ratio > 0.99:
        logger.info(
            f"Dominant component has {dominant_ratio:.1%} of faces "
            f"({len(dominant.faces):,} faces), using height-based splitting"
        )
        return _split_single_component(dominant, config)

    objects: list[trimesh.Trimesh] = []
    min_faces = config.min_face_count

    for i, comp in enumerate(components):
        if i == ground_idx:
            continue
        if len(comp.faces) < min_faces:
            continue
        if _should_split(comp, config):
            sub_parts = _dbscan_split(comp, config)
            objects.extend(sub_parts)
        else:
            objects.append(comp)

    logger.info(f"Split into {len(objects)} individual objects (excluding ground)")
    return objects


def _split_single_component(
    mesh: trimesh.Trimesh,
    config: Stage3Config,
) -> list[trimesh.Trimesh]:
    """Split a single fused mesh into ground + objects using height."""
    t0 = time.time()
    logger.info(f"[1/5] Analyzing mesh normals ({len(mesh.faces):,} faces)...")

    centroids = mesh.triangles_center
    normals = mesh.face_normals

    up_axis = _detect_up_axis(normals)
    logger.info(f"  Detected up axis: {['X', 'Y', 'Z'][up_axis]}")

    heights = centroids[:, up_axis]
    height_range = heights.max() - heights.min()

    if height_range < 0.01:
        logger.warning("Mesh is nearly flat, returning as single object")
        return [mesh]

    # Try normal+height based ground detection
    logger.info("[2/5] Detecting ground faces...")
    up_threshold = config.ground_normal_threshold
    height_cutoff = heights.min() + height_range * 0.3

    is_ground_face = (normals[:, up_axis] > up_threshold) & (heights < height_cutoff)
    ground_ratio = is_ground_face.sum() / len(is_ground_face)

    if ground_ratio < 0.01:
        logger.info(
            f"  Normal-based ground ratio too low ({ground_ratio:.4f}), "
            f"using height-only splitting"
        )
        height_cutoff_low = heights.min() + height_range * 0.15
        is_ground_face = heights < height_cutoff_low
        ground_ratio = is_ground_face.sum() / len(is_ground_face)
        logger.info(f"  Height-only ground ratio: {ground_ratio:.3f}")

    if ground_ratio < 0.005 or ground_ratio > 0.95:
        logger.info(f"  Ground ratio {ground_ratio:.3f} out of range, using DBSCAN")
        parts = _dbscan_split(mesh, config)
        if len(parts) <= 1:
            logger.warning("DBSCAN found only 1 cluster, returning whole mesh")
            return [mesh]
        parts.sort(key=lambda p: len(p.faces), reverse=True)
        return parts[1:]

    # Extract non-ground faces
    object_face_indices = np.where(~is_ground_face)[0]
    logger.info(
        f"  Ground: {is_ground_face.sum():,} faces, "
        f"Objects: {len(object_face_indices):,} faces"
    )

    if len(object_face_indices) < config.min_face_count:
        logger.warning("Too few non-ground faces, returning whole mesh")
        return [mesh]

    logger.info(f"[3/5] Extracting object sub-mesh ({len(object_face_indices):,} faces)...")
    obj_mesh = _extract_submesh(mesh, object_face_indices)
    logger.info(f"  Done in {time.time() - t0:.1f}s")

    # Split into connected components
    logger.info(f"[4/5] Splitting into connected components ({len(obj_mesh.faces):,} faces)...")
    t1 = time.time()
    sub_components = obj_mesh.split(only_watertight=False)
    logger.info(f"  Found {len(sub_components) if sub_components else 0} components in {time.time() - t1:.1f}s")

    if not sub_components:
        return [obj_mesh]

    objects = [c for c in sub_components if len(c.faces) >= config.min_face_count]
    logger.info(f"  After filtering tiny fragments: {len(objects)} objects")

    if not objects:
        return [obj_mesh]

    # If still just one big piece, try DBSCAN
    if len(objects) == 1 and _should_split(objects[0], config):
        logger.info(f"[5/5] Single large object ({len(objects[0].faces):,} faces), running DBSCAN...")
        objects = _dbscan_split(objects[0], config)
    else:
        logger.info("[5/5] Skipping DBSCAN (already split)")

    total_time = time.time() - t0
    logger.info(f"Height-based split found {len(objects)} objects in {total_time:.1f}s")
    return objects


def _detect_up_axis(normals: np.ndarray) -> int:
    """Detect which axis is 'up' by checking normal distributions."""
    threshold = 0.7
    ratios = []
    for axis in range(3):
        up_ratio = (normals[:, axis] > threshold).mean()
        ratios.append(up_ratio)

    max_axis = int(np.argmax(ratios))
    if ratios[max_axis] > 0.1:
        return max_axis
    return 1


def _extract_submesh(
    mesh: trimesh.Trimesh, face_indices: np.ndarray
) -> trimesh.Trimesh:
    """Extract a sub-mesh from face indices."""
    sub_faces = mesh.faces[face_indices]
    unique_verts = np.unique(sub_faces.flatten())
    vert_map = {old: new for new, old in enumerate(unique_verts)}
    new_faces = np.vectorize(vert_map.get)(sub_faces)
    new_verts = mesh.vertices[unique_verts]

    sub_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)

    if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
        sub_mesh.visual.vertex_colors = mesh.visual.vertex_colors[unique_verts]

    return sub_mesh


def _should_split(mesh: trimesh.Trimesh, config: Stage3Config) -> bool:
    """Determine if a component is likely multiple fused objects."""
    face_count = len(mesh.faces)
    if face_count > 5000:
        return True

    bb_volume = np.prod(mesh.bounding_box.extents)
    if bb_volume > 0 and mesh.volume > 0:
        fill_ratio = abs(mesh.volume) / bb_volume
        if fill_ratio < 0.1:
            return True

    return False


def _dbscan_split(
    mesh: trimesh.Trimesh,
    config: Stage3Config,
) -> list[trimesh.Trimesh]:
    """Split a mesh using DBSCAN clustering on face centroids."""
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("Open3D not available, skipping DBSCAN split")
        return [mesh]

    t0 = time.time()
    logger.info(f"  DBSCAN: clustering {len(mesh.faces):,} faces (eps={config.dbscan_eps})...")

    centroids = np.array(mesh.triangles_center, copy=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centroids)

    labels = np.array(
        pcd.cluster_dbscan(
            eps=config.dbscan_eps,
            min_points=config.dbscan_min_samples,
        )
    )
    logger.info(f"  DBSCAN done in {time.time() - t0:.1f}s")

    unique_labels = set(labels)
    unique_labels.discard(-1)

    if len(unique_labels) <= 1:
        logger.info(f"  DBSCAN found only {len(unique_labels)} cluster(s)")
        return [mesh]

    logger.info(f"  DBSCAN found {len(unique_labels)} clusters")

    parts = []
    for label in sorted(unique_labels):
        face_mask = labels == label
        n_faces = face_mask.sum()
        if n_faces < config.min_face_count:
            continue
        face_indices = np.where(face_mask)[0]
        parts.append(_extract_submesh(mesh, face_indices))
        logger.info(f"    Cluster {label}: {n_faces:,} faces")

    if not parts:
        return [mesh]

    return parts
