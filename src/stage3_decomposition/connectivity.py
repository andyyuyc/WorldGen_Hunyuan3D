"""Connectivity analysis for mesh decomposition.

Analyzes connected components and builds a proximity graph to determine
connectivity degree (how many other components each component touches).
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh
from scipy.spatial import KDTree

from src.config import Stage3Config

logger = logging.getLogger(__name__)


def analyze_connectivity(
    mesh: trimesh.Trimesh,
    config: Stage3Config,
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Analyze mesh connectivity: split into components and build proximity graph.

    Args:
        mesh: Input monolithic scene mesh.
        config: Stage 3 configuration.

    Returns:
        Tuple of:
        - List of component meshes (sorted by face count descending)
        - Connectivity degrees (1D array, one per component)
        - Connectivity matrix (2D bool array, NxN)
    """
    # Weld close vertices to ensure topological connectivity
    mesh.merge_vertices(merge_tex=False, merge_norm=False)

    # Split into connected components
    components = mesh.split(only_watertight=False)

    if not components:
        logger.warning("No components found after splitting")
        return [mesh], np.array([0]), np.array([[False]])

    # Sort by face count descending (largest first, usually ground)
    components.sort(key=lambda c: len(c.faces), reverse=True)

    n_raw = len(components)
    logger.info(f"Found {n_raw} connected components (before filtering)")

    # Filter out tiny fragments (< min_face_count) and merge them into
    # the nearest large component later. This prevents O(N^2) blowup
    # in the proximity graph when there are thousands of tiny pieces.
    min_faces = config.min_face_count
    kept = [c for c in components if len(c.faces) >= min_faces]
    tiny = [c for c in components if len(c.faces) < min_faces]

    if not kept:
        # All components are tiny — just keep the largest ones
        kept = components[:max(1, min(20, n_raw))]
        tiny = components[len(kept):]

    # Merge tiny fragments into the nearest kept component
    if tiny:
        tiny_verts = np.vstack([c.vertices.mean(axis=0, keepdims=True) for c in tiny])
        kept_centroids = np.array([c.vertices.mean(axis=0) for c in kept])
        kept_tree = KDTree(kept_centroids)
        _, nearest_idx = kept_tree.query(tiny_verts, k=1)
        for frag, idx in zip(tiny, nearest_idx):
            kept[idx] = trimesh.util.concatenate([kept[idx], frag])
        logger.info(f"Merged {len(tiny)} tiny fragments into {len(kept)} components")

    components = kept
    n = len(components)
    logger.info(f"Working with {n} components (min {min_faces} faces)")

    if n == 1:
        return components, np.array([0]), np.array([[False]])

    # Build proximity graph: two components are "connected" if any of their
    # vertices are within proximity_threshold of each other
    proximity_threshold = config.proximity_threshold
    connectivity = np.zeros((n, n), dtype=bool)

    # Build KDTrees for each component (sample if too many vertices)
    max_samples = 5000
    trees = []
    sampled_points = []
    for comp in components:
        verts = comp.vertices
        if len(verts) > max_samples:
            indices = np.random.choice(len(verts), max_samples, replace=False)
            verts = verts[indices]
        trees.append(KDTree(verts))
        sampled_points.append(verts)

    for i in range(n):
        for j in range(i + 1, n):
            distances, _ = trees[i].query(sampled_points[j], k=1)
            min_dist = distances.min()
            if min_dist < proximity_threshold:
                connectivity[i, j] = True
                connectivity[j, i] = True

    degrees = connectivity.sum(axis=1).astype(np.int32)

    logger.info(
        f"Connectivity analysis: {n} components, "
        f"degrees range [{degrees.min()}, {degrees.max()}]"
    )

    return components, degrees, connectivity
