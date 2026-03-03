"""Merge small fragmented parts into their nearest neighbor."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from src.config import Stage3Config

logger = logging.getLogger(__name__)


def merge_small_parts(
    objects: list[trimesh.Trimesh],
    config: Stage3Config,
) -> list[trimesh.Trimesh]:
    """Merge parts that are too small into their nearest neighbor.

    Args:
        objects: List of object meshes from splitting.
        config: Stage 3 configuration.

    Returns:
        Cleaned list with small parts merged.
    """
    if len(objects) <= 1:
        return objects

    min_faces = config.min_face_count
    min_part, max_part = config.target_part_range

    # Separate into "keep" and "merge" lists
    keep: list[trimesh.Trimesh] = []
    to_merge: list[trimesh.Trimesh] = []

    for obj in objects:
        if len(obj.faces) >= min_faces:
            keep.append(obj)
        else:
            to_merge.append(obj)

    if not to_merge:
        return _trim_to_target(keep, max_part)

    if not keep:
        # All parts are small; just return them all
        return objects

    logger.info(f"Merging {len(to_merge)} small parts into {len(keep)} larger parts")

    # Compute centroids of kept parts
    keep_centroids = np.array([k.centroid for k in keep])

    # Merge each small part into the nearest kept part
    for small_part in to_merge:
        small_centroid = small_part.centroid
        distances = np.linalg.norm(keep_centroids - small_centroid, axis=1)
        nearest_idx = int(np.argmin(distances))

        # Merge meshes
        merged = trimesh.util.concatenate([keep[nearest_idx], small_part])
        keep[nearest_idx] = merged

    result = _trim_to_target(keep, max_part)

    logger.info(f"After merging: {len(result)} objects")
    return result


def _trim_to_target(
    objects: list[trimesh.Trimesh],
    max_parts: int,
) -> list[trimesh.Trimesh]:
    """If there are too many parts, merge the smallest ones.

    Iteratively merges the two closest small parts until count is within target.
    """
    while len(objects) > max_parts:
        # Find the smallest part
        sizes = [len(o.faces) for o in objects]
        smallest_idx = int(np.argmin(sizes))
        smallest = objects.pop(smallest_idx)

        if not objects:
            objects.append(smallest)
            break

        # Find nearest neighbor
        centroids = np.array([o.centroid for o in objects])
        distances = np.linalg.norm(centroids - smallest.centroid, axis=1)
        nearest_idx = int(np.argmin(distances))

        # Merge
        objects[nearest_idx] = trimesh.util.concatenate([objects[nearest_idx], smallest])

    return objects
