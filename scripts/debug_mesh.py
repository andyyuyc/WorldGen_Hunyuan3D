"""Debug: inspect the Hunyuan3D mesh to understand its structure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import trimesh

mesh_path = Path("data/intermediate/stage2/scene_mesh_aligned.glb")
print(f"Loading: {mesh_path}")
scene = trimesh.load(str(mesh_path))
print(f"Loaded type: {type(scene)}")

# If it's a Scene, dump its contents
if isinstance(scene, trimesh.Scene):
    print(f"Scene geometries: {list(scene.geometry.keys())}")
    for name, geom in scene.geometry.items():
        print(f"  {name}: {len(geom.vertices)} verts, {len(geom.faces)} faces")
    # Force to single mesh
    mesh = scene.dump(concatenate=True) if len(scene.geometry) > 0 else None
    if mesh is None:
        print("ERROR: No geometry in scene!")
        sys.exit(1)
else:
    mesh = scene

print(f"\nMesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
print(f"Bounds: {mesh.bounds}")
print(f"Extents: {mesh.bounding_box.extents}")
print(f"Visual kind: {mesh.visual.kind}")

# Check connected components
components = mesh.split(only_watertight=False)
print(f"\nConnected components: {len(components)}")
for i, c in enumerate(components[:10]):
    print(f"  Component {i}: {len(c.faces)} faces, bounds={c.bounds.tolist()}")

# Analyze normals and heights
centroids = mesh.triangles_center
normals = mesh.face_normals
print(f"\nCentroid ranges:")
for axis, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
    vals = centroids[:, axis]
    print(f"  {name}: [{vals.min():.3f}, {vals.max():.3f}]")

print(f"\nNormal ranges:")
for axis, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
    vals = normals[:, axis]
    print(f"  {name}: [{vals.min():.3f}, {vals.max():.3f}], mean={vals.mean():.3f}")

# Check which axis is "up" by looking at upward-facing normals
for axis, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
    up_ratio = (normals[:, axis] > 0.7).mean()
    down_ratio = (normals[:, axis] < -0.7).mean()
    print(f"  {name}-up ratio: {up_ratio:.3f}, {name}-down ratio: {down_ratio:.3f}")

# Test height-based split with Y-up
heights = centroids[:, 1]
height_range = heights.max() - heights.min()
height_cutoff = heights.min() + height_range * 0.3
is_ground = (normals[:, 1] > 0.7) & (heights < height_cutoff)
ground_ratio = is_ground.sum() / len(is_ground)
print(f"\nY-up ground detection: ratio={ground_ratio:.4f} ({is_ground.sum()}/{len(is_ground)} faces)")

# Test with Z-up
heights_z = centroids[:, 2]
height_range_z = heights_z.max() - heights_z.min()
height_cutoff_z = heights_z.min() + height_range_z * 0.3
is_ground_z = (normals[:, 2] > 0.7) & (heights_z < height_cutoff_z)
ground_ratio_z = is_ground_z.sum() / len(is_ground_z)
print(f"Z-up ground detection: ratio={ground_ratio_z:.4f} ({is_ground_z.sum()}/{len(is_ground_z)} faces)")
