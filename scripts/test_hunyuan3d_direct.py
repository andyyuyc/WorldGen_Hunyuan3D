"""Direct test: run Hunyuan3D shapegen + texgen on a single image, with timing."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "Hunyuan3D-2"))

# Force unbuffered output so logs appear in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
print(f"CUDA arch: sm_{props.major}{props.minor}")

def gpu_mem():
    """Return current GPU memory usage string."""
    alloc = torch.cuda.memory_allocated() / 1024**3
    resv = torch.cuda.memory_reserved() / 1024**3
    return f"GPU mem: {alloc:.1f}GB allocated, {resv:.1f}GB reserved"

# Check for reference image, create a test image if not found
image_path = str(project_root / "data" / "intermediate" / "stage1" / "reference.png")
if not os.path.exists(image_path):
    print(f"\nNo reference image at {image_path}, creating test image...")
    from PIL import Image
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    # Create a simple test image (white cube-like shape on gray background)
    img = Image.new("RGB", (512, 512), (180, 180, 180))
    img.save(image_path)
    print(f"  Created test image at {image_path}")

output_dir = project_root / "data" / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Shape generation
print("\n=== Step 1: Shapegen ===")
print(f"  {gpu_mem()}")
t0 = time.time()

print("  Importing shapegen module...")
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
print(f"  Import took {time.time() - t0:.1f}s")

t1 = time.time()
# Hunyuan3D-2.1: 3.0B model (vs 1.1B in v2.0), higher quality shape generation
MODEL_PATH = "tencent/Hunyuan3D-2.1"
SUBFOLDER = "hunyuan3d-dit-v2-1"
print(f"  Loading shapegen model ({MODEL_PATH}, {SUBFOLDER})...")
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    MODEL_PATH, subfolder=SUBFOLDER, use_safetensors=False
)
print(f"  Model load took {time.time() - t1:.1f}s")
print(f"  {gpu_mem()}")

t2 = time.time()
# High-quality settings: more steps, higher resolution grid, stronger guidance
OCTREE_RES = 512          # default 384, max 512 — finer mesh detail
NUM_STEPS = 100            # default 50 — more denoising steps for better shape
GUIDANCE_SCALE = 7.5       # default 5.0 — stronger adherence to input image
NUM_CHUNKS = 20000         # default 8000 — faster volume decoding (uses more VRAM)
MC_ALGO = 'mc'             # marching cubes (vs 'dmc' dual marching cubes)

print(f"  Running shapegen inference on {image_path}...")
print(f"  Settings: octree_res={OCTREE_RES}, steps={NUM_STEPS}, cfg={GUIDANCE_SCALE}, chunks={NUM_CHUNKS}")
mesh = pipeline(
    image=image_path,
    num_inference_steps=NUM_STEPS,
    octree_resolution=OCTREE_RES,
    guidance_scale=GUIDANCE_SCALE,
    num_chunks=NUM_CHUNKS,
)[0]
print(f"  Inference took {time.time() - t2:.1f}s")
print(f"  Result: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
print(f"  {gpu_mem()}")

# Save shape-only
shape_path = output_dir / "test_shape_only.glb"
mesh.export(str(shape_path))
print(f"  Shape saved to {shape_path}")

# Free shapegen VRAM
del pipeline
torch.cuda.empty_cache()
print(f"  Freed shapegen model. {gpu_mem()}")
print(f"  Shapegen total: {time.time() - t0:.1f}s")

# Step 2: Texture generation
print("\n=== Step 2: Texgen ===")
t3 = time.time()

try:
    print("  Importing texgen module...")
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    print(f"  Import took {time.time() - t3:.1f}s")

    t4 = time.time()
    print("  Loading texgen model (hunyuan3d-paint-v2-0-turbo)...")
    tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-paint-v2-0-turbo",
    )
    print(f"  Model load took {time.time() - t4:.1f}s")
    print(f"  {gpu_mem()}")

    from PIL import Image
    image = Image.open(image_path).convert("RGBA")

    # Verify C++ extensions are loaded (not slow Python fallbacks)
    mp_mod = sys.modules.get("hy3dgen.texgen.differentiable_renderer.mesh_processor")
    if mp_mod and hasattr(mp_mod, "__file__"):
        is_cpp = mp_mod.__file__.endswith((".pyd", ".so"))
        print(f"  mesh_processor: {'C++ extension' if is_cpp else 'PYTHON FALLBACK (SLOW!)'} — {mp_mod.__file__}")

    # Run texgen with per-step timing
    t5 = time.time()
    print("  Running texgen inference with detailed timing...")

    # Step 2a: Delight (remove lighting/shadows)
    t_step = time.time()
    images_prompt = [tex_pipeline.recenter_image(image)]
    images_prompt = [tex_pipeline.models['delight_model'](img) for img in images_prompt]
    print(f"    [2a] Delight:       {time.time() - t_step:.1f}s  {gpu_mem()}")

    # Step 2b: Decimate mesh for texgen (texture resolution is only 1-2K pixels,
    # so 100K faces is more than enough; 1.7M faces makes xatlas take 30+ min)
    t_step = time.time()
    import trimesh as _trimesh
    _m = mesh if not isinstance(_m := mesh, _trimesh.Scene) else mesh.dump(concatenate=True)
    orig_faces = len(_m.faces)
    TEXGEN_MAX_FACES = 200_000
    if orig_faces > TEXGEN_MAX_FACES:
        print(f"    [2b] Decimating mesh for texgen: {orig_faces:,} -> {TEXGEN_MAX_FACES:,} faces...")
        sys.stdout.flush()
        _m = _m.simplify_quadric_decimation(1.0 - TEXGEN_MAX_FACES / orig_faces)
        print(f"    [2b] Decimated: {len(_m.vertices):,} verts, {len(_m.faces):,} faces ({time.time() - t_step:.1f}s)")
    else:
        print(f"    [2b] Mesh: {len(_m.vertices):,} verts, {len(_m.faces):,} faces (no decimation needed)")

    # UV wrap with xatlas
    import xatlas
    print(f"    [2b] Running xatlas.parametrize...")
    sys.stdout.flush()
    _v = np.ascontiguousarray(_m.vertices, dtype=np.float32)
    _f = np.ascontiguousarray(_m.faces, dtype=np.int32)
    _vmapping, _indices, _uvs = xatlas.parametrize(_v, _f)
    _m.vertices = _m.vertices[_vmapping]
    _m.faces = _indices
    _m.visual.uv = _uvs
    print(f"    [2b] xatlas done, loading mesh into renderer...")
    tex_pipeline.render.load_mesh(_m)
    print(f"    [2b] UV wrap+load:  {time.time() - t_step:.1f}s")

    # Step 2c: Render normal maps
    t_step = time.time()
    cfg = tex_pipeline.config
    normal_maps = tex_pipeline.render_normal_multiview(
        cfg.candidate_camera_elevs, cfg.candidate_camera_azims, use_abs_coor=True)
    print(f"    [2c] Normal maps:   {time.time() - t_step:.1f}s  {gpu_mem()}")

    # Step 2d: Render position maps
    t_step = time.time()
    position_maps = tex_pipeline.render_position_multiview(
        cfg.candidate_camera_elevs, cfg.candidate_camera_azims)
    print(f"    [2d] Position maps: {time.time() - t_step:.1f}s")

    # Step 2e: Multiview diffusion
    t_step = time.time()
    camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
        elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                   zip(cfg.candidate_camera_azims, cfg.candidate_camera_elevs)]
    multiviews = tex_pipeline.models['multiview_model'](images_prompt, normal_maps + position_maps, camera_info)
    print(f"    [2e] Multiview:     {time.time() - t_step:.1f}s  {gpu_mem()}")

    # Step 2f: Resize multiviews
    t_step = time.time()
    for i in range(len(multiviews)):
        multiviews[i] = multiviews[i].resize((cfg.render_size, cfg.render_size))
    print(f"    [2f] Resize:        {time.time() - t_step:.1f}s")

    # Step 2g: Bake textures
    t_step = time.time()
    texture, mask = tex_pipeline.bake_from_multiview(
        multiviews, cfg.candidate_camera_elevs, cfg.candidate_camera_azims,
        cfg.candidate_view_weights, method=cfg.merge_method)
    print(f"    [2g] Bake:          {time.time() - t_step:.1f}s  {gpu_mem()}")

    # Step 2h: UV inpaint
    t_step = time.time()
    import numpy as np
    mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    texture = tex_pipeline.texture_inpaint(texture, mask_np)
    print(f"    [2h] UV inpaint:    {time.time() - t_step:.1f}s")

    # Step 2i: Save
    t_step = time.time()
    tex_pipeline.render.set_texture(texture)
    textured_mesh = tex_pipeline.render.save_mesh()
    print(f"    [2i] Save mesh:     {time.time() - t_step:.1f}s")

    print(f"  Texgen inference total: {time.time() - t5:.1f}s")
    print(f"  {gpu_mem()}")

    tex_path = output_dir / "test_textured.glb"
    textured_mesh.export(str(tex_path))
    print(f"  Textured mesh saved to {tex_path}")
    print(f"  Texgen total: {time.time() - t3:.1f}s")

except Exception as e:
    print(f"  Texgen failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n=== Total time: {time.time() - t0:.1f}s ===")
print("Done!")
