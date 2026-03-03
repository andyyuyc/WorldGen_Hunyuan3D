from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from torchmcubes import marching_cubes as _torchmcubes_mc
except ImportError:
    _torchmcubes_mc = None

try:
    import mcubes
except ImportError:
    mcubes = None


def _marching_cubes_pymcubes(vol, threshold):
    """PyMCubes-based marching cubes that returns torch tensors."""
    vol_np = vol.detach().cpu().numpy().astype(np.float32)
    vertices, triangles = mcubes.marching_cubes(vol_np, threshold)
    return torch.from_numpy(vertices).float(), torch.from_numpy(triangles.astype(np.int64)).long()


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self._grid_vertices: Optional[torch.FloatTensor] = None

        if _torchmcubes_mc is not None:
            self.mc_func: Callable = _torchmcubes_mc
            self._use_pymcubes = False
        elif mcubes is not None:
            self.mc_func = _marching_cubes_pymcubes
            self._use_pymcubes = True
        else:
            raise ImportError("Neither torchmcubes nor PyMCubes is installed.")

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        try:
            v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
        except AttributeError:
            v_pos, t_pos_idx = self.mc_func(level.detach().cpu(), 0.0)
        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
