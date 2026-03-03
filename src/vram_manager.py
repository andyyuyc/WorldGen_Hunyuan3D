"""GPU VRAM manager ensuring only one heavy model is loaded at a time."""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)


class VRAMManager:
    """Manages GPU memory by ensuring only one heavy ML model is loaded at a time.

    On a 24GB GPU, most stages require 10-20GB VRAM. This manager enforces
    sequential model loading with explicit cleanup between transitions.
    """

    def __init__(self, device: str = "cuda", vram_limit_gb: float = 24.0):
        self.device = device
        self.vram_limit_gb = vram_limit_gb
        self._current_model_name: str | None = None
        self._current_model: Any = None

    def get_free_vram_gb(self) -> float:
        """Return free GPU memory in GB."""
        try:
            import torch
            free, _total = torch.cuda.mem_get_info()
            return free / (1024**3)
        except (ImportError, RuntimeError):
            return -1.0

    def get_used_vram_gb(self) -> float:
        """Return used GPU memory in GB."""
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            return (total - free) / (1024**3)
        except (ImportError, RuntimeError):
            return -1.0

    def unload_current(self) -> None:
        """Fully unload the current model from GPU and reclaim memory."""
        if self._current_model is not None:
            name = self._current_model_name
            logger.info(f"Unloading model '{name}'...")

            # Move to CPU first if possible
            if hasattr(self._current_model, "to"):
                try:
                    self._current_model.to("cpu")
                except Exception:
                    pass

            del self._current_model
            self._current_model = None
            self._current_model_name = None

            gc.collect()

            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except (ImportError, RuntimeError):
                pass

            logger.info(
                f"Model '{name}' unloaded. Free VRAM: {self.get_free_vram_gb():.1f} GB"
            )

    @contextmanager
    def load_model(
        self, name: str, loader_fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Context manager that loads a model, yields it, then keeps it cached.

        If a different model is currently loaded, it will be unloaded first.
        If the same model is already loaded, it's reused without reloading.

        Usage:
            with vram.load_model("sdxl", load_sdxl_pipeline) as pipe:
                result = pipe(prompt="...")
        """
        if self._current_model_name == name and self._current_model is not None:
            logger.info(f"Reusing cached model '{name}'")
            yield self._current_model
            return

        # Unload previous model
        self.unload_current()

        free = self.get_free_vram_gb()
        logger.info(f"Loading model '{name}'... (Free VRAM: {free:.1f} GB)")

        model = loader_fn(*args, **kwargs)
        self._current_model = model
        self._current_model_name = name

        free_after = self.get_free_vram_gb()
        logger.info(
            f"Model '{name}' loaded. "
            f"(Free VRAM: {free_after:.1f} GB, Used: {free - free_after:.1f} GB)"
        )

        try:
            yield model
        finally:
            # Keep model cached; explicit unload_current() to release
            pass

    def force_unload_all(self) -> None:
        """Force unload everything and reclaim all GPU memory."""
        self.unload_current()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass
        logger.info(f"All models unloaded. Free VRAM: {self.get_free_vram_gb():.1f} GB")

    @property
    def current_model_name(self) -> str | None:
        return self._current_model_name

    def __repr__(self) -> str:
        used = self.get_used_vram_gb()
        free = self.get_free_vram_gb()
        model = self._current_model_name or "none"
        return f"VRAMManager(model={model}, used={used:.1f}GB, free={free:.1f}GB)"
