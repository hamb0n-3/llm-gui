from __future__ import annotations

import atexit
from typing import Dict, List, Optional

from backend_base import Backend
from backend_gguf import GgufBackend
from backend_mlx import MlxBackend
from backend_vlm import VlmBackend
from config import BACKEND_MLX, BACKEND_GGUF, BACKEND_VLM

MLX = MlxBackend()
MLX.id = BACKEND_MLX
GGUF = GgufBackend()
GGUF.id = BACKEND_GGUF
VLM = VlmBackend()
VLM.id = BACKEND_VLM

REGISTRY: Dict[str, Backend] = {
    BACKEND_MLX: MLX,
    BACKEND_GGUF: GGUF,
    BACKEND_VLM: VLM,
}


def get_backend(backend_id: str) -> Optional[Backend]:
    return REGISTRY.get(backend_id)


def loaded_backends() -> List[str]:
    return [bid for bid, b in REGISTRY.items() if b.is_loaded()]


@atexit.register
def _release_all() -> None:
    # Ctrl+C otherwise leaves the native handles to interpreter teardown, where
    # ggml's Metal device is already gone and llama.cpp aborts on an assert.
    for b in REGISTRY.values():
        try:
            if b.is_loaded():
                b.unload()
        except Exception:
            pass
