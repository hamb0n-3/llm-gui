from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import deps
from config import DEFAULT_MODELS_DIR


def hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path("~/.cache/huggingface/hub").expanduser()


def has_local_weights(d: Path) -> bool:
    # p.exists() follows symlinks: an interrupted HF download leaves the config
    # but no resolvable blob, so a real *.safetensors is the only weights signal.
    try:
        return any(p.exists() for p in d.glob("*.safetensors"))
    except Exception:
        return False


def hf_cached_snapshot(cache_dir: Path, require_weights: bool = True) -> Optional[Path]:
    # Takes the cache dir, not a repo id: the `/` -> `--` encoding is lossy for
    # repo names containing `--` and can't be reversed. Prefers refs/main.
    try:
        if not cache_dir.is_dir():
            return None
        candidates: List[Path] = []
        try:
            head = (cache_dir / "refs" / "main").read_text(encoding="utf-8").strip()
            if head:
                candidates.append(cache_dir / "snapshots" / head)
        except Exception:
            pass
        candidates.extend(sorted(cache_dir.glob("snapshots/*"), reverse=True))
        for snap in candidates:
            if not snap.is_dir() or not (snap / "config.json").is_file():
                continue
            if require_weights and not has_local_weights(snap):
                continue
            return snap
    except Exception:
        return None
    return None


def hf_snapshot_dir(repo_id: str, require_weights: bool = True) -> Optional[Path]:
    s = str(repo_id or "").strip()
    if "/" not in s or s.startswith((".", "/", "~")):
        return None
    return hf_cached_snapshot(
        hf_hub_cache() / ("models--" + s.replace("/", "--")), require_weights
    )


def local_model_dir(path_or_repo: str) -> Optional[Path]:
    # No Hub request. Weights not required, so the config-only multimodal probe
    # still resolves a config-only cache entry.
    s = str(path_or_repo or "").strip()
    if not s:
        return None
    try:
        local = Path(os.path.expanduser(s))
        if local.is_dir() and (local / "config.json").is_file():
            return local
    except Exception:
        pass
    return hf_snapshot_dir(s, require_weights=False)


def resolve_local_model(path_or_repo: str) -> str:
    # mlx_lm/mlx_vlm skip the Hub only for a path that exists() on disk; a repo
    # id costs a revision lookup every load. Uncached ids pass through unchanged.
    snap = hf_snapshot_dir(path_or_repo)
    return str(snap) if snap is not None else str(path_or_repo or "").strip()


def is_mlx_model_dir(p: Path) -> bool:
    return p.is_dir() and (p / "config.json").is_file() and has_local_weights(p)


def scan_mlx_models(models_dir: str) -> List[str]:
    found: List[str] = []
    root = Path(str(models_dir or "").strip() or DEFAULT_MODELS_DIR).expanduser()
    try:
        if is_mlx_model_dir(root):
            found.append(str(root))
        elif root.is_dir():
            for child in sorted(root.iterdir()):
                if is_mlx_model_dir(child):
                    found.append(str(child))
                elif child.is_dir():
                    for sub in sorted(child.iterdir()):
                        if is_mlx_model_dir(sub):
                            found.append(str(sub))
    except Exception:
        pass
    try:
        for d in sorted(hf_hub_cache().glob("models--*")):
            # Requires fetched weights: an aborted download leaves only the config.
            if hf_cached_snapshot(d) is not None:
                found.append(d.name[len("models--"):].replace("--", "/"))
    except Exception:
        pass
    return list(dict.fromkeys(found))


# `…-00002-of-00003.gguf` — llama.cpp's shard naming.
_SPLIT_SUFFIX_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def is_mmproj_file(path: Any) -> bool:
    return "mmproj" in Path(str(path)).name.lower()


def split_shard_index(path: Any) -> Optional[int]:
    m = _SPLIT_SUFFIX_RE.search(Path(str(path)).name)
    return int(m.group(1)) if m else None


def gguf_is_selectable(path: Any) -> bool:
    # A split model is ONE entry (its first shard); llama.cpp loads the siblings
    # itself. An mmproj is loaded alongside a model, never on its own.
    if is_mmproj_file(path):
        return False
    idx = split_shard_index(path)
    return idx is None or idx == 1


def find_mmproj_for(model_path: str) -> Optional[str]:
    # Vision GGUFs ship the projector as a separate file beside the model.
    try:
        d = Path(os.path.expanduser(str(model_path or ""))).expanduser().parent
        cands = sorted(str(p) for p in d.glob("*.gguf") if is_mmproj_file(p))
    except Exception:
        return None
    return cands[0] if cands else None


def scan_gguf_models(models_dir: str) -> List[str]:
    found: List[str] = []
    root = Path(str(models_dir or "").strip() or DEFAULT_MODELS_DIR).expanduser()
    try:
        # An explicitly pointed-at file is taken as given, filter or not.
        if root.is_file() and root.suffix.lower() == ".gguf":
            found.append(str(root))
        elif root.is_dir():
            for pattern in ("*.gguf", "*/*.gguf", "*/*/*.gguf"):
                found.extend(str(p) for p in sorted(root.glob(pattern)) if gguf_is_selectable(p))
    except Exception:
        pass
    try:
        # p.exists() follows the symlink into the blob store: an interrupted
        # download has no blob behind it and is not loadable.
        found.extend(
            str(p) for p in sorted(hf_hub_cache().glob("models--*/snapshots/*/*.gguf"))
            if p.exists() and gguf_is_selectable(p)
        )
    except Exception:
        pass
    return list(dict.fromkeys(found))


def read_local_model_config(path_or_repo: str) -> Optional[Dict[str, Any]]:
    d = local_model_dir(path_or_repo)
    if d is None:
        return None
    try:
        with open(d / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else None
    except Exception:
        return None


def read_model_config(path_or_repo: str) -> Optional[Dict[str, Any]]:
    # Local dir, then HF cache, then a single config.json Hub pull as last resort.
    # Not mlx_vlm.load_config(): it snapshot-downloads the whole repo.
    s = str(path_or_repo or "").strip()
    if not s:
        return None

    def _load(p: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg if isinstance(cfg, dict) else None
        except Exception:
            return None

    cfg = read_local_model_config(s)
    if cfg is not None:
        return cfg

    if "/" in s and not s.startswith((".", "/", "~")):
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            return _load(Path(hf_hub_download(repo_id=s, filename="config.json")))
        except Exception:
            return None
    return None


def mlx_lm_supports_model_type(model_type: str) -> Optional[bool]:
    # mlx_lm has modules for several VL architectures and runs their language
    # half text-only, so a declared vision tower alone doesn't disqualify an MLX
    # load. None when undeterminable.
    mt = str(model_type or "").strip()
    if not mt:
        return None
    try:
        import importlib.util

        return importlib.util.find_spec(f"mlx_lm.models.{mt}") is not None
    except Exception:
        return None


def config_model_type(cfg: Optional[Dict[str, Any]]) -> str:
    if not isinstance(cfg, dict):
        return ""
    return str(cfg.get("model_type") or "")


def config_is_multimodal(cfg: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(cfg, dict):
        return False
    if deps.vlm_is_text_only_config is not None:
        try:
            return not deps.vlm_is_text_only_config(cfg)
        except Exception:
            pass
    return any(
        cfg.get(key) not in (None, {})
        for key in ("vision_config", "audio_config", "dflash_config")
    )


def scan_vlm_models(models_dir: str) -> List[str]:
    found: List[str] = []
    for entry in scan_mlx_models(models_dir):
        try:
            if config_is_multimodal(read_local_model_config(entry)):
                found.append(entry)
        except Exception:
            continue
    return list(dict.fromkeys(found))


def looks_like_gguf(s: str) -> bool:
    # Heuristic; false positives are fine — only gates an early "wrong backend"
    # message, never a real load.
    if not s:
        return False
    try:
        p = Path(s)
        if p.is_file() and p.suffix.lower() == ".gguf":
            return True
        if p.is_dir():
            try:
                for child in p.iterdir():
                    if child.is_file() and child.suffix.lower() == ".gguf":
                        return True
            except Exception:
                pass
    except Exception:
        pass
    s_low = str(s).lower()
    if "gguf" in s_low:
        return True
    if "-q" in s_low or "_q" in s_low:
        # only if a digit follows q, e.g. -Q4_K_M
        idx = s_low.find("-q")
        if idx == -1:
            idx = s_low.find("_q")
        if idx != -1 and idx + 2 < len(s_low) and s_low[idx + 2].isdigit():
            return True
    return False
