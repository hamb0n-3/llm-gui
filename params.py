from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Tuple

from config import (
    DEFAULT_MLX_MODEL, DEFAULT_VLM_MODEL, DEFAULT_GGUF_MODEL,
    MLX_MAX_TOKENS, MLX_TEMPERATURE, MLX_TOP_P, MLX_TOP_K, MLX_REPEAT_PENALTY, MLX_N_CTX_ESTIMATE,
    GGUF_N_CTX, GGUF_N_GPU_LAYERS, GGUF_SEED, GGUF_MAX_TOKENS, GGUF_TEMPERATURE, GGUF_TOP_P,
    GGUF_TOP_K, GGUF_REPEAT_PENALTY, GGUF_MMPROJ,
    VLM_MAX_TOKENS, VLM_TEMPERATURE, VLM_TOP_P, VLM_TOP_K, VLM_REPEAT_PENALTY, VLM_N_CTX_ESTIMATE,
    VLM_RESEND_HISTORY_MEDIA, VLM_MEDIA_HISTORY_CAP, ENABLE_THINKING, THINKING_BUDGET,
    safe_int, is_vlm, is_mlx,
)


@dataclass
class MLXParams:
    model_id_or_path: str = DEFAULT_MLX_MODEL
    max_tokens: int = MLX_MAX_TOKENS
    temperature: float = MLX_TEMPERATURE
    top_p: float = MLX_TOP_P
    top_k: int = MLX_TOP_K
    repeat_penalty: float = MLX_REPEAT_PENALTY
    n_ctx_estimate: int = MLX_N_CTX_ESTIMATE
    enable_thinking: bool = ENABLE_THINKING
    thinking_budget: int = THINKING_BUDGET


@dataclass
class VLMParams:
    model_id_or_path: str = DEFAULT_VLM_MODEL
    max_tokens: int = VLM_MAX_TOKENS
    temperature: float = VLM_TEMPERATURE
    top_p: float = VLM_TOP_P
    top_k: int = VLM_TOP_K
    repeat_penalty: float = VLM_REPEAT_PENALTY
    n_ctx_estimate: int = VLM_N_CTX_ESTIMATE
    resend_history_media: bool = VLM_RESEND_HISTORY_MEDIA
    media_history_cap: int = VLM_MEDIA_HISTORY_CAP
    enable_thinking: bool = ENABLE_THINKING
    thinking_budget: int = THINKING_BUDGET


@dataclass
class GGUFParams:
    model_path: str = DEFAULT_GGUF_MODEL
    n_ctx: int = GGUF_N_CTX
    n_gpu_layers: int = GGUF_N_GPU_LAYERS
    seed: int = GGUF_SEED
    max_tokens: int = GGUF_MAX_TOKENS
    temperature: float = GGUF_TEMPERATURE
    top_p: float = GGUF_TOP_P
    top_k: int = GGUF_TOP_K
    repeat_penalty: float = GGUF_REPEAT_PENALTY
    mmproj_path: str = GGUF_MMPROJ  # "" = autodetect beside the model
    enable_thinking: bool = ENABLE_THINKING
    thinking_budget: int = THINKING_BUDGET


# Must stay in lockstep with `vlm_inputs` in build_ui() (handlers get *vlm_args).
VLM_INPUT_ORDER = (
    "model_id_or_path", "max_tokens", "temperature", "top_p", "top_k",
    "repeat_penalty", "n_ctx_estimate", "resend_history_media",
    "media_history_cap",
)


def build_vlm_params(*vlm_args: Any) -> VLMParams:
    d = VLMParams()
    # Empty block = not wired; anything else must be complete and in order, else
    # a short/reordered zip mis-assigns values with no visible error.
    if vlm_args and len(vlm_args) != len(VLM_INPUT_ORDER):
        raise ValueError(
            f"VLM parameter block has {len(vlm_args)} values, expected "
            f"{len(VLM_INPUT_ORDER)} ({', '.join(VLM_INPUT_ORDER)}). "
            "vlm_inputs in build_ui() and VLM_INPUT_ORDER have drifted apart."
        )
    vals = dict(zip(VLM_INPUT_ORDER, vlm_args))
    casts = {
        # No fallback: an empty box must stay empty so the loader can say so
        # instead of silently loading the default model.
        "model_id_or_path": lambda v: os.path.expanduser(str(v or "").strip()),
        "max_tokens": lambda v: safe_int(v, d.max_tokens),
        "temperature": lambda v: float(v),
        "top_p": lambda v: float(v),
        "top_k": lambda v: int(v),
        "repeat_penalty": lambda v: float(v),
        "n_ctx_estimate": lambda v: safe_int(v, d.n_ctx_estimate),
        "resend_history_media": lambda v: bool(v),
        "media_history_cap": lambda v: max(0, safe_int(v, d.media_history_cap)),
    }
    out = VLMParams()
    for name, raw in vals.items():
        try:
            setattr(out, name, casts[name](raw))
        except Exception:
            pass  # keep the default for anything unparseable
    return out


def build_params(
    mlx_model_id: str,
    mlx_max_tokens: int,
    mlx_temperature: float,
    mlx_top_p: float,
    mlx_top_k: int,
    mlx_repeat_penalty: float,
    mlx_n_ctx_estimate: int,
    gguf_model_path: str,
    gguf_n_ctx: int,
    gguf_n_gpu_layers: int,
    gguf_seed: int,
    gguf_max_tokens: int,
    gguf_temperature: float,
    gguf_top_p: float,
    gguf_top_k: int,
    gguf_repeat_penalty: float,
    enable_thinking: bool,
    thinking_budget: int,
    *vlm_args: Any,
) -> Tuple[MLXParams, GGUFParams, VLMParams]:
    mlx_params = MLXParams(
        model_id_or_path=os.path.expanduser((mlx_model_id or "").strip()),
        max_tokens=safe_int(mlx_max_tokens, MLX_MAX_TOKENS),
        temperature=float(mlx_temperature),
        top_p=float(mlx_top_p),
        top_k=int(mlx_top_k),
        repeat_penalty=float(mlx_repeat_penalty),
        n_ctx_estimate=safe_int(mlx_n_ctx_estimate, MLX_N_CTX_ESTIMATE),
    )
    gguf_params = GGUFParams(
        model_path=os.path.expanduser((gguf_model_path or "").strip()),
        n_ctx=safe_int(gguf_n_ctx, GGUF_N_CTX),
        n_gpu_layers=int(gguf_n_gpu_layers),
        seed=int(gguf_seed),
        max_tokens=safe_int(gguf_max_tokens, GGUF_MAX_TOKENS),
        temperature=float(gguf_temperature),
        top_p=float(gguf_top_p),
        top_k=int(gguf_top_k),
        repeat_penalty=float(gguf_repeat_penalty),
    )
    # A tab left open across a restart posts the previous input order at the
    # same arity, so Gradio's own length check passes and every value lands one
    # slot off. This pair is typed, so a string here means the page is stale.
    if isinstance(enable_thinking, str):
        raise ValueError(
            "This page was built by an older version of the UI, so its inputs no "
            "longer line up with the server. Hard-refresh the browser tab "
            "(Cmd-Shift-R) to reload the current layout."
        )
    vlm_params = build_vlm_params(*vlm_args)
    try:
        budget = max(0, int(thinking_budget or 0))
    except Exception:
        budget = 0
    think = bool(enable_thinking)
    for p in (mlx_params, gguf_params, vlm_params):
        p.enable_thinking, p.thinking_budget = think, budget
    return mlx_params, gguf_params, vlm_params


def params_for(backend: str, mlx_params: MLXParams, gguf_params: GGUFParams, vlm_params: VLMParams):
    if is_vlm(backend):
        return vlm_params
    if is_mlx(backend):
        return mlx_params
    return gguf_params


def model_ref(params: Any) -> str:
    # VLM/MLX use model_id_or_path, GGUF uses model_path.
    return str(getattr(params, "model_id_or_path", None) or getattr(params, "model_path", "") or "")
