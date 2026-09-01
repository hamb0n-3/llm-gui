from __future__ import annotations

# MLX (Apple) — optional
_MLX_AVAILABLE = False
mlx_load = None
mlx_generate = None
mlx_stream_generate = None
mlx_make_sampler = None
mlx_make_logits_processors = None
try:
    from mlx_lm import load as _mlx_load_func  # type: ignore
    from mlx_lm import generate as _mlx_generate_func  # type: ignore
    try:
        from mlx_lm import stream_generate as _mlx_stream_generate_func  # type: ignore
    except Exception:
        _mlx_stream_generate_func = None
    # mlx-lm >= 0.19 only accepts sampling via sampler/logits_processors
    try:
        from mlx_lm.sample_utils import make_sampler as _mlx_make_sampler_func  # type: ignore
        from mlx_lm.sample_utils import make_logits_processors as _mlx_make_logits_processors_func  # type: ignore
    except Exception:
        _mlx_make_sampler_func = None
        _mlx_make_logits_processors_func = None

    _MLX_AVAILABLE = True
    mlx_load = _mlx_load_func
    mlx_generate = _mlx_generate_func
    mlx_stream_generate = _mlx_stream_generate_func
    mlx_make_sampler = _mlx_make_sampler_func
    mlx_make_logits_processors = _mlx_make_logits_processors_func
except Exception:
    _MLX_AVAILABLE = False

# mlx-vlm (Apple, vision/audio/video language models) — optional
_VLM_AVAILABLE = False
_VLM_IMPORT_ERROR = ""
vlm_load = None
vlm_generate = None
vlm_stream_generate = None
vlm_apply_chat_template = None
vlm_load_config = None
vlm_is_text_only_config = None
vlm_make_sampler = None
vlm_make_logits_processors = None
vlm_get_chat_template = None
try:
    from mlx_vlm import load as _vlm_load_func  # type: ignore
    from mlx_vlm import generate as _vlm_generate_func  # type: ignore
    from mlx_vlm import stream_generate as _vlm_stream_generate_func  # type: ignore
    from mlx_vlm.prompt_utils import apply_chat_template as _vlm_act_func  # type: ignore
    from mlx_vlm.utils import load_config as _vlm_load_config_func  # type: ignore
    try:
        # Private helper; mirrored by config_is_multimodal() if it disappears
        from mlx_vlm.utils import _is_text_only_config as _vlm_text_only_func  # type: ignore
    except Exception:
        _vlm_text_only_func = None
    try:
        from mlx_vlm.sample_utils import make_sampler as _vlm_make_sampler_func  # type: ignore
        from mlx_vlm.sample_utils import make_logits_processors as _vlm_make_lp_func  # type: ignore
    except Exception:
        _vlm_make_sampler_func = None
        _vlm_make_lp_func = None
    try:
        # Needed to re-render a message list after pruning stray video items
        from mlx_vlm.prompt_utils import get_chat_template as _vlm_get_template_func  # type: ignore
    except Exception:
        _vlm_get_template_func = None

    _VLM_AVAILABLE = True
    vlm_load = _vlm_load_func
    vlm_generate = _vlm_generate_func
    vlm_stream_generate = _vlm_stream_generate_func
    vlm_apply_chat_template = _vlm_act_func
    vlm_load_config = _vlm_load_config_func
    vlm_is_text_only_config = _vlm_text_only_func
    vlm_make_sampler = _vlm_make_sampler_func
    vlm_make_logits_processors = _vlm_make_lp_func
    vlm_get_chat_template = _vlm_get_template_func
except Exception as _vlm_e:
    _VLM_AVAILABLE = False
    _VLM_IMPORT_ERROR = str(_vlm_e)

# llama.cpp (GGUF) — optional
_LLAMA_AVAILABLE = False
Llama = None
try:
    from llama_cpp import Llama as _LlamaClass  # type: ignore

    _LLAMA_AVAILABLE = True
    Llama = _LlamaClass
except Exception:
    _LLAMA_AVAILABLE = False

# MCP (Model Context Protocol) — optional, API may vary
_MCP_AVAILABLE = False
try:
    import mcp  # type: ignore  # noqa: F401 — probe only: a successful import sets the flag
    _MCP_AVAILABLE = True
except Exception:
    _MCP_AVAILABLE = False

# Web search (web_search.py, same directory) — optional
_WEB_AVAILABLE = False
_WEB_IMPORT_ERROR = ""
web = None
try:
    import web_search as _web_mod
    web = _web_mod
    _WEB_AVAILABLE = True
except Exception as _web_e:
    _WEB_IMPORT_ERROR = str(_web_e)
