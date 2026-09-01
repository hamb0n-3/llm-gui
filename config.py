from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any, NamedTuple

BACKEND_MLX = "MLX (mlx_lm)"
BACKEND_GGUF = "GGUF (llama.cpp)"
BACKEND_VLM = "MLX-VLM (vision)"
BACKENDS = [BACKEND_MLX, BACKEND_GGUF, BACKEND_VLM]


def is_vlm(backend: str) -> bool:
    return str(backend or "").startswith("MLX-VLM")


def is_mlx(backend: str) -> bool:
    # BACKEND_VLM also starts with "MLX"; exclude it.
    b = str(backend or "")
    return b.startswith("MLX") and not is_vlm(b)


def is_gguf(backend: str) -> bool:
    return str(backend or "").startswith("GGUF")


class Range(NamedTuple):
    min: float
    max: float
    step: float


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7100  # busy → Gradio picks a free port

DEFAULT_BACKEND = BACKEND_GGUF
DEFAULT_SYSTEM_PROMPT = "You are an extremely intelligence assistant."
SHOW_ADVANCED_AT_STARTUP = False

# Basic view (advanced checkbox off): backend is chosen for the user — text to
# llama.cpp, attaching media switches to vision for that send.
BASIC_TEXT_BACKEND = BACKEND_GGUF
BASIC_MEDIA_BACKEND = BACKEND_VLM
BASIC_AUTOLOAD = True
# Transient bubbles; live in Session.pending_note, never history, so they can't
# reach the prompt.
LOADING_NOTE = "⏳ **Loading model…**"
THINKING_NOTE = "💭 *Thinking…*"
DEFAULT_MODELS_DIR = Path(os.environ.get("LLM_GUI_MODELS_DIR", "~/.cache/Models")).expanduser()

DEFAULT_MLX_MODEL = "ailexleon/Huihui-Qwen3.8-27B-abliterated-mlx-8Bit"
DEFAULT_VLM_MODEL = "huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated"
DEFAULT_GGUF_MODEL = os.environ.get(
    "GGUF_MODEL",
    os.path.expanduser(
        "~/.cache/Models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS/"
        "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf"
    ),
)

MLX_MAX_TOKENS = 15000
MLX_TEMPERATURE = 1.0
MLX_TOP_P = 0.95
MLX_TOP_K = 40
MLX_REPEAT_PENALTY = 1.0
MLX_N_CTX_ESTIMATE = 125000  # only when context auto-detect fails

# n_ctx / n_gpu_layers / seed only apply on model load.
# KV cache is f16 and linear in n_ctx: the default model costs ~96 KiB/token
# (48 layers x 2 kv heads x 256 head dim), so 250k ctx alone is ~25 GB on top
# of ~97 GB of weights. The load ladder steps this down further on OOM.
GGUF_N_CTX = 32768
GGUF_N_GPU_LAYERS = 200
GGUF_SEED = 0  # 0 = random
GGUF_MAX_TOKENS = 15000
GGUF_TEMPERATURE = 1.0
GGUF_TOP_P = 0.95
GGUF_TOP_K = 40
GGUF_REPEAT_PENALTY = 1.0
GGUF_CTX_FALLBACK = 4096  # last resort when nothing reports a context size
GGUF_CTX_MIN_FALLBACK = 4096  # floor for the Metal-OOM n_ctx fallback ladder
# Empty + autodetect on = use any mmproj*.gguf in the model's own folder.
GGUF_MMPROJ = os.environ.get("GGUF_MMPROJ", "")
GGUF_MMPROJ_AUTODETECT = True

VLM_MAX_TOKENS = 15000
VLM_TEMPERATURE = 1.0
VLM_TOP_P = 0.95
VLM_TOP_K = 40
VLM_REPEAT_PENALTY = 1.0
VLM_N_CTX_ESTIMATE = 250000  # only when context auto-detect fails
# Off by default: re-encoding history media costs a vision-tower pass and a
# large prefill block every generation.
VLM_RESEND_HISTORY_MEDIA = False
VLM_MEDIA_HISTORY_CAP = 4  # most-recent images kept when resending
VLM_ENABLE_THINKING = True
VLM_THINKING_BUDGET = 0  # 0 = no budget cap

WEB_USE_IN_CHAT = False
WEB_N_PAGES = 3

# RANGE_*_N_CTX also clamps detected context lengths, so raising the max lets a
# longer-context model report it.
RANGE_MAX_TOKENS = Range(16, 32000, 1)
RANGE_TEMPERATURE = Range(0.0, 2.0, 0.05)
RANGE_TOP_P = Range(0.0, 1.0, 0.01)
RANGE_TOP_K = Range(0, 200, 1)
RANGE_REPEAT_PENALTY = Range(0.0, 2.0, 0.01)
RANGE_N_CTX = Range(512, 250072, 64)        # MLX and MLX-VLM estimate
RANGE_GGUF_N_CTX = Range(256, 250072, 64)   # real llama.cpp context window
RANGE_N_GPU_LAYERS = Range(0, 200, 1)
RANGE_MEDIA_CAP = Range(1, 16, 1)
RANGE_THINKING_BUDGET = Range(0, 8192, 64)
RANGE_WEB_PAGES = Range(1, 8, 1)

MEDIA_KINDS = ("image", "video", "audio")

# Budget reserves only — running the real image processor is too slow for a
# per-keystroke meter.
VLM_IMAGE_TOKEN_ESTIMATE = 800
VLM_VIDEO_TOKEN_ESTIMATE = 2500  # fallback when a clip's duration can't be read
VLM_AUDIO_TOKEN_ESTIMATE = 800
# ~65 tokens/sec of clip, measured against Qwen2-VL at mlx-vlm's default fps=1.
VLM_VIDEO_TOKENS_PER_SECOND = 65

# Text-like extensions folded into the message body (see inline_text_attachments).
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml", ".html", ".css",
    ".log", ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".hpp",
    ".rs", ".go", ".java", ".rb", ".php", ".swift", ".kt", ".sh", ".zsh",
    ".bash", ".sql", ".r", ".jl", ".lua", ".pl", ".diff", ".patch",
}
# Larger stays an attachment rather than blowing the context budget.
MAX_INLINE_TEXT_BYTES = 200_000

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus"}


def now_ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_int(val: Any, default: int) -> int:
    try:
        v = int(val)
        if v <= 0:
            return default
        return v
    except Exception:
        return default


def clamp_ctx(value: Any, rng: Range, fallback: int) -> int:
    # Gradio rejects a slider value past its bounds; clamp the detected ctx.
    try:
        return max(int(rng.min), min(int(value), int(rng.max)))
    except Exception:
        return int(fallback)
