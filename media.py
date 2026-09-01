from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import (
    MEDIA_KINDS, TEXT_EXTS, MAX_INLINE_TEXT_BYTES, IMAGE_EXTS, VIDEO_EXTS, AUDIO_EXTS,
    VLM_IMAGE_TOKEN_ESTIMATE, VLM_VIDEO_TOKEN_ESTIMATE, VLM_AUDIO_TOKEN_ESTIMATE,
    VLM_VIDEO_TOKENS_PER_SECOND,
)


def looks_like_text(path: str, probe: int = 8192) -> bool:
    # Pasted clipboard files get inconsistent names, so sniff the head: NUL or
    # undecodable UTF-8 means binary.
    try:
        with open(path, "rb") as f:
            head = f.read(probe)
    except Exception:
        return False
    if not head or b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        # A multi-byte char can straddle the probe boundary; retry without the tail.
        try:
            head[:-4].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


def inline_text_attachments(paths: List[str]) -> Tuple[str, List[str]]:
    inlined: List[str] = []
    seen: set = set()
    rest: List[str] = []
    for p in paths:
        name = Path(str(p)).name
        if Path(str(p)).suffix.lower() not in TEXT_EXTS and not looks_like_text(p):
            rest.append(p)
            continue
        try:
            oversized = os.path.getsize(p) > MAX_INLINE_TEXT_BYTES
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                body = f.read(MAX_INLINE_TEXT_BYTES)
            if oversized:
                body += f"\n\n… [attachment truncated at {MAX_INLINE_TEXT_BYTES} bytes]"
        except Exception:
            rest.append(p)
            continue
        key = body.strip()
        if not key or key in seen:
            continue  # empty, or the same paste delivered twice
        seen.add(key)
        # A paste is the message; a real file keeps its name as a header.
        inlined.append(body if name.startswith("pasted") else f"--- {name} ---\n{body}")
    return "\n\n".join(inlined), rest


def classify_media(files: Iterable[Any]) -> Tuple[Dict[str, List[str]], List[str]]:
    out: Dict[str, List[str]] = {k: [] for k in MEDIA_KINDS}
    unknown: List[str] = []
    for f in files or []:
        # Usually plain paths, but FileData dicts/objects appear in some Gradio
        # versions and on chat-JSON reload.
        path = f if isinstance(f, str) else (f.get("path") if isinstance(f, dict) else getattr(f, "path", None) or getattr(f, "name", None))
        if not path:
            continue
        ext = Path(str(path)).suffix.lower()
        if ext in IMAGE_EXTS:
            out["image"].append(str(path))
        elif ext in VIDEO_EXTS:
            out["video"].append(str(path))
        elif ext in AUDIO_EXTS:
            out["audio"].append(str(path))
        else:
            unknown.append(str(path))
    return out, unknown


_video_duration_cache: Dict[Tuple[str, float, int], Optional[float]] = {}


def video_duration_seconds(path: str) -> Optional[float]:
    try:
        st = os.stat(path)
        key = (str(path), st.st_mtime, st.st_size)
    except Exception:
        return None
    if key in _video_duration_cache:
        return _video_duration_cache[key]
    duration: Optional[float] = None
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps > 0 and frames > 0:
                duration = frames / fps
        finally:
            cap.release()
    except Exception:
        duration = None
    _video_duration_cache[key] = duration
    return duration


def video_token_estimate(path: str) -> int:
    secs = video_duration_seconds(path)
    if not secs or secs <= 0:
        return VLM_VIDEO_TOKEN_ESTIMATE
    return max(64, int(secs * VLM_VIDEO_TOKENS_PER_SECOND))


def media_reserve_tokens(images: List[str], videos: List[str], audios: List[str]) -> int:
    return (
        len(images) * VLM_IMAGE_TOKEN_ESTIMATE
        + sum(video_token_estimate(v) for v in videos)
        + len(audios) * VLM_AUDIO_TOKEN_ESTIMATE
    )
