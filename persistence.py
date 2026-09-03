from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backends import get_backend
from config import GGUF_CTX_FALLBACK, now_ts, is_vlm, is_mlx
from params import params_for
from session import Session, chat_view, msg_media


def _context_window(backend_id: str, mlx_params, gguf_params, vlm_params) -> int:
    b = get_backend(backend_id)
    if b is None:
        return GGUF_CTX_FALLBACK
    return b.context_window(params_for(backend_id, mlx_params, gguf_params, vlm_params))


def snapshot_session(
    session: Session,
    backend: str,
    mlx_params,
    gguf_params,
    vlm_params=None,
) -> Dict[str, Any]:
    from params import VLMParams
    vlm_params = vlm_params or VLMParams()
    ctx = _context_window(backend, mlx_params, gguf_params, vlm_params)
    params = params_for(backend, mlx_params, gguf_params, vlm_params)
    mdl_info: Dict[str, Any] = {
        "backend": backend,
        "enable_thinking": params.enable_thinking,
        "thinking_budget": params.thinking_budget,
    }
    if is_vlm(backend):
        mdl_info.update({
            "model_id_or_path": vlm_params.model_id_or_path,
            "ctx_window": ctx,
            "max_tokens": vlm_params.max_tokens,
            "temperature": vlm_params.temperature,
            "top_p": vlm_params.top_p,
            "top_k": vlm_params.top_k,
            "repeat_penalty": vlm_params.repeat_penalty,
            "resend_history_media": vlm_params.resend_history_media,
            "media_history_cap": vlm_params.media_history_cap,
        })
    elif is_mlx(backend):
        mdl_info.update({
            "model_id_or_path": mlx_params.model_id_or_path,
            "ctx_window": ctx,
            "max_tokens": mlx_params.max_tokens,
            "temperature": mlx_params.temperature,
            "top_p": mlx_params.top_p,
            "top_k": mlx_params.top_k,
            "repeat_penalty": mlx_params.repeat_penalty,
        })
    else:
        mdl_info.update({
            "model_path": gguf_params.model_path,
            "ctx_window": ctx,
            "max_tokens": gguf_params.max_tokens,
            "temperature": gguf_params.temperature,
            "top_p": gguf_params.top_p,
            "top_k": gguf_params.top_k,
            "repeat_penalty": gguf_params.repeat_penalty,
        })

    return {
        "created_at": now_ts(),
        "system_prompt": session.system_prompt,
        "backend_info": mdl_info,
        "history": session.history,
    }


def save_chat_json(
    session: Session,
    backend: str,
    mlx_params,
    gguf_params,
    vlm_params=None,
) -> Tuple[str, str]:
    snap = snapshot_session(session, backend, mlx_params, gguf_params, vlm_params)
    out_dir = Path.cwd() / "chats"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"chat_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    fpath = out_dir / fname
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    return str(fpath), f"✅ Saved chat to `{fpath}`"


def load_chat_json(session: Session, fileobj) -> Tuple[List[Dict[str, Any]], str]:
    if fileobj is None:
        return chat_view(session), "No file provided."

    try:
        fpath = fileobj if isinstance(fileobj, str) else fileobj.name
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return chat_view(session), f"❌ Failed to read JSON: {e}"

    sys_prompt = str(data.get("system_prompt", ""))
    hist = data.get("history", [])
    if not isinstance(hist, list):
        return chat_view(session), "❌ Invalid JSON: 'history' must be a list."
    cleaned: List[Dict[str, Any]] = []
    missing_media = 0
    for m in hist:
        if isinstance(m, dict) and "role" in m and "content" in m:
            msg: Dict[str, Any] = {"role": str(m["role"]), "content": str(m["content"])}
            media = msg_media(m)
            # Drop attachment paths that no longer exist: a dead link would abort
            # the load when gr.Chatbot tries to resolve it.
            kept = {k: [p for p in v if os.path.exists(p)] for k, v in media.items()}
            missing_media += sum(len(v) for v in media.values()) - sum(len(v) for v in kept.values())
            if any(kept.values()):
                msg["media"] = {k: v for k, v in kept.items() if v}
            cleaned.append(msg)
    # Mutate in place, never rebind: keeps the gr.State object stable.
    session.system_prompt = sys_prompt
    session.history[:] = cleaned
    note = f"✅ Loaded chat with {len(cleaned)} messages; system prompt updated."
    if missing_media:
        note += f"  \n⚠️ {missing_media} attachment(s) no longer exist on disk and were dropped."
    return chat_view(session), note


def clear_history(session: Session) -> List[Dict[str, Any]]:
    session.history.clear()
    session.pending_note = None
    return chat_view(session)
