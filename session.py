from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import DEFAULT_BACKEND, DEFAULT_SYSTEM_PROMPT, MEDIA_KINDS


@dataclass
class Session:
    backend: str = DEFAULT_BACKEND
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history: List[Dict[str, Any]] = field(default_factory=list)
    pending_note: Optional[str] = None  # chat-view-only status bubble, never in prompt
    stop_requested: bool = False  # plain bool, not Event, to stay deepcopy-safe


def media_is_readable(path: str) -> bool:
    # Gradio uploads live in a temp dir cleared on restart, so an old path can
    # go stale. Remote URLs pass as-is: mlx-vlm fetches them itself.
    s = str(path)
    if s.startswith(("http://", "https://", "data:")):
        return True
    try:
        return os.path.exists(s)
    except Exception:
        return False


def msg_media(m: Dict[str, Any]) -> Dict[str, List[str]]:
    raw = m.get("media") if isinstance(m, dict) else None
    out: Dict[str, List[str]] = {k: [] for k in MEDIA_KINDS}
    if not isinstance(raw, dict):
        return out
    for kind in MEDIA_KINDS:
        vals = raw.get(kind) or []
        if isinstance(vals, str):
            vals = [vals]
        out[kind] = [str(v) for v in vals if v]
    return out


def has_media(m: Dict[str, Any]) -> bool:
    return any(msg_media(m)[k] for k in MEDIA_KINDS)


def chat_view(session: Session) -> List[Dict[str, Any]]:
    # Attachments become their own {"path": ...} messages ahead of the text.
    view: List[Dict[str, Any]] = []
    for m in session.history:
        role = str(m.get("role", "user"))
        media = msg_media(m)
        for kind in MEDIA_KINDS:
            for p in media[kind]:
                view.append({"role": role, "content": {"path": p}})
        content = m.get("content", "")
        if content or not has_media(m):
            view.append({"role": role, "content": str(content)})
    if session.pending_note:
        view.append({"role": "assistant", "content": session.pending_note})
    return view


def build_messages(session: Session, user_input: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if session.system_prompt:
        msgs.append({"role": "system", "content": session.system_prompt})
    msgs.extend(
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in session.history
    )
    if user_input.strip():
        msgs.append({"role": "user", "content": user_input})
    return msgs


def messages_to_prompt_text(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts) + "\nassistant:"


def augment_messages_with_context(messages: List[Dict[str, Any]], ctx: str) -> List[Dict[str, Any]]:
    # Handles both string content and mlx-vlm content lists (text in a
    # {"type": "text"} item).
    if not ctx:
        return messages
    preamble = (
        "Use the retrieved context below when it is relevant to the question; "
        "cite the bracketed source numbers you rely on.\n"
        f"<retrieved_context>\n{ctx}\n</retrieved_context>\n\n"
    )
    msgs = [dict(m) for m in messages]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") != "user":
            continue
        content = msgs[i].get("content", "")
        if isinstance(content, list):
            items = [dict(it) if isinstance(it, dict) else it for it in content]
            for it in items:
                if isinstance(it, dict) and it.get("type") == "text":
                    it["text"] = preamble + str(it.get("text", ""))
                    break
            else:
                items.append({"type": "text", "text": preamble})
            msgs[i]["content"] = items
        else:
            msgs[i]["content"] = preamble + str(content)
        break
    return msgs
