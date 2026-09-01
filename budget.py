from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend_base import Backend
from session import Session, build_messages


def token_meter_markdown(
    backend: Backend,
    session: Session,
    params: Any,
    pending_user_input: str = "",
    pending_media: Optional[Dict[str, List[str]]] = None,
) -> str:
    ctx = backend.context_window(params)
    messages = build_messages(session, pending_user_input)
    prompt_tokens = backend.count_prompt_tokens(messages)
    max_gen = int(params.max_tokens)
    media_est, media_counts = backend.media_reserve_with_counts(session, params, pending_media)

    remaining = ctx - prompt_tokens - max_gen - media_est
    overflow = remaining < 0

    md = [
        "### Token Meter",
        f"- **Context window**: `{ctx}`",
        f"- **Prompt tokens (system + history + input + template)**: `{prompt_tokens}`",
    ]
    if backend.is_vision:
        attached = ", ".join(f"{n}×{k}" for k, n in media_counts.items() if n) or "none"
        md.append(
            f"- **Media attached (next send)**: `{attached}` — "
            f"reserving ~`{media_est}` tokens *(estimate; real cost is model- "
            f"and resolution-dependent)*"
        )
    md += [
        f"- **Reserved for generation (`max_tokens`)**: `{max_gen}`",
        f"- **Estimated remaining**: `{remaining}` {'⚠️ **Overflow**' if overflow else ''}",
    ]
    return "\n".join(md)


def trim_history_to_fit_context(backend: Backend, session: Session, params: Any) -> None:
    ctx = backend.context_window(params)
    max_gen = int(params.max_tokens)

    def current_prompt_tokens() -> int:
        return backend.count_prompt_tokens(build_messages(session, ""))

    def media_reserve() -> int:
        # Recomputed per iteration: trimming turns can drop resent media on the VLM.
        return backend.media_reserve(session, params)

    # Never drop the newest message; shrink_last_user_message_... truncates it.
    guard = 0
    while current_prompt_tokens() + max_gen + media_reserve() > ctx and len(session.history) > 1 and guard < 512:
        remove = 2 if len(session.history) >= 3 else 1
        del session.history[:remove]
        guard += 1


def shrink_last_user_message_to_fit_and_adjust_max_tokens(
    backend: Backend, session: Session, params: Any, extra_reserve_tokens: int = 0,
) -> Tuple[int, int]:
    ctx = backend.context_window(params)
    max_gen = int(params.max_tokens)
    reserve = max(0, int(extra_reserve_tokens))

    # Copies: the binary search must not mutate real history mid-search (the
    # token-meter handler can read it concurrently).
    messages = [dict(m) for m in build_messages(session, "")]
    prompt_tokens = backend.count_prompt_tokens(messages)

    allowed = max(1, ctx - 1 - reserve)  # leave room for >=1 generation token
    if prompt_tokens <= ctx - max_gen - reserve:
        adjusted_max = max(1, min(max_gen, ctx - prompt_tokens - reserve))
        return prompt_tokens, adjusted_max

    last_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_idx = i
            break
    if last_idx is None:
        adjusted_max = max(1, min(max_gen, ctx - prompt_tokens - reserve))
        return prompt_tokens, adjusted_max

    original = messages[last_idx].get("content", "")
    if not isinstance(original, str) or not original:
        adjusted_max = max(1, min(max_gen, ctx - prompt_tokens - reserve))
        return prompt_tokens, adjusted_max

    # Binary search on char length; the suffix is measured so it can't push back over budget.
    suffix = " … [truncated]"
    lo, hi = 0, len(original)
    best_len = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        messages[last_idx]["content"] = original[:mid] + (suffix if mid < len(original) else "")
        tok = backend.count_prompt_tokens(messages)
        if tok <= allowed:
            best_len = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best_len <= 0:
        # Even empty doesn't fit; leave history intact and let the caller refuse (max==0).
        return prompt_tokens, 0

    final_content = original[:best_len] + (suffix if 0 < best_len < len(original) else "")
    messages[last_idx]["content"] = final_content

    # Apply to session.history so the UI reflects the change.
    if best_len < len(original) and session.history:
        for j in range(len(session.history) - 1, -1, -1):
            if session.history[j].get("role") == "user":
                session.history[j]["content"] = final_content
                break

    messages = build_messages(session, "")
    prompt_tokens = backend.count_prompt_tokens(messages)
    adjusted_max = max(1, min(max_gen, ctx - prompt_tokens - reserve))
    return prompt_tokens, adjusted_max
