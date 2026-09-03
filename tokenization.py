from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def apply_mlx_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    enable_thinking: Optional[bool] = None,
) -> str:
    # transformers 5 raises ValueError when the tokenizer has no template; gate
    # like mlx_lm.generate does (has_chat_template also covers callable templates).
    _has_tmpl = getattr(tokenizer, "has_chat_template", None)
    if _has_tmpl is None:
        _has_tmpl = bool(getattr(tokenizer, "chat_template", None))
    # Unknown template vars are ignored by Jinja, so this is a no-op on models
    # that have no reasoning mode.
    extra = {} if enable_thinking is None else {"enable_thinking": bool(enable_thinking)}
    if hasattr(tokenizer, "apply_chat_template") and _has_tmpl:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **extra,
            )
        except TypeError:
            # Older tokenizers may not support add_generation_prompt
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False)
            except (TypeError, ValueError):
                pass
        except ValueError:
            pass  # no usable template: fall through to naive format
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"[{role.upper()}]: {content}")
    if add_generation_prompt:
        parts.append("[ASSISTANT]:")
    return "\n".join(parts)


def count_mlx_tokens(tokenizer: Any, text_or_messages: Any) -> int:
    try:
        if isinstance(text_or_messages, list):
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    toks = tokenizer.apply_chat_template(
                        text_or_messages, tokenize=True, add_generation_prompt=False
                    )
                except TypeError:
                    toks = tokenizer.apply_chat_template(text_or_messages, tokenize=True)
                n = token_seq_len(toks)
                if n is not None:
                    return n
            text = apply_mlx_chat_template(tokenizer, text_or_messages, add_generation_prompt=False)
        else:
            text = str(text_or_messages)

        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            return len(ids)
        if hasattr(tokenizer, "__call__"):
            out = tokenizer(text)
            if isinstance(out, dict) and "input_ids" in out:
                return len(out["input_ids"])
        return max(1, len(text) // 4)  # rough char estimate, rarely hit
    except Exception:
        return max(1, len(str(text_or_messages)) // 4)


def token_seq_len(toks: Any) -> Optional[int]:
    # transformers >= 5 returns a BatchEncoding (len = key count, not tokens);
    # mlx_lm returns a plain list. Both shapes must be handled.
    if toks is None:
        return None
    if hasattr(toks, "keys"):  # BatchEncoding / dict
        try:
            ids = toks["input_ids"]
        except Exception:
            return None
        if ids and isinstance(ids[0], (list, tuple)):  # batched: one seq per input
            ids = ids[0]
        return len(ids)
    if hasattr(toks, "__len__"):
        seq = toks
        if len(seq) and isinstance(seq[0], (list, tuple)):
            seq = seq[0]
        return len(seq)
    try:
        return int(toks)
    except Exception:
        return None


def count_text_tokens_with_tokenizer(tokenizer: Any, messages: List[Dict[str, str]]) -> int:
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                toks = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            except TypeError:
                toks = tokenizer.apply_chat_template(messages, tokenize=True)
            n = token_seq_len(toks)
            if n is not None:
                return n
    except Exception:
        pass
    text = apply_mlx_chat_template(tokenizer, messages, add_generation_prompt=True)
    return count_mlx_tokens(tokenizer, text)


def infer_mlx_context_window(tokenizer: Any, model: Any) -> Optional[int]:
    candidates: List[int] = []

    def _get(obj: Any, path: Tuple[str, ...]) -> Optional[int]:
        try:
            for name in path:
                obj = getattr(obj, name) if not isinstance(obj, dict) else obj[name]
            if isinstance(obj, int):
                return obj
        except Exception:
            return None
        return None

    for attr_path in [
        ("model_max_length",),
        ("init_kwargs", "model_max_length"),
        ("init_kwargs", "max_position_embeddings"),
    ]:
        val = _get(tokenizer, attr_path)
        if isinstance(val, int) and 0 < val < 10_000_000:
            candidates.append(int(val))

    try:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            mpe = getattr(cfg, "max_position_embeddings", None)
            if isinstance(mpe, int) and 0 < mpe < 10_000_000:
                candidates.append(int(mpe))
    except Exception:
        pass

    if candidates:
        return max(1, min(candidates))
    return None


def infer_vlm_context_from_config(config: Any) -> Optional[int]:
    try:
        cfg = config if isinstance(config, dict) else getattr(config, "__dict__", {}) or {}
    except Exception:
        return None
    for holder in (cfg, cfg.get("text_config") or {}, cfg.get("language_config") or {}):
        if not isinstance(holder, dict):
            continue
        val = holder.get("max_position_embeddings")
        if isinstance(val, int) and 0 < val < 10_000_000:
            return int(val)
    return None


def count_llamacpp_tokens(llm: Any, text: str) -> int:
    try:
        toks = llm.tokenize(text.encode("utf-8"))  # llama.cpp tokenizes bytes
        return len(toks)
    except Exception:
        return max(1, len(text) // 4)
