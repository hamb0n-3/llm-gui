from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

import deps
from backend_base import Backend, Prepared, closing_stream
from config import RANGE_N_CTX, VLM_N_CTX_ESTIMATE, MEDIA_KINDS, clamp_ctx
from media import media_reserve_tokens
from modelscan import looks_like_gguf, read_model_config, config_is_multimodal, resolve_local_model
from session import Session, msg_media, media_is_readable, augment_messages_with_context
from tokenization import (
    count_text_tokens_with_tokenizer, infer_mlx_context_window, infer_vlm_context_from_config,
)


def _prune_video_items(messages: List[Dict[str, Any]], keep_index: int) -> List[Dict[str, Any]]:
    # mlx-vlm's video= kwarg emits a video block on every message; keep only one.
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        if i == keep_index or not isinstance(m, dict):
            out.append(m)
            continue
        content = m.get("content")
        if isinstance(content, list):
            kept = [it for it in content
                    if not (isinstance(it, dict) and it.get("type") == "video")]
            m = {**m, "content": kept}
        out.append(m)
    return out


class VlmBackend(Backend):
    id = ""  # set from config in backends.py
    is_vision = True

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.config: Any = None
        self.ctx_detected: Optional[int] = None

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def _tokenizer(self) -> Any:
        proc = self.processor
        return getattr(proc, "tokenizer", None) or proc

    def load(self, params: Any) -> tuple[str, Optional[int]]:
        # Error paths return None for the ctx slider (handler maps to gr.update()).
        p = str(params.model_id_or_path or "").strip()
        if not p:
            return "❌ Enter an MLX-VLM model id or path.", None
        if looks_like_gguf(p):
            return (
                "❌ That value looks like a **GGUF** model name/path.\n"
                "mlx-vlm expects a Hugging Face Transformers-format repo or a local "
                "directory with `config.json`, a processor, and weights.",
                None,
            )
        if not deps._VLM_AVAILABLE:
            return f"❌ mlx-vlm not available (`{deps._VLM_IMPORT_ERROR}`). Install with: pip install mlx-vlm", None

        cfg_probe = read_model_config(p)
        if cfg_probe is not None and not config_is_multimodal(cfg_probe):
            return (
                "❌ That model looks **text-only** (no `vision_config`/`audio_config` "
                "in `config.json`).\nSwitch the backend to **MLX (mlx_lm)** to run it.",
                None,
            )

        local_p = resolve_local_model(p)  # cached repo id -> snapshot path, skips the Hub

        try:
            model, processor = deps.vlm_load(local_p)
        except Exception as e:
            return f"❌ Failed to load MLX-VLM model: {e}", None

        try:
            config = deps.vlm_load_config(local_p)
        except Exception:
            config = getattr(model, "config", None) or cfg_probe or {}

        self.model = model
        self.processor = processor
        self.config = config

        tok = self._tokenizer()
        inferred = infer_mlx_context_window(tok, model)
        if inferred is None:
            inferred = infer_vlm_context_from_config(config)
        self.ctx_detected = inferred

        ctx_value = inferred if inferred is not None else params.n_ctx_estimate
        ctx_value = clamp_ctx(ctx_value, RANGE_N_CTX, params.n_ctx_estimate or VLM_N_CTX_ESTIMATE)

        mtype = ""
        try:
            c = config if isinstance(config, dict) else getattr(config, "__dict__", {})
            mtype = str(c.get("model_type", "") or "")
        except Exception:
            pass
        status = (
            f"✅ MLX-VLM model loaded: **{p}**  \n"
            f"{'Loaded from the local Hugging Face cache (no Hub request).  ' if local_p != p else ''}\n"
            f"{('Model type: `' + mtype + '`  ') if mtype else ''}\n"
            f"Detected/estimated context: **{ctx_value}** "
            f"{'(detected)' if inferred is not None else '(using UI estimate)'}"
        )
        return status, (int(ctx_value) if ctx_value else None)

    def unload(self) -> str:
        self.model = None
        self.processor = None
        self.config = None
        self.ctx_detected = None
        return "🧹 Unloaded MLX-VLM model."

    def context_window(self, params: Any) -> int:
        est = params.n_ctx_estimate or VLM_N_CTX_ESTIMATE
        if self.ctx_detected:
            return clamp_ctx(self.ctx_detected, RANGE_N_CTX, est)
        return clamp_ctx(est, RANGE_N_CTX, VLM_N_CTX_ESTIMATE)

    def count_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        if not self.is_loaded():
            return self.fallback_count(messages)
        try:
            return count_text_tokens_with_tokenizer(self._tokenizer(), messages)
        except Exception:
            return self.fallback_count(messages)

    def media_reserve(self, session: Session, params: Any) -> int:
        try:
            _, imgs, vids, auds, _ = self._plan(session, params)
        except Exception:
            return 0
        return media_reserve_tokens(imgs, vids, auds)

    def media_reserve_with_counts(
        self, session: Session, params: Any, pending_media: Optional[Dict[str, List[str]]] = None,
    ) -> tuple[int, Dict[str, int]]:
        counts = {k: 0 for k in MEDIA_KINDS}
        # Model the new user turn even when empty: current-turn-only policy means
        # an empty composer won't resend the previous turn's media.
        pending = {k: list((pending_media or {}).get(k) or []) for k in MEDIA_KINDS}
        history = list(session.history) + [{
            "role": "user",
            "content": "",
            "media": {k: v for k, v in pending.items() if v},
        }]
        try:
            _, images, videos, audios, _dropped = self._plan(session, params, history)
        except Exception:
            return 0, counts
        counts["image"], counts["video"], counts["audio"] = len(images), len(videos), len(audios)
        return media_reserve_tokens(images, videos, audios), counts

    def media_overflow_message(
        self, session: Session, params: Any, ctx: int, media_reserve: int, web_reserve: int,
    ) -> Optional[str]:
        # Refuse up front when attachments alone can't fit; otherwise the shrink
        # step silently truncates the user's text to "" and clamps generation.
        if media_reserve + web_reserve <= max(0, ctx - 256):
            return None
        try:
            _, images, videos, audios, _ = self._plan(session, params)
        except Exception:
            images, videos, audios = [], [], []
        attached = ", ".join(
            f"{n} {k}{'s' if n != 1 else ''}"
            for k, n in (("image", len(images)), ("video", len(videos)), ("audio", len(audios)))
            if n
        ) or "the attachments"
        return (
            f"❌ Too much media for this context: {attached} is estimated at "
            f"~{media_reserve} tokens against a {ctx}-token window.\n"
            "Send fewer or shorter attachments, lower the resend cap, or raise "
            "the n_ctx estimate if the model supports a larger window."
        )

    def _plan(
        self, session: Session, params: Any, history: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[Dict[str, Any]], List[str], List[str], List[str], List[str]]:
        # Images get per-turn {"type": "image"} markers; the flattened images list
        # stays in marker order. Video/audio have no marker, so current turn only.
        hist = list(session.history if history is None else history)
        dropped: List[str] = []

        def _keep(paths: List[str]) -> List[str]:
            out = []
            for p in paths:
                (out if media_is_readable(p) else dropped).append(p)
            return out

        last_user = -1
        for i in range(len(hist) - 1, -1, -1):
            if hist[i].get("role") == "user":
                last_user = i
                break

        per_msg: List[List[str]] = [[] for _ in hist]
        if last_user >= 0:
            per_msg[last_user] = _keep(msg_media(hist[last_user])["image"])
        if params.resend_history_media:
            budget = max(0, int(params.media_history_cap) - len(per_msg[last_user] if last_user >= 0 else []))
            for i in range(len(hist) - 1, -1, -1):
                if i == last_user or hist[i].get("role") != "user":
                    continue
                if budget <= 0:
                    break
                imgs = _keep(msg_media(hist[i])["image"])
                take = imgs[-budget:] if budget < len(imgs) else list(imgs)
                per_msg[i] = take
                budget -= len(take)

        messages: List[Dict[str, Any]] = []
        if session.system_prompt:
            messages.append({"role": "system", "content": session.system_prompt})

        images: List[str] = []
        for i, m in enumerate(hist):
            role = str(m.get("role", "user"))
            text = str(m.get("content", ""))
            imgs = per_msg[i] if role == "user" else []
            if imgs:
                images.extend(imgs)
                content: Any = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": text}]
            else:
                content = text
            messages.append({"role": role, "content": content})

        cur = msg_media(hist[last_user]) if last_user >= 0 else {k: [] for k in MEDIA_KINDS}
        return messages, images, _keep(cur["video"]), _keep(cur["audio"]), dropped

    def _format_prompt(self, messages: List[Dict[str, Any]], n_images: int, n_audios: int,
                       params: Any, videos: Optional[List[str]] = None) -> Any:
        # Video has no per-turn marker: render the message list, prune video items
        # to the current turn, then re-render.
        videos = list(videos or [])
        base: Dict[str, Any] = dict(
            add_generation_prompt=True,
            num_images=int(n_images),
            num_audios=int(n_audios),
        )
        if params.enable_thinking:
            thinking_variants = [{"enable_thinking": True}, {}]
        else:
            thinking_variants = [{}]

        last_err: Optional[Exception] = None

        if videos and deps.vlm_get_chat_template is not None:
            last_user = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user = i
                    break
            for extra in thinking_variants:
                try:
                    built = deps.vlm_apply_chat_template(
                        self.processor, self.config, messages,
                        **base, **extra, video=videos, return_messages=True,
                    )
                    if isinstance(built, list):
                        built = _prune_video_items(built, last_user)
                        return deps.vlm_get_chat_template(self.processor, built, True, **extra)
                except Exception as e:
                    last_err = e
                    continue

        for extra in thinking_variants:
            try:
                kws = {**base, **extra}
                if videos:
                    kws["video"] = videos
                return deps.vlm_apply_chat_template(
                    self.processor, self.config, messages, **kws
                )
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"chat template failed: {last_err}")

    def prepare(self, session: Session, params: Any, web_ctx: str) -> Prepared:
        # Re-plan: runs after the shrink step, so the last user message may be truncated.
        messages, images, videos, audios, dropped = self._plan(session, params)
        messages = augment_messages_with_context(messages, web_ctx)
        prompt = self._format_prompt(messages, len(images), len(audios), params, videos)
        return Prepared(prompt=prompt, images=images, videos=videos, audios=audios, dropped=dropped)

    def error_message(self, e: Exception) -> str:
        msg = str(e)
        low = msg.lower()
        if "out of memory" in low or "insufficient memory" in low:
            return ("❌ MLX-VLM ran out of memory. Try a smaller/more quantized model, "
                    "fewer or smaller images, or a lower max-tokens.")
        if "audio" in low and ("not support" in low or "no attribute" in low):
            return f"❌ This model has no audio tower — send images instead. ({e})"
        if "video" in low and ("not support" in low or "no attribute" in low):
            return f"❌ This model has no video support — send images instead. ({e})"
        return f"❌ MLX-VLM generation error: {e}"

    def generate_stream(self, session: Session, params: Any, prepared: Prepared) -> Generator[str, None, None]:
        # Yields cumulative text.
        prompt = prepared.prompt or ""
        images, videos, audios = prepared.images, prepared.videos, prepared.audios
        media = {
            "image": list(images) or None,
            "video": list(videos) or None,
            "audio": list(audios) or None,
        }

        # Prefer a prebuilt sampler: raw sampling kwargs leak into processor() and
        # make transformers warn on every generation.
        sampling: Dict[str, Any] = {"max_tokens": int(params.max_tokens)}
        raw_sampling: Dict[str, Any] = {
            "max_tokens": int(params.max_tokens),
            "temperature": float(params.temperature),
            "top_p": float(params.top_p),
            "top_k": int(params.top_k),
        }
        if float(params.repeat_penalty) not in (0.0, 1.0):
            raw_sampling["repetition_penalty"] = float(params.repeat_penalty)

        if deps.vlm_make_sampler is not None:
            try:
                sampling["sampler"] = deps.vlm_make_sampler(
                    temp=float(params.temperature),
                    top_p=float(params.top_p),
                    top_k=int(params.top_k),
                )
            except TypeError:
                sampling["sampler"] = deps.vlm_make_sampler(
                    temp=float(params.temperature), top_p=float(params.top_p)
                )
            if deps.vlm_make_logits_processors is not None and float(params.repeat_penalty) not in (0.0, 1.0):
                sampling["logits_processors"] = deps.vlm_make_logits_processors(
                    repetition_penalty=float(params.repeat_penalty)
                )
        else:
            sampling = dict(raw_sampling)

        thinking: Dict[str, Any] = {}
        if params.enable_thinking:
            thinking["enable_thinking"] = True
            if int(params.thinking_budget) > 0:
                thinking["thinking_budget"] = int(params.thinking_budget)

        attempts: List[Dict[str, Any]] = []
        if thinking:
            attempts.append({**sampling, **thinking})
        attempts.append(dict(sampling))
        if sampling is not raw_sampling and raw_sampling != sampling:
            attempts.append(dict(raw_sampling))  # older mlx-vlm without sample_utils
        attempts.append({"max_tokens": int(params.max_tokens)})

        def _fallback_warning(kws: Dict[str, Any]) -> str:
            if len(attempts) > 1 and kws is attempts[-1]:
                return ("⚠️ *This mlx-vlm version rejected the sampling settings "
                        "(temperature/top-p/…); generated with library defaults.*\n\n")
            return ""

        last_err: Optional[Exception] = None
        for kws in attempts:
            warn = _fallback_warning(kws)
            text = ""
            try:
                stream = deps.vlm_stream_generate(
                    self.model, self.processor, prompt, **media, **kws
                )
                with closing_stream(stream):
                    for resp in stream:
                        if session.stop_requested:
                            return
                        chunk = getattr(resp, "text", None)
                        if chunk is None:
                            chunk = resp if isinstance(resp, str) else ""
                        if chunk:
                            text += chunk
                            yield warn + text
                return
            except TypeError as e:
                last_err = e
                if text:
                    yield warn + text + f"\n\n❌ MLX-VLM generation error: {e}"
                    return
                continue  # signature mismatch before output: retry with simpler kwargs
            except Exception as e:
                last_err = e
                if text:
                    yield warn + text + f"\n\n❌ MLX-VLM generation error: {e}"
                    return
                break  # runtime failure: try the non-streaming path

        # Non-streaming fallback
        for kws in attempts:
            try:
                out = deps.vlm_generate(
                    self.model, self.processor, prompt, **media, verbose=False, **kws
                )
                yield _fallback_warning(kws) + str(getattr(out, "text", None) or out)
                return
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                break

        yield self.error_message(last_err)
