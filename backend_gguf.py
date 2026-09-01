from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import deps
from backend_base import Backend, Prepared, closing_stream
from config import (
    RANGE_GGUF_N_CTX, GGUF_CTX_FALLBACK, GGUF_CTX_MIN_FALLBACK, GGUF_MMPROJ_AUTODETECT,
    MEDIA_KINDS, clamp_ctx,
)
from media import media_reserve_tokens
from modelscan import find_mmproj_for
from session import (
    Session, build_messages, augment_messages_with_context, messages_to_prompt_text,
    media_is_readable, msg_media,
)
from tokenization import count_llamacpp_tokens


def _image_url(path: str) -> str:
    # Local file -> file:// URI: as_uri() escapes spaces/unicode a bare path wouldn't.
    s = str(path)
    if s.startswith(("http://", "https://", "data:", "file://")):
        return s
    return Path(s).expanduser().resolve().as_uri()


class GgufBackend(Backend):
    id = ""  # set from config in backends.py
    is_vision = False

    def __init__(self) -> None:
        self.llm: Any = None
        self.n_ctx: Optional[int] = None
        # A loaded projector makes this instance vision-capable (composer, meter,
        # basic-view routing all key off is_vision).
        self.chat_handler: Any = None
        self.mmproj_path: str = ""
        self.is_vision: bool = False

    # ---- lifecycle -------------------------------------------------------
    def is_loaded(self) -> bool:
        return self.llm is not None

    def load(self, params: Any) -> tuple[str, Optional[int]]:
        # No ctx slider, so the second element is always None.
        try:
            if not os.path.isfile(params.model_path):
                return f"❌ GGUF file not found: {params.model_path}", None
        except Exception:
            pass

        if not deps._LLAMA_AVAILABLE:
            return "❌ llama.cpp not available. Install with: pip install llama-cpp-python", None

        # Build kwargs against the installed llama-cpp-python signature.
        try:
            sig = inspect.signature(deps.Llama)
            supported = set(sig.parameters.keys())
        except Exception:
            supported = set()

        # Vision projector: explicit path wins, else the one beside the model.
        mmproj = str(getattr(params, "mmproj_path", "") or "").strip()
        if not mmproj and GGUF_MMPROJ_AUTODETECT:
            mmproj = find_mmproj_for(params.model_path) or ""
        chat_handler = None
        vision_note = ""
        if mmproj:
            if not os.path.isfile(mmproj):
                vision_note = f"  \n⚠️ Vision projector not found: `{mmproj}` — loaded text-only."
                mmproj = ""
            else:
                try:
                    from llama_cpp.llama_chat_format import MTMDChatHandler

                    chat_handler = MTMDChatHandler(clip_model_path=mmproj, verbose=False)
                except Exception as e:
                    # A bad projector shouldn't cost the whole model.
                    vision_note = f"  \n⚠️ Vision projector failed to load ({e}) — loaded text-only."
                    chat_handler = None
                    mmproj = ""

        kwargs = {
            "model_path": params.model_path,
            "n_ctx": int(params.n_ctx),
            "n_gpu_layers": int(params.n_gpu_layers),
            # llama.cpp treats -1 as random; 0 would be a fixed seed.
            "seed": (int(params.seed) if int(params.seed or 0) != 0 else -1),
            "logits_all": False,
            "vocab_only": False,
            "use_mmap": True,
            "use_mlock": False,
            "embedding": False,
        }

        # Added only if the signature supports them.
        opt_map = {
            "n_threads": int(getattr(params, "n_threads", max(1, (os.cpu_count() or 8) - 2))),
            "n_batch": int(getattr(params, "n_batch", 128)),
            "n_ubatch": int(getattr(params, "n_ubatch", 32)),
            "f16_kv": bool(getattr(params, "f16_kv", True)),
            "flash_attn": (False if getattr(params, "flash_attn", None) is None else bool(getattr(params, "flash_attn"))),
        }
        for k, v in opt_map.items():
            if k in supported:
                kwargs[k] = v
        if chat_handler is not None and "chat_handler" in supported:
            kwargs["chat_handler"] = chat_handler

        # Progressive fallback for Metal OOM: the KV cache is the term that
        # usually blows the budget, so n_ctx steps down before anything else.
        attempts: list[dict] = [dict(kwargs)]
        for div in (2, 4, 8):
            smaller = max(GGUF_CTX_MIN_FALLBACK, int(kwargs["n_ctx"]) // div)
            if smaller < int(kwargs["n_ctx"]):
                a = dict(kwargs); a["n_ctx"] = smaller
                attempts.append(a)
        kwargs = dict(attempts[-1])  # later fallbacks build on the smallest ctx
        if kwargs.get("n_gpu_layers", 0) > 0:
            safer1 = dict(kwargs); safer1["n_gpu_layers"] = max(0, int(kwargs["n_gpu_layers"]) // 2)
            attempts.append(safer1)
            safer2 = dict(kwargs); safer2["n_gpu_layers"] = 0
            attempts.append(safer2)
        safer3 = dict(attempts[-1])  # stack on the most conservative attempt so far
        if "n_batch" in safer3: safer3["n_batch"] = max(16, int(safer3["n_batch"]) // 2)
        if "n_ubatch" in safer3: safer3["n_ubatch"] = max(8, int(safer3["n_ubatch"]) // 2)
        attempts.append(safer3)

        last_err = None
        for kws in attempts:
            try:
                llm = deps.Llama(**kws)
                self.llm = llm
                self.n_ctx = int(kws.get("n_ctx", params.n_ctx))
                self.chat_handler = kws.get("chat_handler")
                self.mmproj_path = mmproj if self.chat_handler is not None else ""
                self.is_vision = self.chat_handler is not None
                vision_line = (
                    f"  \n👁️ Vision enabled via `{Path(self.mmproj_path).name}` — image attachments accepted."
                    if self.is_vision else vision_note
                )
                return (
                    f"✅ Loaded GGUF model: `{params.model_path}`{vision_line}  \n"
                    f"• n_ctx={kws.get('n_ctx')} • n_gpu_layers={kws.get('n_gpu_layers')} "
                    f"• n_threads={kws.get('n_threads','?')} • n_batch={kws.get('n_batch','?')} • n_ubatch={kws.get('n_ubatch','?')} "
                    f"• f16_kv={kws.get('f16_kv','?')} • flash_attn={kws.get('flash_attn','?')}",
                    None,
                )
            except Exception as e:
                last_err = e
                msg = str(e)
                # Retry the next fallback only on Metal OOM; bail otherwise. A
                # KV cache too big for the wired limit reports "failed to
                # allocate", not an OOM string.
                if any(x in msg for x in ("kIOGPUCommandBufferCallbackErrorOutOfMemory", "ggml_metal_graph_compute", "Insufficient Memory", "out of memory", "failed to compute graph", "failed to allocate", "unable to allocate", "failed to create context")):
                    continue
                else:
                    break

        return f"❌ Failed to load GGUF model after fallbacks: {last_err}", None

    def unload(self) -> str:
        # close() frees the Metal context now; leaving it to __del__ can run at
        # interpreter shutdown, after ggml's device is gone (GGML_ASSERT abort).
        llm, self.llm = self.llm, None
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
        self.n_ctx = None
        self.chat_handler = None
        self.mmproj_path = ""
        self.is_vision = False
        return "🧹 Unloaded GGUF model."

    # ---- budgeting -------------------------------------------------------
    def context_window(self, params: Any) -> int:
        est = params.n_ctx or GGUF_CTX_FALLBACK
        if self.n_ctx:
            return clamp_ctx(self.n_ctx, RANGE_GGUF_N_CTX, est)
        return clamp_ctx(est, RANGE_GGUF_N_CTX, GGUF_CTX_FALLBACK)

    def count_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        if not self.is_loaded():
            return self.fallback_count(messages)
        try:
            prompt_text = None
            try:
                if hasattr(self.llm, "apply_chat_template"):
                    try:
                        prompt_text = self.llm.apply_chat_template(messages, add_generation_prompt=True)
                    except TypeError:
                        prompt_text = self.llm.apply_chat_template(messages)
            except Exception:
                prompt_text = None
            if not isinstance(prompt_text, str):
                prompt_text = messages_to_prompt_text(messages)
            return count_llamacpp_tokens(self.llm, prompt_text)
        except Exception:
            return self.fallback_count(messages)

    # ---- vision ----------------------------------------------------------
    def _current_images(self, session: Session) -> tuple[List[str], List[str]]:
        # Newest user turn only: llama.cpp re-encodes every image each turn, so
        # replaying the whole album would cost prefill on every send.
        if not self.is_vision:
            return [], []
        last = next((m for m in reversed(list(session.history)) if m.get("role") == "user"), None)
        if not last:
            return [], []
        keep: List[str] = []
        dropped: List[str] = []
        for p in msg_media(last)["image"]:
            (keep if media_is_readable(p) else dropped).append(p)
        return keep, dropped

    def _attach_images(
        self, messages: List[Dict[str, Any]], images: List[str],
    ) -> List[Dict[str, Any]]:
        idx = next((i for i in range(len(messages) - 1, -1, -1)
                    if messages[i].get("role") == "user"), None)
        if idx is None or not images:
            return messages
        out = [dict(m) for m in messages]
        text = str(out[idx].get("content") or "")
        parts: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _image_url(p)}} for p in images
        ]
        parts.append({"type": "text", "text": text})
        out[idx]["content"] = parts
        return out

    def media_reserve(self, session: Session, params: Any) -> int:
        imgs, _ = self._current_images(session)
        return media_reserve_tokens(imgs, [], [])

    def media_reserve_with_counts(
        self, session: Session, params: Any, pending_media: Optional[Dict[str, List[str]]] = None,
    ) -> tuple[int, Dict[str, int]]:
        if not self.is_vision:
            return 0, {}
        # Images are current-turn only, so only the composer's images get encoded.
        imgs = list((pending_media or {}).get("image") or [])
        counts = {k: 0 for k in MEDIA_KINDS}
        counts["image"] = len(imgs)
        return media_reserve_tokens(imgs, [], []), counts

    def media_overflow_message(
        self, session: Session, params: Any, ctx: int, media_reserve: int, web_reserve: int,
    ) -> Optional[str]:
        if media_reserve + web_reserve <= max(0, ctx - 256):
            return None
        imgs, _ = self._current_images(session)
        n = len(imgs)
        return (
            f"❌ Too much media for this context: {n} image{'s' if n != 1 else ''} is "
            f"estimated at ~{media_reserve} tokens against a {ctx}-token window.\n"
            "Send fewer or smaller images, or raise n_ctx and reload."
        )

    # ---- generation ------------------------------------------------------
    def prepare(self, session: Session, params: Any, web_ctx: str) -> Prepared:
        messages = augment_messages_with_context(build_messages(session, ""), web_ctx)
        images, dropped = self._current_images(session)
        if images:
            messages = self._attach_images(messages, images)
        return Prepared(messages=messages, images=images, dropped=dropped)

    def error_message(self, e: Exception) -> str:
        emsg = str(e)
        if any(x in emsg for x in ("kIOGPUCommandBufferCallbackErrorOutOfMemory", "ggml_metal_graph_compute", "Insufficient Memory", "failed to compute graph")):
            return "❌ Metal ran out of GPU memory during generation. Lower n_gpu_layers, n_batch / n_ubatch, or n_ctx; disabling flash_attn can also help."
        return f"❌ GGUF generation error: {e}"

    def generate_stream(self, session: Session, params: Any, prepared: Prepared) -> Generator[str, None, None]:
        # Chat completion, else plain completion. Falls back only if the chat API
        # failed before emitting anything; a mid-stream break reports the error.
        messages = prepared.messages or []
        text_accum = ""

        try:
            res = self.llm.create_chat_completion(
                messages=messages,
                stream=True,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                repeat_penalty=params.repeat_penalty,
            )
            with closing_stream(res):
                for chunk in res:
                    if session.stop_requested:
                        break
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        text_accum += delta
                        yield text_accum
            return
        except Exception as e:
            if text_accum:
                yield text_accum + "\n\n" + self.error_message(e)
                return
            # Chat API failed before any output: fall through to completion API.

        prompt = messages_to_prompt_text(messages)
        try:
            res = self.llm(
                prompt,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                repeat_penalty=params.repeat_penalty,
                stream=True,
            )
            with closing_stream(res):
                for t in res:
                    if session.stop_requested:
                        break
                    token_text = t.get("choices", [{}])[0].get("text", "")
                    if token_text:
                        text_accum += token_text
                        yield text_accum
        except Exception as e:
            if text_accum:
                yield text_accum + "\n\n" + self.error_message(e)
            else:
                yield self.error_message(e)
