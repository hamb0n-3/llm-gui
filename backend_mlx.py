from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

import deps
from backend_base import Backend, Prepared, budget_thinking, closing_stream, with_max_tokens
from config import RANGE_N_CTX, MLX_N_CTX_ESTIMATE, clamp_ctx
from modelscan import (
    looks_like_gguf, read_model_config, config_is_multimodal,
    config_model_type, mlx_lm_supports_model_type, resolve_local_model,
)
from session import Session, build_messages, augment_messages_with_context
from tokenization import (
    apply_mlx_chat_template, count_mlx_tokens, token_seq_len, infer_mlx_context_window,
)


class MlxBackend(Backend):
    id = ""  # set from config in backends.py to avoid an import cycle
    is_vision = False

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.ctx_detected: Optional[int] = None

    # ---- lifecycle -------------------------------------------------------
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self, params: Any) -> tuple[str, Optional[int]]:
        # Error paths return None for the ctx slider so a failed load doesn't
        # clamp it (handler maps None -> gr.update()).
        p = str(params.model_id_or_path or "").strip()
        if looks_like_gguf(p):
            return (
                "❌ That value looks like a **GGUF** model name/path.\n"
                "Please switch the backend to **GGUF (llama.cpp)** and load it via the GGUF tab instead.\n"
                "MLX expects a Hugging Face Transformers-format repo or a local directory containing "
                "`config.json`, `tokenizer.json`, and weights.",
                None,
            )

        if not deps._MLX_AVAILABLE:
            return "❌ MLX not available. Install with: pip install mlx-lm", None

        # Redirect only when mlx_lm has no implementation; a vision_config alone
        # is fine since mlx_lm runs several VL types' language half text-only.
        _cfg = read_model_config(p)
        if config_is_multimodal(_cfg) and mlx_lm_supports_model_type(config_model_type(_cfg)) is False:
            return (
                f"❌ That model is **multimodal** (`{config_model_type(_cfg)}`) and "
                "mlx_lm has no implementation for it.\n"
                "Switch the backend to **MLX-VLM (vision)** and load it from the "
                "MLX-VLM tab instead.",
                None,
            )

        # Cached repo id -> snapshot path so mlx_lm skips the Hub.
        local_p = resolve_local_model(p)

        try:
            model, tokenizer = deps.mlx_load(local_p)
        except Exception as e:
            return f"❌ Failed to load MLX model: {e}", None

        self.model = model
        self.tokenizer = tokenizer

        inferred = infer_mlx_context_window(tokenizer, model)
        self.ctx_detected = inferred

        # Never exceed the slider's max.
        ctx_value = inferred if inferred is not None else params.n_ctx_estimate
        ctx_value = clamp_ctx(ctx_value, RANGE_N_CTX, params.n_ctx_estimate or MLX_N_CTX_ESTIMATE)
        status = (
            f"✅ MLX model loaded: **{params.model_id_or_path}**  \n"
            f"{'Loaded from the local Hugging Face cache (no Hub request).  ' if local_p != p else ''}\n"
            f"Detected/estimated context: **{ctx_value}** "
            f"{'(detected)' if inferred is not None else '(using UI estimate)'}"
        )
        return status, (int(ctx_value) if ctx_value else None)

    def unload(self) -> str:
        self.model = None
        self.tokenizer = None
        self.ctx_detected = None
        return "🧹 Unloaded MLX model."

    # ---- budgeting -------------------------------------------------------
    def context_window(self, params: Any) -> int:
        est = params.n_ctx_estimate or MLX_N_CTX_ESTIMATE
        if self.ctx_detected:
            return clamp_ctx(self.ctx_detected, RANGE_N_CTX, est)
        return clamp_ctx(est, RANGE_N_CTX, MLX_N_CTX_ESTIMATE)

    def count_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        if not self.is_loaded():
            return self.fallback_count(messages)
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    toks = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                except TypeError:
                    toks = self.tokenizer.apply_chat_template(messages, tokenize=True)
                n = token_seq_len(toks)
                if n is not None:
                    return n
            prompt_text = apply_mlx_chat_template(self.tokenizer, messages, add_generation_prompt=True)
            return count_mlx_tokens(self.tokenizer, prompt_text)
        except Exception:
            return self.fallback_count(messages)

    # ---- generation ------------------------------------------------------
    def prepare(self, session: Session, params: Any, web_ctx: str) -> Prepared:
        messages = augment_messages_with_context(build_messages(session, ""), web_ctx)
        prompt = apply_mlx_chat_template(
            self.tokenizer, messages, add_generation_prompt=True,
            enable_thinking=bool(params.enable_thinking),
        )
        return Prepared(prompt=prompt)

    def generate_stream(self, session: Session, params: Any, prepared: Prepared) -> Generator[str, None, None]:
        prompt = prepared.prompt or ""
        # The resume re-prefills, so it must generate against what is left of
        # max_tokens: the shrink step already sized that to the context window.
        return budget_thinking(
            session,
            self._stream(session, params, prompt),
            int(params.thinking_budget),
            lambda head, spent: self._stream(
                session, with_max_tokens(params, int(params.max_tokens) - spent),
                prompt + head, head,
            ),
        )

    def _stream(self, session: Session, params: Any, prompt: str, prefix: str = "") -> Generator[str, None, None]:
        # mlx-lm >= 0.19 takes sampling via sampler/logits_processors; older
        # releases took temp/top_p/... as kwargs. Falls back to non-streaming.

        def _chunk_text(resp: Any) -> str:
            # GenerationResponse (modern) or plain str (legacy); both deltas.
            if hasattr(resp, "text"):
                return getattr(resp, "text") or ""
            if isinstance(resp, str):
                return resp
            return str(resp)

        # kwargs attempts, most capable first; the bare one is a last resort
        attempts: List[Dict[str, Any]] = []
        if deps.mlx_make_sampler is not None:
            kwargs: Dict[str, Any] = {"max_tokens": int(params.max_tokens)}
            try:
                kwargs["sampler"] = deps.mlx_make_sampler(
                    temp=float(params.temperature),
                    top_p=float(params.top_p),
                    top_k=int(params.top_k),
                )
            except TypeError:
                kwargs["sampler"] = deps.mlx_make_sampler(
                    temp=float(params.temperature), top_p=float(params.top_p)
                )
            if deps.mlx_make_logits_processors is not None and float(params.repeat_penalty) not in (0.0, 1.0):
                kwargs["logits_processors"] = deps.mlx_make_logits_processors(
                    repetition_penalty=float(params.repeat_penalty)
                )
            attempts.append(kwargs)
        else:
            # Legacy mlx-lm (< 0.19): sampling passed directly to generate_step.
            attempts.append({
                "max_tokens": int(params.max_tokens),
                "temp": float(params.temperature),
                "top_p": float(params.top_p),
                "repetition_penalty": float(params.repeat_penalty),
            })
        attempts.append({"max_tokens": int(params.max_tokens)})

        last_err: Optional[Exception] = None

        def _fallback_warning(kws: Dict[str, Any]) -> str:
            # Reaching the bare fallback means the sampling settings were rejected.
            if len(attempts) > 1 and kws is attempts[-1]:
                return ("⚠️ *This mlx-lm version rejected the sampling settings "
                        "(temperature/top-p/…); generated with library defaults.*\n\n")
            return ""

        # Streaming path
        if deps.mlx_stream_generate is not None:
            for kws in attempts:
                warn = _fallback_warning(kws)
                text = ""
                try:
                    stream = deps.mlx_stream_generate(self.model, self.tokenizer, prompt, **kws)
                    with closing_stream(stream):
                        for resp in stream:
                            if session.stop_requested:
                                return
                            chunk = _chunk_text(resp)
                            if chunk:
                                text += chunk
                                yield prefix + warn + text
                    return
                except TypeError as e:
                    last_err = e
                    if text:
                        # Failed after output: report rather than regenerate.
                        yield prefix + warn + text + f"\n\n❌ MLX generation error: {e}"
                        return
                    continue  # signature mismatch before output: try simpler kwargs
                except Exception as e:
                    last_err = e
                    if text:
                        yield prefix + warn + text + f"\n\n❌ MLX generation error: {e}"
                        return
                    break  # runtime failure: fall through to non-streaming

        # Non-streaming fallback
        for kws in attempts:
            try:
                out = deps.mlx_generate(self.model, self.tokenizer, prompt=prompt, **kws)
                yield prefix + _fallback_warning(kws) + str(out)
                return
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                break

        yield prefix + f"❌ MLX generation error: {last_err}"
