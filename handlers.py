from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr

import deps
from backends import get_backend, loaded_backends
from budget import (
    token_meter_markdown,
    trim_history_to_fit_context,
    shrink_last_user_message_to_fit_and_adjust_max_tokens,
)
from config import (
    BACKEND_MLX, BACKEND_GGUF, BACKEND_VLM,
    BASIC_AUTOLOAD, BASIC_TEXT_BACKEND, BASIC_MEDIA_BACKEND,
    LOADING_NOTE, THINKING_NOTE, DEFAULT_MODELS_DIR, MEDIA_KINDS,
    MLX_MAX_TOKENS, MLX_N_CTX_ESTIMATE, GGUF_MAX_TOKENS, GGUF_N_CTX,
    is_vlm, is_mlx, safe_int,
)
from media import classify_media, inline_text_attachments
from mcp_client import MCP
from modelscan import scan_mlx_models, scan_gguf_models, scan_vlm_models
from params import MLXParams, GGUFParams, build_params, build_vlm_params, params_for, model_ref
from persistence import save_chat_json, load_chat_json
from session import Session, chat_view


def uses_mm_composer(backend: str, advanced: bool = True) -> bool:
    # The send path must use this too, or a visible box's content is discarded.
    if not advanced:
        return True
    be = get_backend(backend)
    return is_vlm(backend) or bool(be is not None and be.is_vision)


def _composer_updates(backend: str) -> Tuple[Any, Any]:
    v = uses_mm_composer(backend)
    return gr.update(visible=not v), gr.update(visible=v)


def _select_backend_updates(session: Session, backend: str) -> Tuple[Any, Any, Any]:
    session.backend = backend
    plain, mm = _composer_updates(backend)
    return gr.update(value=backend), plain, mm


def _no_backend_change() -> Tuple[Any, Any, Any]:
    return gr.update(), gr.update(), gr.update()


def _after_load(session: Session, loaded_backend: str, current_backend: str, status: str, ok: bool) -> Tuple[str, Tuple[Any, Any, Any]]:
    # Switch only when the selected backend has nothing loaded, so an explicit
    # user choice isn't hijacked.
    if not ok:
        return status, _no_backend_change()
    current = str(current_backend or "")
    if current == loaded_backend or current not in loaded_backends():
        return status, _select_backend_updates(session, loaded_backend)
    status += (
        f"  \nℹ️ Backend left on **{current}** (it has a model loaded). "
        f"Select **{loaded_backend}** in the Backend radio to chat with this model."
    )
    return status, _no_backend_change()


def _no_model_message(backend: str) -> str:
    if is_vlm(backend):
        base = "❌ No MLX-VLM model is loaded. Load one from the **MLX-VLM** tab."
    elif is_mlx(backend):
        base = "❌ No MLX model is loaded. Load one from the **MLX** tab."
    else:
        base = "❌ No GGUF model is loaded. Load one from the **GGUF** tab."
    others = [b for b in loaded_backends() if b != backend]
    if others:
        base += (
            f"\n\nA model is loaded for **{', '.join(others)}** — switch the "
            "Backend radio to it, or load a model for the selected backend."
        )
    return base


def _effective_backend(backend: str, advanced: bool, media: Dict[str, List[str]]) -> str:
    # Advanced view follows the radio; basic view (radio hidden) decides here.
    if advanced:
        return backend
    media = media or {}
    if not any(media.get(k) for k in MEDIA_KINDS):
        return BASIC_TEXT_BACKEND
    # Images stay on the text backend if it has a vision projector; video/audio
    # have no mmproj path so they go to the vision backend.
    only_images = not (media.get("video") or media.get("audio"))
    be = get_backend(BASIC_TEXT_BACKEND)
    if only_images and be is not None and be.is_vision:
        return BASIC_TEXT_BACKEND
    return BASIC_MEDIA_BACKEND


def _model_name_for(params: Any) -> str:
    raw = str(model_ref(params) or "").strip()
    return Path(raw).name if ("/" in raw or "\\" in raw) else (raw or "(none)")


def _autoload_model(backend, params) -> Optional[str]:
    try:
        status, _ = backend.load(params)
    except Exception as e:  # a broken model dir shouldn't kill the send
        return f"❌ Could not load the model: {e}"
    return None if backend.is_loaded() else status


def on_apply_system_prompt(session: Session, prompt: str, *chat_args: Any) -> Tuple[str, str]:
    session.system_prompt = prompt or ""
    return "✅ System prompt updated.", on_input_change(session, *chat_args)


def on_backend_change(session: Session, backend: str):
    session.backend = backend
    note = ""
    if is_vlm(backend) and not deps._VLM_AVAILABLE:
        note = f"  \n⚠️ mlx-vlm is not importable (`{deps._VLM_IMPORT_ERROR}`)."
    plain, mm = _composer_updates(backend)
    return f"Backend set to: **{backend}**{note}", plain, mm


def on_toggle_advanced(show: bool, backend: str):
    # Controls stay mounted while hidden so their values keep feeding chat_inputs.
    # Basic view always uses the mm composer (the Backend radio is off screen).
    show = bool(show)
    plain, mm = _composer_updates(backend) if show else (gr.update(visible=False), gr.update(visible=True))
    return (
        gr.update(visible=show),   # mlx_advanced
        gr.update(visible=show),   # gguf_advanced
        gr.update(visible=show),   # vlm_advanced
        gr.update(visible=show),   # web_advanced
        gr.update(visible=show),   # settings_col
        gr.update(visible=show),   # token_meter
        gr.update(visible=show),   # saveload_box
        plain,                     # user_input
        mm,                        # user_input_mm
    )


def on_refresh_models(models_dir: str, mlx_current: str, gguf_current: str, vlm_current: str = ""):
    mlx_choices = scan_mlx_models(models_dir)
    gguf_choices = scan_gguf_models(models_dir)
    vlm_choices = scan_vlm_models(models_dir)
    n_mlx, n_gguf, n_vlm = len(mlx_choices), len(gguf_choices), len(vlm_choices)
    # Keep current selections available even if not found on disk. Vision models
    # stay listed under MLX too (mlx_lm can run the VL language half).
    mlx_current = (mlx_current or "").strip()
    gguf_current = (gguf_current or "").strip()
    vlm_current = (vlm_current or "").strip()
    if mlx_current and mlx_current not in mlx_choices:
        mlx_choices.insert(0, mlx_current)
    if gguf_current and gguf_current not in gguf_choices:
        gguf_choices.insert(0, gguf_current)
    if vlm_current and vlm_current not in vlm_choices:
        vlm_choices.insert(0, vlm_current)
    scanned = (str(models_dir or "").strip() or str(DEFAULT_MODELS_DIR))
    msg = (
        f"🔎 Found **{n_mlx}** MLX, **{n_gguf}** GGUF and **{n_vlm}** vision "
        f"entries under `{scanned}` (incl. Hugging Face cache)."
    )
    return (
        gr.update(choices=mlx_choices, value=mlx_current or (mlx_choices[0] if mlx_choices else None)),
        gr.update(choices=gguf_choices, value=gguf_current or (gguf_choices[0] if gguf_choices else None)),
        gr.update(choices=vlm_choices, value=vlm_current or (vlm_choices[0] if vlm_choices else None)),
        msg,
    )


def on_load_mlx(
    session: Session,
    current_backend: str,
    model_id_or_path: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    n_ctx_estimate: int,
) -> Tuple[str, Any, Any, Any, Any]:
    params = MLXParams(
        model_id_or_path=os.path.expanduser((model_id_or_path or "").strip()),
        max_tokens=safe_int(max_tokens, MLX_MAX_TOKENS),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
        n_ctx_estimate=safe_int(n_ctx_estimate, MLX_N_CTX_ESTIMATE),
    )
    be = get_backend(BACKEND_MLX)
    status, detected_ctx = be.load(params)
    ok = be.is_loaded() and status.startswith("✅")
    status, switch = _after_load(session, BACKEND_MLX, current_backend, status, ok)
    ctx_update = detected_ctx if detected_ctx is not None else gr.update()
    return (status, ctx_update, *switch)


def on_unload_mlx() -> str:
    return get_backend(BACKEND_MLX).unload()


def on_load_vlm(session: Session, current_backend: str, *vlm_args: Any) -> Tuple[str, Any, Any, Any, Any]:
    params = build_vlm_params(*vlm_args)
    be = get_backend(BACKEND_VLM)
    status, detected_ctx = be.load(params)
    ok = be.is_loaded() and status.startswith("✅")
    status, switch = _after_load(session, BACKEND_VLM, current_backend, status, ok)
    ctx_update = detected_ctx if detected_ctx is not None else gr.update()
    return (status, ctx_update, *switch)


def on_unload_vlm() -> str:
    return get_backend(BACKEND_VLM).unload()


def on_load_gguf(
    session: Session,
    current_backend: str,
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    seed: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    mmproj_path: str = "",
) -> Tuple[str, Any, Any, Any]:
    params = GGUFParams(
        model_path=os.path.expanduser((model_path or "").strip()),
        n_ctx=safe_int(n_ctx, GGUF_N_CTX),
        n_gpu_layers=int(n_gpu_layers),
        seed=int(seed),
        max_tokens=safe_int(max_tokens, GGUF_MAX_TOKENS),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
        mmproj_path=os.path.expanduser((mmproj_path or "").strip()),
    )
    be = get_backend(BACKEND_GGUF)
    status, _ = be.load(params)
    ok = be.is_loaded() and status.startswith("✅")
    status, switch = _after_load(session, BACKEND_GGUF, current_backend, status, ok)
    return (status, *switch)


def on_unload_gguf() -> str:
    return get_backend(BACKEND_GGUF).unload()


def on_user_submit(
    session: Session,
    user_input: str,
    user_input_mm: Any,
    backend: str,
    mlx_model_id: str,
    mlx_max_tokens: int,
    mlx_temperature: float,
    mlx_top_p: float,
    mlx_top_k: int,
    mlx_repeat_penalty: float,
    mlx_n_ctx_estimate: int,
    gguf_model_path: str,
    gguf_n_ctx: int,
    gguf_n_gpu_layers: int,
    gguf_seed: int,
    gguf_max_tokens: int,
    gguf_temperature: float,
    gguf_top_p: float,
    gguf_top_k: int,
    gguf_repeat_penalty: float,
    enable_thinking: bool = True,
    thinking_budget: int = 0,
    use_web: bool = False,
    web_n_pages: int = 3,
    show_advanced: bool = True,
    *vlm_args: Any,
) -> Generator[Tuple[List[Dict[str, Any]], Any, Any], None, None]:
    # Yields (chat view, plain-input update, mm-input update); the active
    # composer is cleared on the first yield after a message is accepted.
    advanced = bool(show_advanced)

    use_mm = uses_mm_composer(backend, advanced)
    mm = user_input_mm if isinstance(user_input_mm, dict) else {}
    if use_mm:
        text = str(mm.get("text") or "")
        media, unknown = classify_media(mm.get("files") or [])
    else:
        text = str(user_input or "")
        media, unknown = {k: [] for k in MEDIA_KINDS}, []
        # A paste lands in the mm box even with the plain box live; keep its
        # non-media files (folded into the message below), drop real media.
        _, unknown = classify_media(mm.get("files") or [])

    # A paste can arrive as a text/plain file: fold it back into the message.
    pasted_text, unknown = inline_text_attachments(unknown)
    if pasted_text:
        text = f"{text.rstrip()}\n\n{pasted_text}" if text.strip() else pasted_text

    # Fall back to the other box if the expected one is empty.
    if not text.strip() and not any(media[k] for k in MEDIA_KINDS):
        spare = str(user_input or "") if use_mm else str(mm.get("text") or "")
        if spare.strip():
            text = spare

    has_media = any(media[k] for k in MEDIA_KINDS)
    backend = _effective_backend(backend, advanced, media)
    be = get_backend(backend)

    # Ignore empty sends; attachments alone are valid for the vision backend.
    if not text.strip() and not has_media:
        if unknown:
            names = ", ".join(Path(p).name for p in unknown)
            session.history.append({
                "role": "assistant",
                "content": (f"⚠️ Nothing to send — {len(unknown)} unsupported "
                            f"attachment(s) skipped: {names}"),
            })
            yield chat_view(session), gr.update(), ({"text": "", "files": []} if use_mm else gr.update())
        else:
            yield chat_view(session), gr.update(), gr.update()
        return

    mlx_params, gguf_params, vlm_params = build_params(
        mlx_model_id, mlx_max_tokens, mlx_temperature, mlx_top_p, mlx_top_k, mlx_repeat_penalty, mlx_n_ctx_estimate,
        gguf_model_path, gguf_n_ctx, gguf_n_gpu_layers, gguf_seed, gguf_max_tokens, gguf_temperature, gguf_top_p, gguf_top_k, gguf_repeat_penalty,
        enable_thinking, thinking_budget,
        *vlm_args,
    )

    if be is None:
        session.pending_note = None
        session.history.append({"role": "assistant", "content": f"❌ Unknown backend: {backend}"})
        yield chat_view(session), gr.update(), gr.update()
        return

    params = params_for(backend, mlx_params, gguf_params, vlm_params)

    msg: Dict[str, Any] = {"role": "user", "content": text}
    if has_media:
        msg["media"] = {k: v for k, v in media.items() if v}
    session.history.append(msg)
    if use_mm:
        yield chat_view(session), gr.update(), {"text": "", "files": []}
    else:
        yield chat_view(session), "", gr.update()

    # Basic view has no Load button, so the first message brings the model up.
    if BASIC_AUTOLOAD and not advanced and not be.is_loaded():
        session.pending_note = (
            f"{LOADING_NOTE}\n\n`{_model_name_for(params)}`"
            "\n\n*The first load can take a minute.*"
        )
        yield chat_view(session), gr.update(), gr.update()
        err = _autoload_model(be, params)
        session.pending_note = None
        if err:
            session.history.append({"role": "assistant", "content": err})
            yield chat_view(session), gr.update(), gr.update()
            return

    # Shown until the first token arrives, so a slow prefill doesn't look hung.
    session.pending_note = THINKING_NOTE
    yield chat_view(session), gr.update(), gr.update()

    # Reported on the finished reply rather than as its own turn.
    skipped_footer = ""
    if unknown:
        names = ", ".join(Path(p).name for p in unknown)
        skipped_footer = f"\n\n---\n⚠️ Skipped {len(unknown)} unsupported attachment(s): {names}"

    trim_history_to_fit_context(be, session, params)

    # Web context is injected into the prompt only, not history; sources are
    # appended to the reply afterwards.
    web_ctx = ""
    web_sources_footer = ""
    if use_web and deps._WEB_AVAILABLE and text.strip():
        try:
            r = deps.web.search_and_fetch(text, limit=int(web_n_pages))
            if r.get("sources"):
                web_ctx = r["context"]
                web_sources_footer = "\n\n---\n🌐 **Sources**\n" + _web_sources_md(r["sources"])
            elif r.get("errors"):
                web_sources_footer = "\n\n---\n⚠️ Web search found no readable pages."
        except Exception as e:
            web_sources_footer = f"\n\n---\n⚠️ Web search failed: {e}"
    # Injected context isn't visible to the token budgeter; hold back an estimate.
    web_reserve = (len(web_ctx) // 3 + 100) if web_ctx else 0

    def _emit(chunk: str) -> None:
        session.pending_note = None
        if session.history and session.history[-1].get("role") == "assistant":
            session.history[-1]["content"] = chunk
        else:
            session.history.append({"role": "assistant", "content": chunk})

    def _close_turn() -> None:
        # Ensure the turn ends with an assistant message: a Stop during prefill
        # (or first-token EOS) would else leave user/user alternation that
        # strict chat templates reject.
        session.pending_note = None
        if not session.history or session.history[-1].get("role") != "assistant":
            session.history.append({
                "role": "assistant",
                "content": "⏹️ Stopped." if session.stop_requested else "*(no output)*",
            })

    def _cannot_fit(extra: int) -> str:
        parts = []
        if web_reserve:
            parts.append(f"~{web_reserve} tokens of web-search context")
        if extra > web_reserve:
            parts.append(f"~{extra - web_reserve} tokens of attachments")
        held = " and ".join(parts) or "the generation reserve"
        return (
            f"❌ This message cannot fit: {held} against a "
            f"{be.context_window(params)}-token window leaves no room for your "
            "text. Shorten it, turn off web search, send fewer attachments, or "
            "raise the n_ctx estimate."
        )

    try:
        if not be.is_loaded():
            session.pending_note = None
            session.history.append({"role": "assistant", "content": _no_model_message(backend) + skipped_footer})
            yield chat_view(session), gr.update(), gr.update()
            return

        # Media pre-check (no-op for text backends). Catch attachments that can't
        # fit before the shrink step truncates the user's own text to "".
        media_reserve = be.media_reserve(session, params)
        ctx_window = be.context_window(params)
        over = be.media_overflow_message(session, params, ctx_window, media_reserve, web_reserve)
        if over:
            session.pending_note = None
            session.history.append({"role": "assistant", "content": over + skipped_footer})
            yield chat_view(session), gr.update(), gr.update()
            return

        reserve = web_reserve + media_reserve
        _, adjusted_max = shrink_last_user_message_to_fit_and_adjust_max_tokens(be, session, params, reserve)
        if adjusted_max <= 0:
            session.pending_note = None
            session.history.append({"role": "assistant", "content": _cannot_fit(reserve) + skipped_footer})
            yield chat_view(session), gr.update(), gr.update()
            return
        if adjusted_max != int(params.max_tokens):
            params.max_tokens = int(adjusted_max)

        prepared = be.prepare(session, params, web_ctx)
        if prepared.dropped:
            names = ", ".join(Path(p).name for p in prepared.dropped)
            skipped_footer += (
                f"\n\n---\n⚠️ {len(prepared.dropped)} earlier attachment(s) are no "
                f"longer readable and were not sent: {names}"
            )

        session.stop_requested = False
        for chunk in be.generate_stream(session, params, prepared):
            if session.stop_requested:
                break
            _emit(chunk)
            yield chat_view(session), gr.update(), gr.update()
        _close_turn()

    except Exception as e:
        # Fold the error into the partial reply rather than appending a second
        # assistant turn that strict templates reject.
        session.pending_note = None
        err = f"❌ Generation error: {e}"
        if session.history and session.history[-1].get("role") == "assistant" and session.history[-1].get("content"):
            session.history[-1]["content"] += "\n\n" + err
        else:
            _emit(err)

    # Append web sources / attachment notes to the completed reply.
    session.pending_note = None
    footer = web_sources_footer + skipped_footer
    if footer and session.history and session.history[-1].get("role") == "assistant":
        session.history[-1]["content"] += footer
    yield chat_view(session), gr.update(), gr.update()


def on_stop_stream(session: Session) -> str:
    session.stop_requested = True
    session.pending_note = None  # cancelled event abandons the generator; clear the bubble
    return "⏹️ Stopping (the model will finish the current token)."


def close_dangling_turn(session: Session) -> List[Dict[str, Any]]:
    # `cancels=` abandons on_user_submit, so a Stop before the first token
    # leaves a trailing user message (_close_turn never ran). Wired on the chat
    # concurrency id so this runs only after the generator releases its slot.
    session.pending_note = None
    if session.history and session.history[-1].get("role") == "user":
        session.history.append({"role": "assistant", "content": "⏹️ Stopped."})
    return chat_view(session)


def on_input_change(
    session: Session,
    text: str,
    text_mm: Any,
    backend: str,
    mlx_model_id: str,
    mlx_max_tokens: int,
    mlx_temperature: float,
    mlx_top_p: float,
    mlx_top_k: int,
    mlx_repeat_penalty: float,
    mlx_n_ctx_estimate: int,
    gguf_model_path: str,
    gguf_n_ctx: int,
    gguf_n_gpu_layers: int,
    gguf_seed: int,
    gguf_max_tokens: int,
    gguf_temperature: float,
    gguf_top_p: float,
    gguf_top_k: int,
    gguf_repeat_penalty: float,
    enable_thinking: bool = True,
    thinking_budget: int = 0,
    *vlm_args: Any,
) -> str:
    mlx_params, gguf_params, vlm_params = build_params(
        mlx_model_id, mlx_max_tokens, mlx_temperature, mlx_top_p, mlx_top_k, mlx_repeat_penalty, mlx_n_ctx_estimate,
        gguf_model_path, gguf_n_ctx, gguf_n_gpu_layers, gguf_seed, gguf_max_tokens, gguf_temperature, gguf_top_p, gguf_top_k, gguf_repeat_penalty,
        enable_thinking, thinking_budget,
        *vlm_args,
    )
    be = get_backend(backend)
    if be is None:
        return ""
    params = params_for(backend, mlx_params, gguf_params, vlm_params)
    # Only the visible composer contributes pending text/attachments.
    pending_media = None
    if is_vlm(backend) or be.is_vision:
        mm = text_mm if isinstance(text_mm, dict) else {}
        pending = str(mm.get("text") or "")
        pending_media = classify_media(mm.get("files") or [])[0]
    else:
        pending = str(text or "")
    return token_meter_markdown(be, session, params, pending_user_input=pending, pending_media=pending_media)


def on_save_json(
    session: Session,
    backend: str,
    mlx_model_id: str,
    mlx_max_tokens: int,
    mlx_temperature: float,
    mlx_top_p: float,
    mlx_top_k: int,
    mlx_repeat_penalty: float,
    mlx_n_ctx_estimate: int,
    gguf_model_path: str,
    gguf_n_ctx: int,
    gguf_n_gpu_layers: int,
    gguf_seed: int,
    gguf_max_tokens: int,
    gguf_temperature: float,
    gguf_top_p: float,
    gguf_top_k: int,
    gguf_repeat_penalty: float,
    enable_thinking: bool = True,
    thinking_budget: int = 0,
    *vlm_args: Any,
) -> Tuple[str, str]:
    mlx_params, gguf_params, vlm_params = build_params(
        mlx_model_id, mlx_max_tokens, mlx_temperature, mlx_top_p, mlx_top_k, mlx_repeat_penalty, mlx_n_ctx_estimate,
        gguf_model_path, gguf_n_ctx, gguf_n_gpu_layers, gguf_seed, gguf_max_tokens, gguf_temperature, gguf_top_p, gguf_top_k, gguf_repeat_penalty,
        enable_thinking, thinking_budget,
        *vlm_args,
    )
    return save_chat_json(session, backend, mlx_params, gguf_params, vlm_params)


def on_load_json(session: Session, fileobj) -> Tuple[List[Dict[str, Any]], str, str]:
    hist, msg = load_chat_json(session, fileobj)
    return hist, msg, session.system_prompt  # return system prompt so the textbox stays in sync


def on_mcp_connect(command: str, args: str, env_json: str) -> str:
    return MCP.connect(command, args, env_json)


def on_mcp_list_tools() -> str:
    return MCP.list_tools()


def on_mcp_run_tool(name: str, args_json: str) -> str:
    return MCP.run_tool(name, args_json)


def _web_sources_md(sources: List[Dict[str, Any]]) -> str:
    lines = []
    for s in sources:
        label = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
        lines.append(f"{s['n']}. {label}")
    return "\n".join(lines)


def on_web_test(query: str, n: int) -> str:
    if not deps._WEB_AVAILABLE:
        return f"❌ Web search unavailable: `{deps._WEB_IMPORT_ERROR}`"
    if not (query or "").strip():
        return "Enter a search query first."
    try:
        r = deps.web.search_and_fetch(query.strip(), limit=int(n))
    except Exception as e:
        return f"❌ Search failed: {e}"
    if not r.get("sources"):
        errs = r.get("errors", [])
        if errs:
            return "❌ No readable pages:\n" + "\n".join(f"- {e['url']}: {e['error']}" for e in errs)
        return "ℹ️ No results."
    out = "### Sources\n" + _web_sources_md(r["sources"])
    for err in r.get("errors", []):
        out += f"\n- ⚠️ {err['url']}: {err['error']}"
    out += "\n\n### Context preview\n```\n" + r["context"][:3000] + "\n```"
    return out
