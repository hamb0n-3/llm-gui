from __future__ import annotations

import inspect
from typing import Any

import gradio as gr

import deps
import handlers as h
from backends import get_backend
from budget import token_meter_markdown
from config import (
    BACKENDS, DEFAULT_BACKEND, DEFAULT_SYSTEM_PROMPT, DEFAULT_MODELS_DIR,
    SHOW_ADVANCED_AT_STARTUP, WEB_USE_IN_CHAT, WEB_N_PAGES, Range,
    ENABLE_THINKING, THINKING_BUDGET,
    RANGE_MAX_TOKENS, RANGE_TEMPERATURE, RANGE_TOP_P, RANGE_TOP_K, RANGE_REPEAT_PENALTY,
    RANGE_N_CTX, RANGE_GGUF_N_CTX, RANGE_N_GPU_LAYERS, RANGE_MEDIA_CAP,
    RANGE_THINKING_BUDGET, RANGE_WEB_PAGES, is_vlm,
)
from modelscan import scan_mlx_models, scan_gguf_models, scan_vlm_models
from params import MLXParams, GGUFParams, VLMParams, params_for
from persistence import clear_history
from session import Session


def _knob(rng: Range, value: Any, label: str, **kwargs: Any) -> "gr.Slider":
    return gr.Slider(rng.min, rng.max, value=value, step=rng.step, label=label, **kwargs)


# build_ui must stay one function: handlers wire across tabs, so every component
# has to share one closure.
def build_ui() -> gr.Blocks:
    mlx_d, gguf_d, vlm_d = MLXParams(), GGUFParams(), VLMParams()
    with gr.Blocks(title="MLX/LLAMA LLM GUI", fill_height=True) as demo:
        # Per-browser state. Gradio deep-copies this default per session; every
        # Session field is deepcopy-safe.
        session_state = gr.State(Session())

        # Toggle lives outside settings_col — the only way back from basic view,
        # so it stays on screen when that column is hidden.
        with gr.Row():
            gr.Markdown("# 🧠 Local LLM GUI — MLX, GGUF & MLX-VLM")
            show_advanced = gr.Checkbox(
                label="Show advanced settings",
                value=SHOW_ADVANCED_AT_STARTUP,
                scale=0,
                min_width=260,
                info=(
                    "Off: just the chat — the model loads itself on your first "
                    "message. On: every setting, model picker and knob."
                ),
            )
        with gr.Row():
            with gr.Column(scale=1, min_width=360, visible=SHOW_ADVANCED_AT_STARTUP) as settings_col:
                with gr.Tab("Model & Backend"):
                    backend = gr.Radio(
                        BACKENDS,
                        label="Backend",
                        value=DEFAULT_BACKEND,
                    )
                    with gr.Row():
                        models_dir = gr.Textbox(
                            label="Models folder (scanned for MLX folders and .gguf files)",
                            value=str(DEFAULT_MODELS_DIR),
                            scale=4,
                        )
                        refresh_models_btn = gr.Button("↻ Rescan", scale=1)

                    _default_mlx = mlx_d.model_id_or_path
                    _default_gguf = gguf_d.model_path
                    _default_vlm = vlm_d.model_id_or_path
                    _vlm_choices = scan_vlm_models(str(DEFAULT_MODELS_DIR))
                    _mlx_choices = scan_mlx_models(str(DEFAULT_MODELS_DIR))
                    _gguf_choices = scan_gguf_models(str(DEFAULT_MODELS_DIR))
                    if _default_mlx not in _mlx_choices:
                        _mlx_choices.insert(0, _default_mlx)
                    if _default_gguf not in _gguf_choices:
                        _gguf_choices.insert(0, _default_gguf)
                    if _default_vlm not in _vlm_choices:
                        _vlm_choices.insert(0, _default_vlm)

                    with gr.Tabs():
                        with gr.Tab("MLX"):
                            mlx_model_id = gr.Dropdown(
                                label="MLX model (local folder or HF repo id — type to enter a custom one)",
                                choices=_mlx_choices,
                                value=_default_mlx,
                                allow_custom_value=True,
                            )
                            mlx_max_tokens = _knob(RANGE_MAX_TOKENS, mlx_d.max_tokens, "Max tokens (generation)")
                            mlx_temperature = _knob(RANGE_TEMPERATURE, mlx_d.temperature, "Temperature")

                            with gr.Group(visible=SHOW_ADVANCED_AT_STARTUP) as mlx_advanced:
                                gr.Markdown("**Advanced — sampling**")
                                mlx_top_p = _knob(RANGE_TOP_P, mlx_d.top_p, "Top-p")
                                mlx_top_k = _knob(RANGE_TOP_K, mlx_d.top_k, "Top-k")
                                mlx_repeat_penalty = _knob(RANGE_REPEAT_PENALTY, mlx_d.repeat_penalty, "Repetition penalty")
                                gr.Markdown("**Advanced — context**")
                                mlx_n_ctx_estimate = _knob(RANGE_N_CTX, mlx_d.n_ctx_estimate, "n_ctx estimate (for token meter)")

                            with gr.Row():
                                mlx_load_btn = gr.Button("Load MLX model")
                                mlx_unload_btn = gr.Button("Unload MLX model")

                        with gr.Tab("GGUF"):
                            gguf_model_path = gr.Dropdown(
                                label="GGUF model file (type to enter a custom path)",
                                choices=_gguf_choices,
                                value=_default_gguf,
                                allow_custom_value=True,
                            )
                            gguf_max_tokens = _knob(RANGE_MAX_TOKENS, gguf_d.max_tokens, "Max tokens (generation)")
                            gguf_temperature = _knob(RANGE_TEMPERATURE, gguf_d.temperature, "Temperature")

                            with gr.Group(visible=SHOW_ADVANCED_AT_STARTUP) as gguf_advanced:
                                gr.Markdown("**Advanced — sampling**")
                                gguf_top_p = _knob(RANGE_TOP_P, gguf_d.top_p, "Top-p")
                                gguf_top_k = _knob(RANGE_TOP_K, gguf_d.top_k, "Top-k")
                                gguf_repeat_penalty = _knob(RANGE_REPEAT_PENALTY, gguf_d.repeat_penalty, "Repetition penalty")
                                gr.Markdown("**Advanced — runtime** (applied on the next model load)")
                                gguf_n_ctx = _knob(RANGE_GGUF_N_CTX, gguf_d.n_ctx, "n_ctx (context window)")
                                gguf_n_gpu_layers = _knob(RANGE_N_GPU_LAYERS, gguf_d.n_gpu_layers, f"n_gpu_layers ({int(RANGE_N_GPU_LAYERS.max)} = all on GPU)")
                                gguf_seed = gr.Number(value=gguf_d.seed, precision=0, label="Seed (0 = random)")
                                gr.Markdown("---\n**Advanced — vision (mmproj)**")
                                gguf_mmproj = gr.Textbox(
                                    label="Vision projector (mmproj .gguf)",
                                    value=gguf_d.mmproj_path,
                                    placeholder="leave empty to auto-detect beside the model",
                                    info=(
                                        "Vision GGUFs ship the projector as a separate file. "
                                        "Empty means: use any mmproj*.gguf in the model's folder. "
                                        "Loaded, it lets this backend accept image attachments."
                                    ),
                                )

                            with gr.Row():
                                gguf_load_btn = gr.Button("Load GGUF model")
                                gguf_unload_btn = gr.Button("Unload GGUF model")

                        with gr.Tab("MLX-VLM"):
                            if deps._VLM_AVAILABLE:
                                gr.Markdown(
                                    "### Vision / audio / video\n"
                                    "Select the **MLX-VLM (vision)** backend, then drag, "
                                    "paste or browse attachments in the chat box. Video and "
                                    "audio need a model with that tower (e.g. Qwen3-VL for "
                                    "video); images work on any VLM."
                                )
                            else:
                                gr.Markdown(
                                    f"### mlx-vlm unavailable\n`{deps._VLM_IMPORT_ERROR}`\n\n"
                                    "Install with: `pip install mlx-vlm`"
                                )
                            vlm_model_id = gr.Dropdown(
                                label="Vision model (local folder or HF repo id — type to enter a custom one)",
                                choices=_vlm_choices,
                                value=_default_vlm,
                                allow_custom_value=True,
                            )
                            vlm_max_tokens = _knob(RANGE_MAX_TOKENS, vlm_d.max_tokens, "Max tokens (generation)")
                            vlm_temperature = _knob(RANGE_TEMPERATURE, vlm_d.temperature, "Temperature")

                            with gr.Group(visible=SHOW_ADVANCED_AT_STARTUP) as vlm_advanced:
                                gr.Markdown("**Advanced — sampling**")
                                vlm_top_p = _knob(RANGE_TOP_P, vlm_d.top_p, "Top-p")
                                vlm_top_k = _knob(RANGE_TOP_K, vlm_d.top_k, "Top-k")
                                vlm_repeat_penalty = _knob(RANGE_REPEAT_PENALTY, vlm_d.repeat_penalty, "Repetition penalty")
                                gr.Markdown("**Advanced — context**")
                                vlm_n_ctx_estimate = _knob(RANGE_N_CTX, vlm_d.n_ctx_estimate, "n_ctx estimate (for token meter)")

                                gr.Markdown("---\n**Advanced — attachments across turns**")
                                vlm_resend_media = gr.Checkbox(
                                    label="Resend earlier images on follow-up turns",
                                    value=vlm_d.resend_history_media,
                                    info=(
                                        "Off: images apply only to the message they were sent "
                                        "with. On: earlier images are re-encoded every "
                                        "generation, which costs prefill. Video and audio are "
                                        "always current-turn only."
                                    ),
                                )
                                vlm_media_cap = _knob(
                                    RANGE_MEDIA_CAP, vlm_d.media_history_cap,
                                    "Max images resent (most recent first)",
                                )

                            with gr.Row():
                                vlm_load_btn = gr.Button("Load vision model")
                                vlm_unload_btn = gr.Button("Unload vision model")

                    status = gr.Markdown("Ready.")

                with gr.Tab("System Prompt"):
                    sys_prompt = gr.Textbox(
                        label="System prompt",
                        value=DEFAULT_SYSTEM_PROMPT,
                        lines=10,
                        placeholder="Enter your system prompt…",
                    )
                    apply_sys = gr.Button("Apply System Prompt")

                with gr.Tab("Web"):
                    if deps._WEB_AVAILABLE:
                        gr.Markdown(
                            "### Web search\n"
                            "Searches the web (DDG/Mojeek, no-JS scrape), fetches the "
                            "top hits, and injects the page text into the prompt with "
                            "numbered sources. No models, no database."
                        )
                    else:
                        gr.Markdown(f"### Web search unavailable\n`{deps._WEB_IMPORT_ERROR}`")
                    use_web = gr.Checkbox(
                        label="Use web search in chat (inject fetched pages, cite sources)",
                        value=WEB_USE_IN_CHAT,
                    )
                    with gr.Group(visible=SHOW_ADVANCED_AT_STARTUP) as web_advanced:
                        gr.Markdown("**Advanced — fetching**")
                        web_n_pages = _knob(RANGE_WEB_PAGES, WEB_N_PAGES, "Pages to fetch")
                    gr.Markdown("---")
                    web_test_q = gr.Textbox(label="Test a search", placeholder="query…")
                    web_test_btn = gr.Button("🔍 Search & preview")
                    web_out = gr.Markdown("")

                with gr.Tab("MCP"):
                    gr.Markdown("### Model Context Protocol (optional)")
                    mcp_cmd = gr.Textbox(label="Server command (e.g., `node server.js` or `python server.py`)", value="")
                    mcp_args = gr.Textbox(label="Extra args", value="")
                    mcp_env = gr.Textbox(label="Env (JSON object)", value="{}")
                    mcp_connect_btn = gr.Button("Connect (save settings)")
                    mcp_status = gr.Markdown("Not connected.")
                    gr.Markdown("---")
                    mcp_list_btn = gr.Button("List Tools")
                    mcp_list_out = gr.Markdown("")
                with gr.Tab("Run a Tool"):
                    tool_name = gr.Textbox(label="Tool name", placeholder="e.g., search")
                    tool_args = gr.Textbox(label="Args (JSON)", value="{}", lines=5)
                    run_tool_btn = gr.Button("Run Tool")
                    tool_output = gr.Markdown("")

            with gr.Column(scale=2, min_width=600):
                with gr.Row():
                    # Gradio 4.x requires type="messages"; 6.x removed the param.
                    _chatbot_kwargs = dict(label="Chat", height=500, autoscroll=False)
                    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
                        _chatbot_kwargs["type"] = "messages"
                    chatbot = gr.Chatbot(**_chatbot_kwargs)
                with gr.Row():
                    # One composer visible at a time; on_backend_change swaps them.
                    _start_mm = (not SHOW_ADVANCED_AT_STARTUP) or is_vlm(DEFAULT_BACKEND)
                    user_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type and press Enter…",
                        lines=4,
                        scale=4,
                        visible=not _start_mm,
                    )
                    # No file_types filter on purpose: Gradio raises on a
                    # disallowed paste (e.g. clipboard text/plain), aborting the
                    # event. classify_media sorts them out and reports "skipped".
                    user_input_mm = gr.MultimodalTextbox(
                        label="Your message (drag, paste or browse attachments)",
                        placeholder="Type a message, attach images/video/audio…",
                        lines=4,
                        scale=4,
                        visible=_start_mm,
                        file_count="multiple",
                    )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    stop_btn = gr.Button("Stop")
                    clear_btn = gr.Button("Clear")
                # Outside settings_col: thinking is worth flipping per message,
                # so it stays reachable in the basic view.
                with gr.Row():
                    think_on = gr.Checkbox(
                        label="💭 Thinking", value=ENABLE_THINKING, scale=0, min_width=150,
                        info="Reasoning models only, e.g. Qwen3.",
                    )
                    think_budget = _knob(
                        RANGE_THINKING_BUDGET, THINKING_BUDGET,
                        "Thinking budget in tokens (0 = uncapped)", scale=2,
                    )

                _init_backend = get_backend(DEFAULT_BACKEND)
                _init_params = params_for(DEFAULT_BACKEND, mlx_d, gguf_d, vlm_d)
                token_meter = gr.Markdown(
                    token_meter_markdown(_init_backend, Session(), _init_params, pending_user_input=""),
                    visible=SHOW_ADVANCED_AT_STARTUP,
                )

                with gr.Column(visible=SHOW_ADVANCED_AT_STARTUP) as saveload_box:
                    gr.Markdown("---")
                    with gr.Row():
                        save_btn = gr.Button("💾 Save chat (JSON)")
                        saved_file = gr.File(label="Download saved chat", interactive=False)
                        load_file = gr.File(label="Load chat JSON", file_types=[".json"])
                        load_status = gr.Markdown("")

        # vlm_inputs must stay in VLM_INPUT_ORDER: handlers receive it as trailing *vlm_args.
        vlm_inputs = [
            vlm_model_id, vlm_max_tokens, vlm_temperature, vlm_top_p, vlm_top_k,
            vlm_repeat_penalty, vlm_n_ctx_estimate, vlm_resend_media,
            vlm_media_cap,
        ]
        chat_core = [
            user_input, user_input_mm,
            backend,
            mlx_model_id, mlx_max_tokens, mlx_temperature, mlx_top_p, mlx_top_k, mlx_repeat_penalty, mlx_n_ctx_estimate,
            gguf_model_path, gguf_n_ctx, gguf_n_gpu_layers, gguf_seed, gguf_max_tokens, gguf_temperature, gguf_top_p, gguf_top_k, gguf_repeat_penalty,
            think_on, think_budget,
        ]
        chat_inputs = chat_core + vlm_inputs
        # web-search inputs sit before the VLM block so on_user_submit keeps it as *vlm_args.
        chat_submit_inputs = chat_core + [use_web, web_n_pages, show_advanced] + vlm_inputs

        meter_inputs = [session_state] + chat_inputs
        submit_inputs = [session_state] + chat_submit_inputs

        show_advanced.change(
            fn=h.on_toggle_advanced,
            inputs=[show_advanced, backend],
            outputs=[
                mlx_advanced, gguf_advanced, vlm_advanced, web_advanced,
                settings_col, token_meter, saveload_box,
                user_input, user_input_mm,
            ],
        )

        refresh_models_btn.click(
            fn=h.on_refresh_models,
            inputs=[models_dir, mlx_model_id, gguf_model_path, vlm_model_id],
            outputs=[mlx_model_id, gguf_model_path, vlm_model_id, status],
        )
        models_dir.submit(
            fn=h.on_refresh_models,
            inputs=[models_dir, mlx_model_id, gguf_model_path, vlm_model_id],
            outputs=[mlx_model_id, gguf_model_path, vlm_model_id, status],
        )

        apply_sys.click(
            fn=h.on_apply_system_prompt,
            inputs=[session_state, sys_prompt] + chat_inputs,
            outputs=[status, token_meter],
        )

        # A successful load also selects its backend in the radio and swaps the composer.
        mlx_load_event = mlx_load_btn.click(
            fn=h.on_load_mlx,
            inputs=[session_state, backend, mlx_model_id, mlx_max_tokens, mlx_temperature, mlx_top_p, mlx_top_k, mlx_repeat_penalty, mlx_n_ctx_estimate],
            outputs=[status, mlx_n_ctx_estimate, backend, user_input, user_input_mm],
        )
        mlx_load_event.then(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])
        mlx_unload_event = mlx_unload_btn.click(fn=h.on_unload_mlx, inputs=[], outputs=[status])
        mlx_unload_event.then(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])

        vlm_load_event = vlm_load_btn.click(
            fn=h.on_load_vlm,
            inputs=[session_state, backend] + vlm_inputs,
            outputs=[status, vlm_n_ctx_estimate, backend, user_input, user_input_mm],
        )
        vlm_load_event.then(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])
        vlm_unload_event = vlm_unload_btn.click(fn=h.on_unload_vlm, inputs=[], outputs=[status])
        vlm_unload_event.then(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])

        gguf_load_event = gguf_load_btn.click(
            fn=h.on_load_gguf,
            inputs=[session_state, backend, gguf_model_path, gguf_n_ctx, gguf_n_gpu_layers, gguf_seed, gguf_max_tokens, gguf_temperature, gguf_top_p, gguf_top_k, gguf_repeat_penalty, gguf_mmproj],
            outputs=[status, backend, user_input, user_input_mm],
        )
        gguf_load_event.then(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])
        gguf_unload_event = gguf_unload_btn.click(fn=h.on_unload_gguf, inputs=[], outputs=[status])
        gguf_unload_event.then(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])

        web_test_btn.click(fn=h.on_web_test, inputs=[web_test_q, web_n_pages], outputs=[web_out])
        web_test_q.submit(fn=h.on_web_test, inputs=[web_test_q, web_n_pages], outputs=[web_out])

        mcp_connect_btn.click(fn=h.on_mcp_connect, inputs=[mcp_cmd, mcp_args, mcp_env], outputs=[mcp_status])
        mcp_list_btn.click(fn=h.on_mcp_list_tools, inputs=[], outputs=[mcp_list_out])
        run_tool_btn.click(fn=h.on_mcp_run_tool, inputs=[tool_name, tool_args], outputs=[tool_output])

        chat_outputs = [chatbot, user_input, user_input_mm]

        # concurrency_id="chat-generate" serializes generation process-wide so two
        # calls can't drive the shared model handles (or one session's history) at once.
        chat_event = send_btn.click(fn=h.on_user_submit, inputs=submit_inputs, outputs=chat_outputs, queue=True, concurrency_id="chat-generate")
        enter_event = user_input.submit(fn=h.on_user_submit, inputs=submit_inputs, outputs=chat_outputs, queue=True, concurrency_id="chat-generate")
        enter_mm_event = user_input_mm.submit(fn=h.on_user_submit, inputs=submit_inputs, outputs=chat_outputs, queue=True, concurrency_id="chat-generate")

        # on_stop_stream stays OFF the chat concurrency id so it fires immediately;
        # close_dangling_turn runs ON it, only after the cancelled generator frees its slot.
        stop_event = stop_btn.click(fn=h.on_stop_stream, inputs=[session_state], outputs=[status], cancels=[chat_event, enter_event, enter_mm_event])
        stop_event.then(fn=h.close_dangling_turn, inputs=[session_state], outputs=[chatbot], concurrency_id="chat-generate")

        # Shares the chat concurrency_id so it can't mutate history mid-stream.
        clear_event = clear_btn.click(fn=clear_history, inputs=[session_state], outputs=[chatbot], concurrency_id="chat-generate")
        clear_event.then(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])

        # always_last coalesces bursts of keystrokes.
        user_input.change(
            fn=h.on_input_change,
            inputs=meter_inputs,
            outputs=[token_meter],
            trigger_mode="always_last",
        )
        user_input_mm.change(
            fn=h.on_input_change,
            inputs=meter_inputs,
            outputs=[token_meter],
            trigger_mode="always_last",
        )
        # .input() (not .change()) so a programmatic load setting the radio doesn't overwrite status.
        backend.input(
            fn=h.on_backend_change,
            inputs=[session_state, backend],
            outputs=[status, user_input, user_input_mm],
        )
        # Meter refreshes on both user and programmatic backend switches.
        backend.change(
            fn=h.on_input_change,
            inputs=meter_inputs,
            outputs=[token_meter],
        )

        mlx_n_ctx_estimate.change(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])
        gguf_n_ctx.change(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])
        mlx_max_tokens.change(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])
        gguf_max_tokens.change(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])
        vlm_n_ctx_estimate.change(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])
        vlm_max_tokens.change(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])
        # Media policy changes what the next prompt sends, so the meter follows.
        vlm_resend_media.change(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])
        vlm_media_cap.change(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])

        save_btn.click(
            fn=h.on_save_json,
            inputs=[session_state] + chat_core[2:] + vlm_inputs,
            outputs=[saved_file, status],
        )
        # Shares the chat concurrency_id so it can't mutate history mid-stream.
        load_event = load_file.change(fn=h.on_load_json, inputs=[session_state, load_file], outputs=[chatbot, load_status, sys_prompt], concurrency_id="chat-generate")
        load_event.then(fn=h.on_input_change, inputs=meter_inputs, outputs=[token_meter])

    return demo
