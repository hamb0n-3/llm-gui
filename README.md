# Local LLM GUI — MLX, GGUF & MLX-VLM

A Gradio chat GUI for running local models on Apple Silicon, with three
interchangeable backends:

- **MLX (`mlx_lm`)** — text
- **GGUF (`llama.cpp` via `llama-cpp-python`)** — text
- **MLX-VLM (`mlx_vlm`)** — vision / audio / video

Features: streaming generation with a **Stop** button, a per-backend **token
meter** with context auto-detect, automatic history trimming / last-message
truncation to fit the window, model discovery (a models folder + the Hugging
Face cache), **save/load** chat sessions as JSON, optional **web search** injected
into the prompt with cited sources, a **thinking toggle** with a token budget
that works on all three backends, and a **basic/advanced** two-mode UI (basic
view auto-picks the backend and loads the model on your first message).
Conversation state is per-browser-session; loaded model weights are shared
process-wide.

## Run

```bash
pip install -r ../requirements.txt # MLX/MLX-VLM require Apple Silicon
python llm-gui.py
```

Open the local URL Gradio prints (default `http://127.0.0.1:7100`, or a free port
if that one is busy). All tuning defaults live at the top of `config.py`.

## Module layout

The app is split into flat modules imported by `llm-gui.py` (the entry point):

| Module | Responsibility |
|---|---|
| `config.py` | All tuning constants, backend ids, slider ranges, small pure helpers |
| `deps.py` | Optional-dependency probing (mlx_lm / mlx_vlm / llama_cpp / mcp / web_search) |
| `params.py` | `MLXParams` / `GGUFParams` / `VLMParams` + UI-values params adapters |
| `session.py` | `Session` (per-browser gr.State) + message/conversation helpers |
| `media.py` | Upload classification, pasted-text inlining, media token pricing |
| `tokenization.py` | Tokenizer-based counting and context-window inference |
| `modelscan.py` | Model discovery, HF-cache resolution, config probing |
| `backend_base.py` | `Backend` ABC + `Prepared` result type + the thinking-budget wrapper |
| `backend_mlx.py` / `backend_gguf.py` / `backend_vlm.py` | The three concrete backends |
| `backends.py` | Process-global registry of the three backend singletons |
| `budget.py` | Token meter, history trimming, last-message shrink (backend-polymorphic) |
| `persistence.py` | Save / load / clear a chat as JSON |
| `mcp_client.py` | MCP placeholder (non-functional stub) |
| `handlers.py` | Gradio event handlers (thread `Session` in) + the chat orchestrator |
| `ui.py` | `build_ui()` — Gradio Blocks construction and event wiring |
| `web_search.py` | Standalone web search + page fetch (used by the Web tab) |

## Thinking

The 💭 toggle under the composer sits outside the settings column, so it's
reachable in basic view. Qwen3 and friends are binary — there's no low/medium/high
— so the toggle just switches reasoning on or off, and each backend gets there
differently: MLX and MLX-VLM pass `enable_thinking` as a chat-template kwarg,
while GGUF can't, because `llama-cpp-python` renders the template internally and
forwards no template kwargs. GGUF therefore uses Qwen's `/no_think` soft switch,
appended to the last user turn at prompt-build time so it never lands in your
history or a saved chat.

The budget is a token cap, not a mode, and the harness enforces it rather than
the model: once the reasoning block passes the cap, generation stops, `</think>`
is injected, and the answer resumes from the truncated reasoning — against what
is left of `max_tokens`, since the shrink step already sized that to the context
window and the reasoning already spent part of it. MLX-VLM has
this natively (`thinking_budget`); MLX and GGUF get it from `budget_thinking()`
in `backend_base.py`. GGUF image turns skip the budget — they have to stay on
llama.cpp's chat handler, which owns the mmproj path.

Reasoning is folded into a collapsible block in the transcript. That's display
only: history keeps the raw `<think>` text, so the prompt is unchanged.

Backends are UI-free (they never import Gradio): `Backend.load()` returns a plain
`(status, detected_ctx)` pair and the handler translates it into widget updates.
