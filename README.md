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
into the prompt with cited sources, and a **basic/advanced** two-mode UI (basic
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
| `backend_base.py` | `Backend` ABC + `Prepared` result type |
| `backend_mlx.py` / `backend_gguf.py` / `backend_vlm.py` | The three concrete backends |
| `backends.py` | Process-global registry of the three backend singletons |
| `budget.py` | Token meter, history trimming, last-message shrink (backend-polymorphic) |
| `persistence.py` | Save / load / clear a chat as JSON |
| `mcp_client.py` | MCP placeholder (non-functional stub) |
| `handlers.py` | Gradio event handlers (thread `Session` in) + the chat orchestrator |
| `ui.py` | `build_ui()` — Gradio Blocks construction and event wiring |
| `web_search.py` | Standalone web search + page fetch (used by the Web tab) |

Backends are UI-free (they never import Gradio): `Backend.load()` returns a plain
`(status, detected_ctx)` pair and the handler translates it into widget updates.
