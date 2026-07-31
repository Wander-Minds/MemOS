# Local LLM Custom Provider Example (`examples\llm_local\`)

> **TL;DR** — a self-contained example that registers a *local* WebSocket-served LLM as a
> custom MemOS provider (`backend="local"`), then drives MemOS modules (NaiveTextMemory,
> MemChat, MOS) and a compact replica of the Wander-Memory flat-memory pipeline through it.
> Two inference backends are shipped: a **deterministic dummy** (no model needed) and a
> **real llama.cpp backend** (`--backend llama`) that runs a local GGUF model from `models/`
> with the `llama-server` binary built from the `ext/llama.cpp` submodule.

This example brings the Wander-Memory local-LLM stack
(`C:\dev\Wander-Memory\src\llm\llm_client.py` + `llm_server.py`, Apache-2.0) into MemOS
as a **runnable custom provider example**. The WebSocket protocol (sessions, heartbeats,
queue back-pressure, connection pooling) is shared by both backends, so switching between
them never touches the client, provider, or pipeline code.

## Architecture

```
┌───────────────────────────┐      WebSocket (ws://127.0.0.1:18081)
│  llm_server.py            │ ◄──────────────────────────────┐
│  ┌─────────────────────┐  │   {"prompt", "reset",          │
│  │ DummyServerManager  │  │    "system_prompt",            │
│  │   └ DummyBackend    │  │    "temperature", "reasoning"} │
│  │   (deterministic,   │  │   {"reasoning", "content"}     │
│  │    rule-based)      │  │   {"type": "heartbeat"}        │
│  └─────────────────────┘  │                                │
└───────────────────────────┘                                │
                                                              │
┌─────────────────────────────────────────────────────────────┴──────────┐
│  llm_client.py — persistent event loop + ConnectionPool + prompt_func  │
└────────────────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼───────────────────────────────┐
        ▼                 ▼                               ▼
┌──────────────┐   ┌───────────────┐              ┌──────────────────┐
│ provider.py  │   │ wander_       │              │ (optional)       │
│ (backend     │   │ pipeline.py   │              │ standalone CLI   │
│  "local" →   │   │ extract →     │              │ llm_client.py    │
│  LocalLLM)   │   │ dedup → store │              │ --prompt ...     │
└──────────────┘   │ → retrieve →  │              └──────────────────┘
        │          │ summarize →   │
        ▼          │ maintain      │
┌───────────────┐  └───────────────┘
│ MemOS modules │
│ NaiveTextMem  │
│ MemChat       │
│ MOS (guarded) │
└───────────────┘
```

## Hardware note

Two backends are available:

* `--backend dummy` (default) — a pure-Python, deterministic, rule-based stand-in that
  needs no model, GPU, or binary.  Use it to exercise the full protocol / pipeline stack
  on any machine.
* `--backend llama` — a **real llama.cpp backend**.  It launches the `llama-server`
  binary built from the `ext/llama.cpp` git submodule (`ext/build_llama_cpp.py`;
  prebuilt binaries ship in `ext/build-windows` / `ext/build-linux`) and serves a local
  GGUF model from the (git-ignored) `models/` directory — e.g.
  `models/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf`.  CUDA / Vulkan are auto-detected by
  llama-server; on a GPU you can offload all layers with `--ngl 99`.

## Run it

Requirements: Python 3.10+ and the `websockets` package (not a MemOS core dependency):

```bash
pip install websockets        # only if `import websockets` fails
```

Run everything end-to-end from the repository root (`C:\dev\MemOS`):

```bash
python examples\llm_local\workflow.py
```

The workflow script starts the dummy WebSocket server in a background thread and runs
seven scenarios: provider registration + factory, direct `generate()` (multi-turn reset
semantics), `generate_stream()`, NaiveTextMemory via the custom provider, MemChat,
an optional (guarded) MOS demo, and the Wander-style flat-memory pipeline.

### What you should see

* **Scenarios 1–5 and 7** print deterministic output (dummy backend replies, extracted
  JSON memories, BM25 hits, the summarization context block, maintenance report).
* **Scenario 6 (MOS)** is *guarded*: `MOS.chat` hard-requires a registered `cube_root`
  memory cube, which is the vector-DB/embedder-backed textual-memory stack (external
  services). On this hardware the scenario prints a clear skip note and falls back to
  Scenarios 4/5 — the local provider is used for `chat_model` and `mem_reader` during
  the attempt.
* The server logs a single line to stderr: `WebSocket server listening on
  ws://127.0.0.1:18081 (backend=dummy)`.

### Optional: run the server / client standalone

```bash
python examples\llm_local\llm_server.py --backend dummy --ws-port 18081
python examples\llm_local\llm_client.py --prompt prompt.txt
```

The client CLI sends one prompt and prints the reply (run the server in another terminal
first; `prompt.txt` is any text file).

## Real llama.cpp backend (`--backend llama`)

The real backend lives in `llama_backend.py` and mirrors the original Wander-Memory
`LlamaServerManager` (Apache-2.0): it spawns a persistent `llama-server` subprocess,
polls `GET /health`, and speaks the OpenAI-compatible streaming chat API, splitting
`reasoning_content` (thinking) from `content` so the WebSocket
`{"reasoning", "content"}` protocol is preserved.  The same `infer(...)` signature as
`DummyBackend` means the server/worker/client code is identical.

Requirements: a built `llama-server` binary and a GGUF model.  With the shipped
`ext/` contents and `models/` files, run:

```bash
# server (defaults: models/Qwen3.5-9B, ext/build-<platform> binary, ngl=99, ctx=32768)
python examples\llm_local\llm_server.py --backend llama --ws-port 18081

# use the smaller model or point at a specific binary / gguf
python examples\llm_local\llm_server.py --backend llama --model models\Qwen3.5-4B
python examples\llm_local\llm_server.py --backend llama --server ext\build-windows\bin\llama-server.exe
```

### Real end-to-end test (`real_test.py`)

`real_test.py` runs **real inference** through both halves of the stack and exits with a
clear PASS/FAIL:

```bash
python examples\llm_local\real_test.py                      # full: server + client checks
python examples\llm_local\real_test.py --model models\Qwen3.5-4B
python examples\llm_local\real_test.py --server-only        # just start the WS server (llama)
python examples\llm_local\real_test.py --client-only         # attach to a running server
```

It verifies: (1) llama-server starts and `/health` returns ok; (2) a direct HTTP streaming
chat request returns real tokens (with `reasoning=on`, `reasoning_content` is captured);
(3) the WebSocket server accepts connections; (4) `llm_client` single-turn
(`reset=True`) and multi-turn (`reset=False`, server-side conversation continuity)
round trips return non-empty content; (5) the sync wrappers (`prompt_func`,
`LocalLLMClient`) work through the connection pool.  Everything is shut down cleanly
(`llama-server` terminated) on exit.

### Attach to an externally running server

Set `WM_WS_URL` to skip the in-process server and talk to a server you started yourself:

```powershell
$env:WM_WS_URL = "ws://127.0.0.1:18081"
python examples\llm_local\workflow.py
```

## How the real backend is wired in

`llama_backend.LlamaServerManager` implements the same `infer(...)` interface as
`DummyBackend`:

```python
def infer(self, prompt, reset=True, system_prompt="", temperature=0.7, reasoning="off") -> tuple[str, str]
```

`llm_server.run_server(backend="llama", ...)` constructs it (or accepts a pre-built
`manager=`), calls `start()` (idempotent — the test harness can start llama-server
first and reuse it), and drives it through the same request queue / WebSocket protocol.
The client, the custom provider, and both pipeline layers are transport-agnostic — they
only speak the WebSocket protocol.  To serve a different local engine (Ollama, an
OpenAI-compatible endpoint, …), implement a manager with that `infer` signature and
register it in `run_server`.

## Custom provider registration (runtime-only)

`provider.register_local_llm()` performs the AGENTS.md "Adding a New Provider" steps
(config + factory registration) **at runtime** instead of editing `src\memos`:

```python
LLMConfigFactory.backend_to_class["local"] = LocalLLMConfig
LLMFactory.backend_to_class["local"] = LocalLLM
```

It is idempotent and must be called **before** any
`LLMConfigFactory.model_validate({"backend": "local", ...})`, because the pydantic
`backend` validator rejects unknown backends. `LLMFactory.from_config` is cached by the
singleton factory (keyed by the config dump): identical configs reuse one `LocalLLM`
instance, different configs (e.g. a different `ws_url`) create separate instances. If
cache reuse is ever undesirable, construct `LocalLLM(LocalLLMConfig(...))` directly.

## Files

| File | Purpose |
| --- | --- |
| `dummy_backend.py` | Deterministic rule-based inference backend (`DummyBackend` + `DummyServerManager`). |
| `llama_backend.py` | **Real llama.cpp backend**: `LlamaServerManager` (spawns `llama-server`, `/health` polling, streaming HTTP chat, `infer()`), model/binary resolution. Adapted from `C:\dev\Wander-Memory\src\llm\llm_server.py` (Apache-2.0). |
| `llm_server.py` | WebSocket server (protocol, sessions, heartbeats, queue back-pressure). `--backend dummy` or `--backend llama`. Adapted from `C:\dev\Wander-Memory\src\llm\llm_server.py` (Apache-2.0). |
| `llm_client.py` | Persistent event loop + connection pool + sync `prompt_func`. Adapted from `C:\dev\Wander-Memory\src\llm\llm_client.py` (Apache-2.0). |
| `provider.py` | MemOS custom provider: `LocalLLMConfig`, `LocalLLM`, `register_local_llm()`. |
| `real_test.py` | Real end-to-end test: starts llama-server + WS server, runs server- and client-side checks (modes: `all` / `server-only` / `client-only`). |
| `wander_pipeline.py` | Compact replica of the Wander-Memory flat-memory pipeline (extract → dedup → store → retrieve → summarize → maintain), stdlib only. |
| `workflow.py` | Orchestrated end-to-end demo (Scenarios 1–7), dummy backend by default. |

`ext/` (repo root) holds the `llama.cpp` git submodule (`ext/llama.cpp`), the cross-platform
build script (`ext/build_llama_cpp.py`), and the git-ignored build directories
(`ext/build-*`).  Local GGUF models live in `models/` (git-ignored).

## Verification

Run from the repository root (`D:\MemOS`):

```bash
# dummy backend (no model needed)
python examples\llm_local\workflow.py
# real llama.cpp backend, end-to-end (server + client)
python examples\llm_local\real_test.py
ruff check examples\llm_local
ruff format --check examples\llm_local
```

Sanity import (registration-before-validation):

```bash
python -c "import sys; sys.path.insert(0,'src'); from examples.llm_local.provider import register_local_llm; register_local_llm(); from memos.configs.llm import LLMConfigFactory; c=LLMConfigFactory.model_validate({'backend':'local','config':{'model_name_or_path':'dummy'}}); print(c.config)"
```

All three checks pass on the shipped code (ruff 0.16.x, Python 3.11).
