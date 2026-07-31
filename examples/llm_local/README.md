# Local LLM Custom Provider Example (`examples\llm_local\`)

> **TL;DR** — a self-contained example that registers a *local* WebSocket-served LLM as a
> custom MemOS provider (`backend="local"`), then drives MemOS modules (NaiveTextMemory,
> MemChat, MOS) and a compact replica of the Wander-Memory flat-memory pipeline through it.

This example brings the Wander-Memory local-LLM stack
(`C:\dev\Wander-Memory\src\llm\llm_client.py` + `llm_server.py`, Apache-2.0) into MemOS
as a **runnable custom provider example**. It requires no external services, no GPU, and
no model download: the server runs a **deterministic dummy inference backend** that
implements the exact same WebSocket protocol (sessions, heartbeats, queue back-pressure,
connection pooling) as the real `llama-server`-based server, so a real backend can be
plugged in later without touching the client, provider, or pipeline code.

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

The current development hardware cannot run real LLM inference (no `llama-server`
binary / no usable GPU backend). `llm_server.py` therefore defaults to `--backend dummy`:
a pure-Python, deterministic, rule-based stand-in that keeps the full WebSocket protocol
so a real backend can be swapped in later (see "How to swap in a real backend" below).

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

### Attach to an externally running server

Set `WM_WS_URL` to skip the in-process server and talk to a server you started yourself:

```powershell
$env:WM_WS_URL = "ws://127.0.0.1:18081"
python examples\llm_local\workflow.py
```

## How to swap in a real backend

`DummyBackend` implements the same `infer(...)` interface the original
`LlamaServerManager` exposed:

```python
def infer(self, prompt, reset=True, system_prompt="", temperature=0.7, reasoning="off") -> tuple[str, str]
```

To use a real model: implement a manager with that `infer` signature (e.g. wrap
`llama-server`, Ollama, or any OpenAI-compatible endpoint) and wire it into
`llm_server.run_server(...)`. The client, the custom provider, and both pipeline layers
are transport-agnostic — they only speak the WebSocket protocol.

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
| `llm_server.py` | WebSocket server (protocol, sessions, heartbeats, queue back-pressure). Adapted from `C:\dev\Wander-Memory\src\llm\llm_server.py` (Apache-2.0). |
| `llm_client.py` | Persistent event loop + connection pool + sync `prompt_func`. Adapted from `C:\dev\Wander-Memory\src\llm\llm_client.py` (Apache-2.0). |
| `provider.py` | MemOS custom provider: `LocalLLMConfig`, `LocalLLM`, `register_local_llm()`. |
| `wander_pipeline.py` | Compact replica of the Wander-Memory flat-memory pipeline (extract → dedup → store → retrieve → summarize → maintain), stdlib only. |
| `workflow.py` | Orchestrated end-to-end demo (Scenarios 1–7). |

## Verification

Run from the repository root (`C:\dev\MemOS`):

```bash
python examples\llm_local\workflow.py
ruff check examples\llm_local
ruff format --check examples\llm_local
```

Sanity import (registration-before-validation):

```bash
python -c "import sys; sys.path.insert(0,'src'); from examples.llm_local.provider import register_local_llm; register_local_llm(); from memos.configs.llm import LLMConfigFactory; c=LLMConfigFactory.model_validate({'backend':'local','config':{'model_name_or_path':'dummy'}}); print(c.config)"
```

All three checks pass on the shipped code (ruff 0.16.x, Python 3.11).
