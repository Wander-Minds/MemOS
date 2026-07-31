#!/usr/bin/env python3
"""WebSocket server for the local LLM (dummy or llama.cpp backend).

Adapted from ``C:\\dev\\Wander-Memory\\src\\llm\\llm_server.py`` (Apache-2.0).

The original server wrapped a persistent ``llama-server`` subprocess; this
example keeps the exact same protocol and lets you choose the backend:

* ``--backend dummy`` (default) — :class:`dummy_backend.DummyServerManager`,
  a deterministic, rule-based stand-in that needs no model or GPU.
* ``--backend llama`` — :class:`llama_backend.LlamaServerManager`, a real
  llama.cpp backend: it launches the ``llama-server`` binary built from
  ``ext/llama.cpp`` and serves a local GGUF model from ``models/``.

The WebSocket protocol, session semantics, heartbeats, and queue
back-pressure are preserved exactly:

* client -> server: ``{"prompt": str, "reset": bool, "system_prompt": str,
  "temperature": float, "reasoning": "on"|"off"|"auto"}``
* server -> client (result): ``{"reasoning": str, "content": str}``
* server -> client (while queued): ``{"type": "heartbeat", "status": "queued"}``
* server -> client (errors): ``{"error": str}``

Run standalone::

    python examples\\llm_local\\llm_server.py --backend dummy --ws-port 18081
    python examples\\llm_local\\llm_server.py --backend llama --ws-port 18081

or start it in-process from :mod:`workflow` via :func:`run_server`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import threading


from dummy_backend import DummyBackend, DummyServerManager


try:  # AGENTS.md: guard optional third-party deps with a clear install message
    import websockets
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "The 'websockets' package is required by this example. "
        "Install it with: pip install websockets"
    ) from exc

from llama_backend import (
    DEFAULT_LLAMA_PORT,
    DEFAULT_MODEL,
    LlamaServerManager,
)


DEFAULT_WS_PORT = 18081
DEFAULT_QUEUE_MAXSIZE = 4


@dataclass
class _RequestItem:
    """A queued inference request with a future for the response."""

    prompt: str
    reset: bool
    system_prompt: str
    temperature: float
    reasoning: str
    response_future: asyncio.Future[tuple[str, str]]


def wait_for_server(base_url: str, proc: object | None = None, timeout: float = 600.0) -> None:
    """No-op for the dummy backend (there is no HTTP health endpoint).

    Kept for provenance with the original llama-server health polling; a real
    backend implementation should restore the polling logic here.
    """


async def handle_client(websocket, request_queue: asyncio.Queue) -> None:
    """Handle a WebSocket client connection.

    Submits requests to a shared queue and waits for the result.
    If the queue is full, sends a "busy" error to the client.
    """
    try:
        async for message in websocket:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
                continue

            prompt = payload.get("prompt")
            if not isinstance(prompt, str):
                await websocket.send(json.dumps({"error": "Missing or invalid 'prompt' field"}))
                continue

            reset = payload.get("reset", True)
            system_prompt = payload.get("system_prompt", "")
            temperature = payload.get("temperature", 0.7)
            reasoning = payload.get("reasoning", "off")

            # Try to queue the request
            response_future: asyncio.Future[tuple[str, str]] = asyncio.Future()
            item = _RequestItem(
                prompt=prompt,
                reset=reset,
                system_prompt=system_prompt,
                temperature=temperature,
                reasoning=reasoning,
                response_future=response_future,
            )

            try:
                await asyncio.wait_for(request_queue.put(item), timeout=5.0)
            except asyncio.TimeoutError:
                await websocket.send(
                    json.dumps({"error": ("Server busy — request queue full. Please retry later.")})
                )
                continue

            # Send periodic heartbeats while waiting for inference to complete
            while not response_future.done():
                done, _ = await asyncio.wait(
                    [response_future],
                    timeout=30,  # heartbeat interval in seconds
                )
                if done:
                    break
                try:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "heartbeat",
                                "status": "queued",
                            }
                        )
                    )
                except websockets.exceptions.ConnectionClosed:
                    response_future.cancel()
                    return

            try:
                reasoning_text, content_text = await response_future
            except Exception as exc:
                await websocket.send(json.dumps({"error": str(exc)}))
                continue

            try:
                await websocket.send(
                    json.dumps(
                        {
                            "reasoning": reasoning_text,
                            "content": content_text,
                        }
                    )
                )
            except websockets.exceptions.ConnectionClosed:
                response_future.cancel()
                return
    except websockets.exceptions.ConnectionClosed:
        # Client disconnected abruptly (e.g. network dropped, tab closed).
        # This is expected; log nothing and release the connection.
        return


async def request_worker(
    manager: DummyServerManager | LlamaServerManager,
    request_queue: asyncio.Queue,
) -> None:
    """Background worker that processes queued inference requests one by one."""
    loop = asyncio.get_running_loop()
    while True:
        item: _RequestItem = await request_queue.get()
        try:
            reasoning_text, content_text = await loop.run_in_executor(
                None,
                manager.infer,
                item.prompt,
                item.reset,
                item.system_prompt,
                item.temperature,
                item.reasoning,
            )
            item.response_future.set_result((reasoning_text, content_text))
        except Exception as exc:
            if not item.response_future.done():
                item.response_future.set_exception(exc)
        finally:
            request_queue.task_done()


async def run_server(
    *,
    ws_host: str = "127.0.0.1",
    ws_port: int = DEFAULT_WS_PORT,
    queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    reasoning: bool = False,
    backend: str = "dummy",
    stop_event: asyncio.Event | None = None,
    ready_event: threading.Event | None = None,
    manager: DummyServerManager | LlamaServerManager | None = None,
    model: Path | None = None,
    server: Path | None = None,
    ngl: int = 99,
    ctx: int = 32768,
    tokens: int = -1,
    build_dir: Path | None = None,
    chat_template: str | None = None,
    device: str | None = None,
    quiet: bool = False,
    llama_port: int = DEFAULT_LLAMA_PORT,
) -> None:
    """Run the WebSocket server until stopped (or forever when no stop event).

    Parameters
    ----------
    ws_host, ws_port : str, int
        Bind address for the WebSocket server.
    queue_maxsize : int
        Max pending requests in the shared queue (back-pressure).
    reasoning : bool
        Enable reasoning/thinking output by default.
    backend : str
        ``"dummy"`` (default, deterministic rule-based) or ``"llama"``
        (real llama.cpp backend — launches the ``llama-server`` binary built
        from ``ext/llama.cpp`` and serves a GGUF model from ``models/``).
    stop_event : asyncio.Event or None
        When provided, the server runs until the event is set (used by
        :mod:`workflow` for in-process startup/shutdown).  ``None`` runs
        forever.
    ready_event : threading.Event or None
        When provided, it is set as soon as the WebSocket server is
        accepting connections (used by :mod:`workflow` to wait for startup
        without probing the port).
    manager : DummyServerManager, LlamaServerManager or None
        Pre-built backend manager.  When provided, the backend is used
        as-is (``start()`` is still called for the llama backend — it is a
        no-op when the subprocess is already running); otherwise a manager
        is constructed from the llama options below.
    model : Path or None
        GGUF file, or a directory containing one (default:
        ``models/Qwen3.5-9B`` — the first ``*.gguf`` inside is used).
    server : Path or None
        Path to the llama-server binary (default: searched under
        ``--build-dir`` and then on ``$PATH``).
    ngl : int
        GPU layers offloaded to the device (default: 99 = all).
    ctx : int
        Context size in tokens (default: 32768).
    tokens : int
        Max tokens to generate (default: -1 = unlimited).
    build_dir : Path or None
        llama.cpp build directory (default: ``ext/build-<platform>``).
    chat_template : str or None
        Optional ``--chat-template`` passed to llama-server.
    device : str or None
        Device passed to llama-server ``--device`` (e.g. CUDA0, Vulkan0;
        auto-detected if omitted).
    quiet : bool
        Suppress llama-server stdout/stderr output.
    llama_port : int
        Port for the llama-server HTTP API (default: 18080).
    """
    if backend == "llama":
        if manager is None:
            manager = LlamaServerManager(
                model=model if model is not None else DEFAULT_MODEL,
                llama_port=llama_port,
                ngl=ngl,
                ctx=ctx,
                tokens=tokens,
                build_dir=build_dir,
                chat_template=chat_template,
                device=device,
                quiet=quiet,
                reasoning=reasoning,
                server_path=server,
            )
        manager.start()  # no-op when the subprocess is already running
        print(f"llama-server ready at {manager.base_url}", file=sys.stderr, flush=True)
    else:
        if manager is None:
            manager = DummyServerManager(backend=DummyBackend(), reasoning=reasoning)

    # Shared request queue with back-pressure
    request_queue: asyncio.Queue[_RequestItem] = asyncio.Queue(maxsize=queue_maxsize)

    # Start the background worker
    worker_task = asyncio.create_task(request_worker(manager, request_queue))

    try:
        async with websockets.serve(
            lambda ws: handle_client(ws, request_queue),
            ws_host,
            ws_port,
        ):
            print(
                f"WebSocket server listening on ws://{ws_host}:{ws_port} (backend={backend})",
                file=sys.stderr,
                flush=True,
            )
            if ready_event is not None:
                ready_event.set()
            if stop_event is None:
                await asyncio.Future()  # run forever
            else:
                await stop_event.wait()
    finally:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        manager.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "WebSocket server for the local LLM example. "
            "Default backend is 'dummy' (deterministic, no model required)."
        )
    )
    parser.add_argument(
        "--ws-host",
        default="127.0.0.1",
        help="Host for the WebSocket server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=DEFAULT_WS_PORT,
        help=f"Port for the WebSocket server (default: {DEFAULT_WS_PORT})",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable model reasoning/thinking by default (default: disabled)",
    )
    parser.add_argument(
        "--queue-maxsize",
        type=int,
        default=DEFAULT_QUEUE_MAXSIZE,
        help=f"Max pending requests in queue (default: {DEFAULT_QUEUE_MAXSIZE})",
    )
    parser.add_argument(
        "--backend",
        choices=["dummy", "llama"],
        default="dummy",
        help=(
            "Inference backend: 'dummy' (default, deterministic, no model) or "
            "'llama' (real llama.cpp backend — runs the llama-server binary "
            "from ext/build-<platform> with a GGUF model from models/)."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=None,
        help=(
            "Path to a .gguf file or a directory containing one "
            "(default: models/Qwen3.5-9B; the first *.gguf in the dir is used)"
        ),
    )
    parser.add_argument(
        "--server",
        type=Path,
        default=None,
        help=(
            "Path to the llama-server binary (default: searched under "
            "--build-dir and then on $PATH)"
        ),
    )
    parser.add_argument("--ngl", "--gpu-layers", type=int, default=99)
    parser.add_argument(
        "--ctx",
        "--ctx-size",
        type=int,
        default=32768,
        help="Context size in tokens (default: 32768)",
    )
    parser.add_argument("--tokens", "-n", type=int, default=-1)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="llama.cpp build directory (default: ext/build-<platform>)",
    )
    parser.add_argument("--chat-template", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device passed to llama-server --device (e.g. CUDA0, Vulkan0; auto-detected if omitted)"
        ),
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress llama-server stdout/stderr output",
    )
    parser.add_argument(
        "--llama-port",
        type=int,
        default=DEFAULT_LLAMA_PORT,
        help="Port for llama-server HTTP API (default: 18080)",
    )
    args = parser.parse_args()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(
            run_server(
                ws_host=args.ws_host,
                ws_port=args.ws_port,
                queue_maxsize=args.queue_maxsize,
                reasoning=args.reasoning,
                backend=args.backend,
                model=args.model,
                server=args.server,
                ngl=args.ngl,
                ctx=args.ctx,
                tokens=args.tokens,
                build_dir=args.build_dir,
                chat_template=args.chat_template,
                device=args.device,
                quiet=args.quiet,
                llama_port=args.llama_port,
            )
        )


if __name__ == "__main__":
    main()
