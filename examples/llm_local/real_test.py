#!/usr/bin/env python3
"""Real llama.cpp backend end-to-end test: server + client.

Unlike the deterministic dummy workflow, this script runs **real inference**
with the model shipped in ``D:\\MemOS\\models`` (git-ignored): it launches the
``llama-server`` binary built from ``ext/llama.cpp`` (``ext/build-<platform>``)
and verifies both halves of the stack:

* **server** — llama-server ``/health``, a direct OpenAI-compatible streaming
  chat request (``reasoning_content`` vs ``content``), and the WebSocket
  server (:mod:`llm_server` with ``backend="llama"``).
* **client** — :mod:`llm_client` single-turn and multi-turn (``reset=False``)
  round trips through the connection pool, plus the ``LocalLLMClient``.

Usage (from the repository root ``D:\\MemOS``)::

    python examples\\llm_local\\real_test.py                          # full test
    python examples\\llm_local\\real_test.py --model models\\Qwen3.5-4B
    python examples\\llm_local\\real_test.py --server-only            # start server, keep running
    python examples\\llm_local\\real_test.py --client-only            # attach to a running server
    python examples\\llm_local\\real_test.py --client-only --ws-url ws://127.0.0.1:18081
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import threading

from pathlib import Path


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _path in (str(_HERE), str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import llm_client as llm_client_module

from llama_backend import DEFAULT_BUILD_DIR, DEFAULT_MODEL, LlamaServerManager, chat_request
from llm_client import DEFAULT_WS_URL, LocalLLMClient, prompt_func, send_prompt
from llm_server import run_server


DEFAULT_LLAMA_PORT = 18080
DEFAULT_WS_PORT = 18081


# ---------------------------------------------------------------------------
# Server-side checks (direct HTTP against llama-server)
# ---------------------------------------------------------------------------


def server_http_checks(manager: LlamaServerManager) -> None:
    """Talk to llama-server over its OpenAI-compatible HTTP API directly."""
    print(f"[server] llama-server ready at {manager.base_url} (model: {manager.model})")

    reasoning_text, content_text = chat_request(
        manager.base_url,
        [{"role": "user", "content": "用一句话介绍你自己"}],
        temperature=0.7,
        reasoning="off",
    )
    print(f"[server] HTTP chat (reasoning=off) -> {content_text[:80]!r}...")
    assert content_text.strip(), "HTTP chat returned empty content"
    assert not reasoning_text, "reasoning=off should not produce reasoning content"

    reasoning_text, content_text = chat_request(
        manager.base_url,
        [{"role": "user", "content": "2+2=? 只回答数字。"}],
        temperature=0.0,
        reasoning="on",
    )
    print(
        f"[server] HTTP chat (reasoning=on) -> content={content_text[:40]!r} "
        f"reasoning_len={len(reasoning_text)}"
    )
    assert content_text.strip(), "HTTP chat (reasoning=on) returned empty content"
    print("[server] HTTP checks passed")


# ---------------------------------------------------------------------------
# Client-side checks (WebSocket transport via llm_client)
# ---------------------------------------------------------------------------


def _sync_client_checks(ws_url: str) -> None:
    """Sync-API checks (prompt_func / LocalLLMClient).

    Runs on the persistent ``llm-loop`` thread via :func:`prompt_func`, which
    is loop-bound: it must never be called directly from an async function
    (it would block the event loop that also serves the WebSocket server).
    """
    # The module-level ConnectionPool is loop-bound: drop it so the sync API
    # builds its own pool on the llm-loop thread.
    llm_client_module._pool = None  # type: ignore[attr-defined]
    content3 = prompt_func(
        "请用一句话介绍你自己。",
        ws_url=ws_url,
        reset=True,
        system_prompt="你是一个乐于助人的中文助手。",
        timeout=300,
    )
    print(f"[client] prompt_func(reset=True) -> {content3[:80]!r}...")
    assert content3.strip(), "prompt_func returned empty content"

    client = LocalLLMClient(ws_url=ws_url)
    content4 = client.prompt("你叫什么名字？", system_prompt="简短回答。", reset=True)
    print(f"[client] LocalLLMClient.prompt -> {content4[:80]!r}...")
    assert content4.strip(), "LocalLLMClient.prompt returned empty content"


async def client_checks(ws_url: str) -> None:
    """Exercise the WebSocket client against the example WS server."""
    # 1. Single turn (reset=True) with a system prompt.
    reasoning, content = await send_prompt(
        ws_url,
        "你好，请用一句话介绍你自己。",
        reset=True,
        system_prompt="你是一个乐于助人的中文助手。",
        timeout=300,
    )
    print(
        f"[client] send_prompt(reset=True) -> reasoning_len={len(reasoning)} "
        f"content={content[:80]!r}..."
    )
    assert content.strip(), "single-turn send_prompt returned empty content"

    # 2. Multi-turn: reset=False must keep the server-side conversation.
    reasoning, content2 = await send_prompt(
        ws_url,
        "我在上一轮问了什么？",
        reset=False,
        timeout=300,
    )
    print(f"[client] send_prompt(reset=False, follow-up) -> {content2[:80]!r}...")
    assert content2.strip(), "multi-turn send_prompt returned empty content"

    # 3. Sync wrappers in a worker thread (they run on the persistent
    #    llm-loop thread and would block this event loop otherwise).
    await asyncio.to_thread(_sync_client_checks, ws_url)

    print("[client] WebSocket client checks passed")


async def run_with_server(manager: LlamaServerManager, ws_port: int) -> None:
    """Run the WS server (reusing *manager*) and the client checks."""
    stop_event = asyncio.Event()
    ready_event = threading.Event()

    server_task = asyncio.create_task(
        run_server(
            ws_host="127.0.0.1",
            ws_port=ws_port,
            backend="llama",
            manager=manager,
            stop_event=stop_event,
            ready_event=ready_event,
        )
    )
    # Wait in a worker thread so the event loop keeps running the server task.
    if not await asyncio.to_thread(ready_event.wait, 30):
        if server_task.done():
            with contextlib.suppress(Exception):
                server_task.result()  # re-raise the real startup error
        raise RuntimeError("WebSocket server did not become ready in time")

    ws_url = f"ws://127.0.0.1:{ws_port}"
    print(f"[server] WebSocket server listening on {ws_url} (backend=llama)")
    try:
        await client_checks(ws_url)
    finally:
        stop_event.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(server_task, timeout=15)


async def run_client_only(ws_url: str) -> None:
    """Attach to an externally running server and run the client checks."""
    print(f"[client] attaching to external server at {ws_url}")
    await client_checks(ws_url)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Real llama.cpp backend end-to-end test (server + client). "
            "Defaults to the models/ GGUF files and the ext/build-<platform> binary."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["all", "server-only", "client-only"],
        default="all",
        help=(
            "'all' (default) starts llama-server + the WS server and runs "
            "server- and client-side checks; 'server-only' just starts the "
            "WS server (llama backend) and keeps it running; 'client-only' "
            "attaches to a running server and runs only the client checks."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=DEFAULT_MODEL,
        help=(f"Path to a .gguf file or a directory containing one (default: {DEFAULT_MODEL})"),
    )
    parser.add_argument(
        "--server",
        type=Path,
        default=None,
        help="Path to the llama-server binary (default: searched under --build-dir)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=f"llama.cpp build directory (default: {DEFAULT_BUILD_DIR})",
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
    parser.add_argument("--llama-port", type=int, default=DEFAULT_LLAMA_PORT)
    parser.add_argument("--ws-port", type=int, default=DEFAULT_WS_PORT)
    parser.add_argument(
        "--ws-url",
        default=DEFAULT_WS_URL,
        help=f"WebSocket URL for --mode client-only (default: {DEFAULT_WS_URL})",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress llama-server stdout/stderr output",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "client-only":
        asyncio.run(run_client_only(args.ws_url))
        return

    manager = LlamaServerManager(
        model=args.model,
        llama_port=args.llama_port,
        ngl=args.ngl,
        ctx=args.ctx,
        tokens=args.tokens,
        build_dir=args.build_dir,
        quiet=args.quiet,
        server_path=args.server,
    )
    try:
        manager.start()
        if args.mode == "server-only":
            print(
                f"llama-server ready at {manager.base_url} — "
                f"WS server on ws://127.0.0.1:{args.ws_port} (backend=llama). "
                "Press Ctrl+C to stop.",
                file=sys.stderr,
            )
            with contextlib.suppress(KeyboardInterrupt):
                asyncio.run(
                    run_server(
                        ws_host="127.0.0.1",
                        ws_port=args.ws_port,
                        backend="llama",
                        manager=manager,
                    )
                )
        else:
            server_http_checks(manager)
            asyncio.run(run_with_server(manager, args.ws_port))
            print("\nREAL TEST PASSED ✔ (server + client, llama.cpp backend)")
    finally:
        manager.stop()
        print("[cleanup] llama-server stopped.")


if __name__ == "__main__":
    main()
