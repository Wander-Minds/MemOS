#!/usr/bin/env python3
"""Orchestrated end-to-end demo for the local LLM custom provider example.

Run from the repository root::

    python examples\\llm_local\\workflow.py

What it does:

1. Starts the local WebSocket LLM server with the deterministic dummy backend
   in a background thread (or attaches to an external server when the
   ``WM_WS_URL`` environment variable is set).
2. Registers the custom provider (``backend="local"``) at runtime via
   :func:`provider.register_local_llm`.
3. Runs seven scenarios:

   * Scenario 1 — custom provider registration + factory construction.
   * Scenario 2 — direct ``generate()`` with multi-turn reset semantics.
   * Scenario 3 — ``generate_stream()`` chunked output.
   * Scenario 4 — MemOS ``NaiveTextMemory`` whose extractor runs through the
     custom provider.
   * Scenario 5 — MemOS ``MemChat`` (simple backend, textual memory off),
     one scripted non-interactive turn.
   * Scenario 6 — MemOS ``MOS`` (guarded: needs a vector-DB/embedder-backed
     textual memory stack; skips gracefully when unavailable).
   * Scenario 7 — the Wander-style flat-memory pipeline (extract → dedup →
     store → retrieve → summarize → maintain) over the same WebSocket
     transport.

No external services, model downloads, or API keys are required.
"""

from __future__ import annotations

import asyncio
import builtins
import os
import tempfile
import threading
import traceback

from contextlib import contextmanager
from typing import TYPE_CHECKING

from llm_server import DEFAULT_WS_PORT, run_server
from provider import register_local_llm
from wander_pipeline import WanderConfig, WanderPipeline

from memos.configs.llm import LLMConfigFactory
from memos.configs.mem_chat import MemChatConfigFactory
from memos.configs.mem_os import MOSConfig
from memos.configs.memory import MemoryConfigFactory
from memos.llms.factory import LLMFactory
from memos.mem_chat.factory import MemChatFactory
from memos.mem_os.main import MOS
from memos.memories.factory import MemoryFactory


if TYPE_CHECKING:
    from collections.abc import Generator


def _default_ws_url(port: int) -> str:
    return f"ws://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# In-process dummy server lifecycle
# ---------------------------------------------------------------------------


def _wait_server_ready(ready_event: threading.Event, timeout: float = 20.0) -> None:
    """Block until the server signals it is accepting connections."""
    if not ready_event.wait(timeout):
        raise TimeoutError("local LLM server did not become ready in time")


@contextmanager
def start_dummy_server(port: int = DEFAULT_WS_PORT) -> Generator[None, None, None]:
    """Start the dummy WebSocket server in a background thread.

    When ``WM_WS_URL`` is set, no server is started — the workflow attaches to
    the externally running server instead.
    """
    external = os.environ.get("WM_WS_URL")
    if external:
        print(f"[server] WM_WS_URL set — attaching to external server: {external}")
        yield
        return

    loop = asyncio.new_event_loop()
    stop_box: dict[str, asyncio.Event] = {}
    ready_event = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        stop_event = asyncio.Event()
        stop_box["event"] = stop_event
        try:
            loop.run_until_complete(
                run_server(
                    ws_host="127.0.0.1",
                    ws_port=port,
                    backend="dummy",
                    stop_event=stop_event,
                    ready_event=ready_event,
                )
            )
        except Exception:
            traceback.print_exc()

    thread = threading.Thread(target=_run, daemon=True, name="llm-local-server")
    thread.start()
    print(f"[server] starting dummy backend on ws://127.0.0.1:{port} ...")
    _wait_server_ready(ready_event)
    print(f"[server] ready (dummy backend, port {port})")
    try:
        yield
    finally:
        if stop_box.get("event") is not None:
            loop.call_soon_threadsafe(stop_box["event"].set)
        thread.join(timeout=10)
        print("[server] stopped")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_1_provider_registration(port: int) -> LLMFactory:
    """Register the custom provider and build an LLM through the factory."""
    print("=" * 70)
    print("Scenario 1 — custom provider registration + factory")
    print("=" * 70)
    register_local_llm()  # must run before any LLMConfigFactory.model_validate
    config = LLMConfigFactory.model_validate(
        {
            "backend": "local",
            "config": {
                "model_name_or_path": "local-dummy",
                "ws_url": _default_ws_url(port),
                "temperature": 0.7,
                "max_tokens": 1024,
                "reasoning": "on",
            },
        }
    )
    llm = LLMFactory.from_config(config)
    print(f"registered backend 'local' -> config={type(config.config).__name__}")
    print(
        f"factory returned instance: {type(llm).__name__} (config={config.config.model_name_or_path})"
    )
    return llm


def scenario_2_generate(port: int, llm: LLMFactory) -> None:
    """Direct generate() calls with multi-turn reset semantics."""
    print()
    print("=" * 70)
    print("Scenario 2 — direct generate() (multi-turn reset semantics)")
    print("=" * 70)

    print("-- reset_each_turn=True (default): each call is a fresh session --")
    print("turn 1:", llm.generate([{"role": "user", "content": "你好，我叫小明。"}]))
    print("turn 2:", llm.generate([{"role": "user", "content": "我下周要去北京出差。"}]))

    print("-- reset_each_turn=False: server-side history accumulates --")
    hist_config = LLMConfigFactory.model_validate(
        {
            "backend": "local",
            "config": {
                "model_name_or_path": "local-dummy",
                "ws_url": _default_ws_url(port),
                "reset_each_turn": False,
            },
        }
    )
    hist_llm = LLMFactory.from_config(hist_config)
    print("history turn 1:", hist_llm.generate([{"role": "user", "content": "第一轮：你好"}]))
    print(
        "history turn 2:",
        hist_llm.generate([{"role": "user", "content": "第二轮：我下周要去北京"}]),
    )
    print(
        "history turn 3:", hist_llm.generate([{"role": "user", "content": "第三轮：记得我是小明"}])
    )
    print(
        "same config via factory is a singleton:",
        type(LLMFactory.from_config(hist_config)).__name__,
    )


def scenario_3_generate_stream(port: int, llm: LLMFactory) -> None:
    """Streaming (chunked) output through the same transport."""
    print()
    print("=" * 70)
    print("Scenario 3 — generate_stream() (chunked demo output)")
    print("=" * 70)
    chunks: list[str] = []
    for chunk in llm.generate_stream(
        [{"role": "user", "content": "请用一段话介绍这个本地LLM示例。"}],
        temperature=0.7,
    ):
        chunks.append(chunk)
        print(f"  chunk: {chunk!r}")
    print("joined:", "".join(chunks))


def scenario_4_naive_text_memory(port: int) -> None:
    """MemOS NaiveTextMemory whose extractor runs through the local provider."""
    print()
    print("=" * 70)
    print("Scenario 4 — NaiveTextMemory via the custom provider")
    print("=" * 70)
    mem_config = MemoryConfigFactory(
        backend="naive_text",
        config={
            "extractor_llm": {
                "backend": "local",
                "config": {
                    "model_name_or_path": "local-dummy",
                    "ws_url": _default_ws_url(port),
                    "temperature": 0.0,
                },
            }
        },
    )
    mem = MemoryFactory.from_config(mem_config)

    print("-- extract() runs the LLM extractor through the local provider --")
    messages = [
        {"role": "user", "content": "I plan to visit Paris next week. I love the Eiffel Tower."},
        {"role": "assistant", "content": "Paris is a beautiful city with many attractions."},
    ]
    extracted = mem.extract(messages)
    for item in extracted:
        print(f"  extracted: {item.memory} | {item.metadata.model_dump()}")

    print("-- add() extracted + manual memories --")
    mem.add(extracted)
    mem.add(
        [
            {"memory": "MemOS is awesome!", "metadata": {"type": "opinion"}},
            {"memory": "User is Chinese.", "metadata": {"type": "opinion"}},
        ]
    )
    print("all memories:", [m.memory for m in mem.get_all()])

    print("-- search() (word-overlap matcher; English text matches) --")
    print("search 'MemOS':", [m.memory for m in mem.search("MemOS", top_k=2)])
    print("search 'user':", [m.memory for m in mem.search("user", top_k=2)])


def scenario_5_mem_chat(port: int) -> None:
    """MemChat (simple backend, textual memory off), one scripted turn."""
    print()
    print("=" * 70)
    print("Scenario 5 — MemChat (simple backend, textual memory off)")
    print("=" * 70)
    chat_config = MemChatConfigFactory.model_validate(
        {
            "backend": "simple",
            "config": {
                "user_id": "llm_local_demo",
                "chat_llm": {
                    "backend": "local",
                    "config": {
                        "model_name_or_path": "local-dummy",
                        "ws_url": _default_ws_url(port),
                        "temperature": 0.7,
                    },
                },
                "max_turns_window": 10,
                "top_k": 5,
                "enable_textual_memory": False,
                "enable_activation_memory": False,
                "enable_parametric_memory": False,
            },
        }
    )
    mem_chat = MemChatFactory.from_config(chat_config)

    print("-- running one scripted turn, then 'bye' (non-interactive) --")
    scripted_inputs = iter(["你好，请介绍一下你自己", "bye"])
    original_input = builtins.input
    builtins.input = lambda *args, **kwargs: next(scripted_inputs)  # type: ignore[assignment]
    try:
        mem_chat.run()
    finally:
        builtins.input = original_input


def scenario_6_mos(port: int) -> None:
    """MOS (guarded): ``MOS.chat`` requires a registered ``cube_root`` mem cube.

    The ``cube_root`` mem cube is the vector-DB/embedder-backed textual-memory
    stack, which needs external services that this example deliberately does
    not require — so the scenario constructs MOS with the local provider and
    skips gracefully (falling back to Scenarios 4/5) when the cube is missing.
    """
    print()
    print("=" * 70)
    print("Scenario 6 — MOS (optional; guarded)")
    print("=" * 70)
    try:
        mos_config = MOSConfig(
            user_id="root",
            chat_model={
                "backend": "local",
                "config": {
                    "model_name_or_path": "local-dummy",
                    "ws_url": _default_ws_url(port),
                    "temperature": 0.7,
                },
            },
            mem_reader={
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "local",
                        "config": {
                            "model_name_or_path": "local-dummy",
                            "ws_url": _default_ws_url(port),
                            "temperature": 0.0,
                        },
                    },
                    "embedder": {
                        "backend": "sentence_transformer",
                        "config": {"model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2"},
                    },
                    "chunker": {"backend": "sentence", "config": {}},
                },
            },
            enable_textual_memory=False,
            enable_activation_memory=False,
            enable_parametric_memory=False,
            enable_preference_memory=False,
            enable_mem_scheduler=False,
        )
        mos = MOS(mos_config)
        reply = mos.chat("你好，你是谁？")
        print("MOS chat:", reply)
    except Exception as exc:
        print(
            "Scenario 6 skipped: MOS.chat requires a registered 'cube_root' "
            "mem cube (vector-DB/embedder-backed textual-memory stack) which "
            f"needs external services not available on this hardware "
            f"({type(exc).__name__}: {exc})."
        )
        print("Falling back to Scenarios 4/5 for the MemOS integration demo.")


def scenario_7_wander_pipeline(port: int) -> None:
    """Wander-style flat-memory pipeline over the same WebSocket transport."""
    print()
    print("=" * 70)
    print("Scenario 7 — Wander-style flat-memory pipeline")
    print("=" * 70)
    dialogue = (
        "用户: 我下周要去北京出差，大概待三天。\n"
        "助手: 好的，祝您出差顺利！\n"
        "用户: 另外我吃花生会过敏，请帮我记住这一点。"
    )
    with tempfile.TemporaryDirectory(prefix="llm_local_") as tmpdir:
        pipeline = WanderPipeline(
            os.path.join(tmpdir, "memories.db"),
            WanderConfig(ws_url=_default_ws_url(port)),
        )
        try:
            print("-- extract (LLM via prompt_func) --")
            candidates = pipeline.extract(dialogue)
            for cand in candidates:
                print(
                    f"  {cand['text']} (importance={cand['importance']}, entities={cand['entities']})"
                )

            print("-- dedup + store (ADD/UPDATE/DELETE/NOOP per candidate) --")
            results = pipeline.run_write(dialogue)
            for res in results:
                mem = res.get("memory") or {}
                print(f"  {res['action']} -> {mem.get('text', res.get('detail', ''))}")

            print(f"-- retrieve (BM25-ish) from {pipeline.store.count()} stored memories --")
            for mem in pipeline.retrieve("用户对什么过敏", top_k=3):
                print(f"  {mem['text']} (score above 0, importance={mem['importance']})")

            print("-- summarize (LLM) into an injectable context block --")
            context = pipeline.build_context("用户对什么过敏")
            print("  " + context.replace("\n", "\n  "))

            print("-- maintain (merge/decay/archive, LLM merge) --")
            report = pipeline.maintain()
            print(f"  {report}")
        finally:
            pipeline.close()
    print("  (scratch sqlite db cleaned up with the temp dir)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    port = int(os.environ.get("WM_WS_PORT", str(DEFAULT_WS_PORT)))
    print(f"[workflow] local LLM custom provider example (port {port})")
    with start_dummy_server(port):
        llm = scenario_1_provider_registration(port)
        scenario_2_generate(port, llm)
        scenario_3_generate_stream(port, llm)
        scenario_4_naive_text_memory(port)
        scenario_5_mem_chat(port)
        scenario_6_mos(port)
        scenario_7_wander_pipeline(port)
    print()
    print("All scenarios completed.")


if __name__ == "__main__":
    main()
