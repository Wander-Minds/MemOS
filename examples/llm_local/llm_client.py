#!/usr/bin/env python3
"""LLM client: WebSocket transport and client abstraction layer.

Adapted from ``C:\\dev\\Wander-Memory\\src\\llm\\llm_client.py`` (Apache-2.0).

This module combines two layers:

1. A single-turn inference client for the WebSocket server
   (:mod:`llm_server`).  It maintains a persistent event loop in a daemon
   thread with a WebSocket connection pool, eliminating connection
   setup/teardown overhead per call.
2. A common interface (``LLMClient``) that the concrete WebSocket client
   implements, so that callers (e.g. :mod:`provider` and
   :mod:`wander_pipeline`) can operate without knowing which backend is
   active.  Use :func:`create_llm_client` to build a client by type name.

The protocol and semantics are identical to the original; the concrete client
was renamed ``QwenClient`` -> :class:`LocalLLMClient` and the factory now
builds ``"local"`` clients against the example server.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import threading

from abc import ABC, abstractmethod
from asyncio import Queue
from pathlib import Path


try:  # AGENTS.md: guard optional third-party deps with a clear install message
    import websockets
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "The 'websockets' package is required by this example. "
        "Install it with: pip install websockets"
    ) from exc

# websockets >= 13 renamed WebSocketClientProtocol -> ClientConnection; keep a
# version-tolerant alias for the pool's type annotations.
try:
    from websockets.asyncio.client import ClientConnection as _WSConnection
except ImportError:  # pragma: no cover - websockets < 13
    from websockets import WebSocketClientProtocol as _WSConnection  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

DEFAULT_WS_URL = "ws://127.0.0.1:18081"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_REASONING = "off"
DEFAULT_POOL_MAXSIZE = 1

# ---------------------------------------------------------------------------
# Persistent event loop + connection pool
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()
_loop_thread: threading.Thread | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    """Get or create the persistent event loop (runs in a daemon thread)."""
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        return _loop

    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()

        def _run_loop() -> None:
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        _loop_thread = threading.Thread(target=_run_loop, daemon=True, name="llm-loop")
        _loop_thread.start()
        return _loop


class ConnectionPool:
    """A per-URL pool of reusable WebSocket connections.

    Connections are long-lived and multiplexed across calls.  When
    ``reset=True``, any idle connection can be reused.  When
    ``reset=False``, the conversation state must be preserved, so the
    caller must keep using the same connection.
    """

    def __init__(self, maxsize: int = DEFAULT_POOL_MAXSIZE) -> None:
        self._maxsize = maxsize
        self._pools: dict[str, Queue[_WSConnection]] = {}
        self._in_use: dict[str, int] = {}  # ws_url -> count of borrowed conns
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        ws_url: str,
        *,
        reset: bool = True,
        ping_interval: float = 60,
        ping_timeout: float = 600,
        close_timeout: float = 600,
    ) -> _WSConnection:
        """Get a connection from the pool (or create one)."""
        async with self._lock:
            if ws_url not in self._pools:
                self._pools[ws_url] = Queue(maxsize=self._maxsize)
                self._in_use[ws_url] = 0

            # If reset=True, try to reuse an idle connection
            if reset and not self._pools[ws_url].empty():
                conn = self._pools[ws_url].get_nowait()
                self._in_use[ws_url] += 1
                return conn

            # Check if we can create a new connection
            in_use = self._in_use.get(ws_url, 0)
            pool_size = self._pools[ws_url].qsize()
            total = in_use + pool_size

            if total < self._maxsize:
                conn = await websockets.connect(
                    ws_url,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    close_timeout=close_timeout,
                )
                self._in_use[ws_url] += 1
                return conn

        # All connections are busy — wait for one to come back
        # We release the lock while waiting to avoid deadlock.
        conn = await self._pools[ws_url].get()
        async with self._lock:
            self._in_use[ws_url] += 1
        return conn

    async def release(
        self,
        ws_url: str,
        conn: _WSConnection,
        *,
        reset: bool = True,
    ) -> None:
        """Return a connection to the pool (or close it)."""
        async with self._lock:
            self._in_use[ws_url] = max(0, self._in_use.get(ws_url, 0) - 1)

            if reset and not self._pools[ws_url].full():
                # Connection is clean — put it back
                await self._pools[ws_url].put(conn)
                return

        # Either conversation state must persist (reset=False) or pool is full
        with contextlib.suppress(Exception):
            await conn.close()

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            for queue in self._pools.values():
                while not queue.empty():
                    conn = queue.get_nowait()
                    with contextlib.suppress(Exception):
                        await conn.close()
            self._pools.clear()
            self._in_use.clear()


# Module-level pool singleton
_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool(maxsize: int = DEFAULT_POOL_MAXSIZE) -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(maxsize=maxsize)
    return _pool


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def send_prompt(
    ws_url: str,
    prompt_text: str,
    reset: bool = True,
    system_prompt: str = "",
    timeout: float | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    reasoning: str = DEFAULT_REASONING,
    pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
) -> tuple[str, str]:
    """Send a prompt to the server over WebSocket and wait for the result.

    Uses the global connection pool to avoid connection setup/teardown
    overhead on each call.

    Parameters
    ----------
    ws_url : str
        WebSocket URL of the server.
    prompt_text : str
        The prompt to send.
    reset : bool
        Whether to reset the conversation before this turn.
    system_prompt : str
        System prompt to set when ``reset=True``. Ignored when
        ``reset=False``.
    timeout : float or None
        Maximum total time to wait for the final response (including
        inference). Heartbeat messages from the server are ignored and
        do not reset this timer. ``None`` means wait indefinitely.
    temperature : float
        Sampling temperature passed to the model server.
    reasoning : str
        Reasoning mode passed to the model server: ``on``, ``off``, or
        ``auto``.
    pool_maxsize : int
        Maximum number of WebSocket connections in the pool per URL.
    """
    pool = _get_pool(maxsize=pool_maxsize)
    conn = await pool.acquire(ws_url, reset=reset)

    try:
        payload: dict = {
            "prompt": prompt_text,
            "reset": reset,
            "temperature": temperature,
            "reasoning": reasoning,
        }
        if reset:
            payload["system_prompt"] = system_prompt
        await conn.send(json.dumps(payload))

        while True:
            try:
                response = await asyncio.wait_for(conn.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError("Request timed out waiting for server response") from None

            payload = json.loads(response)

            # Skip heartbeat messages
            if payload.get("type") == "heartbeat":
                continue

            if "error" in payload:
                raise RuntimeError(payload["error"])

            return payload.get("reasoning", ""), payload.get("content", "")
    finally:
        await pool.release(ws_url, conn, reset=reset)


# ---------------------------------------------------------------------------
# Synchronous wrapper (used by LLMClient implementations and CLI)
# ---------------------------------------------------------------------------


def prompt_func(
    prompt_str: str,
    ws_url: str = DEFAULT_WS_URL,
    reset: bool = True,
    system_prompt: str = "",
    timeout: float | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    reasoning: str = DEFAULT_REASONING,
    pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
) -> str:
    """Send a prompt and return the model's content output (no reasoning).

    Runs on the persistent event loop so the connection pool can be
    reused across calls.

    Parameters
    ----------
    prompt_str : str
        The prompt text.
    ws_url : str
        WebSocket URL of the server.
    reset : bool
        Whether to reset the conversation before this turn.
    system_prompt : str
        System prompt to set when ``reset=True``. Ignored when
        ``reset=False``.
    timeout : float or None
        Maximum total time to wait for the final response (including
        inference). ``None`` means wait indefinitely.
    temperature : float
        Sampling temperature passed to the model server.
    reasoning : str
        Reasoning mode passed to the model server: ``on``, ``off``, or
        ``auto``.
    pool_maxsize : int
        Maximum number of WebSocket connections in the pool per URL.
    """
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(
        send_prompt(
            ws_url,
            prompt_str,
            reset=reset,
            system_prompt=system_prompt,
            timeout=timeout,
            temperature=temperature,
            reasoning=reasoning,
            pool_maxsize=pool_maxsize,
        ),
        loop,
    )
    reasoning, content = future.result()
    return content


# ---------------------------------------------------------------------------
# Client abstraction layer
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract base class for LLM clients.

    Subclasses must implement :meth:`prompt`.
    """

    @abstractmethod
    def prompt(
        self,
        prompt_str: str,
        system_prompt: str = "",
        *,
        reset: bool = True,
        reasoning: str = "off",
        temperature: float | None = None,
    ) -> str:
        """Send a prompt to the LLM and return the text response.

        Parameters
        ----------
        prompt_str : str
            The user prompt / article text.
        system_prompt : str
            Optional system-level instruction.
        reset : bool
            If True, reset the model's conversation context before this turn.
        reasoning : str
            Reasoning mode passed to the model server: ``on``, ``off``, or
            ``auto``. The workflows always use ``"off"``.
        temperature : float | None
            Optional sampling temperature. ``None`` means use the client default.

        Returns
        -------
        str
            The LLM's text output.
        """
        ...


class LocalLLMClient(LLMClient):
    """Concrete WebSocket client for the local LLM server.

    Wraps the WebSocket transport :func:`prompt_func` (renamed from the
    original ``QwenClient``; the transport is backend-agnostic).
    """

    def __init__(
        self,
        ws_url: str = DEFAULT_WS_URL,
        pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
        max_context_tokens: int = 131072,
    ) -> None:
        self.ws_url = ws_url
        self.pool_maxsize = pool_maxsize
        self.max_context_tokens = max_context_tokens

    def prompt(
        self,
        prompt_str: str,
        system_prompt: str = "",
        *,
        reset: bool = True,
        reasoning: str = DEFAULT_REASONING,
        temperature: float | None = None,
    ) -> str:
        return prompt_func(
            prompt_str=prompt_str,
            system_prompt=system_prompt,
            ws_url=self.ws_url,
            reset=reset,
            reasoning=reasoning,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            pool_maxsize=self.pool_maxsize,
        )


def create_llm_client(client_type: str = "local", **kwargs) -> LLMClient:
    """Factory: create an :class:`LLMClient` by type name.

    Parameters
    ----------
    client_type : str
        Must be ``"local"`` (the only supported client in this example).
    **kwargs
        Passed to the client constructor (e.g. ``ws_url``, ``pool_maxsize``,
        ``max_context_tokens``).

    Returns
    -------
    LLMClient

    Raises
    ------
    ValueError
        If *client_type* is not ``"local"``.
    """
    if client_type == "local":
        return LocalLLMClient(**kwargs)
    raise ValueError(f"Unknown client type: {client_type!r}. Use 'local'.")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-turn inference client for the LLM WebSocket server."
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        required=True,
        help="Path to a text file containing the prompt.",
    )
    parser.add_argument(
        "--ws-url",
        default=DEFAULT_WS_URL,
        help="WebSocket URL of the LLM server (default: ws://127.0.0.1:18081)",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="System prompt (only used when --reset is true)",
    )
    parser.add_argument(
        "--reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset conversation (default: true)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=("Maximum total seconds to wait for the server response (default: wait indefinitely)"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--reasoning",
        choices=["on", "off", "auto"],
        default=DEFAULT_REASONING,
        help=f"Reasoning mode (default: {DEFAULT_REASONING})",
    )
    args = parser.parse_args()

    if not args.prompt.exists():
        print(f"Prompt file not found: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    prompt_text = args.prompt.read_text(encoding="utf-8")

    try:
        reasoning_text, content_text = asyncio.run(
            send_prompt(
                args.ws_url,
                prompt_text,
                reset=args.reset,
                system_prompt=args.system_prompt,
                timeout=args.timeout,
                temperature=args.temperature,
                reasoning=args.reasoning,
            )
        )
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)
    if reasoning_text.strip():
        print("[Thinking]")
        print(reasoning_text)
        print("[/Thinking]")
        print()
    print(content_text)


if __name__ == "__main__":
    main()
