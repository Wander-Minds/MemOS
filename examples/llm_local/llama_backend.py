#!/usr/bin/env python3
"""Real llama.cpp inference backend (persistent ``llama-server`` subprocess).

Adapted from ``C:\\dev\\Wander-Memory\\src\\llm\\llm_server.py`` (Apache-2.0).

This module is the real counterpart of :mod:`dummy_backend`: it implements
the **exact same interface** — ::

    infer(prompt, reset=True, system_prompt="", temperature=0.7,
          reasoning="off") -> (reasoning_text, content_text)

but drives an actual local model through a persistent ``llama-server``
subprocess (built from ``ext/llama.cpp`` via ``ext/build_llama_cpp.py``; the
prebuilt binary lives in ``ext/build-windows/bin`` or ``ext/build-linux/bin``).
The WebSocket server (:mod:`llm_server`) picks this backend with
``--backend llama``; the client, the custom provider, and the pipeline code
are transport-agnostic and do not change.

Model & binary resolution:

* ``--model`` may point to a ``.gguf`` file or a directory (the first
  ``*.gguf`` inside is used).  Default: ``models/Qwen3.5-9B`` (models shipped
  in ``D:\\MemOS\\models``, git-ignored).
* ``--server`` may point at a ``llama-server`` binary directly; otherwise the
  binary is searched in ``--build-dir`` (default ``ext/build-<platform>``)
  and then on ``$PATH``.

The manager speaks the llama.cpp OpenAI-compatible HTTP API: it polls
``/health`` on startup and sends streaming ``/v1/chat/completions`` requests,
capturing ``reasoning_content`` (thinking) separately from ``content`` so the
WebSocket ``{"reasoning", "content"}`` protocol is preserved.
"""

from __future__ import annotations

import contextlib
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
# https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
DEFAULT_MODEL = REPO_ROOT / "models" / "Qwen3.5-9B"

DEFAULT_LLAMA_PORT = 18080


def _platform_tag() -> str:
    """Return the build-directory tag used by ``ext/build_llama_cpp.py``."""
    return {"darwin": "macos"}.get(platform.system().lower(), platform.system().lower())


DEFAULT_BUILD_DIR = REPO_ROOT / "ext" / f"build-{_platform_tag()}"


def _default_server_name() -> str:
    """Return the platform-appropriate llama-server executable name."""
    return "llama-server.exe" if platform.system() == "Windows" else "llama-server"


def _find_llama_server(build_dir: Path | None, server_path: Path | None = None) -> Path:
    """Locate the llama-server binary.

    Resolution order:
    1. If *server_path* is provided and exists → use it directly.
    2. If *server_path* is provided but does not exist → search ``$PATH``
       via ``shutil.which()`` (useful on Linux when the binary is installed
       system-wide).
    3. Otherwise, search in *build_dir* (or the default build directory).
    """
    # --- User-provided path ---
    if server_path is not None:
        exe = Path(server_path).resolve()
        if exe.exists():
            return exe
        resolved = shutil.which(server_path)
        if resolved is not None:
            return Path(resolved).resolve()
        print(f"llama-server not found at {exe}", file=sys.stderr)
        sys.exit(1)

    # --- Search in build directory ---
    directory = Path(build_dir).resolve() if build_dir else DEFAULT_BUILD_DIR
    exe_name = _default_server_name()
    candidates = [
        directory / exe_name,
        directory / "bin" / exe_name,
        directory / "bin" / "Release" / exe_name,
    ]
    for cli in candidates:
        if cli.exists():
            return cli

    # --- Last resort: try bare name on PATH (Linux/macOS) ---
    resolved = shutil.which(exe_name)
    if resolved is not None:
        return Path(resolved).resolve()

    print(
        f"llama-server not found under {directory} (searched {[str(c) for c in candidates]}). "
        f"Build it first with python ext\\build_llama_cpp.py (or pass --server <path>).",
        file=sys.stderr,
    )
    sys.exit(1)


def _resolve_model(model: Path) -> Path:
    """Resolve a model path to a usable GGUF file.

    If the path is a directory, the first .gguf file inside it is used.
    If the path is already a file, it is returned as-is.
    """
    path = Path(model).resolve()
    if path.is_dir():
        ggufs = sorted(path.glob("*.gguf"))
        if not ggufs:
            print(f"No .gguf model found in directory: {path}", file=sys.stderr)
            sys.exit(1)
        return ggufs[0]
    if not path.exists():
        print(f"Model not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def wait_for_server(
    base_url: str,
    proc: subprocess.Popen | None = None,
    timeout: float = 600.0,
) -> None:
    """Poll /health until the llama-server is ready.

    If ``proc`` is provided and exits before becoming healthy, raise an
    error immediately. Any captured stderr is included in the message;
    otherwise the user is told to check the visible output above.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            stderr = ""
            if proc.stderr is not None and hasattr(proc.stderr, "read"):
                with contextlib.suppress(Exception):  # pragma: no cover - defensive
                    stderr = proc.stderr.read().decode("utf-8", errors="replace")
            if stderr:
                print(
                    f"llama-server exited early with code {proc.returncode}.\n{stderr}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"llama-server exited early with code {proc.returncode}. "
                    "See output above for details.",
                    file=sys.stderr,
                )
            sys.exit(1)
        try:
            resp = urllib.request.urlopen(f"{base_url}/health", timeout=2)
            if resp.status == 200:
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    print("Server did not become ready in time.", file=sys.stderr)
    sys.exit(1)


def chat_request(
    base_url: str,
    messages: list[dict],
    temperature: float,
    reasoning: str,
) -> tuple[str, str]:
    """Send a streaming chat request and return (reasoning_text, content_text)."""
    body = {
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": False},
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": reasoning == "on"},
        "cache_prompt": True,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    reasoning_text = ""
    content_text = ""
    with urllib.request.urlopen(req, timeout=7200) as resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw or raw == "data: [DONE]":
                    continue
                raw = raw.removeprefix("data: ")
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                reasoning_content = delta.get("reasoning_content")
                if reasoning_content:
                    reasoning_text += reasoning_content
                content = delta.get("content", "")
                if content:
                    content_text += content
    return reasoning_text, content_text


class LlamaServerManager:
    """Manages the lifecycle of a llama-server subprocess.

    Implements the same ``infer(...)`` / ``stop()`` surface as
    :class:`dummy_backend.DummyServerManager`, so the WebSocket server can
    swap backends without changing the protocol, worker, or client code.

    Cross-platform: works on both Windows and Linux.
    """

    def __init__(
        self,
        model: Path,
        llama_port: int = DEFAULT_LLAMA_PORT,
        ngl: int = 99,
        ctx: int = 32768,
        tokens: int = -1,
        build_dir: Path | None = None,
        chat_template: str | None = None,
        device: str | None = None,
        quiet: bool = False,
        reasoning: bool = False,
        server_path: Path | None = None,
    ) -> None:
        self.model = _resolve_model(model)
        self.llama_port = llama_port
        self.ngl = ngl
        self.ctx = ctx
        self.tokens = tokens
        self.build_dir = build_dir
        self.chat_template = chat_template
        self.device = device
        self.quiet = quiet
        self.reasoning = reasoning
        self.server_path = server_path
        self.proc: subprocess.Popen | None = None
        self.base_url = f"http://127.0.0.1:{llama_port}"
        self.messages: list[dict] = []

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Launch llama-server and wait until /health reports ready.

        Idempotent: calling ``start()`` on an already-running manager is a
        no-op (useful when the server is started by the test harness and then
        handed to the WebSocket server via ``run_server(manager=...)``).
        """
        if self.proc is not None and self.proc.poll() is None:
            return

        server_bin = _find_llama_server(self.build_dir, self.server_path)

        cmd = [
            str(server_bin),
            "-m",
            str(self.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.llama_port),
            "-ngl",
            str(self.ngl),
            "-c",
            str(self.ctx),
            "-n",
            str(self.tokens),
            "--no-webui",
        ]
        if self.device is not None:
            cmd += ["--device", self.device]
        if self.chat_template:
            cmd += ["--chat-template", self.chat_template]
        cmd += ["--reasoning", "on" if self.reasoning else "off"]

        # Use shell=False on all platforms (list form is preferred)
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL if self.quiet else None,
            stderr=subprocess.PIPE if self.quiet else None,
        )
        wait_for_server(self.base_url, proc=self.proc)

    def stop(self) -> None:
        """Terminate the llama-server subprocess (no-op if not running)."""
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None

    def infer(
        self,
        prompt: str,
        reset: bool = True,
        system_prompt: str = "",
        temperature: float = 0.7,
        reasoning: str = "off",
    ) -> tuple[str, str]:
        """Send one prompt to the model and return ``(reasoning, content)``.

        Parameters mirror :meth:`DummyBackend.infer` exactly.  ``reset=True``
        starts a fresh conversation (with the system prompt); ``reset=False``
        appends to the server-side history (multi-turn).
        """
        if not self.reasoning:
            reasoning = "off"
        if reset:
            self.messages = [{"role": "system", "content": system_prompt}]
            self.messages.append({"role": "user", "content": prompt})
        else:
            if not self.messages:
                self.messages = [{"role": "system", "content": system_prompt}]
            self.messages.append({"role": "user", "content": prompt})
        reasoning_text, content_text = chat_request(
            self.base_url,
            self.messages,
            temperature,
            reasoning,
        )

        self.messages.append({"role": "assistant", "content": content_text})
        return reasoning_text, content_text
