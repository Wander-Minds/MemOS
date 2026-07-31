#!/usr/bin/env python3
"""Deterministic dummy inference backend for the local LLM WebSocket server.

The current development hardware cannot run real LLM inference (no
``llama-server`` binary, no usable GPU backend).  This module provides a
pure-Python, rule-based stand-in that implements the **exact same interface**
the original ``LlamaServerManager`` exposed::

    infer(prompt, reset=True, system_prompt="", temperature=0.7,
          reasoning="off") -> (reasoning_text, content_text)

The WebSocket protocol (sessions, heartbeats, queue back-pressure) and the
connection pooling in :mod:`llm_client` are preserved by the server, so a real
backend can be plugged in later by implementing the same ``infer`` signature
(see the example README, "How to swap in a real backend").

Response rules (all deterministic):

* ``reset=True`` clears the in-memory conversation history; ``reset=False``
  keeps it (mirrors the real server's session semantics).
* ``reasoning="on"`` prepends a canned ``<think>...</think>`` block to the
  *reasoning* output (the *content* output is unaffected).
* Management prompts (extraction / dedup / merge / summarize) return valid
  canned JSON (or the summarize plain-text block) matching each prompt
  contract, so the memory pipelines parse cleanly.
* Anything else gets an echo-style assistant reply that references the
  conversation turn count when history is present.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Canned deterministic responses (per prompt contract)
# ---------------------------------------------------------------------------

# MemOS NaiveTextMemory extraction contract (English prompt in
# ``memos.memories.textual.naive``: "You are a memory extractor ...").
CANNED_MEMOS_EXTRACT = """\
[
  {"memory": "用户喜欢本地大语言模型，并且正在使用MemOS记忆系统。", "metadata": {"type": "opinion"}},
  {"memory": "用户对花生过敏。", "metadata": {"type": "fact"}}
]"""

# Wander-Memory extraction contract
# (``{"text": ..., "importance": 0~1, "entities": [...]}``).
CANNED_WANDER_EXTRACT = """\
[
  {"text": "用户对花生过敏", "importance": 0.9, "entities": ["花生", "用户"]},
  {"text": "用户下周要去北京出差", "importance": 0.7, "entities": ["北京", "用户"]}
]"""

# Dedup decision contract (ADD / UPDATE / DELETE / NOOP).
CANNED_DEDUP = """\
{"action": "ADD"}"""

# Merge contract.
CANNED_MERGE = """\
{"text": "用户对花生过敏，外出时需要携带抗过敏药物", "importance": 0.8, "entities": ["花生", "用户"]}"""

# Summarize contract (plain text, one "- " line per fact — NOT JSON).
CANNED_SUMMARIZE = """\
- 用户对花生过敏，需要注意饮食安全。
- 用户下周要去北京出差。"""

# Canned reasoning block emitted when ``reasoning == "on"``.
CANNED_REASONING = "<think>用户发来一条消息，我正在规划一个恰当的回复。</think>"


class DummyBackend:
    """Rule-based stand-in for a real local LLM.

    Implements the same ``infer(...)`` interface as the original
    ``LlamaServerManager`` so the server/worker code does not change when a
    real backend is plugged in later.
    """

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def infer(
        self,
        prompt: str,
        reset: bool = True,
        system_prompt: str = "",
        temperature: float = 0.7,
        reasoning: str = "off",
    ) -> tuple[str, str]:
        """Return a deterministic ``(reasoning_text, content_text)`` pair.

        Parameters mirror :meth:`LlamaServerManager.infer` exactly:
        ``temperature`` is accepted for interface compatibility and ignored
        (the dummy backend is fully deterministic).
        """
        if reset:
            self.messages = []
            if system_prompt:
                self.messages.append({"role": "system", "content": system_prompt})
        elif not self.messages and system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "user", "content": prompt})

        prompt_type = self._detect_prompt_type(prompt)
        content = self._respond(prompt_type, prompt)
        self.messages.append({"role": "assistant", "content": content})

        reasoning_text = CANNED_REASONING if reasoning == "on" else ""
        return reasoning_text, content

    # ------------------------------------------------------------------
    def _detect_prompt_type(self, prompt: str) -> str:
        """Classify the prompt into a response contract.

        Order matters: the MemOS extractor prompt is English ("You are a
        memory extractor"), the Wander-Memory management prompts are Chinese
        and share keywords such as "记忆" — so the more specific keywords are
        checked first.
        """
        lowered = prompt.lower()
        if "memory extractor" in lowered:
            return "memos_extract"
        if "去重" in prompt or "dedup" in lowered or "请判断" in prompt:
            return "dedup"
        if "合并" in prompt or "merge" in lowered:
            return "merge"
        if "总结" in prompt or "summarize" in lowered:
            return "summarize"
        if "抽取" in prompt or "记忆" in prompt or "extract" in lowered or "json" in lowered:
            return "wander_extract"
        return "echo"

    def _respond(self, prompt_type: str, prompt: str) -> str:
        if prompt_type == "memos_extract":
            return CANNED_MEMOS_EXTRACT
        if prompt_type == "wander_extract":
            return CANNED_WANDER_EXTRACT
        if prompt_type == "dedup":
            return CANNED_DEDUP
        if prompt_type == "merge":
            return CANNED_MERGE
        if prompt_type == "summarize":
            return CANNED_SUMMARIZE
        return self._echo(prompt)

    def _echo(self, prompt: str) -> str:
        """Echo-style assistant reply; references history when present."""
        # After appending the user message: [system?, user, (assistant,
        # user)*] — the number of completed turns is (len - 1) // 2.
        turn_index = (len(self.messages) - 1) // 2
        preview = prompt[:80] + ("…" if len(prompt) > 80 else "")
        reply = f"本地示例模型已收到你的消息：{preview}"
        if turn_index > 0:
            reply += f"（已结合此前 {turn_index} 轮对话上下文；当前是第 {turn_index + 1} 轮）"
        reply += " 这是确定性模拟回复，非真实模型推理。"
        return reply


class DummyServerManager:
    """Thin server-side manager wrapping a :class:`DummyBackend`.

    Replaces ``LlamaServerManager`` in the WebSocket server; exposes the same
    ``infer(...)`` / ``stop()`` surface so the request worker and the CLI do
    not change when a real backend is introduced.
    """

    def __init__(self, backend: DummyBackend | None = None, reasoning: bool = False) -> None:
        self.backend = backend if backend is not None else DummyBackend()
        self.reasoning = reasoning

    def infer(
        self,
        prompt: str,
        reset: bool = True,
        system_prompt: str = "",
        temperature: float = 0.7,
        reasoning: str = "off",
    ) -> tuple[str, str]:
        """Delegate to the backend, honoring the server-level reasoning flag."""
        if not self.reasoning:
            reasoning = "off"
        return self.backend.infer(
            prompt,
            reset=reset,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning=reasoning,
        )

    def stop(self) -> None:
        """Release resources (no-op for the pure-Python dummy backend)."""
