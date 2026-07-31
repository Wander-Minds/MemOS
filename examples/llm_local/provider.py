#!/usr/bin/env python3
"""MemOS custom LLM provider: ``backend="local"``.

This module glues the copied local WebSocket client (:mod:`llm_client`) into
the MemOS LLM provider framework without touching ``src\\memos``: it defines a
pydantic config (:class:`LocalLLMConfig`), an implementation
(:class:`LocalLLM`), and :func:`register_local_llm` which performs the same
registration that AGENTS.md's "Adding a New Provider" steps do permanently
(``configs/llm.py`` + ``llms/factory.py``) — but at runtime, from example
code only.

IMPORTANT: call :func:`register_local_llm` **before** any
``LLMConfigFactory.model_validate({"backend": "local", ...})`` — the pydantic
``backend`` validator rejects unknown backends.
"""

from __future__ import annotations

import time

from typing import TYPE_CHECKING

from llm_client import create_llm_client
from pydantic import Field

from memos.configs.llm import BaseLLMConfig, LLMConfigFactory
from memos.llms.base import BaseLLM
from memos.llms.factory import LLMFactory
from memos.llms.utils import remove_thinking_tags
from memos.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Generator

    from memos.types import MessageList


logger = get_logger(__name__)

BACKEND_NAME = "local"
DEFAULT_WS_URL = "ws://127.0.0.1:18081"


class LocalLLMConfig(BaseLLMConfig):
    """Configuration for the local WebSocket-served LLM provider.

    Inherits the standard ``BaseLLMConfig`` fields (``model_name_or_path``,
    ``temperature``, ``max_tokens``, ``top_p``, ``top_k``,
    ``remove_think_prefix``) and adds the local WebSocket transport options.
    """

    ws_url: str = Field(
        default=DEFAULT_WS_URL,
        description="WebSocket URL of the local LLM server",
    )
    ws_pool_maxsize: int = Field(
        default=1,
        description="Max WebSocket connections in the client pool per URL",
    )
    reasoning: str = Field(
        default="off",
        description="Reasoning mode passed to the model server: 'on', 'off' or 'auto'",
    )
    reset_each_turn: bool = Field(
        default=True,
        description=(
            "Reset the server-side conversation before every call. "
            "True matches MemOS semantics (clients send the full context); "
            "False keeps server-side history across calls."
        ),
    )
    simulate_stream_delay: float = Field(
        default=0.0,
        description="Seconds to sleep between demo stream chunks (0 = no delay)",
    )


class LocalLLM(BaseLLM):
    """MemOS LLM provider that talks to the local WebSocket server.

    ``generate`` converts the MemOS ``MessageList`` into a system prompt + a
    rendered user prompt and sends it over the shared WebSocket transport.
    ``generate_stream`` obtains the full response through the same transport
    and yields it in small chunks (the WS protocol returns the complete
    response; chunking is a demo-side effect).
    """

    def __init__(self, config: LocalLLMConfig):
        self.config = config
        self.client = create_llm_client(
            "local",
            ws_url=config.ws_url,
            pool_maxsize=config.ws_pool_maxsize,
        )

    # ------------------------------------------------------------------
    def _messages_to_prompt(self, messages: MessageList) -> tuple[str, str]:
        """Split a ``MessageList`` into ``(system_prompt, user_prompt)``.

        All ``role == "system"`` contents are concatenated into the system
        prompt; every other turn is rendered as ``"{role}: {content}"`` lines
        in order.  If there are no non-system turns, the last system message
        is sent as the user prompt instead (so the transport always has
        something to say).
        """
        system_parts: list[str] = []
        turn_parts: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            else:
                turn_parts.append(f"{role}: {content}")

        system_prompt = "\n".join(system_parts)
        user_prompt = "\n".join(turn_parts)
        if not user_prompt and system_parts:
            user_prompt = system_parts[-1]
            system_prompt = ""
        return system_prompt, user_prompt

    @staticmethod
    def _chunk_text(content: str) -> list[str]:
        """Split content into demo-sized chunks for streaming."""
        if not content:
            return []
        if " " in content:
            chunks: list[str] = []
            current = ""
            for word in content.split(" "):
                if current and len(current) + len(word) + 1 > 12:
                    chunks.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
            if current:
                chunks.append(current)
            return chunks
        return [content[i : i + 6] for i in range(0, len(content), 6)]

    # ------------------------------------------------------------------
    def generate(self, messages: MessageList, **kwargs) -> str:
        """Generate a response from the local LLM."""
        if kwargs.get("tools"):
            logger.info(
                "local provider does not support tools; ignoring %s", type(kwargs["tools"]).__name__
            )
        system_prompt, user_prompt = self._messages_to_prompt(messages)
        content = self.client.prompt(
            user_prompt,
            system_prompt=system_prompt,
            reset=self.config.reset_each_turn,
            reasoning=self.config.reasoning,
            temperature=kwargs.get("temperature", self.config.temperature),
        )
        if self.config.remove_think_prefix:
            content = remove_thinking_tags(content)
        return content

    def generate_stream(self, messages: MessageList, **kwargs) -> Generator[str, None, None]:
        """Generate a streaming (chunked) response from the local LLM.

        The WebSocket protocol returns the full response; chunks are a
        demo-side effect so callers can exercise stream consumption.
        """
        if kwargs.get("tools"):
            logger.info("stream api not support tools")
            return
        content = self.generate(messages, **kwargs)
        for chunk in self._chunk_text(content):
            yield chunk
            if self.config.simulate_stream_delay > 0:
                time.sleep(self.config.simulate_stream_delay)


def register_local_llm() -> None:
    """Register the ``"local"`` backend into MemOS's LLM registries.

    Idempotent and runtime-only (no ``src\\memos`` edits): mirrors the
    permanent provider-addition steps in AGENTS.md ("Adding a New Provider")
    at runtime.  Must be called before any
    ``LLMConfigFactory.model_validate({"backend": "local", ...})``.
    """
    LLMConfigFactory.backend_to_class[BACKEND_NAME] = LocalLLMConfig
    LLMFactory.backend_to_class[BACKEND_NAME] = LocalLLM
    logger.info(
        "registered LLM backend %r (config=%s, impl=%s)",
        BACKEND_NAME,
        LocalLLMConfig.__name__,
        LocalLLM.__name__,
    )
