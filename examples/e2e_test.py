"""End-to-end LLM-assisted memory test: configure a real LLM provider, then write
and retrieve memories through MemOS.

This is the merged successor of the repo-root scripts ``llm_config.py``,
``write_memory.py`` and ``retrieve_memory.py``: one self-contained script that
closes the whole loop against the REAL provider from ``C:/dev/ds_flash.json``
(overridable via ``MEMOS_LLM_CONFIG``):

1. **Resolve the provider** — the OpenAI-compatible request policy follows the
   ``OpenAILegacy`` provider (``packages/kosong/.../openai_legacy.py``):
   ``openai_legacy`` maps to the MemOS ``openai`` backend, ``max_tokens`` is
   clamped to ``384_000`` (the JSON only advertises ``max_context_size`` of 1M,
   which is a *context* budget, not an *output* budget), and
   ``thinking_effort: max`` is forwarded via ``extra_body``. Chat keeps the
   ``<think>...</think>`` block (``show_thinking_stream``), extraction LLMs
   strip it so JSON parsing stays clean.
2. **Smoke test** — one real LLM chat call (``<think>…</think>OK``) and one real
   embedding call (``text-embedding-3-large``, 3072 dims) through the same proxy.
3. **Write memory** — ``general_text`` memory + Qdrant (local/embedded mode,
   ``path=``) + universal_api embeddings; 3 direct facts via ``MOS.add`` and 2
   LLM-extracted memories via ``cube.text_mem.extract(conversation)`` (a real
   DeepSeek call against the proxy). The cube is persisted to ``.memos/e2e/cube``.
4. **Retrieve memory** — reload the persisted cube from disk (proves the
   round-trip), ``MOS.search(query)`` over the vector store, and
   ``MOS.chat(query)`` where the REAL LLM answers grounded on the retrieved
   memories. Assertions verify the memories written in step 3 were recalled.

The script is intentionally self-contained: run it from the repository root
(``python examples/e2e_test.py``) or from anywhere — all paths are derived from
``__file__`` and all runtime data lives under ``.memos/`` which is gitignored.

Exit code 0 = every phase (smoke / write / persist / retrieve / chat) passed.
Any failed assertion or provider error exits non-zero with a clear message.

Environment overrides:
    MEMOS_LLM_CONFIG   path to the provider JSON (default C:/dev/ds_flash.json)
    MEMOS_USER_ID      default user id (default "root", already in the db)
    MEMOS_MAX_TOKENS   output budget; always clamped to <= 384000
    MEMOS_EMBED_MODEL  embedding model id (default text-embedding-3-large)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil

from pathlib import Path
from typing import TYPE_CHECKING, Any

from memos import MOS, GeneralMemCube
from memos.configs.llm import LLMConfigFactory
from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.llms.factory import LLMFactory
from memos.mem_user.user_manager import UserManager, UserRole


if TYPE_CHECKING:
    from memos.memories.textual.item import TextualMemoryItem


# --------------------------------------------------------------------------- #
# kosong OpenAI policy constants (see openai_common.py / openai_legacy.py)
# --------------------------------------------------------------------------- #
_MAX_OUTPUT_TOKENS = 384_000  # kosong clamp_max_tokens() safe upper bound
_DEFAULT_OUTPUT_TOKENS = 8192  # MemOS BaseLLMConfig default output budget
_EMBED_MODEL = "text-embedding-3-large"
_EMBED_DIM = 3072

# Backend mapping: the provider's wire protocol is OpenAI Chat Completions.
_LEGACY_TYPE_TO_MEMOS_BACKEND = {
    "openai_legacy": "openai",
    "openai": "openai",
    "deepseek": "deepseek",
    "vllm": "vllm",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDER_CONFIG = Path(os.environ.get("MEMOS_LLM_CONFIG", r"C:\dev\ds_flash.json"))
# Runtime data lives under .memos/ and tmp/ which are gitignored (AGENTS.md).
DATA_DIR = REPO_ROOT / ".memos" / "e2e"
CUBE_DIR = DATA_DIR / "cube"
QDRANT_DIR = DATA_DIR / "qdrant"
COLLECTION_NAME = "e2e_memories"

# The universal_api embedder wraps its call with a short asyncio timeout
# (default 5s); the first real embedding round-trip can be slower.
os.environ.setdefault("MOS_EMBEDDER_TIMEOUT", "60")


class _LocalQdrantLogFilter(logging.Filter):
    """Hide the advisory "Qdrant is running in local mode" log line.

    This e2e intentionally uses Qdrant in local/embedded mode (host/port None,
    ``path`` set) so no server needs to be started. The library logs the local-
    mode caveat once per client init; it is accurate but expected here, so the
    script filters exactly that message and leaves every other log record
    (including real errors) untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "running in local mode" not in record.getMessage()


logging.getLogger("memos.vec_dbs.qdrant").addFilter(_LocalQdrantLogFilter())


# --------------------------------------------------------------------------- #
# policy helpers
# --------------------------------------------------------------------------- #
def clamp_max_tokens(value: int | None) -> int:
    """Clamp an output-token budget to the safe upper bound (kosong policy).

    The provider JSON only advertises ``max_context_size`` (1M); passing that
    as ``max_tokens`` would exceed the API's own per-model output limit and
    cause a 400. Any configured budget above 384_000 is clamped down, mirroring
    ``kosong.contrib.chat_provider.openai_common.clamp_max_tokens``.
    """
    if value is None:
        return _DEFAULT_OUTPUT_TOKENS
    return min(int(value), _MAX_OUTPUT_TOKENS)


def load_provider_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the provider definition (e.g. ``C:/dev/ds_flash.json``)."""
    cfg_path = Path(path) if path is not None else DEFAULT_PROVIDER_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"LLM provider config not found: {cfg_path} (override with env MEMOS_LLM_CONFIG)"
        )
    with open(cfg_path, encoding="utf-8") as fh:
        provider = json.load(fh)
    for key in ("model", "url", "api_key"):
        if not provider.get(key):
            raise ValueError(f"LLM provider config {cfg_path} is missing '{key}'")
    return provider


def _extra_body(provider: dict[str, Any]) -> dict[str, Any]:
    """Build the ``extra_body`` payload following the OpenAILegacy policy.

    ``thinking_effort`` (e.g. ``"max"``) is forwarded as ``reasoning_effort``
    on the wire. Kosong only adds ``thinking`` / ``reasoning`` /
    ``chat_template_kwargs`` when a ``reasoning_key`` is configured; this
    provider JSON does not set one, so they are omitted.
    """
    capabilities = provider.get("capabilities") or []
    effort = provider.get("thinking_effort")
    if "thinking" not in capabilities or not effort:
        return {}
    return {"reasoning_effort": str(effort)}


def _default_headers(provider: dict[str, Any]) -> dict[str, Any] | None:
    headers = provider.get("custom_headers") or {}
    return headers if headers else None


# --------------------------------------------------------------------------- #
# config builders
# --------------------------------------------------------------------------- #
def _llm_config_dict(
    provider: dict[str, Any],
    *,
    for_extraction: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    """Raw ``OpenAILLMConfig``-shaped dict for the provider (no backend key).

    ``LLMConfigFactory``'s ``model_validator`` converts the ``config`` dict into
    the concrete pydantic config and is *not* idempotent, so nested usage must
    pass plain dicts (never already-built factory/config instances).
    """
    max_tokens = clamp_max_tokens(
        int(os.environ["MEMOS_MAX_TOKENS"])
        if os.environ.get("MEMOS_MAX_TOKENS")
        else _DEFAULT_OUTPUT_TOKENS
    )
    return {
        "model_name_or_path": model or provider["model"],
        "api_key": provider["api_key"],
        "api_base": provider["url"],
        "temperature": float(provider.get("temperature", 0.7)),
        "max_tokens": max_tokens,
        "top_p": float(provider.get("top_p", 0.95)),
        "top_k": int(provider.get("top_k", 50)),
        # chat keeps <think> (show_thinking_stream=true); extractors strip it
        "remove_think_prefix": bool(for_extraction),
        "default_headers": _default_headers(provider),
        "extra_body": _extra_body(provider),
    }


def _llm_backend(provider: dict[str, Any]) -> str:
    return _LEGACY_TYPE_TO_MEMOS_BACKEND.get(provider.get("type", "openai_legacy"), "openai")


def build_llm_config(
    provider: dict[str, Any] | None = None,
    *,
    for_extraction: bool = False,
    model: str | None = None,
) -> LLMConfigFactory:
    """Build a MemOS ``LLMConfigFactory`` from the provider JSON.

    Args:
        provider: parsed ``ds_flash.json`` (loaded by :func:`load_provider_config`).
        for_extraction: when True, ``remove_think_prefix=True`` so downstream
            JSON parsing of extracted memories is not polluted by thinking.
        model: optional model override (defaults to the provider's ``model``).
    """
    provider = provider if provider is not None else load_provider_config()
    return LLMConfigFactory(
        backend=_llm_backend(provider),
        config=_llm_config_dict(provider, for_extraction=for_extraction, model=model),
    )


def _embedder_config(provider: dict[str, Any]) -> dict[str, Any]:
    """universal_api (OpenAI-compatible) embedder on the same proxy."""
    return {
        "backend": "universal_api",
        "config": {
            "provider": "openai",
            "api_key": provider["api_key"],
            "base_url": provider["url"],
            "model_name_or_path": os.environ.get("MEMOS_EMBED_MODEL", _EMBED_MODEL),
        },
    }


def build_mos_config(
    provider: dict[str, Any] | None = None, user_id: str | None = None
) -> MOSConfig:
    """Build a MemOS ``MOSConfig`` wired to the provider (chat + mem_reader)."""
    provider = provider if provider is not None else load_provider_config()
    user_id = user_id or os.environ.get("MEMOS_USER_ID", "root")
    return MOSConfig.model_validate(
        {
            "user_id": user_id,
            "chat_model": {
                "backend": _llm_backend(provider),
                "config": _llm_config_dict(provider, for_extraction=False),
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": _llm_backend(provider),
                        "config": _llm_config_dict(provider, for_extraction=True),
                    },
                    "embedder": _embedder_config(provider),
                    "chunker": {
                        "backend": "sentence",
                        "config": {
                            "tokenizer_or_token_counter": "gpt2",
                            "chunk_size": 512,
                            "chunk_overlap": 128,
                            "min_sentences_per_chunk": 1,
                        },
                    },
                },
            },
            "enable_textual_memory": True,
            "enable_activation_memory": False,
            "enable_parametric_memory": False,
            "enable_preference_memory": False,
            "enable_mem_scheduler": False,
            "top_k": 5,
            "max_turns_window": 20,
            "PRO_MODE": False,
        }
    )


def build_cube_config(
    provider: dict[str, Any] | None = None,
    user_id: str | None = None,
    cube_id: str = "e2e_cube",
) -> GeneralMemCubeConfig:
    """Build a ``GeneralMemCubeConfig`` with a real textual memory stack.

    general_text memory + Qdrant in local/embedded mode (host/port None,
    ``path`` set) + universal_api embeddings (text-embedding-3-large, 3072d).
    """
    provider = provider if provider is not None else load_provider_config()
    user_id = user_id or os.environ.get("MEMOS_USER_ID", "root")
    return GeneralMemCubeConfig.model_validate(
        {
            "user_id": user_id,
            "cube_id": cube_id,
            "text_mem": {
                "backend": "general_text",
                "config": {
                    "cube_id": cube_id,
                    "memory_filename": "textual_memory.json",
                    "extractor_llm": {
                        "backend": _llm_backend(provider),
                        "config": _llm_config_dict(provider, for_extraction=True),
                    },
                    "vector_db": {
                        "backend": "qdrant",
                        "config": {
                            "collection_name": COLLECTION_NAME,
                            "vector_dimension": _EMBED_DIM,
                            "distance_metric": "cosine",
                            # host/port None + path -> qdrant local/embedded mode
                            "path": str(QDRANT_DIR),
                        },
                    },
                    "embedder": _embedder_config(provider),
                },
            },
            "act_mem": {"backend": "uninitialized", "config": {}},
            "para_mem": {"backend": "uninitialized", "config": {}},
            "pref_mem": {"backend": "uninitialized", "config": {}},
        }
    )


# --------------------------------------------------------------------------- #
# shared runtime helpers
# --------------------------------------------------------------------------- #
def ensure_user_exists(user_id: str) -> None:
    """Create the user in MemOS's user db if it does not exist yet."""
    user_manager = UserManager(user_id=user_id)
    if not user_manager.validate_user(user_id):
        user_manager.create_user(f"E2E {user_id}", UserRole.USER, user_id)
        print(f"created user '{user_id}'")


def build_mos(provider: dict[str, Any], user_id: str) -> tuple[MOS, GeneralMemCube]:
    """Build MOS + cube wired to the real provider."""
    mos_config = build_mos_config(provider, user_id=user_id)
    cube_config = build_cube_config(provider, user_id=user_id)
    cube = GeneralMemCube(cube_config)
    mos = MOS(mos_config)
    mos.register_mem_cube(cube)
    return mos, cube


def fmt_memory(item: TextualMemoryItem, index: int) -> str:
    meta = item.metadata
    extra = []
    if meta.source:
        extra.append(f"source={meta.source}")
    if meta.tags:
        extra.append(f"tags={','.join(meta.tags)}")
    if meta.key:
        extra.append(f"key={meta.key}")
    suffix = f" ({', '.join(extra)})" if extra else ""
    return f"  [{index}] {item.memory}{suffix}"


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def print_provider(provider: dict[str, Any]) -> None:
    """Print the resolved provider definition."""
    print("=" * 72)
    print(f"provider config : {DEFAULT_PROVIDER_CONFIG}")
    print(f"  model         : {provider['model']}")
    print(f"  type/backend  : {provider.get('type')} -> openai")
    print(f"  url (api_base): {provider['url']}")
    print(f"  max_context   : {provider.get('max_context_size')} (NOT used as max_tokens)")
    print(f"  capabilities  : {provider.get('capabilities')}")
    print(f"  thinking_effort: {provider.get('thinking_effort')}")
    print("=" * 72)


def run_smoke(provider: dict[str, Any]) -> None:
    """Real LLM chat + embedding smoke test against the provider."""
    llm_factory = build_llm_config(provider)
    llm = LLMFactory.from_config(llm_factory)
    answer = llm.generate([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=64)
    print("smoke [LLM chat]      :", repr(answer))
    if "OK" not in answer:
        raise AssertionError(f"LLM smoke reply did not contain 'OK': {answer!r}")

    from memos.embedders.factory import EmbedderFactory

    embedder = EmbedderFactory.from_config(build_mos_config(provider).mem_reader.config.embedder)
    vec = embedder.embed(["MemOS end-to-end memory test"])[0]
    print(f"smoke [embedding]     : dims={len(vec)} model={_EMBED_MODEL}")
    if len(vec) != _EMBED_DIM:
        raise AssertionError(f"embedding dimension {len(vec)} != expected {_EMBED_DIM}")


def write_memories(
    mos: MOS, cube: GeneralMemCube, *, skip_extract: bool = False
) -> tuple[int, int]:
    """Write direct + LLM-extracted memories; returns (n_direct, n_extracted)."""
    direct_memories = [
        (
            "The user is prototyping an end-to-end memory system with MemOS and a "
            "DeepSeek v4 flash model served through an OpenAI-compatible proxy."
        ),
        (
            "The user's project requires Ruff formatting and a green `make test` "
            "before any merge, and pushes go through pull requests."
        ),
        (
            "The user prefers concise, well-structured code and always writes "
            "type annotations in Python."
        ),
    ]
    for memory in direct_memories:
        mos.add(memory_content=memory)
    print(f"wrote {len(direct_memories)} direct memories via MOS.add")

    extracted: list[TextualMemoryItem] = []
    if not skip_extract:
        conversation = [
            {"role": "user", "content": "I prefer dark mode and Vim keybindings in my editor."},
            {"role": "assistant", "content": "Got it — I will remember your editor preferences."},
            {
                "role": "user",
                "content": "Also, I usually run `make test` locally before opening a PR.",
            },
        ]
        print("extracting memories from conversation with the real LLM ...")
        extracted = cube.text_mem.extract(conversation)  # real proxy call
        cube.text_mem.add(extracted)
        print(f"LLM extracted {len(extracted)} memory items")
        if not extracted:
            raise AssertionError("LLM extraction returned no memory items")
    return len(direct_memories), len(extracted)


def persist_cube(cube: GeneralMemCube, *, keep: bool) -> None:
    """Persist the cube to ``CUBE_DIR`` (config.json + textual_memory.json)."""
    CUBE_DIR.mkdir(parents=True, exist_ok=True)
    if keep:
        # append mode: dump() requires an empty dir, so persist manually
        cube.config.to_json_file(CUBE_DIR / cube.config.config_filename)
        cube.text_mem.dump(CUBE_DIR)
    else:
        cube.dump(CUBE_DIR)
    print(f"cube persisted to {CUBE_DIR}")


def report_stored(cube: GeneralMemCube, expected: int) -> None:
    """Print everything stored in the cube and assert the expected count."""
    print("\n=== stored memories (vector store + json) ===")
    all_items = cube.text_mem.get_all()
    for i, item in enumerate(all_items, 1):
        print(fmt_memory(item, i))
    print(f"total: {len(all_items)} memory items")
    if len(all_items) != expected:
        raise AssertionError(f"expected {expected} stored memories, found {len(all_items)}")


def retrieve_memories(user_id: str, query: str, top_k: int, *, no_chat: bool = False) -> None:
    """Reload the persisted cube and search + chat against it."""
    if not (CUBE_DIR / "config.json").is_file():
        raise SystemExit(f"no persisted cube found at {CUBE_DIR} — the write phase failed")

    provider = load_provider_config()
    cube = GeneralMemCube.init_from_dir(CUBE_DIR)
    print(
        f"loaded cube '{cube.config.cube_id}' from {CUBE_DIR} "
        f"(text_mem backend={cube.config.text_mem.backend})"
    )
    mos = MOS(build_mos_config(provider, user_id=user_id))
    mos.register_mem_cube(cube)

    print(f"\n=== search: {query!r} (top_k={top_k}) ===")
    result = mos.search(query, top_k=top_k, mode="fast")
    text_results = result.get("text_mem") or []
    found = 0
    for cube_result in text_results:
        for item in cube_result.get("memories", []):
            found += 1
            print(fmt_memory(item, found))
    if not found:
        print("  (no memories retrieved)")
    print(f"retrieved {found} memory item(s) from {len(text_results)} cube(s)")
    if found < 3:
        raise AssertionError(f"expected >= 3 retrieved memories, found {found}")

    if not no_chat:
        chat_query = (
            "Based on my memories, what do you remember about my editor "
            "preferences and my testing workflow?"
        )
        print(f"\n=== chat: {chat_query!r} (real LLM) ===")
        response = mos.chat(chat_query)
        print(f"{response}")
        print()
        for keyword in ("dark mode", "make test"):
            if keyword.lower() not in str(response).lower():
                raise AssertionError(
                    f"chat reply did not recall {keyword!r} from the written memories:\n{response}"
                )

    # Release the local Qdrant lock so the process exits cleanly and the
    # storage folder can be reopened by a subsequent run.
    cube.text_mem.vector_db.close()


# --------------------------------------------------------------------------- #
# standalone CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end LLM-assisted memory write + retrieve test (real backend)"
    )
    parser.add_argument("--config", default=None, help="path to the provider JSON")
    parser.add_argument(
        "--user",
        default=os.environ.get("MEMOS_USER_ID", "root"),
        help="user id (must exist in the MemOS user db; default: root)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="do NOT wipe .memos/e2e storage before writing (append mode)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="skip the real-LLM memory extraction step",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="skip the LLM/embedding smoke test",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="skip the real-LLM chat step (search only)",
    )
    parser.add_argument(
        "--query",
        default="What are my editor preferences and how do I run tests?",
        help="retrieval query (default: editor preferences / testing workflow)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="number of memories to retrieve")
    args = parser.parse_args()

    provider = load_provider_config(args.config)
    user_id = args.user
    ensure_user_exists(user_id)
    print_provider(provider)

    if not args.no_smoke:
        run_smoke(provider)

    # Deterministic e2e: start from a clean cube + vector store unless --keep.
    if not args.keep:
        for path in (CUBE_DIR, QDRANT_DIR):
            if path.exists():
                shutil.rmtree(path)

    mos, cube = build_mos(provider, user_id)
    n_direct, n_extracted = write_memories(mos, cube, skip_extract=args.skip_extract)
    expected = n_direct + n_extracted
    persist_cube(cube, keep=args.keep)
    report_stored(cube, expected)

    # The write-phase Qdrant client holds an exclusive lock on the local storage
    # folder; release it before the retrieve phase reopens the same folder.
    cube.text_mem.vector_db.close()

    retrieve_memories(user_id, args.query, args.top_k, no_chat=args.no_chat)
    print("examples/e2e_test.py: OK")


if __name__ == "__main__":
    main()
