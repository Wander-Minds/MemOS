# MemOS LLM Backend Implementation

> Source of truth: `src/memos/llms/` (implementations), `src/memos/configs/llm.py` (configs).
> Scope: how LLM backends are structured, configured, instantiated, and used across MemOS.

---

## 1. Overview

MemOS talks to LLMs exclusively through a thin, pluggable layer:

```
┌──────────────────────────────────────────────────────────────┐
│  Callers (MOSCore.chat, cot_decompose, deepsearch_agent,     │
│  mem_reader, extractor_llm, api handlers)                    │
│              │ generate(messages, **kwargs)                  │
│              ▼                                               │
│  LLMFactory.from_config(LLMConfigFactory)   (singleton)      │
│              │ backend_to_class[backend]                     │
│              ▼                                               │
│  Backend class (OpenAILLM, OllamaLLM, HFLLM, ...)             │
│              │ client (openai SDK / ollama Client / HF)      │
│              ▼                                               │
│  Upstream API / local model                                  │
└──────────────────────────────────────────────────────────────┘
```

Design follows the project-wide **three-piece pattern** (also used by embedders, vec_dbs, etc.):

1. **Config class** (pydantic v2) — `memos.configs.llm`
2. **Implementation class** — `memos.llms.<backend>`
3. **Factory registry** — `memos.llms.factory.LLMFactory`

---

## 2. Base Interface (`src/memos/llms/base.py`)

`BaseLLM(ABC)` mandates exactly three members:

| Member | Signature | Notes |
|---|---|---|
| `__init__` | `(config: BaseLLMConfig)` | Every backend takes exactly one config object |
| `generate` | `(messages: MessageList, **kwargs) -> str` | Non-streaming generation (main path) |
| `generate_stream` | `(messages: MessageList, **kwargs) -> Generator[str, None, None]` | Streaming; docstring marks it "(Optional)" — subclasses override it |

There is **no tool-calling, reasoning, or JSON-mode requirement** in the contract. Anything a caller needs beyond messages is passed through `**kwargs` (e.g. `tools`, `temperature`, `max_tokens`, `model_name_or_path`).

---

## 3. Configuration Layer (`src/memos/configs/llm.py`)

### 3.1 `BaseLLMConfig` — shared fields

| Field | Default | Meaning |
|---|---|---|
| `model_name_or_path` | *(required)* | Model identifier |
| `temperature` | `0.7` | Sampling temperature |
| `max_tokens` | `8192` | Generation length cap |
| `top_p` | `0.95` | Nucleus sampling |
| `top_k` | `50` | Top-k sampling |
| `remove_think_prefix` | `False` | Strip `<think>...</think>` from output (`remove_thinking_tags`) |
| `default_headers` | `None` | Extra HTTP headers |

### 3.2 Per-backend configs

| Config | Extra fields |
|---|---|
| `OpenAILLMConfig` | `api_key`, `api_base` (default `https://api.openai.com/v1`), `extra_body`, **backup client**: `backup_client`, `backup_api_key`, `backup_api_base`, `backup_model_name_or_path`, `backup_headers` |
| `OpenAIResponsesLLMConfig` | `api_key`, `api_base`, `extra_body`, `enable_thinking` |
| `AzureLLMConfig` | `base_url` (`https://api.openai.azure.com/`), `api_version` (`2024-03-01-preview`), `api_key` |
| `OllamaLLMConfig` | `api_base` (`http://localhost:11434`), `enable_thinking` |
| `HFLLMConfig` | `do_sample` (default `False` → greedy decoding), `add_generation_prompt` |
| `VLLMLLMConfig` | `api_key` (optional), `api_base` (`http://localhost:8088/v1`), `enable_thinking`, `extra_body` |
| `QwenLLMConfig` | *(inherits OpenAI)* `api_base` → `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| `DeepSeekLLMConfig` | *(inherits OpenAI)* `api_base` → `https://api.deepseek.com` |
| `MinimaxLLMConfig` | *(inherits OpenAI)* `api_base` → `https://api.minimax.io/v1` |

### 3.3 `LLMConfigFactory` — config dispatch

```python
class LLMConfigFactory(BaseConfig):
    backend: str                       # e.g. "openai"
    config: dict[str, Any]             # raw dict for that backend

    backend_to_class = { "openai": OpenAILLMConfig, "ollama": OllamaLLMConfig, ... }

    @field_validator("backend")        # rejects unknown backends (ValueError)
    @model_validator(mode="after")     # builds concrete Config from the dict
```

The registry mirrors `LLMFactory.backend_to_class` exactly (10 entries each).

---

## 4. Factory (`src/memos/llms/factory.py`)

```python
class LLMFactory(BaseLLM):
    backend_to_class = {
        "openai": OpenAILLM,          "azure": AzureLLM,
        "ollama": OllamaLLM,          "huggingface": HFLLM,
        "huggingface_singleton": HFSingletonLLM,
        "vllm": VLLMLLM,              "qwen": QwenLLM,
        "deepseek": DeepSeekLLM,      "minimax": MinimaxLLM,
        "openai_new": OpenAIResponsesLLM,
    }

    @classmethod
    @singleton_factory()              # caches instances per config
    def from_config(cls, config_factory: LLMConfigFactory) -> BaseLLM:
        backend = config_factory.backend
        if backend not in cls.backend_to_class:
            raise ValueError(f"Invalid backend: {backend}")
        return cls.backend_to_class[backend](config_factory.config)
```

- Decorated with `singleton_factory()` (from `memos.memos_tools.singleton`) → repeated `from_config` calls with equal configs return the **same instance** (important for `huggingface_singleton`, which shares one loaded model).
- Unknown backend → `ValueError` at both config and factory level.

---

## 5. Backend Matrix

| backend | Class | Transport | Tool calls | Reasoning/thinking |
|---|---|---|---|---|
| `openai` | `OpenAILLM` | `openai.Client` (chat.completions) | ✅ via kwargs | ✅ `reasoning_content` → `<think>` |
| `openai_new` | `OpenAIResponsesLLM` | OpenAI **Responses** API | ✅ (`ResponseFunctionToolCall`) | ✅ |
| `azure` | `AzureLLM` | `openai.AzureOpenAI` | ✅ | ✅ |
| `ollama` | `OllamaLLM` | `ollama.Client` | ✅ | ✅ `message.thinking` (config `enable_thinking`) |
| `vllm` | `VLLMLLM` | OpenAI-compatible (`http://localhost:8088/v1`) | ✅ | ✅ `enable_thinking` |
| `huggingface` | `HFLLM` | Local transformers pipeline | ❌ | ✅ strips `<think>` tags |
| `huggingface_singleton` | `HFSingletonLLM` | Shared local model singleton | ❌ | ✅ |
| `qwen` | `QwenLLM` | OpenAI-compatible (DashScope) | ✅ (inherits OpenAI) | ✅ |
| `deepseek` | `DeepSeekLLM` | OpenAI-compatible | ✅ | ✅ |
| `minimax` | `MinimaxLLM` | OpenAI-compatible | ✅ | ✅ |

---

## 6. Common Capabilities

### 6.1 Tool calling — **optional, pass-through**

- Every OpenAI-compatible backend forwards the caller-supplied kwarg:
  ```python
  tools=kwargs.get("tools", NOT_GIVEN)   # openai / openai_new / vllm
  tools=kwargs.get("tools")              # ollama
  ```
- When the response contains `tool_calls`, backends parse them via `tool_call_parser(...)` into `{tool_call_id, function_name, arguments}` (OpenAI) or `{function_name, arguments}` (Ollama).
- **No caller in the core flow uses tools**: a grep for `tools=` across `src/memos` excluding `llms/` yields **zero matches**; `deepsearch_agent.py` only calls plain `llm.generate(messages)`.
- **Streaming does not support tools**: `generate_stream` logs `"stream api not support tools"` and returns when `tools` is passed.
- The **HuggingFace backend has no tool support at all** yet is a first-class backend — proof that tool calling is not a requirement.

### 6.2 Reasoning / thinking — handled, never required

- OpenAI-compatible backends read `reasoning_content` (or `delta.reasoning_content` in streaming) and wrap it in `<think>...</think>`.
- Ollama emits `message.thinking` when `enable_thinking=True`.
- `remove_think_prefix` (or the global `remove_thinking_tags()` util) strips `<think>...</think>` when set:
  ```python
  # src/memos/llms/utils.py
  re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
  ```
- If the model has no native reasoning field, the code simply falls through to `message.content` — **native reasoning is fully optional**.

### 6.3 Failure resilience (OpenAI)

`OpenAILLM.generate` supports a **backup client**: on primary-call exception, it retries the same request against a second endpoint/model (`backup_client`, `backup_api_key`, `backup_api_base`, `backup_model_name_or_path`). The backup is configured purely in `OpenAILLMConfig` and reuses `_parse_response`.

### 6.4 Convenience (Ollama)

- `OllamaLLM.__init__` defaults the model to `llama3.1:latest` if unset.
- `_ensure_model_exists()` lists local models via `client.list()` and **auto-pulls** the model from Ollama when missing.

### 6.5 KV-cache / activation-memory integration (HF)

- `HFLLM` imports `transformers.DynamicCache` and exposes `build_kv_cache(messages) -> DynamicCache` plus `past_key_values` parameters — this is what `MOSCore.chat` uses for **Activation Memory** (injecting a cached prefix to skip prefill).
- KV-trimming code is written to be compatible with both old (`key_cache`/`value_cache`) and new (`layers`) transformers `DynamicCache` layouts (see `llms/hf.py:403-426`, `memories/activation/kv.py`).

---

## 7. How the LLM Layer Is Used

| Caller | Usage |
|---|---|
| `MOSCore.chat` / `MOS.chat` | `chat_llm.generate(current_messages, past_key_values=...)` — builds system prompt from retrieved memories |
| CoT (`mem_os/main.py::cot_decompose`) | **Prompt-based**: `COT_DECOMPOSE_PROMPT` + `llm.generate()`, then `json.loads` with regex fallback → default `{"is_complex": False, "sub_questions": []}`. No native CoT capability needed |
| `deepsearch_agent.py` | plain `llm.generate(messages)` |
| Memory extractors / `mem_reader` | `extractor_llm` / `general_llm` from factory |

### Hard requirements vs optional features

| Capability | Required? | If missing |
|---|---|---|
| `generate(messages) -> str` | ✅ **Only hard requirement** | Backend unusable |
| `generate_stream` | ❌ optional (interface marks it Optional) | Non-streaming path still works |
| Native CoT/reasoning | ❌ | Query decomposition degrades to prompt-based or falls back to standard chat |
| Native tool calling | ❌ | `tools=` simply not passed; no core flow depends on it |
| OpenAI-compatible API | ⚠️ recommended | Tool calls + `reasoning_content` unavailable; local HF backend still works |

---

## 8. Adding a New Backend (project convention — AGENTS.md)

1. Implement `src/memos/llms/<backend>.py`, subclass `BaseLLM`, match existing signatures.
2. Add a pydantic config in `src/memos/configs/llm.py`; register in `LLMConfigFactory.backend_to_class`.
3. Register the class in `LLMFactory.backend_to_class` (`src/memos/llms/factory.py`).
4. Third-party SDKs go into an **optional extras group** in `pyproject.toml`; guard imports with `try/except ImportError` and raise a clear "install extras X" message.
5. Add tests under `tests/llms/test_<backend>.py`; mock all external HTTP/model loading.

---

## 9. Verification Status

- `tests/llms/` covers factory creation and backend behavior with mocked clients (e.g. `test_openai.py`, `test_ollama.py`, `test_hf.py` — all pass in the full suite).
- Full suite result (this environment, transformers 4.57.6): **736 passed / 2 failed / 4 skipped**; the 2 failures are in `tests/memories/activation/test_kv.py` (stale `DynamicCache.key_cache` test helper vs. new transformers API), unrelated to the LLM layer.

---

## 10. Key Takeaways

1. **Uniform contract**: every backend implements `BaseLLM`; everything else is `**kwargs` pass-through.
2. **Double registry**: `LLMConfigFactory` (configs) and `LLMFactory` (classes) both keyed by the same `backend` string; both reject unknown backends.
3. **Singleton caching** via `singleton_factory()` — especially relevant for local-model backends (`huggingface_singleton`).
4. **No capability gatekeeping**: tool calling and native reasoning are optional enrichments; the only hard requirement is text generation.
5. **OpenAI-compatible backends share one code family** (openai/azure/qwen/deepseek/minimax/vllm), keeping maintenance small.
