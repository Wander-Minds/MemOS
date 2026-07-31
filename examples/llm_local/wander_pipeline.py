#!/usr/bin/env python3
"""Compact replica of the Wander-Memory flat atomic-memory pipeline.

Adapted from ``C:\\dev\\Wander-Memory\\src\\memory\\*`` (Apache-2.0): the
write path (extract -> dedup -> store), the read path (retrieve -> summarize)
and background maintenance (merge / decay / archive), re-implemented in one
self-contained module using **stdlib only** (``sqlite3``, ``json``, ``re``,
``math``).

Every LLM call goes through :func:`llm_client.prompt_func` against the local
WebSocket server (driven by the deterministic dummy backend in this example),
exactly like the original ``src\\memory`` modules did.  The four prompt
templates are concise-English adaptations of ``src\\memory\\prompts.py``.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import unicodedata
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_client import DEFAULT_WS_URL, prompt_func
from typing_extensions import Self


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates (adapted from Wander-Memory src/memory/prompts.py)
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """\
You are a memory extraction assistant. Extract info worth remembering long-term from the dialogue.

Rules:
- Extract only info with cross-session value: user preferences, personal facts, plans, important events, explicit user rules.
- Ignore small talk, one-off context, and transient info already in the question. Prefer under-extraction.
- Each memory is one standalone atomic fact, understandable without dialogue context.
- importance 0~1: health/safety/explicit requests > 0.8; normal preferences 0.5~0.7; weak relevance < 0.3.
- entities: key entities in the memory (people, things, topics).
- Output an empty list if nothing is worth remembering.

Output strict JSON only, no other text:
[{"text": "...", "importance": 0.7, "entities": ["..."]}, ...]"""

DEDUP_SYSTEM = """\
You are a memory dedup assistant. Given one new memory and several similar existing memories, decide the action for the new memory.

Choose one action:
- ADD: entirely new info, duplicates nothing.
- UPDATE: supplements or corrects an old memory that stays partly valid; replace it with new_text (merged full statement). Output old_id and new_text.
- DELETE: new memory directly contradicts an old memory and the old one is stale; delete the old one. Output old_id.
- NOOP: an existing memory equals the new one; do nothing.

Output strict JSON only, no other text:
{"action": "ADD"}
{"action": "UPDATE", "old_id": "...", "new_text": "..."}
{"action": "DELETE", "old_id": "..."}
{"action": "NOOP"}"""

SUMMARIZE_SYSTEM = """\
You are a memory summarizer. Given the user's question and retrieved memories, condense them into concise background info fit for a system prompt.

Rules:
- Keep only facts relevant to the question; list one per line, each starting with "- ".
- Keep each fact's date prefix so recency is clear.
- Do not invent info; do not answer the question.
- If no memory is relevant, output one line: (no relevant memories)
- Respond in English."""

MERGE_SYSTEM = """\
You are a memory merge assistant. Merge several similar memories into one more complete statement.

Rules:
- Result is one standalone sentence keeping all non-conflicting info.
- When info differs by time, prefer the newer.
- importance: take the highest among inputs.
- entities: union of all input entities.

Output strict JSON only, no other text:
{"text": "...", "importance": 0.7, "entities": ["..."]}"""


def build_extraction_prompt(dialogue: str) -> str:
    return f"Extract long-term memorable facts from the recent dialogue:\n\n{dialogue}"


def build_dedup_prompt(new_text: str, similar: list[dict]) -> str:
    lines = [f'New memory: "{new_text}"', "", "Similar existing memories:"]
    for m in similar:
        lines.append(f"- [{m['id']}] {m['text']}")
    lines.append("")
    lines.append("Decide the dedup action (ADD/UPDATE/DELETE/NOOP).")
    return "\n".join(lines)


def build_summarize_prompt(query: str, memories: list[dict]) -> str:
    """Build the summarization prompt listing every retrieved memory."""
    lines = [f"User question: {query}", "", "Retrieved memories:"]
    for m in memories:
        date = m.get("created_at", "unknown date")
        lines.append(f"- [{date}] {m['text']}")
    lines.append("")
    lines.append("Summarize the memories relevant to the question as background info.")
    return "\n".join(lines)


def build_merge_prompt(memories: list[dict]) -> str:
    lines = ["Merge the following similar memories into one:"]
    for m in memories:
        lines.append(f"- {m['text']} (recorded on {m.get('created_at', 'unknown date')})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tolerant JSON parsing (stdlib version of src/memory/parsing.py)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _scan(text: str, opener: str, closer: str) -> str | None:
    """Return the first balanced opener..closer substring, or None."""
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _salvage_truncated_array(text: str) -> list | None:
    """Recover complete objects from an unterminated ``[ {...}, {...`` array."""
    start = text.find("[")
    if start < 0:
        return None
    items: list = []
    pos = start + 1
    while True:
        obj = _scan(text[pos:], "{", "}")
        if obj is None:
            break
        try:
            value = json.loads(obj)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            items.append(value)
        pos += text[pos:].find(obj) + len(obj)
    return items or None


def parse_json_array(text: str, fallback: Any | None = None) -> Any:
    """Parse the first JSON array found in *text*; return *fallback* on failure."""
    if fallback is None:
        fallback = []
    candidate = _scan(_strip_fences(text), "[", "]")
    if candidate is None:
        salvaged = _salvage_truncated_array(text)
        if salvaged is not None:
            return salvaged
        logger.warning("no JSON array found in model output: %.200r", text)
        return fallback
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("malformed JSON array in model output: %.200r", candidate)
        return fallback
    return value if isinstance(value, list) else fallback


def parse_json_object(text: str, fallback: Any | None = None) -> Any:
    """Parse the first JSON object found in *text*; return *fallback* on failure."""
    if fallback is None:
        fallback = {}
    candidate = _scan(_strip_fences(text), "{", "}")
    if candidate is None:
        logger.warning("no JSON object found in model output: %.200r", text)
        return fallback
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("malformed JSON object in model output: %.200r", candidate)
        return fallback
    return value if isinstance(value, dict) else fallback


# ---------------------------------------------------------------------------
# Lexical helpers (BM25-ish, stdlib only)
# ---------------------------------------------------------------------------


class NgramTokenizer:
    """Overlapping n-gram tokenizer with normalization (CJK-aware)."""

    def __init__(self, n: int = 2) -> None:
        self.n = n

    @staticmethod
    def _is_cjk(char: str) -> bool:
        cp = ord(char)
        return (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2EBEF)
            or (0x3040 <= cp <= 0x309F)
            or (0x30A0 <= cp <= 0x30FF)
            or (0xAC00 <= cp <= 0xD7AF)
        )

    def _detect_n(self, text: str) -> int:
        if text.isascii():
            return max(self.n, 3)
        threshold = len(text) * 3 // 10
        cjk_count = sum(1 for c in text if self._is_cjk(c))
        return 2 if cjk_count > threshold else max(self.n, 3)

    def tokenize(self, text: str) -> list[str]:
        """Normalize and generate overlapping character n-grams."""
        lowered = text.lower()
        if not lowered.isascii():
            lowered = unicodedata.normalize("NFKC", lowered)
        text = lowered.strip()
        if not text:
            return []
        n = self._detect_n(text)
        if len(text) < n:
            return [text]
        return [text[i : i + n] for i in range(len(text) - n + 1)]


def sorensen_dice_coefficient(a: str, b: str) -> float:
    """Sørensen-Dice similarity over character bigrams (0..1)."""
    ba = {a[i : i + 2] for i in range(len(a) - 1)}
    bb = {b[i : i + 2] for i in range(len(b) - 1)}
    if not ba or not bb:
        return 0.0
    return 2 * len(ba & bb) / (len(ba) + len(bb))


def _bm25_scores(
    query_tokens: list[str],
    index: dict[str, dict[int, int]],
    doc_lengths: dict[int, int],
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[int, float]:
    """Score all docs containing any query term with BM25."""
    n_docs = len(doc_lengths)
    if n_docs == 0 or avgdl <= 0:
        return {}
    scores: dict[int, float] = {}
    for term in set(query_tokens):
        postings = index.get(term)
        if not postings:
            continue
        df = len(postings)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        for doc_id, tf in postings.items():
            doc_len = doc_lengths.get(doc_id, 1)
            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (k1 + 1)) / denom
    return scores


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WanderConfig:
    """Tunable knobs for the flat memory pipeline (mirrors src/memory/config.py)."""

    ws_url: str = DEFAULT_WS_URL
    ws_pool_maxsize: int = 1
    llm_temperature: float = 0.0  # deterministic JSON for management tasks
    extract_max_chars: int = 6000
    dedup_top_k: int = 5
    dedup_sim_threshold: float = 0.45
    recall_top_k: int = 10
    merge_sim_threshold: float = 0.85
    merge_max_cluster: int = 6
    decay_factor: float = 0.98
    archive_importance: float = 0.15
    archive_stale_days: float = 90.0
    date_prefix_format: str = "[{date}记录] "


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PipelineStore:
    """Thread-safe SQLite store with BM25-ish lexical search.

    Table: ``memories(id, text, importance, entities, created_at,
    accessed_at)`` — a deliberately minimal version of Wander-Memory's
    ``src/memory/store.py`` schema (status/soft-delete omitted for brevity;
    rows are hard-deleted).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id         TEXT PRIMARY KEY,
                    text       TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    entities   TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    def add(
        self,
        text: str,
        *,
        importance: float = 0.5,
        entities: list[str] | None = None,
        memory_id: str | None = None,
    ) -> dict:
        """Insert one atomic memory and return it as a dict."""
        now = _utcnow()
        mid = memory_id or str(uuid.uuid4())
        ent = entities or []
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (id, text, importance, entities, created_at, accessed_at) "
                "VALUES (?,?,?,?,?,?)",
                (mid, text, float(importance), json.dumps(ent, ensure_ascii=False), now, now),
            )
            self._conn.commit()
        logger.info("stored memory %s", mid)
        return {
            "id": mid,
            "text": text,
            "importance": float(importance),
            "entities": ent,
            "created_at": now,
            "accessed_at": now,
        }

    def get(self, memory_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def delete(self, memory_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()

    def update_importance(self, memory_id: str, importance: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (float(importance), memory_id),
            )
            self._conn.commit()

    def record_access(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = _utcnow()
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET accessed_at = ? WHERE id = ?",
                [(now, mid) for mid in memory_ids],
            )
            self._conn.commit()

    def list_active(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM memories ORDER BY created_at").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """BM25-ish lexical retrieval; returns ``[(memory_dict, score)]``."""
        tokenizer = NgramTokenizer()
        query_tokens = tokenizer.tokenize(query)
        if not query_tokens:
            return []
        with self._lock:
            rows = self._conn.execute("SELECT rowid, * FROM memories").fetchall()
        if not rows:
            return []

        index: dict[str, dict[int, int]] = {}
        doc_lengths: dict[int, int] = {}
        total = 0
        for row in rows:
            rowid = int(row["rowid"])
            tokens = tokenizer.tokenize(str(row["text"]))
            doc_lengths[rowid] = len(tokens) or 1
            total += len(tokens)
            for term in tokens:
                index.setdefault(term, {}).setdefault(rowid, 0)
                index[term][rowid] += 1
        avgdl = total / len(rows)

        scores = _bm25_scores(query_tokens, index, doc_lengths, avgdl)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        row_by_rowid = {int(r["rowid"]): self._row_to_dict(r) for r in rows}
        return [(row_by_rowid[rowid], score) for rowid, score in ranked]

    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "text": row["text"],
            "importance": row["importance"],
            "entities": json.loads(row["entities"] or "[]"),
            "created_at": row["created_at"],
            "accessed_at": row["accessed_at"],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Pipeline facade (all LLM calls via llm_client.prompt_func)
# ---------------------------------------------------------------------------


class WanderPipeline:
    """Flat atomic-memory pipeline: write path + read path + maintenance."""

    def __init__(self, db_path: str | Path, config: WanderConfig | None = None) -> None:
        self.config = config or WanderConfig()
        self.store = PipelineStore(db_path)

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------
    def extract(self, dialogue: str) -> list[dict]:
        """Step 1 — extract candidate atomic memories from dialogue."""
        dialogue = dialogue.strip()
        if not dialogue:
            return []
        if len(dialogue) > self.config.extract_max_chars:
            dialogue = dialogue[-self.config.extract_max_chars :]

        raw = prompt_func(
            prompt_str=build_extraction_prompt(dialogue),
            system_prompt=EXTRACTION_SYSTEM,
            ws_url=self.config.ws_url,
            reset=True,
            reasoning="off",
            temperature=self.config.llm_temperature,
            pool_maxsize=self.config.ws_pool_maxsize,
        )
        items = parse_json_array(raw, fallback=[])

        candidates: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            try:
                importance = float(item.get("importance", 0.5))
            except (TypeError, ValueError):
                importance = 0.5
            importance = max(0.0, min(1.0, importance))
            entities = item.get("entities") or []
            if not isinstance(entities, list):
                entities = [str(entities)]
            candidates.append(
                {
                    "text": text,
                    "importance": importance,
                    "entities": [str(e) for e in entities],
                }
            )
        logger.info("extracted %d candidate memories", len(candidates))
        return candidates

    def dedup(self, candidate: dict) -> dict:
        """Step 2 — dedup decision (ADD/UPDATE/DELETE/NOOP) + execute."""
        similar = self.store.search(candidate["text"], top_k=self.config.dedup_top_k)

        # Fast path: nothing remotely similar -> ADD without an LLM call.
        if not similar or (
            sorensen_dice_coefficient(candidate["text"], similar[0][0]["text"])
            < self.config.dedup_sim_threshold
        ):
            mem = self.store.add(**candidate)
            return {"action": "ADD", "memory": mem, "detail": "no similar memory"}

        decision = parse_json_object(
            prompt_func(
                prompt_str=build_dedup_prompt(
                    candidate["text"], [{"id": m["id"], "text": m["text"]} for m, _ in similar]
                ),
                system_prompt=DEDUP_SYSTEM,
                ws_url=self.config.ws_url,
                reset=True,
                reasoning="off",
                temperature=self.config.llm_temperature,
                pool_maxsize=self.config.ws_pool_maxsize,
            ),
            fallback={"action": "ADD"},
        )
        action = str(decision.get("action", "ADD")).upper()
        if action not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
            logger.warning("invalid dedup action %r; defaulting to ADD", action)
            action = "ADD"

        if action == "NOOP":
            return {"action": "NOOP", "memory": None, "detail": candidate["text"]}
        if action == "ADD":
            mem = self.store.add(**candidate)
            return {"action": "ADD", "memory": mem}

        old_id = str(decision.get("old_id", ""))
        old = self.store.get(old_id) if old_id else None
        if old is None:
            # Model referenced a non-existent/old id — fall back to ADD.
            logger.warning("dedup referenced unknown id %r; defaulting to ADD", old_id)
            mem = self.store.add(**candidate)
            return {"action": "ADD", "memory": mem, "detail": "unknown old_id"}
        if action == "DELETE":
            self.store.delete(old_id)
            return {"action": "DELETE", "memory": None, "detail": old_id}

        # UPDATE: supersede the old record, insert the merged text.
        new_text = str(decision.get("new_text", "")).strip() or candidate["text"]
        merged_entities = sorted(set(old["entities"]) | set(candidate["entities"]))
        importance = max(old["importance"], candidate["importance"])
        self.store.delete(old_id)
        mem = self.store.add(new_text, importance=importance, entities=merged_entities)
        return {"action": "UPDATE", "memory": mem, "detail": f"supersedes {old_id}"}

    def run_write(self, dialogue: str) -> list[dict]:
        """Run extract -> dedup -> store for one dialogue transcript."""
        results: list[dict] = []
        for candidate in self.extract(dialogue):
            results.append(self.dedup(candidate))
        return results

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """BM25-ish recall, best first."""
        return [m for m, _ in self.store.search(query, top_k or self.config.recall_top_k)]

    def build_context(self, query: str) -> str:
        """Summarize retrieved memories into an injectable context block.

        Every retrieved memory is listed in one summarization prompt; the
        small model condenses it.  On any LLM failure the raw timestamp
        prefixed lines are returned instead (fail-open).
        """
        memories = self.retrieve(query)
        if not memories:
            return ""
        self.store.record_access([m["id"] for m in memories])
        mem_dicts = [{"text": m["text"], "created_at": m["created_at"][:10]} for m in memories]
        try:
            summary = prompt_func(
                prompt_str=build_summarize_prompt(query, mem_dicts),
                system_prompt=SUMMARIZE_SYSTEM,
                ws_url=self.config.ws_url,
                reset=True,
                reasoning="off",
                temperature=self.config.llm_temperature,
                pool_maxsize=self.config.ws_pool_maxsize,
            ).strip()
        except Exception:
            logger.exception("memory summarization failed; using raw lines")
            summary = ""
        if summary:
            return summary
        return "\n".join(
            self.config.date_prefix_format.format(date=m["created_at"][:10]) + m["text"]
            for m in memories
        )

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------
    def maintain(self, *, merge: bool = True, decay: bool = True, archive: bool = True) -> dict:
        """Merge near-duplicate clusters, decay importance, archive stale rows."""
        report = {"merged": 0, "merged_away": 0, "decayed": 0, "archived": 0, "errors": []}
        if merge:
            self._merge_pass(report)
        if decay:
            self._decay_pass(report)
        if archive:
            self._archive_pass(report)
        logger.info("maintenance: %s", report)
        return report

    def _merge_pass(self, report: dict) -> None:
        actives = self.store.list_active()
        if len(actives) < 2:
            return
        consumed: set[str] = set()
        for i, mem in enumerate(actives):
            if mem["id"] in consumed:
                continue
            cluster = [mem]
            for other in actives[i + 1 :]:
                if other["id"] in consumed or len(cluster) >= self.config.merge_max_cluster:
                    continue
                if sorensen_dice_coefficient(mem["text"], other["text"]) >= (
                    self.config.merge_sim_threshold
                ):
                    cluster.append(other)
            if len(cluster) < 2:
                continue
            consumed.update(m["id"] for m in cluster)
            try:
                self._merge_cluster(cluster, report)
            except Exception as exc:  # keep maintenance best-effort
                logger.exception("merge failed for cluster seeded by %s", mem["id"])
                report["errors"].append(f"merge {mem['id']}: {exc}")

    def _merge_cluster(self, cluster: list[dict], report: dict) -> None:
        raw = prompt_func(
            prompt_str=build_merge_prompt(
                [{"text": m["text"], "created_at": m["created_at"][:10]} for m in cluster]
            ),
            system_prompt=MERGE_SYSTEM,
            ws_url=self.config.ws_url,
            reset=True,
            reasoning="off",
            temperature=self.config.llm_temperature,
            pool_maxsize=self.config.ws_pool_maxsize,
        )
        obj = parse_json_object(raw, fallback={})
        text = str(obj.get("text", "")).strip()
        if not text:
            report["errors"].append(f"merge produced empty text for {cluster[0]['id']}")
            return
        try:
            importance = float(obj.get("importance", 0.0))
        except (TypeError, ValueError):
            importance = 0.0
        importance = max(importance, max(m["importance"] for m in cluster))
        entities = sorted({e for m in cluster for e in m["entities"]})
        for m in cluster:
            self.store.delete(m["id"])
        self.store.add(text, importance=min(1.0, importance), entities=entities)
        report["merged"] += 1
        report["merged_away"] += len(cluster)

    def _decay_pass(self, report: dict) -> None:
        for m in self.store.list_active():
            new_importance = m["importance"] * self.config.decay_factor
            new_importance = max(0.0, min(1.0, new_importance))
            if abs(new_importance - m["importance"]) > 1e-6:
                self.store.update_importance(m["id"], new_importance)
                report["decayed"] += 1

    def _archive_pass(self, report: dict) -> None:
        now = datetime.now(timezone.utc)
        for m in self.store.list_active():
            if m["importance"] >= self.config.archive_importance:
                continue
            try:
                last = datetime.fromisoformat(m["accessed_at"])
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                stale_days = (now - last).total_seconds() / 86400.0
            except ValueError:
                stale_days = float("inf")
            if stale_days >= self.config.archive_stale_days:
                self.store.delete(m["id"])
                report["archived"] += 1

    def close(self) -> None:
        self.store.close()
