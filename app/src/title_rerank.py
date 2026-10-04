"""
Optional second retrieval stage: an LLM reorders the candidate pages by their titles.

OFF BY DEFAULT. With TITLE_RERANK_ENABLED unset or false, `maybe_enable_title_rerank(engine)`
returns the engine object unchanged, so /search behaves exactly as without this module.

How it works when enabled
  1. The query is embedded and searched exactly as in Engine.search_documents, but with a wider
     paragraph pool (TITLE_RERANK_POOL_PARAGRAPHS, default 200), deduplicated to the first
     TITLE_RERANK_CANDIDATES pages (default 30), each represented by its best paragraph.
  2. The chat model (any provider from llm_factory) receives the question and the numbered list of
     those page titles and returns the numbers of the pages that answer it, best first.
  3. Pages it chose come first in its order, the remaining candidates follow in retrieval order,
     and the first `num_of_pages` go to the answering LLM as before.
  Any failure (no key, timeout, unparsable reply) falls back to the plain retrieval order, logs a
  warning, and never fails the request.

Measured (retrieval_eval/title_llm.py, gpt-4o-mini, on top of the title-in-text index, hit@1):
  vendor held-out 296: 0.463 -> 0.601; agent-written clean set 150: 0.600 -> 0.667;
  IdoAgai's public generated set 300: 0.537 -> 0.607. Median added latency ~0.8 s per query.

Turn it on (app/.env):
  TITLE_RERANK_ENABLED=true
  TITLE_RERANK_PROVIDER=openai            # openai | anthropic | gemini | any registered name
  TITLE_RERANK_MODEL=gpt-4o-mini
  TITLE_RERANK_API_KEY=                   # empty: the provider's usual variable (OAI_API_KEY, ...)
  TITLE_RERANK_BASE_URL=                  # only for OpenAI-compatible servers (Azure, Ollama, vLLM)
  TITLE_RERANK_CANDIDATES=30
  TITLE_RERANK_POOL_PARAGRAPHS=200
  TITLE_RERANK_TIMEOUT_SECS=5
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from llm_factory import create_chat_model

# Kept byte-identical to the prompt the measurements were made with (retrieval_eval/title_llm.py imports it).
PICK_PROMPT = ("לפניך שאלה של אזרח ורשימה ממוספרת של כותרות עמודים מאתר כל-זכות. "
               "בחר את העמודים שהכי סביר שעונים על השאלה, מהמתאים ביותר לפחות מתאים. "
               "שים לב במיוחד לאוכלוסייה ולתת-המקרה שהשאלה מתארת. "
               "החזר JSON בלבד בצורה {\"ranking\": [מספרים]} עם 1 עד 10 מספרים מהרשימה.")


def build_user_message(question: str, titles: List[str]) -> str:
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    return f"שאלה: {question}\n\nכותרות:\n{listing}"


def parse_ranking(reply: str, n_candidates: int, max_picks: int = 10) -> List[int]:
    """Zero-based candidate indices from the model's reply. Tolerates prose around the JSON and
    bare number lists; drops out-of-range and repeated numbers."""
    nums: list = []
    if reply:
        match = re.search(r"\{.*\}", reply, flags=re.S)
        try:
            raw = json.loads(match.group(0) if match else reply)
            nums = raw.get("ranking", []) if isinstance(raw, dict) else raw if isinstance(raw, list) else []
        except (ValueError, AttributeError):
            nums = re.findall(r"\d+", reply)
    out: List[int] = []
    for n in nums:
        try:
            i = int(n) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_candidates and i not in out:
            out.append(i)
        if len(out) >= max_picks:
            break
    return out


@dataclass
class TitleRerankSettings:
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    candidates: int = 30
    pool_paragraphs: int = 200
    timeout_secs: float = 5.0
    title_field: str = "title"

    @classmethod
    def from_env(cls) -> "TitleRerankSettings":
        env = os.getenv
        return cls(enabled=env("TITLE_RERANK_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on"),
                   provider=env("TITLE_RERANK_PROVIDER", "openai"),
                   model=env("TITLE_RERANK_MODEL", "gpt-4o-mini"),
                   api_key=env("TITLE_RERANK_API_KEY") or None,
                   base_url=env("TITLE_RERANK_BASE_URL") or None,
                   candidates=int(env("TITLE_RERANK_CANDIDATES", "30")),
                   pool_paragraphs=int(env("TITLE_RERANK_POOL_PARAGRAPHS", "200")),
                   timeout_secs=float(env("TITLE_RERANK_TIMEOUT_SECS", "5")))


class TitleReranker:
    def __init__(self, chat_model, timeout_secs: float = 5.0, title_field: str = "title"):
        self.chat_model, self.timeout_secs, self.title_field = chat_model, timeout_secs, title_field

    def rerank(self, question: str, docs: List[dict]) -> List[dict]:
        """Return `docs` reordered; on any failure return them unchanged."""
        if len(docs) < 2:
            return docs
        titles = [str(d.get(self.title_field, "")) for d in docs]
        try:
            reply = self.chat_model.complete(PICK_PROMPT, build_user_message(question, titles),
                                             max_tokens=120, json_mode=True, timeout=self.timeout_secs)
        except Exception as e:  # network, auth, timeout, provider errors
            logging.warning(f"title rerank skipped, LLM call failed: {e}")
            return docs
        picked = parse_ranking(reply, len(docs))
        if not picked:
            logging.warning(f"title rerank skipped, unparsable reply: {reply[:200]!r}")
            return docs
        chosen = set(picked)
        return [docs[i] for i in picked] + [d for i, d in enumerate(docs) if i not in chosen]


class TitleRerankingEngine:
    """Wraps a webiks_hebrew_ragbot Engine; only answer_query changes, everything else is delegated."""

    def __init__(self, engine, reranker: TitleReranker, candidates: int = 30, pool_paragraphs: int = 200):
        self._engine, self.reranker = engine, reranker
        self.candidates, self.pool_paragraphs = candidates, pool_paragraphs

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def candidate_documents(self, query: str) -> List[dict]:
        vector = self._engine.retrieval_model.encode(query)
        hits = self._engine.elastic_model.search(vector, size=self.pool_paragraphs)
        docs, seen = [], set()
        for hit in hits:
            src = hit["_source"]
            if src["doc_id"] not in seen:
                seen.add(src["doc_id"]); docs.append(src)
            if len(docs) >= self.candidates:
                break
        return docs

    def answer_query(self, query, top_k: int, model):
        t0 = time.perf_counter()
        candidates = self.candidate_documents(query)
        t1 = time.perf_counter()
        top_k_documents = self.reranker.rerank(query, candidates)[:top_k]
        t2 = time.perf_counter()
        retrieval_time = round(t2 - t0, 4)
        logging.info(f"retrieval time: {retrieval_time} (search {t1 - t0:.3f}s, title rerank {t2 - t1:.3f}s)")
        llm_answer, llm_elapsed, tokens = self._engine.llms_client.answer(query, top_k_documents)
        stats = {"retrieval_time": retrieval_time, "llm_model": model, "llm_time": llm_elapsed, "tokens": tokens}
        return top_k_documents, llm_answer, stats


def maybe_enable_title_rerank(engine, settings: Optional[TitleRerankSettings] = None):
    """The engine unchanged unless TITLE_RERANK_ENABLED is true; then the reranking wrapper."""
    settings = settings or TitleRerankSettings.from_env()
    if not settings.enabled:
        return engine
    chat_model = create_chat_model(settings.provider, settings.model,
                                   api_key=settings.api_key, base_url=settings.base_url)
    logging.info(f"title rerank ON: provider={settings.provider} model={settings.model} "
                 f"candidates={settings.candidates} pool={settings.pool_paragraphs}")
    return TitleRerankingEngine(engine, TitleReranker(chat_model, settings.timeout_secs, settings.title_field),
                                settings.candidates, settings.pool_paragraphs)
