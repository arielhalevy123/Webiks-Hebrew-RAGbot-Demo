"""Tests for the optional title reranking stage and the LLM factory. No network, no Elasticsearch."""
import os
from unittest.mock import patch

import pytest

import llm_factory
from llm_factory import create_chat_model, register_provider, available_providers
from title_rerank import (PICK_PROMPT, TitleReranker, TitleRerankSettings, TitleRerankingEngine,
                          build_user_message, maybe_enable_title_rerank, parse_ranking)


class FakeChat:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def complete(self, system, user, *, max_tokens=200, json_mode=False, timeout=10.0):
        self.calls.append({"system": system, "user": user, "json_mode": json_mode, "timeout": timeout})
        if self.error:
            raise self.error
        return self.reply


def docs(*titles):
    return [{"doc_id": i + 1, "title": t, "link": f"l{i}", "content": f"c{i}"} for i, t in enumerate(titles)]


# ---- parsing ---------------------------------------------------------------------------------------

def test_parse_json_reply():
    assert parse_ranking('{"ranking": [3, 1]}', 5) == [2, 0]


def test_parse_drops_out_of_range_duplicates_and_junk():
    assert parse_ranking('{"ranking": [7, 2, 2, "x", 0, 1]}', 3) == [1, 0]


def test_parse_tolerates_prose_and_bare_numbers():
    assert parse_ranking('Sure: {"ranking": [2]} hope that helps', 4) == [1]
    assert parse_ranking("2, 4 and 1", 4) == [1, 3, 0]


def test_parse_empty_or_garbage():
    assert parse_ranking("", 4) == [] and parse_ranking("no idea", 4) == []


def test_parse_caps_picks():
    assert len(parse_ranking('{"ranking": [1,2,3,4,5,6,7,8,9,10,11,12]}', 30)) == 10


# ---- reranker ---------------------------------------------------------------------------------------

def test_rerank_moves_picked_pages_first_and_keeps_the_rest_in_order():
    chat = FakeChat('{"ranking": [3, 1]}')
    out = TitleReranker(chat).rerank("q", docs("a", "b", "c", "d"))
    assert [d["title"] for d in out] == ["c", "a", "b", "d"]
    assert chat.calls[0]["system"] == PICK_PROMPT and chat.calls[0]["json_mode"] is True


def test_rerank_sends_numbered_titles():
    chat = FakeChat('{"ranking": [1]}')
    TitleReranker(chat).rerank("מי זכאי?", docs("הנחה בארנונה לנכים", "הנחה בארנונה לנכי עבודה"))
    assert chat.calls[0]["user"] == build_user_message("מי זכאי?", ["הנחה בארנונה לנכים", "הנחה בארנונה לנכי עבודה"])
    assert "1. הנחה בארנונה לנכים\n2. הנחה בארנונה לנכי עבודה" in chat.calls[0]["user"]


def test_rerank_falls_back_on_llm_error():
    d = docs("a", "b", "c")
    assert TitleReranker(FakeChat(error=TimeoutError("slow"))).rerank("q", d) == d


def test_rerank_falls_back_on_unparsable_reply():
    d = docs("a", "b", "c")
    assert TitleReranker(FakeChat("I cannot help with that")).rerank("q", d) == d


def test_rerank_skips_the_call_for_fewer_than_two_pages():
    chat = FakeChat('{"ranking": [1]}')
    assert TitleReranker(chat).rerank("q", docs("only")) == docs("only") and chat.calls == []


# ---- engine wrapper -------------------------------------------------------------------------------

class FakeEngine:
    def __init__(self, hits):
        self.hits, self.searched_with, self.answered_with = hits, None, None
        engine = self

        class R:
            def encode(self, q):
                return [0.1, 0.2]

        class E:
            def search(self, vector, size=50):
                engine.searched_with = size
                return engine.hits

        class L:
            def answer(self, query, top_docs):
                engine.answered_with = top_docs
                return "answer", 0.5, 42

        self.retrieval_model, self.elastic_model, self.llms_client = R(), E(), L()

    def update_docs(self, *a, **k):
        return "delegated"


def hits_for(*pairs):
    return [{"_source": {"doc_id": d, "title": t, "content": f"p{n}", "link": ""}} for n, (d, t) in enumerate(pairs)]


def test_wrapper_dedupes_pages_reranks_and_cuts_to_top_k():
    eng = FakeEngine(hits_for((1, "a"), (1, "a"), (2, "b"), (3, "c"), (4, "d")))
    wrapped = TitleRerankingEngine(eng, TitleReranker(FakeChat('{"ranking": [3]}')), candidates=30, pool_paragraphs=200)
    top, answer, stats = wrapped.answer_query("q", 2, "gpt")
    assert [d["doc_id"] for d in top] == [3, 1]
    assert eng.answered_with == top and answer == "answer"
    assert eng.searched_with == 200 and stats["llm_model"] == "gpt" and stats["tokens"] == 42
    assert top[0]["content"] == "p3" and [d["content"] for d in top] == ["p3", "p0"]   # best paragraph per page


def test_wrapper_respects_candidate_limit():
    eng = FakeEngine(hits_for(*[(i, f"t{i}") for i in range(1, 50)]))
    chat = FakeChat('{"ranking": [1]}')
    TitleRerankingEngine(eng, TitleReranker(chat), candidates=5).answer_query("q", 3, "m")
    assert chat.calls[0]["user"].count("\n") == 2 + 5 - 1 + 1


def test_wrapper_delegates_everything_else():
    eng = FakeEngine([])
    assert TitleRerankingEngine(eng, TitleReranker(FakeChat("{}"))).update_docs() == "delegated"


# ---- switch and factory ---------------------------------------------------------------------------

def test_disabled_by_default_returns_the_same_engine_object():
    with patch.dict(os.environ, {}, clear=True):
        assert TitleRerankSettings.from_env().enabled is False
        eng = object()
        assert maybe_enable_title_rerank(eng) is eng


def test_enabled_from_env_with_a_registered_provider():
    @register_provider("unit-test-llm")
    class _Fake(FakeChat):
        def __init__(self, model, **options):
            super().__init__('{"ranking": [1]}'); self.model, self.options = model, options

    env = {"TITLE_RERANK_ENABLED": "true", "TITLE_RERANK_PROVIDER": "unit-test-llm", "TITLE_RERANK_MODEL": "m1",
           "TITLE_RERANK_CANDIDATES": "12", "TITLE_RERANK_TIMEOUT_SECS": "2.5"}
    with patch.dict(os.environ, env, clear=True):
        wrapped = maybe_enable_title_rerank(FakeEngine([]))
    assert isinstance(wrapped, TitleRerankingEngine)
    assert wrapped.candidates == 12 and wrapped.reranker.timeout_secs == 2.5
    assert wrapped.reranker.chat_model.model == "m1"


def test_factory_rejects_unknown_provider_and_lists_known_ones():
    with pytest.raises(ValueError) as e:
        create_chat_model("no-such-llm", "x")
    assert "openai" in str(e.value) and "anthropic" in str(e.value) and "gemini" in str(e.value)


def test_factory_requires_a_model_name():
    with pytest.raises(ValueError):
        create_chat_model("openai", "")


def test_builtin_providers_registered():
    assert {"openai", "anthropic", "gemini"} <= set(available_providers())


def test_openai_provider_builds_the_request_without_network():
    class FakeCompletions:
        def create(self, **kw):
            FakeCompletions.kw = kw
            class M: content = ' {"ranking": [2]} '
            class C: message = M()
            class R: choices = [C()]
            return R()

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            FakeOpenAI.args = (api_key, base_url)
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    with patch("openai.OpenAI", FakeOpenAI):
        chat = create_chat_model("openai", "gpt-4o-mini", api_key="k", base_url="http://localhost:11434/v1")
        out = chat.complete("sys", "user", max_tokens=50, json_mode=True, timeout=3)
    assert out == '{"ranking": [2]}'
    assert FakeOpenAI.args == ("k", "http://localhost:11434/v1")
    kw = FakeCompletions.kw
    assert kw["model"] == "gpt-4o-mini" and kw["temperature"] == 0.0 and kw["max_tokens"] == 50
    assert kw["response_format"] == {"type": "json_object"} and kw["timeout"] == 3
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
