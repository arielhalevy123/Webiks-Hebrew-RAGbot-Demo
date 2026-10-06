"""Tests for the TypeSafe Jev title scorer (TITLE_RERANK_PROVIDER=typesafe). HTTP is mocked; no network."""
import ast
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from llm_factory import JEV_CHOICE_INSTRUCTIONS, TypeSafeJev, available_providers, create_chat_model
from title_rerank import TitleReranker, TitleRerankSettings, TitleRerankingEngine, maybe_enable_title_rerank

REPO = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code, self._payload = status, payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def reply(probs):
    return FakeResponse(200, {"model": "jev-x", "answers": {"answering_page": {"probabilities": probs}},
                              "usage": {"input_tokens": 100, "output_tokens": 0}})


def docs(*titles):
    return [{"doc_id": 100 + i, "title": t, "link": f"l{i}", "content": f"c{i}"} for i, t in enumerate(titles)]


def jev(api_key="k"):
    return create_chat_model("typesafe", "jev-latest", api_key=api_key)


def run(post_result, titles, timeout=5.0):
    """Rerank `titles` with requests.post mocked; returns (reordered titles, captured post kwargs)."""
    seen = {}

    def fake_post(url, **kw):
        seen.update(kw, url=url)
        if isinstance(post_result, Exception):
            raise post_result
        return post_result

    d = docs(*titles)
    with patch("requests.post", fake_post):
        out = TitleReranker(jev(), timeout_secs=timeout).rerank("דרכון דחוף", d)
    return [x["title"] for x in out], seen


# ---- request ----------------------------------------------------------------------------------------

def test_registered_and_default_model():
    assert "typesafe" in available_providers()
    env = {"TITLE_RERANK_ENABLED": "true", "TITLE_RERANK_PROVIDER": "typesafe"}
    with patch.dict(os.environ, env, clear=True):
        assert TitleRerankSettings.from_env().model == "jev-latest"


def test_request_is_one_choice_over_all_titles_in_retrieval_order():
    titles, seen = run(reply({"b": 0.9}), ["a", "b", " c "])
    body = seen["json"]
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["headers"] == {"Authorization": "Bearer k"} and seen["timeout"] == 5.0
    assert body["model"] == "jev-latest" and body["state"] == {"question": "דרכון דחוף"}
    q = body["questions"]["answering_page"]
    assert q["type"] == "choice" and q["instructions"] == JEV_CHOICE_INSTRUCTIONS
    assert list(q["criteria"]) == ["a", "b", "c"] and set(q["criteria"].values()) == {None}


def test_duplicate_titles_get_distinct_option_keys():
    body, keys = TypeSafeJev.build_request("jev-latest", "q", ["x", "x", "y"], ids=[7, 8, 9])
    assert keys == ["x", "x (doc 8)", "y"]
    assert list(body["questions"]["answering_page"]["criteria"]) == keys


def test_instruction_matches_the_measured_experiment():
    """The production instruction must equal INSTR["C1"] in retrieval_eval/jev_rerank.py (read without importing it)."""
    tree = ast.parse((REPO / "retrieval_eval" / "jev_rerank.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
                and ast.literal_eval(node.targets[0].slice) == "C1"):
            assert ast.literal_eval(node.value) == JEV_CHOICE_INSTRUCTIONS
            return
    pytest.fail('INSTR["C1"] not found in jev_rerank.py')


# ---- ordering ---------------------------------------------------------------------------------------

def test_sorted_by_probability_highest_first():
    titles, _ = run(reply({"a": 0.05, "b": 0.15, "c": 0.8, "d": 0.0}), ["a", "b", "c", "d"])
    assert titles == ["c", "b", "a", "d"]


def test_ties_and_zero_or_missing_options_keep_retrieval_order():
    titles, _ = run(reply({"b": 0.4, "d": 0.4, "a": 0.0}), ["a", "b", "c", "d", "e"])
    assert titles == ["b", "d", "a", "c", "e"]


# ---- failures keep retrieval order ------------------------------------------------------------------

@pytest.mark.parametrize("result", [
    requests.Timeout("read timed out"),
    requests.ConnectionError("down"),
    FakeResponse(401, {"error": "bad key"}),
    FakeResponse(500, {"error": "boom"}),
    FakeResponse(200, ValueError("not json")),
    FakeResponse(200, {"answers": {}}),
    FakeResponse(200, {"answers": {"answering_page": {"probabilities": {}}}}),
    FakeResponse(200, {"answers": {"answering_page": {"probabilities": "nope"}}}),
])
def test_any_error_or_invalid_reply_keeps_retrieval_order(result):
    titles, _ = run(result, ["a", "b", "c"])
    assert titles == ["a", "b", "c"]


def test_timeout_setting_is_passed_to_http():
    _, seen = run(reply({"a": 1.0}), ["a", "b"], timeout=1.5)
    assert seen["timeout"] == 1.5


def test_missing_key_keeps_retrieval_order_without_calling():
    called = []
    d = docs("a", "b")
    with patch.dict(os.environ, {}, clear=True), patch("requests.post", lambda *a, **k: called.append(1)):
        out = TitleReranker(jev(api_key=None)).rerank("q", d)
    assert out == d and called == []


def test_key_from_typesafe_env_and_title_rerank_key_wins():
    with patch.dict(os.environ, {"TYPESAFE_API_KEY": "env-k"}, clear=True):
        assert jev(api_key=None).api_key == "env-k"
        assert jev(api_key="explicit").api_key == "explicit"


def test_complete_is_not_supported():
    with pytest.raises(NotImplementedError):
        jev().complete("s", "u")


# ---- end to end through the switch ------------------------------------------------------------------

def test_enabled_from_env_reorders_the_pages_sent_to_the_answering_llm():
    try:
        from tests.test_title_rerank import FakeEngine, hits_for
    except ImportError:
        from test_title_rerank import FakeEngine, hits_for
    eng = FakeEngine(hits_for((1, "a"), (2, "b"), (3, "c")))
    env = {"TITLE_RERANK_ENABLED": "true", "TITLE_RERANK_PROVIDER": "typesafe", "TYPESAFE_API_KEY": "k",
           "TITLE_RERANK_TIMEOUT_SECS": "2"}
    with patch.dict(os.environ, env, clear=True):
        wrapped = maybe_enable_title_rerank(eng)
    assert isinstance(wrapped, TitleRerankingEngine) and isinstance(wrapped.reranker.chat_model, TypeSafeJev)
    with patch("requests.post", lambda url, **kw: reply({"c": 0.7, "a": 0.2})):
        top, _, _ = wrapped.answer_query("q", 2, "gpt")
    assert [d["doc_id"] for d in top] == [3, 1] and eng.answered_with == top
