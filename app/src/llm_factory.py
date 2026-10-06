"""
A small provider-agnostic factory for chat LLMs.

Used by the optional title reranking stage (`title_rerank.py`), so that stage is not tied to one
vendor. Every provider exposes the same single method:

    complete(system, user, *, max_tokens=200, json_mode=False, timeout=10.0) -> str

Built-in providers
    openai     OpenAI, and any OpenAI-compatible server through `base_url`
               (Azure OpenAI v1 endpoint, Ollama, vLLM, LM Studio, Together, Groq, ...).
               Key: `api_key`, else OAI_API_KEY, else OPENAI_API_KEY.
    anthropic  Anthropic Messages API. Needs `pip install anthropic`. Key: `api_key`, else ANTHROPIC_API_KEY.
    gemini     Google Gemini through google-generativeai (already in requirements.txt).
               Key: `api_key`, else GOOGLE_API_KEY, else GEMINI_API_KEY.
    typesafe   TypeSafe Jev (model jev-latest) over plain HTTP with `requests`. Not a chat model: it is a
               *scoring* model. It exposes `score_titles(question, titles, *, ids, timeout) -> [probability]`,
               one Jev Choice question over all candidate titles, and title_rerank.py sorts by those
               probabilities. Key: `api_key`, else TYPESAFE_API_KEY.

Adding another model takes a few lines and no change anywhere else:

    from llm_factory import register_provider

    @register_provider("my-llm")
    class MyLLM:
        def __init__(self, model, **options): ...
        def complete(self, system, user, *, max_tokens=200, json_mode=False, timeout=10.0) -> str: ...

and then TITLE_RERANK_PROVIDER=my-llm in .env.

A provider may instead expose `score_titles(question, titles, *, ids=None, timeout=10.0) -> list[float]`,
one score per title in input order; title_rerank.py then sorts the candidates by score (stable, so ties keep
retrieval order) instead of asking for a ranked list.

SDKs are imported lazily, when a provider is created, so an unused provider never needs to be installed.
"""
import os
from typing import Callable, Dict, Optional, Protocol


class ChatModel(Protocol):
    def complete(self, system: str, user: str, *, max_tokens: int = 200, json_mode: bool = False,
                 timeout: float = 10.0) -> str:
        ...


_PROVIDERS: Dict[str, Callable[..., ChatModel]] = {}


def register_provider(name: str):
    """Class or function decorator: makes `name` available to `create_chat_model`."""
    def decorator(factory):
        _PROVIDERS[name.lower()] = factory
        return factory
    return decorator


def available_providers() -> list:
    return sorted(_PROVIDERS)


def create_chat_model(provider: str, model: str, **options) -> ChatModel:
    """Build a chat model. `options` are passed to the provider (api_key, base_url, temperature, ...)."""
    key = (provider or "").strip().lower()
    if key not in _PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider!r}; available: {', '.join(available_providers())}")
    if not model:
        raise ValueError(f"no model name given for LLM provider {provider!r}")
    return _PROVIDERS[key](model=model, **options)


def _first_env(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


@register_provider("openai")
class OpenAIChat:
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 temperature: float = 0.0, **_):
        from openai import OpenAI
        self.model, self.temperature = model, temperature
        self.client = OpenAI(api_key=api_key or _first_env("OAI_API_KEY", "OPENAI_API_KEY") or "missing-key",
                             base_url=base_url or None)

    def complete(self, system, user, *, max_tokens=200, json_mode=False, timeout=10.0):
        kwargs = dict(model=self.model, temperature=self.temperature, max_tokens=max_tokens, timeout=timeout,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = self.client.chat.completions.create(**kwargs)
        return (r.choices[0].message.content or "").strip()


@register_provider("anthropic")
class AnthropicChat:
    def __init__(self, model: str, api_key: Optional[str] = None, temperature: float = 0.0, **_):
        import anthropic  # optional dependency: pip install anthropic
        self.model, self.temperature = model, temperature
        self.client = anthropic.Anthropic(api_key=api_key or _first_env("ANTHROPIC_API_KEY"))

    def complete(self, system, user, *, max_tokens=200, json_mode=False, timeout=10.0):
        r = self.client.messages.create(model=self.model, system=system, max_tokens=max_tokens,
                                        temperature=self.temperature, timeout=timeout,
                                        messages=[{"role": "user", "content": user}])
        return "".join(getattr(b, "text", "") for b in r.content).strip()


@register_provider("gemini")
class GeminiChat:
    def __init__(self, model: str, api_key: Optional[str] = None, temperature: float = 0.0, **_):
        import google.generativeai as genai
        genai.configure(api_key=api_key or _first_env("GOOGLE_API_KEY", "GEMINI_API_KEY"))
        self.temperature = temperature
        self._genai, self.model_name = genai, model

    def complete(self, system, user, *, max_tokens=200, json_mode=False, timeout=10.0):
        config = {"temperature": self.temperature, "max_output_tokens": max_tokens}
        if json_mode:
            config["response_mime_type"] = "application/json"
        model = self._genai.GenerativeModel(self.model_name, system_instruction=system, generation_config=config)
        r = model.generate_content(user, request_options={"timeout": timeout})
        return (r.text or "").strip()


# Kept byte-identical to INSTR["C1"] in retrieval_eval/jev_rerank.py, the instruction the "C1" numbers were
# measured with (a unit test checks this).
JEV_CHOICE_INSTRUCTIONS = (
    "The state holds a question a member of the public asked (in Hebrew). Each option is a candidate page "
    "from Kol-Zchut, the Israeli rights-information website, named by its page title. Which page answers "
    "the question? Choose the page about the specific right, benefit, procedure or situation the question "
    "asks about that fits the population and sub-case the question describes (who the person is, their "
    "status, the specific circumstance), rather than a page on a related or broader topic.")


@register_provider("typesafe")
class TypeSafeJev:
    """TypeSafe Jev as a title scorer: one Choice question whose options are the candidate titles.

    Request and response handling mirror retrieval_eval/jev_rerank.py (variant C1). Any HTTP error, timeout
    or malformed reply raises; the caller (TitleReranker) then keeps the plain retrieval order.
    """
    DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **_):
        import requests  # already a dependency (requirements.txt)
        self._requests = requests
        self.model = model
        self.api_key = api_key or _first_env("TYPESAFE_API_KEY")
        self.url = base_url or self.DEFAULT_URL

    @staticmethod
    def build_request(model: str, question: str, titles: list, ids: Optional[list] = None) -> tuple:
        """The request body and the option key of each title. Option keys are the stripped titles; a repeated
        title gets " (doc <id>)" appended (its doc_id, else its position) so every option stays distinct."""
        options, keys, seen = {}, [], set()
        for i, t in enumerate(titles):
            t = str(t).strip()
            k = t if t not in seen else f"{t} (doc {ids[i] if ids else i + 1})"
            seen.add(t)
            options[k] = None
            keys.append(k)
        body = {"model": model, "state": {"question": question},
                "questions": {"answering_page": {"type": "choice", "instructions": JEV_CHOICE_INSTRUCTIONS,
                                                 "criteria": options}}}
        return body, keys

    def score_titles(self, question: str, titles: list, *, ids: Optional[list] = None, timeout: float = 10.0) -> list:
        if not self.api_key:
            raise RuntimeError("no TypeSafe API key (set TITLE_RERANK_API_KEY or TYPESAFE_API_KEY)")
        body, keys = self.build_request(self.model, question, titles, ids)
        r = self._requests.post(self.url, headers={"Authorization": f"Bearer {self.api_key}"}, json=body,
                                timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"TypeSafe HTTP {r.status_code}")
        probs = r.json()["answers"]["answering_page"].get("probabilities")
        if not isinstance(probs, dict) or not probs:
            raise ValueError("TypeSafe reply has no probabilities")
        return [float(probs.get(k) or 0.0) for k in keys]

    def complete(self, system, user, *, max_tokens=200, json_mode=False, timeout=10.0):
        raise NotImplementedError("the typesafe provider scores titles (score_titles); it is not a chat model")
