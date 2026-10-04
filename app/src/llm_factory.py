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

Adding another model takes a few lines and no change anywhere else:

    from llm_factory import register_provider

    @register_provider("my-llm")
    class MyLLM:
        def __init__(self, model, **options): ...
        def complete(self, system, user, *, max_tokens=200, json_mode=False, timeout=10.0) -> str: ...

and then TITLE_RERANK_PROVIDER=my-llm in .env.

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
