"""LLM providers for the planner & critic agents.

All optional. If no key/provider is configured, ``NullLLM`` is returned and the
agents fall back to deterministic, keyless heuristics — the pipeline still runs.
Free options: Google Gemini, Groq (Llama), OpenRouter free models.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from ..config import Settings
from .base import LLMProvider

log = logging.getLogger(__name__)


class NullLLM(LLMProvider):
    """No LLM configured. Agents detect this and use heuristics."""

    name = "none"
    available = False

    def complete(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 1024) -> str:
        return ""


class _OpenAICompatLLM(LLMProvider):
    """Shared impl for any OpenAI-compatible /chat/completions endpoint."""

    endpoint: str = ""
    model: str = ""
    api_key: str = ""

    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 1024) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.settings.request_timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            log.warning("%s LLM call failed: %s", self.name, exc)
            return ""


class GroqLLM(_OpenAICompatLLM):
    name = "groq"
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model = settings.groq_model
        self.api_key = settings.groq_api_key


class OpenRouterLLM(_OpenAICompatLLM):
    name = "openrouter"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model = settings.openrouter_model
        self.api_key = settings.openrouter_api_key


class GeminiLLM(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.gemini_model
        self.api_key = settings.gemini_api_key

    def complete(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 1024) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        gen_cfg = {"maxOutputTokens": max_tokens, "temperature": 0.7}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen_cfg,
        }
        try:
            r = requests.post(url, json=payload, timeout=self.settings.request_timeout)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            log.warning("Gemini call failed: %s", exc)
            return ""


def build_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.resolved_llm_provider()
    if provider == "gemini" and settings.gemini_api_key:
        return GeminiLLM(settings)
    if provider == "groq" and settings.groq_api_key:
        return GroqLLM(settings)
    if provider == "openrouter" and settings.openrouter_api_key:
        return OpenRouterLLM(settings)
    return NullLLM()
