"""Runtime configuration, sourced from environment variables with safe defaults.

Everything works with **zero configuration** (keyless providers + mock fallback).
Set env vars to upgrade quality. See ``.env.example``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env if python-dotenv is available (optional dependency).
try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONCEPTS_PATH = DATA_DIR / "concepts.json"
CATEGORIES_PATH = DATA_DIR / "categories.txt"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    # --- provider selection -------------------------------------------------
    # image: pollinations | openverse | huggingface | mock
    image_provider: str = field(default_factory=lambda: _env("ADSYNTH_IMAGE_PROVIDER", "pollinations"))
    # llm: gemini | groq | openrouter | none  (none => keyless template planner)
    llm_provider: str = field(default_factory=lambda: _env("ADSYNTH_LLM_PROVIDER", "none"))

    # --- keys (all optional) ------------------------------------------------
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))

    # --- model names --------------------------------------------------------
    hf_image_model: str = field(default_factory=lambda: _env("ADSYNTH_HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell"))
    gemini_model: str = field(default_factory=lambda: _env("ADSYNTH_GEMINI_MODEL", "gemini-1.5-flash"))
    groq_model: str = field(default_factory=lambda: _env("ADSYNTH_GROQ_MODEL", "llama-3.3-70b-versatile"))
    openrouter_model: str = field(default_factory=lambda: _env("ADSYNTH_OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"))

    # --- behaviour ----------------------------------------------------------
    request_timeout: int = field(default_factory=lambda: int(_env("ADSYNTH_TIMEOUT", "60")))
    seed: int = field(default_factory=lambda: int(_env("ADSYNTH_SEED", "42")))
    cache_dir: Path = field(default_factory=lambda: OUTPUT_DIR / ".cache")

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # convenience: auto-pick an LLM provider if a key is present but provider is "none"
    def resolved_llm_provider(self) -> str:
        if self.llm_provider and self.llm_provider != "none":
            return self.llm_provider
        if self.gemini_api_key:
            return "gemini"
        if self.groq_api_key:
            return "groq"
        if self.openrouter_api_key:
            return "openrouter"
        return "none"


def get_settings() -> Settings:
    return Settings()
