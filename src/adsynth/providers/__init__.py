"""Provider registry — build concrete image/LLM providers from Settings."""
from __future__ import annotations

from ..config import Settings, get_settings
from .base import ImageProvider, LLMProvider
from .image import build_image_provider
from .llm import build_llm_provider


class Providers:
    """Bundle of the concrete providers used across the pipeline."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.image: ImageProvider = build_image_provider(self.settings)
        self.llm: LLMProvider = build_llm_provider(self.settings)

    def describe(self) -> dict:
        return {
            "image_provider": self.image.name,
            "llm_provider": self.llm.name,
            "llm_available": self.llm.available,
        }


__all__ = ["Providers", "ImageProvider", "LLMProvider", "build_image_provider", "build_llm_provider"]
