"""Provider interfaces. Nothing in the pipeline depends on a concrete vendor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class ImageProvider(ABC):
    """Generates a raster image from a text prompt."""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, width: int, height: int, seed: Optional[int] = None) -> Image.Image:
        ...


class LLMProvider(ABC):
    """Text-in / text-out chat completion used by the planner & critic agents."""

    name: str = "base"
    available: bool = True

    @abstractmethod
    def complete(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 1024) -> str:
        ...
