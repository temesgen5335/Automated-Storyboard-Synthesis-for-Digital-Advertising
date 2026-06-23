"""Provider interfaces. Nothing in the pipeline depends on a concrete vendor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class ImageProvider(ABC):
    """Produces a raster image for an asset.

    ``prompt`` is the rich, brand-aware description (used by generative
    backends). ``query`` is a short search phrase and ``category`` the asset
    role — used by retrieval backends (e.g. Openverse) that search real images
    rather than generate them. Generative backends ignore the latter two.
    """

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: Optional[int] = None,
        *,
        query: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Image.Image:
        ...


class LLMProvider(ABC):
    """Text-in / text-out chat completion used by the planner & critic agents."""

    name: str = "base"
    available: bool = True

    @abstractmethod
    def complete(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 1024) -> str:
        ...
