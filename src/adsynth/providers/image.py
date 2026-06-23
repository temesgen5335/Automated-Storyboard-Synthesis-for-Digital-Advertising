"""Image-generation providers.

Defaults to **Pollinations** (free, keyless, FLUX-based). Falls back to an
offline deterministic **mock** so the pipeline always produces output, even
with no network or keys.
"""
from __future__ import annotations

import hashlib
import io
import logging
import urllib.parse
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFont

from ..config import Settings
from .base import ImageProvider

log = logging.getLogger(__name__)


def _cache_key(provider: str, prompt: str, w: int, h: int, seed: Optional[int]) -> str:
    raw = f"{provider}|{prompt}|{w}x{h}|{seed}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class _CachingMixin:
    cache_dir: Path

    def _cached(self, key: str) -> Optional[Image.Image]:
        p = self.cache_dir / f"{key}.png"
        if p.exists():
            try:
                return Image.open(p).convert("RGBA")
            except Exception:
                return None
        return None

    def _store(self, key: str, img: Image.Image) -> None:
        try:
            img.convert("RGBA").save(self.cache_dir / f"{key}.png")
        except Exception:
            pass


class MockImageProvider(ImageProvider):
    """Deterministic, offline placeholder generator (gradient + label)."""

    name = "mock"

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, prompt: str, width: int, height: int, seed: Optional[int] = None) -> Image.Image:
        h = int(hashlib.sha1(f"{prompt}{seed}".encode()).hexdigest(), 16)
        c1 = ((h >> 0) & 255, (h >> 8) & 255, (h >> 16) & 255)
        c2 = ((h >> 24) & 255, (h >> 32) & 255, (h >> 40) & 255)
        img = Image.new("RGBA", (width, height))
        px = img.load()
        for y in range(height):
            t = y / max(1, height - 1)
            row = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3)) + (255,)
            for x in range(width):
                px[x, y] = row
        draw = ImageDraw.Draw(img)
        label = prompt[:60] + ("…" if len(prompt) > 60 else "")
        try:
            font = ImageFont.load_default()
        except Exception:  # pragma: no cover
            font = None
        draw.rectangle([4, height - 26, width - 4, height - 4], fill=(0, 0, 0, 120))
        draw.text((8, height - 22), label, fill=(255, 255, 255, 255), font=font)
        return img


class PollinationsImageProvider(_CachingMixin, ImageProvider):
    """Free, keyless text-to-image via image.pollinations.ai."""

    name = "pollinations"
    BASE = "https://image.pollinations.ai/prompt/"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_dir = settings.cache_dir
        self._fallback = MockImageProvider(settings)

    def generate(self, prompt: str, width: int, height: int, seed: Optional[int] = None) -> Image.Image:
        seed = self.settings.seed if seed is None else seed
        key = _cache_key(self.name, prompt, width, height, seed)
        if (hit := self._cached(key)) is not None:
            return hit
        url = self.BASE + urllib.parse.quote(prompt, safe="")
        params = {"width": width, "height": height, "seed": seed, "nologo": "true", "model": "flux"}
        try:
            r = requests.get(url, params=params, timeout=self.settings.request_timeout)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGBA").resize((width, height))
            self._store(key, img)
            return img
        except Exception as exc:  # network / rate-limit → graceful fallback
            log.warning("Pollinations failed (%s); using mock generator.", exc)
            return self._fallback.generate(prompt, width, height, seed)


class HuggingFaceImageProvider(_CachingMixin, ImageProvider):
    """Hugging Face Inference API (free tier). Requires HF_TOKEN."""

    name = "huggingface"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_dir = settings.cache_dir
        self.model = settings.hf_image_model
        self._fallback = MockImageProvider(settings)

    def generate(self, prompt: str, width: int, height: int, seed: Optional[int] = None) -> Image.Image:
        seed = self.settings.seed if seed is None else seed
        key = _cache_key(self.name + self.model, prompt, width, height, seed)
        if (hit := self._cached(key)) is not None:
            return hit
        if not self.settings.hf_token:
            log.warning("HF_TOKEN missing; using mock generator.")
            return self._fallback.generate(prompt, width, height, seed)
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.settings.hf_token}"}
        payload = {"inputs": prompt, "parameters": {"width": width, "height": height}}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.settings.request_timeout)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGBA").resize((width, height))
            self._store(key, img)
            return img
        except Exception as exc:
            log.warning("HuggingFace failed (%s); using mock generator.", exc)
            return self._fallback.generate(prompt, width, height, seed)


def build_image_provider(settings: Settings) -> ImageProvider:
    p = settings.image_provider.lower()
    if p == "pollinations":
        return PollinationsImageProvider(settings)
    if p == "huggingface":
        return HuggingFaceImageProvider(settings)
    return MockImageProvider(settings)
