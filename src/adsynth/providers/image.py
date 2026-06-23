"""Image-generation providers.

Defaults to **Pollinations** (free, keyless, FLUX-based). Falls back to an
offline deterministic **mock** so the pipeline always produces output, even
with no network or keys.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
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

    def _store(self, key: str, img: Image.Image, attribution: Optional[dict] = None) -> None:
        try:
            img.convert("RGBA").save(self.cache_dir / f"{key}.png")
            if attribution is not None:
                (self.cache_dir / f"{key}.json").write_text(json.dumps(attribution))
        except Exception:
            pass

    def _cached_attribution(self, key: str) -> Optional[dict]:
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None


class MockImageProvider(ImageProvider):
    """Deterministic, offline placeholder generator (gradient + label)."""

    name = "mock"

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, prompt, width, height, seed=None, *, query=None, category=None) -> Image.Image:
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

    def generate(self, prompt, width, height, seed=None, *, query=None, category=None) -> Image.Image:
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

    def generate(self, prompt, width, height, seed=None, *, query=None, category=None) -> Image.Image:
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


class OpenverseImageProvider(_CachingMixin, ImageProvider):
    """Retrieval (not generation): fetch a REAL, CC-licensed image that matches
    the asset from the Openverse API (https://openverse.org). Keyless.

    This is the "bring your own / real artifact" path — the same compositor that
    handles generated assets composes these real photos and logos into frames.
    Falls back to Lorem Picsum (real photos, keyless) then the mock generator.
    """

    name = "openverse"
    API = "https://api.openverse.org/v1/images/"
    PICSUM = "https://picsum.photos/seed/{seed}/{w}/{h}"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_dir = settings.cache_dir
        self._fallback = MockImageProvider(settings)
        self._headers = {"User-Agent": "adsynth/0.1 (storyboard synthesis; research)"}
        self.last_attribution: dict | None = None

    _STOP = {
        "the", "a", "an", "of", "and", "with", "to", "for", "in", "on", "is", "are",
        "this", "that", "as", "by", "at", "from", "into", "their", "its", "it", "be",
        "subtle", "animated", "various", "set", "scene", "designed", "encourages",
        "viewers", "featuring", "showcasing", "series", "quick", "exciting",
    }

    def _keywords(self, text: str) -> str:
        """Extract a short, search-friendly query from a prose asset description.

        Prefers proper-noun phrases (brand/product names like "LEGO CITY"); falls
        back to the first salient content words. Openverse needs terse queries.
        """
        text = text or ""
        # proper-noun runs: capitalized or ALL-CAPS word sequences
        proper = re.findall(r"\b(?:[A-Z][A-Za-z0-9]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9]+|[A-Z]{2,}))*\b", text)
        proper = [p for p in proper if len(p) > 2]
        if proper:
            return " ".join(max(proper, key=len).split()[:3])
        words = [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w not in self._STOP and len(w) > 2]
        return " ".join(words[:3])

    def _search_query(self, query: Optional[str], category: Optional[str], prompt: str) -> str:
        kw = self._keywords(query or prompt or "")
        cat = (category or "").lower()
        if "logo" in cat:
            return f"{kw} logo".strip()
        if "background" in cat:
            return f"{kw} landscape".strip()
        if "icon" in cat:
            return "swipe gesture icon" if "swipe" in (query or "").lower() else "tap gesture icon"
        if "product" in cat:
            return f"{kw} product".strip()
        if "mascot" in cat or "illustration" in cat:
            return f"{kw} illustration".strip()
        return kw or "advertising"

    def _openverse_candidates(self, query: str, seed: int) -> list[dict]:
        """Search results, ordered so the seed picks a stable starting point."""
        params = {"q": query, "page_size": 12, "mature": "false"}
        r = requests.get(self.API, params=params, headers=self._headers, timeout=self.settings.request_timeout)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return []
        start = seed % len(results)
        ordered = results[start:] + results[:start]  # rotate, keep all as fallbacks
        return [
            {
                "url": x.get("url"),
                "title": x.get("title"),
                "creator": x.get("creator"),
                "license": x.get("license"),
                "license_url": x.get("license_url"),
                "source": x.get("source"),
                "foreign_landing_url": x.get("foreign_landing_url"),
                "query": query,
            }
            for x in ordered
            if x.get("url")
        ]

    def generate(self, prompt, width, height, seed=None, *, query=None, category=None) -> Image.Image:
        seed = self.settings.seed if seed is None else seed
        q = self._search_query(query, category, prompt)
        key = _cache_key(self.name, q, width, height, seed)
        cached = self._cached(key)
        if cached is not None:
            self.last_attribution = self._cached_attribution(key)
            return cached
        # 1) Openverse search → try candidates until one downloads (some URLs 403/404)
        try:
            for cand in self._openverse_candidates(q, seed)[:6]:
                try:
                    resp = requests.get(cand["url"], headers=self._headers, timeout=self.settings.request_timeout)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                    img = _cover_resize(img, width, height)
                    self.last_attribution = cand
                    self._store(key, img, attribution=cand)
                    return img
                except Exception:
                    continue  # try next candidate
            log.warning("Openverse: no downloadable result for %r; trying Picsum.", q)
        except Exception as exc:
            log.warning("Openverse search failed for %r (%s); trying Picsum.", q, exc)
        # 2) Picsum (real photo, keyless)
        try:
            purl = self.PICSUM.format(seed=abs(seed) % 1000, w=width, h=height)
            resp = requests.get(purl, headers=self._headers, timeout=self.settings.request_timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
            img = _cover_resize(img, width, height)
            attribution = {"source": "Lorem Picsum", "license": "Unsplash", "query": q}
            self.last_attribution = attribution
            self._store(key, img, attribution=attribution)
            return img
        except Exception as exc:
            log.warning("Picsum failed (%s); using mock generator.", exc)
        return self._fallback.generate(prompt, width, height, seed)


def _cover_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    """Resize+center-crop to exactly (w, h) preserving aspect (CSS object-fit: cover)."""
    sw, sh = img.size
    scale = max(w / sw, h / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


def build_image_provider(settings: Settings) -> ImageProvider:
    p = settings.image_provider.lower()
    if p == "pollinations":
        return PollinationsImageProvider(settings)
    if p == "huggingface":
        return HuggingFaceImageProvider(settings)
    if p == "openverse":
        return OpenverseImageProvider(settings)
    return MockImageProvider(settings)
