"""Asset generator — briefs → GeneratedAsset (PIL images).

Text assets are rendered with PIL; visual assets are produced by the image
provider using the planner's brand-aware prompt.
"""
from __future__ import annotations

import hashlib

from ..agents.planner import Planner
from ..formats import AdFormat
from ..providers.base import ImageProvider
from ..schemas import AssetBrief, BrandKit, Concept, Frame, GeneratedAsset
from .text_render import render_text_asset


def _seed(*parts: object) -> int:
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.sha1(raw.encode()).hexdigest(), 16) % (2**31)


def _native_size(brief: AssetBrief, fmt: AdFormat) -> tuple[int, int]:
    """Native generation size per category (compositor resizes to placement)."""
    cat = brief.category.lower()
    if "background" in cat:
        return fmt.width, fmt.height
    if any(k in cat for k in ("logo", "icon", "badge", "qr")):
        return 256, 256
    if "product" in cat or "mascot" in cat or "illustration" in cat or "photo" in cat:
        return 512, 512
    return 512, 512


class AssetGenerator:
    def __init__(self, image_provider: ImageProvider, planner: Planner, brand: BrandKit, variant: int = 0):
        self.image = image_provider
        self.planner = planner
        self.brand = brand
        self.variant = variant

    def generate(self, brief: AssetBrief, concept: Concept, fmt: AdFormat, frame_id: str, idx: int) -> GeneratedAsset:
        if brief.is_text:
            img = render_text_asset(brief, self.brand, max_width=int(fmt.width * 0.9))
            return GeneratedAsset(brief=brief, image=img, source="pil-text")
        prompt = self.planner.image_prompt(brief, concept, self.brand, fmt)
        w, h = _native_size(brief, fmt)
        seed = _seed(concept.name, frame_id, brief.category, idx, self.variant)
        img = self.image.generate(prompt, w, h, seed=seed)
        return GeneratedAsset(brief=brief, image=img, source=self.image.name)

    def generate_frame(self, frame: Frame, concept: Concept, fmt: AdFormat) -> list[GeneratedAsset]:
        return [
            self.generate(brief, concept, fmt, frame.id, i)
            for i, brief in enumerate(frame.assets)
        ]
