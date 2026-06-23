"""Critic / grading agent — scores composed frames on creative-quality heuristics.

Mirrors the original brief's "critic/grading agent". Heuristic by default (no
keys); an LLM, when present, adds a qualitative note. Scores feed an optional
regeneration loop and the DCO variant ranking.
"""
from __future__ import annotations

import colorsys
import logging

from PIL import Image

from ..providers.base import LLMProvider
from ..schemas import BrandKit, ComposedFrame, Concept

log = logging.getLogger(__name__)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = (c / 255 for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _pixels(image: Image.Image) -> list[tuple[int, int, int]]:
    """RGB pixels as a list of tuples (avoids the deprecated getdata())."""
    rgb = image.convert("RGB")
    raw = rgb.tobytes()
    return [(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)]


def _palette_alignment(image: Image.Image, brand: BrandKit, sample: int = 1500) -> float:
    """Fraction of sampled pixels whose hue is near a brand-palette hue."""
    small = image.convert("RGB").resize((40, 40))
    px = _pixels(small)
    brand_hues = []
    for hexc in brand.palette:
        r, g, b = _hex_to_rgb(hexc)
        brand_hues.append(colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0])
    near = 0
    for (r, g, b) in px:
        h = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0]
        if any(min(abs(h - bh), 1 - abs(h - bh)) < 0.08 for bh in brand_hues):
            near += 1
    return near / max(1, len(px))


class Critic:
    def __init__(self, llm: LLMProvider, brand: BrandKit):
        self.llm = llm
        self.brand = brand

    def grade_frame(self, cf: ComposedFrame, *, is_final: bool) -> dict:
        scores: dict[str, float] = {}
        notes: list[str] = []

        # 1. asset density — not empty, not crowded
        n = len(cf.placements)
        scores["composition"] = 1.0 if 1 <= n <= 5 else max(0.3, 1.0 - 0.15 * abs(n - 3))

        # 2. CTA presence on the final frame
        has_cta = any("cta" in p.category.lower() for p in cf.placements)
        if is_final:
            scores["cta"] = 1.0 if has_cta else 0.3
            if not has_cta:
                notes.append("Final frame is missing a clear Call-To-Action.")
        else:
            scores["cta"] = 1.0

        # 3. brand-palette alignment
        align = _palette_alignment(cf.image, self.brand) if cf.image is not None else 0.0
        scores["brand"] = min(1.0, 0.4 + align)  # presence-weighted

        # 4. text legibility — background contrast under text placements
        legible = self._text_contrast(cf)
        scores["legibility"] = legible
        if legible < 0.5:
            notes.append("Low text/background contrast — legibility risk.")

        overall = round(sum(scores.values()) / len(scores), 3)
        return {"overall": overall, "scores": {k: round(v, 3) for k, v in scores.items()}, "notes": notes}

    def _text_contrast(self, cf: ComposedFrame) -> float:
        if cf.image is None:
            return 1.0
        text_places = [p for p in cf.placements if "text" in p.category.lower() or "cta" in p.category.lower()]
        if not text_places:
            return 1.0
        img = cf.image.convert("RGB")
        worst = 1.0
        for p in text_places:
            region = img.crop((p.x, p.y, min(img.width, p.x + p.width), min(img.height, p.y + p.height)))
            if region.width == 0 or region.height == 0:
                continue
            pixels = _pixels(region)
            avg = [sum(c) / len(c) for c in zip(*pixels)]
            lum = _luminance(tuple(int(a) for a in avg))  # type: ignore[arg-type]
            # we render text/CTA on a contrasting plate, so mid-luminance is fine;
            # penalise only extreme mid-grey backgrounds where overlay text vanishes
            worst = min(worst, 1.0 - max(0.0, 1.0 - abs(lum - 0.5) * 2) * 0.5)
        return round(worst, 3)

    def grade_concept(self, concept: Concept, frames: list[ComposedFrame]) -> dict:
        per_frame = []
        for i, cf in enumerate(frames):
            g = self.grade_frame(cf, is_final=(i == len(frames) - 1))
            cf.critique = g
            per_frame.append(g)
        overall = round(sum(g["overall"] for g in per_frame) / max(1, len(per_frame)), 3)
        result = {"overall": overall, "frames": per_frame}
        if self.llm.available:
            result["llm_note"] = self._llm_note(concept)
        return result

    def _llm_note(self, concept: Concept) -> str:
        system = (
            "You are an advertising creative director. In <=2 sentences, critique "
            "the narrative flow and engagement of this ad concept."
        )
        flow = " -> ".join(f"{f.id}({f.interaction_type})" for f in concept.frames)
        user = f"Concept: {concept.name}\nFlow: {flow}\nExplanation: {concept.explanation[:600]}"
        return self.llm.complete(system, user, max_tokens=160).strip()
