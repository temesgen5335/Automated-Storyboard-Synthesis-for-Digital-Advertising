"""Planner agent — turns asset briefs into concrete generation prompts/copy.

Works with zero keys (heuristic prompt construction). If an LLM is configured it
refines ad copy and enriches image prompts for higher quality.
"""
from __future__ import annotations

import json
import logging

from ..formats import AdFormat
from ..providers.base import LLMProvider
from ..schemas import AssetBrief, BrandKit, Concept

log = logging.getLogger(__name__)

_NEGATIVE = "no watermark, no extra text, no logos"


class Planner:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    # -- image prompt ---------------------------------------------------------
    def image_prompt(self, brief: AssetBrief, concept: Concept, brand: BrandKit, fmt: AdFormat) -> str:
        """Deterministic, brand-aware text-to-image prompt for a visual asset."""
        style = ", ".join(brand.style_keywords)
        palette = " / ".join(brand.palette)
        role = {
            "Background Image": "full-bleed background scene, no central subject",
            "Logo": "minimal vector brand logo on transparent-like flat background",
            "Product Image": "centered product hero shot, studio lighting",
            "Mascot": "friendly brand mascot character, full body",
            "Illustration": "flat vector illustration",
            "Icon": "simple flat UI icon",
        }.get(brief.category, "advertising creative element")
        return (
            f"{brief.description}. {role}. Brand style: {style}. "
            f"Color palette: {palette}. Mobile ad creative, {fmt.orientation}, "
            f"clean composition, high quality, {_NEGATIVE}."
        )

    # -- copy refinement (optional LLM) --------------------------------------
    def refine_copy(self, concept: Concept) -> dict[str, str]:
        """Return ``{asset_signature: punchy_copy}`` for text assets.

        Uses the LLM when available; otherwise returns {} and callers keep the
        heuristic copy already on the brief.
        """
        if not self.llm.available:
            return {}
        text_assets = [
            (f.id, a.category, a.description)
            for f in concept.frames
            for a in f.assets
            if a.is_text
        ]
        if not text_assets:
            return {}
        system = (
            "You are a senior advertising copywriter. Given ad text-asset briefs, "
            "return punchy, on-brand copy. Respond ONLY as a JSON object mapping "
            "the index (as string) to the final short copy (<= 8 words)."
        )
        listing = "\n".join(f"{i}: [{c}] {d}" for i, (_, c, d) in enumerate(text_assets))
        raw = self.llm.complete(system, listing, json_mode=True, max_tokens=512)
        try:
            mapping = json.loads(raw) if raw else {}
        except Exception:
            log.warning("Planner: could not parse LLM copy JSON; keeping heuristic copy.")
            return {}
        out: dict[str, str] = {}
        for i, (fid, cat, _desc) in enumerate(text_assets):
            if str(i) in mapping and isinstance(mapping[str(i)], str):
                out[f"{fid}|{cat}"] = mapping[str(i)].strip()
        return out

    def apply_copy(self, concept: Concept) -> Concept:
        """Mutate text-asset ``.text`` in place with refined copy where available."""
        refined = self.refine_copy(concept)
        if not refined:
            return concept
        for f in concept.frames:
            for a in f.assets:
                if a.is_text and (key := f"{f.id}|{a.category}") in refined:
                    a.text = refined[key]
        return concept
