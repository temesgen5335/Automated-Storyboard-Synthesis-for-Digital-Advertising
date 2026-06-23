"""End-to-end orchestration: Concept → assets → frames → storyboard.

This is the spine of the Creative-Automation system. The DCO layer
(:meth:`StoryboardPipeline.run_dco`) produces multiple ranked variants and
multi-format adaptations from a single concept.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from .agents.critic import Critic
from .agents.planner import Planner
from .composition.compositor import compose_frame
from .config import Settings, get_settings
from .assembly.storyboard import assemble_storyboard
from .formats import AdFormat, get_format
from .generation.assets import AssetGenerator
from .providers import Providers
from .schemas import BrandKit, ComposedFrame, Concept, Storyboard

log = logging.getLogger(__name__)
ProgressFn = Callable[[str, float], None]


def _noop(msg: str, frac: float) -> None:  # default progress sink
    log.info("[%3d%%] %s", int(frac * 100), msg)


@dataclass
class DCOResult:
    storyboards: list[Storyboard]  # ranked best-first

    @property
    def best(self) -> Storyboard:
        return self.storyboards[0]


class StoryboardPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        brand: Optional[BrandKit] = None,
        fmt_key: str = "fs",
        variant: int = 0,
    ):
        self.settings = settings or get_settings()
        self.brand = brand or BrandKit()
        self.fmt: AdFormat = get_format(fmt_key)
        self.variant = variant
        self.providers = Providers(self.settings)
        self.planner = Planner(self.providers.llm)
        self.generator = AssetGenerator(self.providers.image, self.planner, self.brand, variant=variant)
        self.critic = Critic(self.providers.llm, self.brand)

    # -- single storyboard ----------------------------------------------------
    def run(self, concept: Concept, progress: ProgressFn = _noop, *, grade: bool = True) -> Storyboard:
        progress("Refining copy", 0.05)
        self.planner.apply_copy(concept)

        composed: list[ComposedFrame] = []
        n = max(1, len(concept.frames))
        for i, frame in enumerate(concept.frames):
            progress(f"Generating assets — {frame.id}", 0.1 + 0.7 * (i / n))
            assets = self.generator.generate_frame(frame, concept, self.fmt)
            progress(f"Composing — {frame.id}", 0.1 + 0.7 * ((i + 0.5) / n))
            composed.append(compose_frame(frame, assets, self.brand, self.fmt))

        grade_result = None
        if grade:
            progress("Grading creative", 0.85)
            grade_result = self.critic.grade_concept(concept, composed)

        progress("Assembling storyboard", 0.93)
        image = assemble_storyboard(concept, composed, self.brand, show_grades=grade)
        progress("Done", 1.0)

        return Storyboard(
            concept_name=concept.name,
            format_key=self.fmt.key,
            frames=composed,
            image=image,
            meta={
                "format": {"key": self.fmt.key, "w": self.fmt.width, "h": self.fmt.height},
                "providers": self.providers.describe(),
                "variant": self.variant,
                "branching": concept.is_branching,
                "grade": grade_result,
            },
        )

    # -- DCO: variants + multi-format ----------------------------------------
    def run_dco(
        self,
        raw_concept: dict,
        *,
        n_variants: int = 3,
        formats: Optional[list[str]] = None,
        brand_variants: Optional[list[BrandKit]] = None,
        progress: ProgressFn = _noop,
    ) -> DCOResult:
        """Generate ranked variants for a concept (the DCO core).

        Each variant draws a different asset-suggestion set and/or brand kit.
        Variants are scored by the critic and returned best-first.
        """
        from .data_loader import parse_concept, variant_count

        vcount = variant_count(raw_concept)
        formats = formats or [self.fmt.key]
        boards: list[Storyboard] = []
        total = n_variants * len(formats)
        done = 0
        for vi in range(n_variants):
            brand = (brand_variants[vi % len(brand_variants)] if brand_variants else self.brand)
            for fkey in formats:
                concept = parse_concept(raw_concept, variant=vi % vcount)
                sub = StoryboardPipeline(self.settings, brand=brand, fmt_key=fkey, variant=vi)

                def _p(msg: str, frac: float, _done=done) -> None:
                    progress(f"v{vi+1}/{fkey}: {msg}", (_done + frac) / total)

                boards.append(sub.run(concept, progress=_p))
                done += 1

        boards.sort(key=lambda b: (b.meta.get("grade") or {}).get("overall", 0.0), reverse=True)
        return DCOResult(storyboards=boards)
