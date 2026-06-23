"""Data models for the AdSynth pipeline.

Two layers:
  * **Spec** models (pydantic) — JSON-serializable description of *what to build*:
    Concept, Frame, AssetBrief, BrandKit, Placement.
  * **Artifact** dataclasses — runtime objects that carry PIL images:
    GeneratedAsset, ComposedFrame, Storyboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------------------------
# Spec models (serializable)
# ----------------------------------------------------------------------------


class AssetBrief(BaseModel):
    """One asset to generate/place within a frame."""

    model_config = ConfigDict(extra="ignore")

    category: str = Field(description="One of the 23 ad-asset categories, e.g. 'Logo'.")
    description: str = Field(description="What the asset should depict.")
    text: Optional[str] = Field(
        default=None,
        description="Literal copy to render, for text assets (headline, CTA label, …).",
    )

    @property
    def is_text(self) -> bool:
        cat = self.category.lower()
        return self.text is not None or any(
            k in cat for k in ("text", "cta", "tagline", "headline", "legal", "disclaimer", "quote")
        )


class Frame(BaseModel):
    """A single AdFrame parsed from a concept's ``implementation``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    description: str = ""
    interaction_type: str = "Tap"
    next_frame: Optional[str] = None
    duration: Optional[str] = None
    assets: list[AssetBrief] = Field(default_factory=list)


class Concept(BaseModel):
    """A complete ad concept = the primary input to the pipeline."""

    model_config = ConfigDict(extra="ignore")

    name: str
    explanation: str = ""
    frames: list[Frame] = Field(default_factory=list)

    @property
    def is_branching(self) -> bool:
        """True if any frame points somewhere other than the next linear frame."""
        ids = [f.id for f in self.frames]
        for i, f in enumerate(self.frames):
            nxt = (f.next_frame or "").strip().lower()
            if not nxt or nxt.startswith("end"):
                continue
            expected = ids[i + 1] if i + 1 < len(ids) else None
            if expected and nxt != expected.lower():
                return True
        return False


class BrandKit(BaseModel):
    """Brand constraints that drive on-brand, consistent generation/composition."""

    model_config = ConfigDict(extra="ignore")

    name: str = "Generic"
    primary: str = "#1A73E8"          # hex
    secondary: str = "#FFFFFF"
    accent: str = "#FF6D00"
    text_color: str = "#111111"
    font: Optional[str] = None         # path to a .ttf, else PIL default
    style_keywords: list[str] = Field(
        default_factory=lambda: ["clean", "modern", "high-contrast", "professional"]
    )

    @property
    def palette(self) -> list[str]:
        return [self.primary, self.secondary, self.accent]


class Placement(BaseModel):
    """Where/how big an asset sits in a frame, in pixel coordinates."""

    model_config = ConfigDict(extra="ignore")

    category: str
    x: int
    y: int
    width: int
    height: int
    z: int = 0


# ----------------------------------------------------------------------------
# Artifact dataclasses (carry PIL images at runtime)
# ----------------------------------------------------------------------------


@dataclass
class GeneratedAsset:
    brief: AssetBrief
    image: Any  # PIL.Image.Image
    source: str = "mock"  # which provider produced it


@dataclass
class ComposedFrame:
    frame: Frame
    image: Any  # PIL.Image.Image
    placements: list[Placement] = field(default_factory=list)
    assets: list[GeneratedAsset] = field(default_factory=list)
    critique: Optional[dict] = None  # set by the critic agent


@dataclass
class Storyboard:
    concept_name: str
    format_key: str
    frames: list[ComposedFrame] = field(default_factory=list)
    image: Any = None  # the assembled storyboard PIL.Image.Image
    meta: dict = field(default_factory=dict)
