"""Ad formats — the display dimensions a creative is rendered into.

The brief names two canonical formats (FS, MPU). We add the common IAB
mobile/display sizes so the DCO layer can adapt one concept to many placements.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdFormat:
    """A named display size. ``key`` is a stable slug used in filenames/URLs."""

    key: str
    label: str
    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def orientation(self) -> str:
        if self.width > self.height:
            return "landscape"
        if self.height > self.width:
            return "portrait"
        return "square"


# Canonical formats from the brief + common IAB display/mobile placements.
AD_FORMATS: dict[str, AdFormat] = {
    "fs": AdFormat("fs", "Full Screen (FS)", 320, 480),
    "mpu": AdFormat("mpu", "Mid-Page Unit (MPU)", 300, 250),
    "leaderboard": AdFormat("leaderboard", "Leaderboard", 728, 90),
    "skyscraper": AdFormat("skyscraper", "Wide Skyscraper", 160, 600),
    "square": AdFormat("square", "Square", 250, 250),
    "story": AdFormat("story", "Story / Vertical", 1080, 1920),
}

DEFAULT_FORMAT = "fs"


def get_format(key: str) -> AdFormat:
    key = (key or DEFAULT_FORMAT).lower()
    if key not in AD_FORMATS:
        raise KeyError(f"Unknown ad format {key!r}. Known: {sorted(AD_FORMATS)}")
    return AD_FORMATS[key]
