"""Prompt styling for the image-generation path.

Turns a thin asset brief into a rich, category-aware, brand-aware prompt.
Deterministic (no LLM required) so results are reproducible and cacheable;
the planner may still layer LLM enrichment on top.

Design notes
------------
* Palettes are passed as *color names*, not hex — diffusion models respond to
  "deep forest green", not "#2F6B52".
* Visual assets explicitly forbid text/lettering: all copy is rendered
  separately by PIL, and generated pseudo-text is the fastest way for a
  composited ad to look fake.
* Each category gets its own photographic/vector vocabulary; a generic
  "high quality" suffix upgrades nothing.
"""
from __future__ import annotations

from ..formats import AdFormat
from ..schemas import AssetBrief, BrandKit, Concept

PROMPT_VERSION = 2  # bump to invalidate provider caches when templates change

# ---------------------------------------------------------------------------
# hex -> nearest basic color name (models understand words, not hex)
# ---------------------------------------------------------------------------

_NAMED = {
    "red": (220, 40, 40), "crimson": (150, 30, 45), "orange": (245, 130, 30),
    "amber": (255, 190, 0), "yellow": (245, 220, 50), "olive": (128, 110, 50),
    "green": (60, 160, 75), "forest green": (40, 100, 70), "teal": (0, 130, 130),
    "cyan": (60, 200, 220), "sky blue": (110, 180, 240), "blue": (40, 90, 200),
    "navy": (25, 40, 90), "purple": (120, 60, 180), "magenta": (200, 50, 160),
    "pink": (240, 140, 180), "brown": (120, 80, 50), "black": (20, 20, 20),
    "charcoal": (55, 55, 60), "gray": (128, 128, 128), "white": (245, 245, 245),
    "cream": (245, 235, 210),
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def color_name(hex_color: str) -> str:
    try:
        r, g, b = _hex_to_rgb(hex_color)
    except Exception:
        return "brand color"
    return min(
        _NAMED,
        key=lambda n: sum((a - b) ** 2 for a, b in zip(_NAMED[n], (r, g, b))),
    )


def palette_phrase(brand: BrandKit) -> str:
    names = []
    for hx in brand.palette:
        n = color_name(hx)
        if n not in names:
            names.append(n)
    return ", ".join(names[:3])


# ---------------------------------------------------------------------------
# per-category templates
# ---------------------------------------------------------------------------

_NO_TEXT = "no text, no words, no letters, no watermark, no logo lettering"

# (keyword match on category, positive template)
# {desc} = brief description, {palette} = color names, {style} = brand keywords,
# {orient} = portrait/landscape/square
_TEMPLATES: list[tuple[tuple[str, ...], str]] = [
    (("background",),
     "{desc}, full-bleed advertising background, cinematic wide shot, no central "
     "subject, atmospheric depth, professional color grading in {palette} tones, "
     "{style}, soft gradient lighting, {orient} composition, negative space for "
     "overlaid copy, award-winning commercial photography, " + _NO_TEXT),
    (("logo", "badge"),
     "{desc}, minimal flat vector logo mark, geometric, bold silhouette, "
     "{palette} on clean solid background, crisp edges, centered, iconic, "
     "professional brand identity design, " + _NO_TEXT),
    (("product",),
     "{desc}, hero product photography, studio lighting with soft key and rim "
     "light, shallow depth of field, seamless {palette} backdrop, centered "
     "composition, ultra-sharp focus, commercial advertising shot, premium "
     "{style} feel, " + _NO_TEXT),
    (("mascot", "character"),
     "{desc}, friendly brand mascot character, full body, expressive pose, "
     "clean {style} character design, {palette} color scheme, flat shading "
     "with subtle depth, centered on simple background, " + _NO_TEXT),
    (("illustration",),
     "{desc}, modern flat vector illustration, {style}, harmonious {palette} "
     "palette, bold shapes, clean lines, editorial advertising style, "
     "{orient} composition, " + _NO_TEXT),
    (("icon",),
     "{desc}, simple flat UI icon, single subject, {palette}, bold rounded "
     "geometry, high contrast on plain background, pictogram style, " + _NO_TEXT),
    (("photo", "hero", "lifestyle", "person", "people"),
     "{desc}, cinematic lifestyle advertising photography, natural candid "
     "moment, golden-hour lighting, shallow depth of field, {palette} color "
     "grade, {style} mood, shot on medium format, {orient} framing, "
     "award-winning campaign photo, " + _NO_TEXT),
]

_DEFAULT = (
    "{desc}, premium advertising creative element, {style}, {palette} color "
    "harmony, clean composition, professional studio quality, {orient} "
    "orientation, " + _NO_TEXT
)


def build_prompt(brief: AssetBrief, concept: Concept, brand: BrandKit, fmt: AdFormat) -> str:
    cat = brief.category.lower()
    template = _DEFAULT
    for keys, tpl in _TEMPLATES:
        if any(k in cat for k in keys):
            template = tpl
            break
    return template.format(
        desc=brief.description.rstrip("."),
        palette=palette_phrase(brand),
        style=", ".join(brand.style_keywords[:4]),
        orient=fmt.orientation,
    )
