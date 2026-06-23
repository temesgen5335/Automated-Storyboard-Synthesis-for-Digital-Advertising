"""Compositor — paste generated assets into a single AdFrame image."""
from __future__ import annotations

from PIL import Image, ImageDraw

from ..formats import AdFormat
from ..schemas import BrandKit, ComposedFrame, Frame, GeneratedAsset, Placement
from .layout import plan_layout


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def compose_frame(
    frame: Frame,
    assets: list[GeneratedAsset],
    brand: BrandKit,
    fmt: AdFormat,
) -> ComposedFrame:
    canvas = Image.new("RGBA", (fmt.width, fmt.height), _hex(brand.secondary) + (255,))
    placements = plan_layout(assets, fmt)
    # plan_layout returns exactly one placement per asset, in asset order.
    paired: list[tuple[GeneratedAsset, Placement]] = list(zip(assets, placements))

    # paint in z-order (background first)
    for ga, p in sorted(paired, key=lambda t: t[1].z):
        tile = ga.image.convert("RGBA").resize((max(1, p.width), max(1, p.height)))
        canvas.alpha_composite(tile, (p.x, p.y))

    return ComposedFrame(frame=frame, image=canvas, placements=[p for _, p in paired], assets=assets)


def annotate_frame(cf: ComposedFrame, brand: BrandKit, *, show_grade: bool = False) -> Image.Image:
    """Return a copy with a caption bar (frame id + interaction) for the storyboard."""
    img = cf.image.convert("RGBA").copy()
    draw = ImageDraw.Draw(img)
    bar_h = max(18, img.height // 16)
    draw.rectangle([0, 0, img.width, bar_h], fill=_hex(brand.primary) + (235,))
    label = f"{cf.frame.id.replace('_', ' ').title()}  ·  {cf.frame.interaction_type}"
    if show_grade and cf.critique:
        label += f"  ·  {cf.critique['overall']:.2f}"
    draw.text((6, 3), label, fill=(255, 255, 255, 255))
    return img
